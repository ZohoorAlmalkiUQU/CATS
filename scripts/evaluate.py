from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable regardless of how the script is invoked.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import csv
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cats.data.dataset import EmbeddingDataset
from cats.data.collate import collate_embeddings
from cats.builders.model_builder import build_model
from cats.utils.metrics import (
    compute_spike_metrics,
    compute_routing_metrics,
    compute_prototype_metrics,
)
from cats.utils.seed import set_seed


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CATS model checkpoint on a data split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: direct checkpoint path
  python scripts/evaluate.py \\
      --checkpoint checkpoints/core_experiments/full_routing/cifar10/cats/carson/run_001/best_run_001.pt

  # Mode 2: experiment identifiers (resolves checkpoint automatically)
  python scripts/evaluate.py \\
      --main_experiment core_experiments \\
      --sub_experiment full_routing \\
      --dataset cifar10 \\
      --routing carson \\
      --run_name run_001

  # Evaluate on validation split instead of test
  python scripts/evaluate.py --checkpoint ... --split val

  # SST-2 test (inference-only; no ground-truth labels in GLUE benchmark):
  python scripts/evaluate.py \\
      --main_experiment core_experiments \\
      --sub_experiment full_routing \\
      --dataset sst2 \\
      --routing carson \\
      --run_name run_001
""",
    )

    ckpt = parser.add_argument_group("Checkpoint  (provide --checkpoint OR experiment args)")
    ckpt.add_argument("--checkpoint", type=str, default=None,
                      help="Direct path to a .pt checkpoint file.")
    ckpt.add_argument("--main_experiment", type=str, default=None,
                      help="Main experiment name, e.g. core_experiments or ablation_study.")
    ckpt.add_argument("--sub_experiment", type=str, default=None,
                      help="Sub-experiment name, e.g. full_routing or no_adaptive. Optional.")
    ckpt.add_argument("--dataset", type=str, default=None,
                      help="Dataset name: cifar10 | mnist | speech_commands | sst2 | all.")
    ckpt.add_argument("--routing", type=str, default=None,
                      help="Routing type: carson | identity | linear | all.")
    ckpt.add_argument("--run_name", type=str, default="run_001",
                      help="Run name (default: run_001).")
    ckpt.add_argument("--checkpoint_type", type=str, default="best",
                      choices=["best", "latest"],
                      help="Which checkpoint to load when using experiment args (default: best).")

    ev = parser.add_argument_group("Evaluation settings")
    ev.add_argument("--split", type=str, default="test",
                    choices=["test", "val", "train"],
                    help="Data split to evaluate on (default: test).")
    ev.add_argument("--batch_size", type=int, default=None,
                    help="Batch size override (default: from checkpoint config).")
    ev.add_argument("--device", type=str, default=None,
                    help="Device override, e.g. cuda or cpu (default: auto from config).")
    ev.add_argument("--output_dir", type=str, default=None,
                    help="Output directory override (default: auto-derived from checkpoint path).")

    return parser.parse_args()


# ============================================================
# Checkpoint resolution
# ============================================================

def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.resolve()

    required = {"main_experiment": args.main_experiment,
                "dataset": args.dataset,
                "routing": args.routing}
    missing = [f"--{k}" for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Provide either --checkpoint PATH or all experiment args.\n"
            f"Missing: {missing}"
        )

    base = Path("checkpoints")
    if args.sub_experiment:
        ckpt_dir = base / args.main_experiment / args.sub_experiment / args.dataset / "cats" / args.routing / args.run_name
    else:
        ckpt_dir = base / args.main_experiment / args.dataset / "cats" / args.routing / args.run_name

    ckpt_file = ckpt_dir / f"{args.checkpoint_type}_{args.run_name}.pt"
    if not ckpt_file.exists():
        # Build a helpful message showing what IS available under the experiment root.
        if args.sub_experiment:
            search_root = base / args.main_experiment / args.sub_experiment
        else:
            search_root = base / args.main_experiment
        available = sorted(search_root.rglob("best_*.pt")) if search_root.exists() else []
        avail_str = "\n  ".join(str(p) for p in available) if available else "(none found)"
        raise FileNotFoundError(
            f"\nCheckpoint not found: {ckpt_file}\n\n"
            f"Available checkpoints under '{search_root}':\n  {avail_str}\n"
        )
    return ckpt_file.resolve()


def infer_checkpoint_type(path: Path) -> str:
    stem = path.stem
    if stem.startswith("best"):
        return "best"
    if stem.startswith("latest"):
        return "latest"
    return "unknown"


def resolve_output_dir(
    checkpoint_path: Path,
    split: str,
    checkpoint_type: str,
    override: Optional[str],
) -> Path:
    if override:
        return Path(override)
    parts = checkpoint_path.parts
    for i, part in enumerate(parts):
        if part == "checkpoints":
            rel = parts[i + 1 : -1]  # strip 'checkpoints/' prefix and filename
            return Path("results").joinpath(*rel) / f"eval_{split}_{checkpoint_type}"
    # Fallback when 'checkpoints' not in path
    return Path("results") / checkpoint_path.parent.name / f"eval_{split}_{checkpoint_type}"


# ============================================================
# I/O helpers
# ============================================================

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_predictions_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


class BatchCSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def log(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ============================================================
# Classification metrics
# ============================================================

def _binary_cls_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _macro_cls_metrics(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2.0 * p * r / (p + r + 1e-8)
        precisions.append(p); recalls.append(r); f1s.append(f)
    return {
        "precision": float(sum(precisions) / len(precisions)),
        "recall": float(sum(recalls) / len(recalls)),
        "f1": float(sum(f1s) / len(f1s)),
    }


def _confusion_matrix(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> List[List[int]]:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def _per_class_report(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum().item())
        fp = int(((preds == c) & (labels != c)).sum().item())
        fn = int(((preds != c) & (labels == c)).sum().item())
        tn = int(((preds != c) & (labels != c)).sum().item())
        support = int((labels == c).sum().item())
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        report[str(c)] = {
            "precision": round(float(p), 6),
            "recall": round(float(r), 6),
            "f1": round(float(f), 6),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    return report


# ============================================================
# Evaluation loop
# ============================================================

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    batch_logger: BatchCSVLogger,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate the model on a dataloader.

    Returns:
        metrics: full aggregated metrics dict
        sample_records: per-sample prediction dicts (for predictions.csv)
    """
    model.eval()

    eval_start = time.perf_counter()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_tokens = 0
    labeled_samples = 0
    loss_batches = 0

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    spike_sums: Dict[str, float] = {}
    routing_sums: Dict[str, float] = {}
    proto_sums: Dict[str, float] = {}
    diag_sums: Dict[str, float] = {}

    spike_n = routing_n = proto_n = diag_n = 0

    sample_records: List[Dict[str, Any]] = []
    global_idx = 0

    pbar = tqdm(loader, desc="[eval]", leave=True, dynamic_ncols=False, ascii=True)

    for batch_idx, batch in enumerate(pbar):
        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)      # None or Tensor[B] (may contain -1)
        sentences = batch.get("sentences", None)

        bsz = embeddings.size(0)
        seq_len = embeddings.size(1)

        outputs = model(embeddings=embeddings, attention_mask=attention_mask)
        logits = outputs["logits"]               # [B, C]
        probs = F.softmax(logits.float(), dim=-1)  # [B, C]

        total_samples += bsz
        total_tokens += bsz * seq_len

        # ---------------------------------------------------
        # Loss (only for valid labels; SST-2 test has -1)
        # ---------------------------------------------------
        batch_loss: Optional[float] = None
        if labels is not None:
            valid_mask = labels >= 0
            n_valid = int(valid_mask.sum().item())
            labeled_samples += n_valid
            if n_valid > 0:
                l = criterion(logits[valid_mask].float(), labels[valid_mask])
                batch_loss = float(l.item())
                total_loss += batch_loss * n_valid
                loss_batches += 1

        # ---------------------------------------------------
        # Collect tensors for global metrics
        # ---------------------------------------------------
        all_logits.append(logits.detach().cpu())
        if labels is not None:
            all_labels.append(labels.detach().cpu())

        # ---------------------------------------------------
        # Per-sample records
        # ---------------------------------------------------
        pred_cls = logits.argmax(dim=-1)         # [B]
        conf = probs.max(dim=-1).values          # [B]

        for i in range(bsz):
            rec: Dict[str, Any] = {
                "sample_index": global_idx + i,
                "predicted_label": int(pred_cls[i].item()),
                "confidence": round(float(conf[i].item()), 6),
            }
            if labels is not None:
                rec["true_label"] = int(labels[i].item())
            for c in range(num_classes):
                rec[f"prob_class_{c}"] = round(float(probs[i, c].item()), 6)
            if sentences is not None:
                rec["sentence"] = sentences[i]
            sample_records.append(rec)

        global_idx += bsz

        # ---------------------------------------------------
        # Model internals
        # ---------------------------------------------------
        spikes = outputs.get("spikes", None)
        group_assignments = outputs.get("group_assignments", None)
        if spikes is not None:
            sm = compute_spike_metrics(spikes, attention_mask, group_assignments)
            for k, v in sm.items():
                spike_sums[k] = spike_sums.get(k, 0.0) + v
            spike_n += 1

        routing_weights = outputs.get("routing_weights", None)
        if routing_weights is not None:
            rm = compute_routing_metrics(
                routing_weights, attention_mask,
                outputs.get("routing_confidence", None),
                outputs.get("agreement", None),
            )
            for k, v in rm.items():
                routing_sums[k] = routing_sums.get(k, 0.0) + v
            routing_n += 1

        gp = outputs.get("group_prototypes", None)
        if gp is not None:
            pm = compute_prototype_metrics(gp)
            for k, v in pm.items():
                proto_sums[k] = proto_sums.get(k, 0.0) + v
            proto_n += 1

        _DIAG_KEYS = [
            ("exc_tau", "exc_tau"),
            ("inh_tau", "inh_tau"),
            ("exc_threshold", "exc_threshold"),
            ("inh_threshold", "inh_threshold"),
            ("exc_effective_threshold", "exc_effective_threshold"),
            ("inh_effective_threshold", "inh_effective_threshold"),
            ("exc_adaptive_threshold", "exc_adaptive_threshold"),
            ("inh_adaptive_threshold", "inh_adaptive_threshold"),
        ]
        for out_key, prefix in _DIAG_KEYS:
            t = outputs.get(out_key, None)
            if t is not None and torch.is_tensor(t) and t.numel() > 0:
                tf = t.detach().float()
                diag_sums[f"{prefix}_mean"] = diag_sums.get(f"{prefix}_mean", 0.0) + float(tf.mean())
                diag_sums[f"{prefix}_std"] = diag_sums.get(f"{prefix}_std", 0.0) + float(tf.std(unbiased=False))
        diag_n += 1

        # ---------------------------------------------------
        # Progress bar + batch CSV log
        # ---------------------------------------------------
        pf: Dict[str, str] = {}
        if batch_loss is not None:
            pf["loss"] = f"{batch_loss:.4f}"
        if spike_n > 0 and "firing_rate" in spike_sums:
            pf["spk_rt"] = f"{spike_sums['firing_rate'] / spike_n:.4f}"
        if routing_n > 0 and "routing_entropy" in routing_sums:
            pf["H"] = f"{routing_sums['routing_entropy'] / routing_n:.3f}"
        if routing_n > 0 and "routing_conf_mean" in routing_sums:
            pf["r_conf"] = f"{routing_sums['routing_conf_mean'] / routing_n:.3f}"
        pbar.set_postfix(pf)

        batch_row: Dict[str, Any] = {
            "batch_idx": batch_idx,
            "batch_size": bsz,
            "cumulative_samples": total_samples,
            "batch_loss": round(batch_loss, 6) if batch_loss is not None else None,
        }
        if spike_n > 0:
            batch_row["running_spk_rt"] = round(spike_sums.get("firing_rate", 0.0) / spike_n, 6)
            batch_row["running_exc_rt"] = round(spike_sums.get("exc_firing_rate", 0.0) / spike_n, 6)
            batch_row["running_inh_rt"] = round(spike_sums.get("inh_firing_rate", 0.0) / spike_n, 6)
            batch_row["running_spk_tok"] = round(spike_sums.get("spikes_per_token", 0.0) / spike_n, 4)
        if routing_n > 0:
            batch_row["running_routing_entropy"] = round(routing_sums.get("routing_entropy", 0.0) / routing_n, 6)
            batch_row["running_routing_conf_mean"] = round(routing_sums.get("routing_conf_mean", 0.0) / routing_n, 6)
            batch_row["running_max_weight_mean"] = round(routing_sums.get("max_weight_mean", 0.0) / routing_n, 6)
            batch_row["running_dominance_fraction"] = round(routing_sums.get("dominance_fraction", 0.0) / routing_n, 6)
        if diag_n > 0:
            for k in ("exc_tau_mean", "inh_tau_mean", "exc_threshold_mean", "inh_threshold_mean",
                      "exc_effective_threshold_mean", "inh_effective_threshold_mean"):
                if k in diag_sums:
                    batch_row[f"running_{k}"] = round(diag_sums[k] / diag_n, 6)
        batch_logger.log(batch_row)

    # ==============================================================
    # Aggregate
    # ==============================================================
    elapsed = time.perf_counter() - eval_start
    unlabeled = labeled_samples == 0

    metrics: Dict[str, Any] = {
        "total_samples": total_samples,
        "labeled_samples": labeled_samples,
        "unlabeled": unlabeled,
        "eval_time_sec": round(elapsed, 4),
        "samples_per_sec": round(total_samples / max(elapsed, 1e-8), 2),
        "tokens_per_sec": round(total_tokens / max(elapsed, 1e-8), 2),
    }

    # -----------------------------------------------------------
    # Classification metrics (skip for unlabeled SST-2 test)
    # -----------------------------------------------------------
    if not unlabeled:
        metrics["loss"] = round(total_loss / max(labeled_samples, 1), 6)

        logits_all = torch.cat(all_logits, dim=0)
        labels_all = torch.cat(all_labels, dim=0)
        valid = labels_all >= 0
        lv = logits_all[valid]
        yv = labels_all[valid]
        pv = lv.argmax(dim=-1)

        metrics["accuracy"] = round(float((pv == yv).float().mean().item()), 6)

        cls = _binary_cls_metrics(pv, yv) if num_classes == 2 else _macro_cls_metrics(pv, yv, num_classes)
        metrics.update({k: round(v, 6) for k, v in cls.items()})

        metrics["confusion_matrix"] = _confusion_matrix(pv, yv, num_classes)
        metrics["per_class"] = _per_class_report(pv, yv, num_classes)

    # -----------------------------------------------------------
    # SNN / routing / prototype / diagnostic internals
    # -----------------------------------------------------------
    if spike_n > 0:
        for k, v in spike_sums.items():
            metrics[f"test_{k}"] = round(v / spike_n, 6)

    if routing_n > 0:
        for k, v in routing_sums.items():
            metrics[f"test_{k}"] = round(v / routing_n, 6)

    if proto_n > 0:
        for k, v in proto_sums.items():
            metrics[f"test_{k}"] = round(v / proto_n, 6)

    if diag_n > 0:
        for k, v in diag_sums.items():
            metrics[f"test_{k}"] = round(v / diag_n, 6)

    return metrics, sample_records


# ============================================================
# All-experiment constants
# ============================================================

_ALL_DATASETS = ["cifar10", "mnist", "speech_commands", "sst2"]
_ALL_ROUTINGS  = ["carson", "identity", "linear"]


# ============================================================
# Single-experiment evaluation (core logic)
# ============================================================

def evaluate_one(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Run evaluation for a single (dataset, routing) combination.

    Returns a compact result dict for the multi-run summary table,
    or None if the checkpoint does not exist (skip silently).
    """

    # ----------------------------------------------------------
    # Resolve checkpoint
    # ----------------------------------------------------------
    checkpoint_path = resolve_checkpoint_path(args)
    checkpoint_type = (
        infer_checkpoint_type(checkpoint_path)
        if args.checkpoint is not None
        else args.checkpoint_type
    )

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------
    device_name = args.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    # ----------------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "config"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: '{key}'")

    config: Dict[str, Any] = checkpoint["config"]
    best_val_f1: float = float(checkpoint.get("best_val_f1", float("nan")))
    best_epoch: int = int(checkpoint.get("best_epoch", -1))
    train_epoch: int = int(checkpoint.get("epoch", -1))

    # ----------------------------------------------------------
    # Config sections
    # ----------------------------------------------------------
    experiment_cfg = config.get("experiment", {})
    data_cfg       = config.get("data", {})
    model_cfg      = config.get("model", {})
    routing_cfg    = config.get("routing", {})
    lif_exc_cfg    = config.get("lif_exc", {})
    lif_inh_cfg    = config.get("lif_inh", {})
    classifier_cfg = config.get("classifier", {})
    training_cfg   = config.get("training", {})
    position_cfg   = config.get("position", {})

    dataset_name  = experiment_cfg.get("dataset_name", "unknown")
    model_name    = experiment_cfg.get("model_name", "cats")
    routing_type  = experiment_cfg.get("routing_type", "unknown")
    run_name      = experiment_cfg.get("run_name", "run_001")
    seed          = int(training_cfg.get("seed", 42))
    num_classes   = int(model_cfg.get("num_classes", 2))

    set_seed(seed, deterministic=False)

    # ----------------------------------------------------------
    # Data path for the requested split
    # ----------------------------------------------------------
    split = args.split
    processed_root = data_cfg.get("processed_root", "data/processed")
    _split_map = {"test": "test_split", "val": "val_split", "train": "train_split"}
    split_folder = data_cfg.get(_split_map[split], split)
    data_path = str(Path(processed_root) / dataset_name / split_folder)

    # SST-2 test: GLUE withholds labels (stored as -1 sentinel)
    sst2_test_mode = dataset_name == "sst2" and split == "test"

    batch_size  = args.batch_size if args.batch_size is not None else int(training_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory  = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"
    max_cached  = int(training_cfg.get("max_cached_shards", 6))

    dataset = EmbeddingDataset(data_path, max_cached_shards=max_cached)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_embeddings,
        drop_last=False,
    )

    # ----------------------------------------------------------
    # Build model and load weights
    # ----------------------------------------------------------
    # Alias map for routing types that were renamed across code versions.
    # carson_v1 / carson_v2 → carson (current unified router).
    _ROUTING_ALIASES: Dict[str, str] = {"carson_v1": "carson", "carson_v2": "carson"}
    orig_routing_type = experiment_cfg.get("routing_type", "")
    if orig_routing_type in _ROUTING_ALIASES:
        aliased = _ROUTING_ALIASES[orig_routing_type]
        print(
            f"\n[WARN] Checkpoint routing_type='{orig_routing_type}' is a legacy alias "
            f"-> remapping to '{aliased}' for model construction.\n"
            f"       Weight compatibility is not guaranteed; any mismatch will be reported.\n"
        )
        experiment_cfg = dict(experiment_cfg)
        experiment_cfg["routing_type"] = aliased

    model = build_model(
        experiment_cfg=experiment_cfg,
        model_cfg=model_cfg,
        routing_cfg=routing_cfg,
        lif_exc_cfg=lif_exc_cfg,
        lif_inh_cfg=lif_inh_cfg,
        classifier_cfg=classifier_cfg,
        position_cfg=position_cfg,
    ).to(device)

    # Load weights — strict first; fall back with a detailed mismatch report.
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
        weights_loaded = "strict"
    except RuntimeError as strict_err:
        print(f"\n[WARN] Strict weight loading failed:\n  {strict_err}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing  = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
        print(f"[WARN] Loaded with strict=False.")
        if missing:
            print(f"  Missing keys  ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        if missing or unexpected:
            print("[WARN] Evaluation results may be unreliable due to weight mismatch.\n")
        weights_loaded = f"non-strict (missing={len(missing)}, unexpected={len(unexpected)})"

    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    # ----------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------
    output_dir = resolve_output_dir(checkpoint_path, split, checkpoint_type, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------
    print("=" * 100)
    print("CATS Evaluation Script")
    print("=" * 100)
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Checkpoint type : {checkpoint_type}")
    print(f"Dataset         : {dataset_name}")
    print(f"Split           : {split}")
    print(f"Routing         : {routing_type}")
    print(f"Run             : {run_name}")
    print(f"Device          : {device}")
    print(f"Data path       : {data_path}")
    print(f"Samples         : {len(dataset)}")
    print(f"Batch size      : {batch_size}")
    print(f"Parameters      : {param_count:,}")
    print(f"Output dir      : {output_dir}")
    print(f"Training best   : val_f1={best_val_f1:.6f}  epoch={best_epoch}  (last epoch={train_epoch})")
    if sst2_test_mode:
        print()
        print("  [SST-2 TEST] GLUE benchmark withholds test labels.")
        print("  Running in INFERENCE-ONLY mode — no classification metrics will be computed.")
        print("  Predictions saved to predictions.csv (format matches GLUE submission).")
    print("=" * 100)

    # ----------------------------------------------------------
    # Save config and run info
    # ----------------------------------------------------------
    save_json(config, output_dir / "config.json")

    run_info: Dict[str, Any] = {
        "evaluated_at":           datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path":        str(checkpoint_path),
        "checkpoint_type":        checkpoint_type,
        "split":                  split,
        "inference_only":         sst2_test_mode,
        "dataset_name":           dataset_name,
        "model_name":             model_name,
        "routing_type":           routing_type,
        "run_name":               run_name,
        "device":                 str(device),
        "data_path":              data_path,
        "num_samples":            len(dataset),
        "batch_size":             batch_size,
        "num_classes":            num_classes,
        "num_parameters":         param_count,
        "seed":                   seed,
        "training_best_val_f1":   best_val_f1,
        "training_best_epoch":    best_epoch,
        "training_last_epoch":    train_epoch,
        "weights_loaded":         weights_loaded,
        "orig_routing_type":      orig_routing_type,
        "output_dir":             str(output_dir),
    }
    save_json(run_info, output_dir / "run_info.json")

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    batch_logger = BatchCSVLogger(output_dir / "eval_batch_log.csv")

    metrics, sample_records = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
        batch_logger=batch_logger,
    )

    # ----------------------------------------------------------
    # Results summary to stdout
    # ----------------------------------------------------------
    print()
    print("=" * 100)
    print("Evaluation Results")
    print("=" * 100)

    if metrics["unlabeled"]:
        print(f"  Mode             : INFERENCE ONLY (no ground-truth labels)")
        print(f"  Samples          : {metrics['total_samples']}")
        print(f"  Eval time        : {metrics['eval_time_sec']:.2f}s")
        print(f"  Samples/sec      : {metrics['samples_per_sec']:.1f}")
    else:
        print(f"  Split            : {split}")
        print(f"  Samples          : {metrics['labeled_samples']}")
        print(f"  Loss             : {metrics.get('loss', float('nan')):.6f}")
        print(f"  Accuracy         : {metrics.get('accuracy', float('nan')):.6f}")
        print(f"  Precision (macro): {metrics.get('precision', float('nan')):.6f}")
        print(f"  Recall    (macro): {metrics.get('recall', float('nan')):.6f}")
        print(f"  F1        (macro): {metrics.get('f1', float('nan')):.6f}")
        print(f"  Eval time        : {metrics['eval_time_sec']:.2f}s")
        print(f"  Samples/sec      : {metrics['samples_per_sec']:.1f}")

        if "confusion_matrix" in metrics:
            print()
            print("  Confusion matrix (rows=true, cols=pred):")
            for row_idx, row in enumerate(metrics["confusion_matrix"]):
                print(f"    class {row_idx}: {row}")

        if "per_class" in metrics:
            print()
            print(f"  {'Class':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>10}")
            for cls_id, cls_m in metrics["per_class"].items():
                print(f"  {cls_id:<8} {cls_m['precision']:>8.4f} {cls_m['recall']:>8.4f} {cls_m['f1']:>8.4f} {cls_m['support']:>10}")

    internals = {k: v for k, v in metrics.items() if k.startswith("test_") and not k.endswith("_std")}
    if internals:
        print()
        print("  --- SNN / Routing internals ---")
        for k in sorted(internals):
            print(f"  {k:<45}: {internals[k]:.6f}")

    print("=" * 100)

    # ----------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------

    # Pop nested objects for flat eval_summary; save them separately too
    confusion_matrix = metrics.pop("confusion_matrix", None)
    per_class        = metrics.pop("per_class", None)

    eval_summary = {**run_info, "metrics": metrics}
    save_json(eval_summary, output_dir / "eval_summary.json")

    if confusion_matrix is not None:
        save_json(
            {"confusion_matrix": confusion_matrix, "num_classes": num_classes},
            output_dir / "confusion_matrix.json",
        )

    if per_class is not None:
        save_json(per_class, output_dir / "class_report.json")

    # Predictions CSV (always; critical for SST-2 inference mode)
    save_predictions_csv(sample_records, output_dir / "predictions.csv")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print()
    print("Saved outputs:")
    for fname in ["config.json", "run_info.json", "eval_summary.json",
                  "eval_batch_log.csv", "predictions.csv",
                  "confusion_matrix.json", "class_report.json"]:
        fp = output_dir / fname
        if fp.exists():
            size_kb = fp.stat().st_size / 1024
            print(f"  {fname:<30} ({size_kb:.1f} KB)")

    print()
    print(f"All results saved to: {output_dir}")
    print("=" * 100)

    return {
        "dataset":        dataset_name,
        "routing":        routing_type,
        "run_name":       run_name,
        "split":          split,
        "checkpoint":     str(checkpoint_path),
        "unlabeled":      metrics.get("unlabeled", False),
        "accuracy":       metrics.get("accuracy", None),
        "f1":             metrics.get("f1", None),
        "precision":      metrics.get("precision", None),
        "recall":         metrics.get("recall", None),
        "loss":           metrics.get("loss", None),
        "output_dir":     str(output_dir),
    }


# ============================================================
# Main — expands "all" and loops over combinations
# ============================================================

def main() -> None:
    args = parse_args()

    # "all" is only meaningful in experiment-args mode
    if args.checkpoint is not None:
        evaluate_one(args)
        return

    datasets = _ALL_DATASETS if args.dataset == "all" else [args.dataset]
    routings  = _ALL_ROUTINGS  if args.routing  == "all" else [args.routing]

    combos = [(ds, rt) for ds in datasets for rt in routings]
    total  = len(combos)

    if total == 1:
        # Single run — behave exactly like before
        evaluate_one(args)
        return

    # ----------------------------------------------------------
    # Multi-run loop
    # ----------------------------------------------------------
    import copy
    results: List[Dict[str, Any]] = []
    failed:  List[str]            = []

    print("=" * 100)
    print(f"CATS Batch Evaluation  ({total} experiments)")
    print(f"  main_experiment : {args.main_experiment}")
    print(f"  sub_experiment  : {args.sub_experiment}")
    print(f"  datasets        : {datasets}")
    print(f"  routings        : {routings}")
    print(f"  split           : {args.split}")
    print(f"  checkpoint_type : {args.checkpoint_type}")
    print("=" * 100)

    for idx, (ds, rt) in enumerate(combos, 1):
        print(f"\n[{idx}/{total}] {ds} / {rt}")
        print("-" * 60)
        run_args = copy.copy(args)
        run_args.dataset = ds
        run_args.routing = rt
        try:
            result = evaluate_one(run_args)
            if result is not None:
                results.append(result)
        except FileNotFoundError as e:
            msg = f"[SKIP] {ds}/{rt} — checkpoint not found: {e}"
            print(msg)
            failed.append(f"{ds}/{rt}")
        except Exception as e:
            msg = f"[ERROR] {ds}/{rt} — {type(e).__name__}: {e}"
            print(msg)
            failed.append(f"{ds}/{rt}")

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------
    print()
    print("=" * 100)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 100)
    header = f"{'Dataset':<20} {'Routing':<12} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Loss':>10}  {'Split'}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["unlabeled"]:
            row = (f"{r['dataset']:<20} {r['routing']:<12}"
                   f"  {'(inference-only — SST-2 test, no labels)'}")
        else:
            acc  = f"{r['accuracy']:.4f}"  if r["accuracy"]  is not None else "  N/A"
            f1   = f"{r['f1']:.4f}"        if r["f1"]        is not None else "  N/A"
            prec = f"{r['precision']:.4f}" if r["precision"]  is not None else "  N/A"
            rec  = f"{r['recall']:.4f}"    if r["recall"]     is not None else "  N/A"
            loss = f"{r['loss']:.4f}"      if r["loss"]       is not None else "  N/A"
            row  = (f"{r['dataset']:<20} {r['routing']:<12}"
                    f" {acc:>8} {f1:>8} {prec:>8} {rec:>8} {loss:>10}  {r['split']}")
        print(row)

    if failed:
        print()
        print(f"Skipped / failed ({len(failed)}): {failed}")

    print()
    print(f"Completed {len(results)}/{total} experiments.")
    print("=" * 100)


if __name__ == "__main__":
    main()
