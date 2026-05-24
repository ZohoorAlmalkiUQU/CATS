from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make src/ importable when this file is placed at tests/diagnostics/check_evaluate.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cats.builders.model_builder import build_model
from cats.data.collate import collate_embeddings
from cats.data.dataset import EmbeddingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic evaluation checker for CATS checkpoints."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to diagnose.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "val", "test", "both"],
        help="Which split to evaluate. Use 'both' for val + test.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override. Defaults to checkpoint training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cuda or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional limit for quick diagnostics. Use full split if omitted.",
    )
    parser.add_argument(
        "--require_explicit_classifier",
        action="store_true",
        help="Fail if classifier config is missing from the checkpoint.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the diagnostic JSON report.",
    )

    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    required_keys = ["model_state_dict", "config"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")

    return checkpoint


def summarize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    classifier_cfg = config.get("classifier", {})
    routing_cfg = config.get("routing", {})
    position_cfg = config.get("position", {})

    return {
        "dataset_name": experiment_cfg.get("dataset_name"),
        "routing_type": experiment_cfg.get("routing_type"),
        "run_name": experiment_cfg.get("run_name"),
        "num_classes": model_cfg.get("num_classes"),
        "embedding_dim": model_cfg.get("embedding_dim"),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "spiking_enabled": model_cfg.get("spiking", {}).get("enabled"),
        "inhibition_enabled": model_cfg.get("inhibition", {}).get("enabled"),
        "classifier_present": bool(classifier_cfg),
        "classifier_cfg": classifier_cfg,
        "routing_cfg_keys": sorted(list(routing_cfg.keys())),
        "position_cfg": position_cfg,
        "processed_root": data_cfg.get("processed_root"),
        "train_split": data_cfg.get("train_split"),
        "val_split": data_cfg.get("val_split"),
        "test_split": data_cfg.get("test_split"),
    }


def build_and_load_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
    require_explicit_classifier: bool,
) -> Tuple[nn.Module, Dict[str, Any]]:
    config = checkpoint["config"]

    experiment_cfg = dict(config.get("experiment", {}))
    model_cfg = config.get("model", {})
    routing_cfg = config.get("routing", {})
    lif_exc_cfg = config.get("lif_exc", {})
    lif_inh_cfg = config.get("lif_inh", {})
    classifier_cfg = config.get("classifier", {})
    position_cfg = config.get("position", {})

    if require_explicit_classifier and not classifier_cfg:
        raise ValueError(
            "Classifier config is missing. Add an explicit classifier section, "
            "for example: classifier: {type: ann}"
        )

    if not classifier_cfg:
        print("[WARN] classifier config is missing. The model builder will use its default.")

    routing_aliases = {"carson_v1": "carson", "carson_v2": "carson"}
    original_routing = experiment_cfg.get("routing_type", None)
    if original_routing in routing_aliases:
        aliased = routing_aliases[original_routing]
        print(f"[WARN] Legacy routing alias detected: {original_routing} -> {aliased}")
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

    state_dict = checkpoint["model_state_dict"]

    # Diagnostic mode must fail on mismatch. No non-strict loading is allowed here.
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    load_info = {
        "weights_loaded": "strict",
        "original_routing_type": original_routing,
        "effective_routing_type": experiment_cfg.get("routing_type"),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    return model, load_info


def resolve_split_path(config: Dict[str, Any], split: str) -> Path:
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})

    dataset_name = experiment_cfg.get("dataset_name")
    processed_root = data_cfg.get("processed_root", "data/processed")

    split_key = {
        "train": "train_split",
        "val": "val_split",
        "test": "test_split",
    }[split]

    split_folder = data_cfg.get(split_key, split)
    return Path(processed_root) / str(dataset_name) / str(split_folder)


def inspect_dataset(dataset: EmbeddingDataset, max_items: int = 2000) -> Dict[str, Any]:
    n = len(dataset)
    indices = list(range(min(n, max_items)))

    labels: List[int] = []
    seq_lens: List[int] = []
    embedding_means: List[float] = []
    embedding_stds: List[float] = []

    first_keys = None
    first_shapes: Dict[str, Any] = {}

    for idx in indices:
        item = dataset[idx]

        if first_keys is None:
            first_keys = sorted(list(item.keys()))
            for k, v in item.items():
                if torch.is_tensor(v):
                    first_shapes[k] = list(v.shape)
                else:
                    first_shapes[k] = type(v).__name__

        if "labels" in item:
            label_value = item["labels"]
            if torch.is_tensor(label_value):
                label_int = int(label_value.item())
            else:
                label_int = int(label_value)
            labels.append(label_int)

        emb = item.get("embeddings", None)
        if torch.is_tensor(emb):
            seq_lens.append(int(emb.shape[0]))
            embedding_means.append(float(emb.float().mean().item()))
            embedding_stds.append(float(emb.float().std(unbiased=False).item()))

    label_counter = Counter(labels)

    return {
        "num_samples": n,
        "inspected_items": len(indices),
        "first_item_keys": first_keys,
        "first_item_shapes": first_shapes,
        "label_distribution_sample": dict(sorted(label_counter.items())),
        "num_unique_labels_sample": len(label_counter),
        "seq_len_min_sample": min(seq_lens) if seq_lens else None,
        "seq_len_max_sample": max(seq_lens) if seq_lens else None,
        "seq_len_mean_sample": sum(seq_lens) / len(seq_lens) if seq_lens else None,
        "embedding_mean_avg_sample": sum(embedding_means) / len(embedding_means) if embedding_means else None,
        "embedding_std_avg_sample": sum(embedding_stds) / len(embedding_stds) if embedding_stds else None,
    }


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    data_path: Path,
    config: Dict[str, Any],
    split: str,
    device: torch.device,
    batch_size_override: Optional[int],
    max_batches: Optional[int],
) -> Dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist for split={split}: {data_path}")

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    batch_size = batch_size_override or int(training_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"
    max_cached = int(training_cfg.get("max_cached_shards", 6))
    num_classes = int(model_cfg.get("num_classes", 2))

    dataset = EmbeddingDataset(str(data_path), max_cached_shards=max_cached)
    dataset_report = inspect_dataset(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_embeddings,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_valid = 0
    total_seen = 0

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_conf: List[torch.Tensor] = []

    output_keys = None
    first_batch_shapes: Dict[str, Any] = {}

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)

        outputs = model(embeddings=embeddings, attention_mask=attention_mask)

        if output_keys is None:
            output_keys = sorted(list(outputs.keys()))
            first_batch_shapes["embeddings"] = list(embeddings.shape)
            first_batch_shapes["attention_mask"] = list(attention_mask.shape)
            for k, v in outputs.items():
                if torch.is_tensor(v):
                    first_batch_shapes[f"outputs.{k}"] = list(v.shape)
                else:
                    first_batch_shapes[f"outputs.{k}"] = type(v).__name__

        logits = outputs["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        conf = probs.max(dim=-1).values

        total_seen += int(embeddings.size(0))

        if labels is not None:
            valid = labels >= 0
            if int(valid.sum().item()) > 0:
                loss = criterion(logits[valid], labels[valid])
                total_loss += float(loss.item()) * int(valid.sum().item())
                total_valid += int(valid.sum().item())

                all_preds.append(preds[valid].detach().cpu())
                all_labels.append(labels[valid].detach().cpu())
                all_conf.append(conf[valid].detach().cpu())

    report: Dict[str, Any] = {
        "split": split,
        "data_path": str(data_path),
        "dataset_report": dataset_report,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "evaluated_samples": total_seen,
        "valid_labeled_samples": total_valid,
        "model_output_keys": output_keys,
        "first_batch_shapes": first_batch_shapes,
    }

    if total_valid == 0:
        report["unlabeled"] = True
        return report

    preds_all = torch.cat(all_preds)
    labels_all = torch.cat(all_labels)
    conf_all = torch.cat(all_conf)

    accuracy = float((preds_all == labels_all).float().mean().item())
    loss = total_loss / max(total_valid, 1)

    pred_counter = Counter(preds_all.tolist())
    label_counter = Counter(labels_all.tolist())

    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels_all.tolist(), preds_all.tolist()):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            confusion[y][p] += 1

    per_class_recall: Dict[str, float] = {}
    for c in range(num_classes):
        support = sum(confusion[c])
        correct = confusion[c][c]
        per_class_recall[str(c)] = correct / support if support > 0 else 0.0

    macro_recall = sum(per_class_recall.values()) / max(num_classes, 1)

    # Macro F1
    per_class_f1: Dict[str, float] = {}
    for c in range(num_classes):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(num_classes) if r != c)
        fn = sum(confusion[c][r] for r in range(num_classes) if r != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[str(c)] = f1

    macro_f1 = sum(per_class_f1.values()) / max(num_classes, 1)

    report.update(
        {
            "unlabeled": False,
            "loss": loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
            "label_distribution": dict(sorted(label_counter.items())),
            "prediction_distribution": dict(sorted(pred_counter.items())),
            "confidence_mean": float(conf_all.mean().item()),
            "confidence_min": float(conf_all.min().item()),
            "confidence_max": float(conf_all.max().item()),
            "confusion_matrix": confusion,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
        }
    )

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("CATS EVALUATION DIAGNOSTIC REPORT")
    print("=" * 100)

    print("\n[Checkpoint]")
    for k, v in report["checkpoint"].items():
        print(f"  {k}: {v}")

    print("\n[Config summary]")
    for k, v in report["config_summary"].items():
        print(f"  {k}: {v}")

    print("\n[Model load]")
    for k, v in report["model_load"].items():
        print(f"  {k}: {v}")

    for split_name, split_report in report["splits"].items():
        print("\n" + "-" * 100)
        print(f"[Split: {split_name}]")
        print(f"  data_path: {split_report['data_path']}")
        print(f"  evaluated_samples: {split_report['evaluated_samples']}")
        print(f"  valid_labeled_samples: {split_report['valid_labeled_samples']}")
        print(f"  unlabeled: {split_report.get('unlabeled')}")

        dsr = split_report["dataset_report"]
        print("  dataset:")
        print(f"    num_samples: {dsr['num_samples']}")
        print(f"    first_item_keys: {dsr['first_item_keys']}")
        print(f"    first_item_shapes: {dsr['first_item_shapes']}")
        print(f"    label_distribution_sample: {dsr['label_distribution_sample']}")
        print(f"    seq_len_min/max/mean: {dsr['seq_len_min_sample']} / {dsr['seq_len_max_sample']} / {dsr['seq_len_mean_sample']}")
        print(f"    embedding_mean_avg_sample: {dsr['embedding_mean_avg_sample']}")
        print(f"    embedding_std_avg_sample: {dsr['embedding_std_avg_sample']}")

        print("  first batch/model:")
        print(f"    model_output_keys: {split_report['model_output_keys']}")
        print(f"    first_batch_shapes: {split_report['first_batch_shapes']}")

        if not split_report.get("unlabeled", False):
            print("  metrics:")
            print(f"    loss: {split_report['loss']:.6f}")
            print(f"    accuracy: {split_report['accuracy']:.6f}")
            print(f"    macro_f1: {split_report['macro_f1']:.6f}")
            print(f"    macro_recall: {split_report['macro_recall']:.6f}")
            print(f"    confidence_mean: {split_report['confidence_mean']:.6f}")
            print(f"    label_distribution: {split_report['label_distribution']}")
            print(f"    prediction_distribution: {split_report['prediction_distribution']}")
            print(f"    per_class_recall: {split_report['per_class_recall']}")

    print("\n" + "=" * 100)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]

    model, load_info = build_and_load_model(
        checkpoint=checkpoint,
        device=device,
        require_explicit_classifier=args.require_explicit_classifier,
    )

    checkpoint_report = {
        "path": str(checkpoint_path),
        "device": str(device),
        "best_val_f1": checkpoint.get("best_val_f1"),
        "best_epoch": checkpoint.get("best_epoch"),
        "last_epoch": checkpoint.get("epoch"),
        "has_model_state_dict": "model_state_dict" in checkpoint,
        "has_config": "config" in checkpoint,
    }

    split_names = ["val", "test"] if args.split == "both" else [args.split]

    split_reports: Dict[str, Any] = {}
    for split in split_names:
        data_path = resolve_split_path(config, split)
        split_reports[split] = evaluate_split(
            model=model,
            data_path=data_path,
            config=config,
            split=split,
            device=device,
            batch_size_override=args.batch_size,
            max_batches=args.max_batches,
        )

    report = {
        "checkpoint": checkpoint_report,
        "config_summary": summarize_config(config),
        "model_load": load_info,
        "splits": split_reports,
    }

    print_report(report)

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved diagnostic report to: {out_path}")


if __name__ == "__main__":
    main()