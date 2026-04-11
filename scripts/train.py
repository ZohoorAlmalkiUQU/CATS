from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cats.data.dataset import EmbeddingDataset
from cats.data.collate import collate_embeddings
from cats.utils.config import load_config
from cats.builders.model_builder import build_model
from cats.utils.metrics import (
    accuracy_from_logits,
    binary_precision_from_logits,
    binary_recall_from_logits,
    binary_f1_from_logits,
    compute_spike_metrics,
    compute_routing_metrics,
    compute_prototype_metrics,
)
from cats.utils.seed import set_seed


# ============================================================
# Basic helpers
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General training script for CATS experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _get_required_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name, None)
    if section is None:
        raise KeyError(f"Missing required config section: '{name}'")
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a dictionary")
    return section


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def add_selected_metrics_to_postfix(
    postfix: Dict[str, str],
    metric_sums: Dict[str, float],
    num_batches: int,
    key_map: Dict[str, str],
    decimals_map: Optional[Dict[str, int]] = None,
) -> None:
    if num_batches <= 0:
        return

    decimals_map = decimals_map or {}

    for original_key, short_name in key_map.items():
        if original_key in metric_sums:
            value = metric_sums[original_key] / num_batches
            decimals = decimals_map.get(original_key, 4)
            postfix[short_name] = f"{value:.{decimals}f}"


class CSVLogger:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = (
            self.csv_path.exists() and self.csv_path.stat().st_size > 0
        )

    def log(self, row: Dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ============================================================
# Path resolution
# ============================================================
def resolve_data_paths(
    experiment_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    dataset_name = experiment_cfg["dataset_name"]

    processed_root = data_cfg.get("processed_root", "data/processed")
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "val")
    test_split = data_cfg.get("test_split", "test")
    has_test = bool(data_cfg.get("has_test", True))

    auto_train = str(Path(processed_root) / dataset_name / train_split)
    auto_val = str(Path(processed_root) / dataset_name / val_split)
    auto_test = (
        str(Path(processed_root) / dataset_name / test_split) if has_test else None
    )

    train_path = data_cfg.get("train_path", auto_train)
    val_path = data_cfg.get("val_path", auto_val)
    test_path = data_cfg.get("test_path", auto_test)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
    }


def resolve_log_dir(
    experiment_cfg: Dict[str, Any],
    logging_cfg: Dict[str, Any],
) -> Path:
    base_dir = Path(logging_cfg.get("base_dir", "logs"))
    dataset_name = experiment_cfg["dataset_name"]
    model_name = experiment_cfg["model_name"]
    routing_type = experiment_cfg["routing_type"]
    run_name = experiment_cfg["run_name"]

    return base_dir / dataset_name / model_name / routing_type / run_name


def resolve_checkpoint_paths(
    experiment_cfg: Dict[str, Any],
    checkpoint_cfg: Dict[str, Any],
) -> Dict[str, Path]:
    base_dir = Path(checkpoint_cfg.get("base_dir", "checkpoints"))
    dataset_name = experiment_cfg["dataset_name"]
    model_name = experiment_cfg["model_name"]
    routing_type = experiment_cfg["routing_type"]
    run_name = experiment_cfg["run_name"]

    ckpt_dir = base_dir / dataset_name / model_name / routing_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_name = checkpoint_cfg.get(
        "best_name_template", "best_{run_name}.pt"
    ).format(
        run_name=run_name,
        dataset_name=dataset_name,
        model_name=model_name,
        routing_type=routing_type,
    )
    latest_name = checkpoint_cfg.get(
        "latest_name_template", "latest_{run_name}.pt"
    ).format(
        run_name=run_name,
        dataset_name=dataset_name,
        model_name=model_name,
        routing_type=routing_type,
    )

    return {
        "dir": ckpt_dir,
        "best": ckpt_dir / best_name,
        "latest": ckpt_dir / latest_name,
    }


# ============================================================
# Data
# ============================================================
def build_dataloader(
    path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    max_cached_shards: int = 6,
    validate_on_load: bool = False,
) -> Tuple[EmbeddingDataset, DataLoader]:
    dataset = EmbeddingDataset(
        path,
        max_cached_shards=max_cached_shards,
        validate_on_load=validate_on_load,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_embeddings,
        drop_last=False,
    )
    return dataset, loader


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


# ============================================================
# Optimizer
# ============================================================
def build_optimizer(
    model: nn.Module,
    training_cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")


# ============================================================
# Classification metrics
# ============================================================
def multiclass_macro_precision_recall_f1_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    return {
        "precision": float(sum(precisions) / len(precisions)),
        "recall": float(sum(recalls) / len(recalls)),
        "f1": float(sum(f1s) / len(f1s)),
    }


def compute_classification_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    accuracy = float(accuracy_from_logits(logits, labels))

    if num_classes == 2:
        precision = float(binary_precision_from_logits(logits, labels))
        recall = float(binary_recall_from_logits(logits, labels))
        f1 = float(binary_f1_from_logits(logits, labels))
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro = multiclass_macro_precision_recall_f1_from_logits(
        logits=logits,
        labels=labels,
        num_classes=num_classes,
    )
    return {
        "accuracy": accuracy,
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1"],
    }


# ============================================================
# Metric helpers
# ============================================================
def accumulate_metric_dict(
    accumulator: Dict[str, float],
    values: Dict[str, float],
) -> None:
    for key, value in values.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def append_cpu_tensors(
    storage: List[torch.Tensor],
    tensor: torch.Tensor,
) -> None:
    storage.append(tensor.detach().cpu())


def maybe_add_tensor_stats(
    accumulator: Dict[str, float],
    outputs: Dict[str, Any],
    key: str,
    prefix: str,
) -> None:
    tensor = outputs.get(key, None)
    if tensor is None or not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return

    t = tensor.detach().float()
    accumulator[f"{prefix}_mean"] = accumulator.get(f"{prefix}_mean", 0.0) + float(t.mean().item())
    accumulator[f"{prefix}_std"] = accumulator.get(f"{prefix}_std", 0.0) + float(t.std(unbiased=False).item())


def build_postfix_train(
    total_loss: float,
    total_samples: int,
    logits_buffer: List[torch.Tensor],
    labels_buffer: List[torch.Tensor],
    num_classes: int,
    spike_metric_sums: Dict[str, float],
    routing_metric_sums: Dict[str, float],
    prototype_metric_sums: Dict[str, float],
    diag_metric_sums: Dict[str, float],
    aux_metric_sums: Dict[str, float],
    spike_batches: int,
    routing_batches: int,
    prototype_batches: int,
    diag_batches: int,
    aux_batches: int,
) -> Dict[str, str]:
    postfix: Dict[str, str] = {
        "loss": f"{(total_loss / max(total_samples, 1)):.4f}",
    }

    if logits_buffer and labels_buffer:
        logits_all = torch.cat(logits_buffer, dim=0)
        labels_all = torch.cat(labels_buffer, dim=0)
        cls_metrics = compute_classification_metrics_from_logits(
            logits=logits_all,
            labels=labels_all,
            num_classes=num_classes,
        )
        postfix["acc"] = f"{cls_metrics['accuracy']:.4f}"
        postfix["f1"] = f"{cls_metrics['f1']:.4f}"

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=spike_metric_sums,
        num_batches=spike_batches,
        key_map={
            "firing_rate": "spk_rt",
            "spikes_per_token": "spk_tok",
            "exc_firing_rate": "exc_rt",
            "inh_firing_rate": "inh_rt",
            "exc_inh_diff": "exc-inh",
        },
        decimals_map={
            "firing_rate": 4,
            "spikes_per_token": 2,
            "exc_firing_rate": 4,
            "inh_firing_rate": 4,
            "exc_inh_diff": 4,
        },
    )

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=routing_metric_sums,
        num_batches=routing_batches,
        key_map={
            "routing_entropy": "H",
            "routing_conf_mean": "r_conf",
            "max_weight_mean": "max_w",
            "dominance_fraction": "dom",
            "exc_weight_mean": "exc_w",
            "inh_weight_mean": "inh_w",
            "agreement_mean": "agree",
        },
        decimals_map={
            "routing_entropy": 3,
            "routing_conf_mean": 3,
            "max_weight_mean": 3,
            "dominance_fraction": 3,
            "exc_weight_mean": 3,
            "inh_weight_mean": 3,
            "agreement_mean": 3,
        },
    )

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=prototype_metric_sums,
        num_batches=prototype_batches,
        key_map={
            "prototype_cosine_distance": "proto_dist",
        },
        decimals_map={
            "prototype_cosine_distance": 3,
        },
    )

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=diag_metric_sums,
        num_batches=diag_batches,
        key_map={
            "exc_tau_mean": "exc_tau",
            "inh_tau_mean": "inh_tau",
            "exc_threshold_mean": "exc_thr",
            "inh_threshold_mean": "inh_thr",
            "exc_effective_threshold_mean": "exc_eff_thr",
            "inh_effective_threshold_mean": "inh_eff_thr",
        },
        decimals_map={
            "exc_tau_mean": 3,
            "inh_tau_mean": 3,
            "exc_threshold_mean": 3,
            "inh_threshold_mean": 3,
            "exc_effective_threshold_mean": 3,
            "inh_effective_threshold_mean": 3,
        },
    )

    if aux_batches > 0:
        postfix["task"] = f"{aux_metric_sums['task_loss'] / aux_batches:.4f}"
        postfix["bal"] = f"{aux_metric_sums['balance_loss'] / aux_batches:.4f}"
        postfix["ent"] = f"{aux_metric_sums['entropy_loss'] / aux_batches:.4f}"


    return postfix


def build_postfix_val(
    total_loss: float,
    total_samples: int,
    logits_buffer: List[torch.Tensor],
    labels_buffer: List[torch.Tensor],
    num_classes: int,
    spike_metric_sums: Dict[str, float],
    routing_metric_sums: Dict[str, float],
    prototype_metric_sums: Dict[str, float],
    diag_metric_sums: Dict[str, float],
    spike_batches: int,
    routing_batches: int,
    prototype_batches: int,
    diag_batches: int,
) -> Dict[str, str]:
    postfix: Dict[str, str] = {
        "loss": f"{(total_loss / max(total_samples, 1)):.4f}",
    }

    if logits_buffer and labels_buffer:
        logits_all = torch.cat(logits_buffer, dim=0)
        labels_all = torch.cat(labels_buffer, dim=0)
        cls_metrics = compute_classification_metrics_from_logits(
            logits=logits_all,
            labels=labels_all,
            num_classes=num_classes,
        )
        postfix["acc"] = f"{cls_metrics['accuracy']:.4f}"
        postfix["f1"] = f"{cls_metrics['f1']:.4f}"

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=spike_metric_sums,
        num_batches=spike_batches,
        key_map={
            "firing_rate": "spk_rt",
            "spikes_per_token": "spk_tok",
            "exc_firing_rate": "exc_rt",
            "inh_firing_rate": "inh_rt",
            "exc_inh_diff": "exc-inh",
        },
        decimals_map={
            "firing_rate": 4,
            "spikes_per_token": 2,
            "exc_firing_rate": 4,
            "inh_firing_rate": 4,
            "exc_inh_diff": 4,
        },
    )

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=routing_metric_sums,
        num_batches=routing_batches,
        key_map={
            "routing_entropy": "H",
            "routing_conf_mean": "r_conf",
            "max_weight_mean": "max_w",
            "dominance_fraction": "dom",
            "exc_weight_mean": "exc_w",
            "inh_weight_mean": "inh_w",
            "agreement_mean": "agree",
        },
        decimals_map={
            "routing_entropy": 3,
            "routing_conf_mean": 3,
            "max_weight_mean": 3,
            "dominance_fraction": 3,
            "exc_weight_mean": 3,
            "inh_weight_mean": 3,
            "agreement_mean": 3,
        },
    )

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=prototype_metric_sums,
        num_batches=prototype_batches,
        key_map={
            "prototype_cosine_distance": "proto_dist",
        },
        decimals_map={
            "prototype_cosine_distance": 3,
        },
    )

    add_selected_metrics_to_postfix(
        postfix=postfix,
        metric_sums=diag_metric_sums,
        num_batches=diag_batches,
        key_map={
            "exc_tau_mean": "exc_tau",
            "inh_tau_mean": "inh_tau",
            "exc_threshold_mean": "exc_thr",
            "inh_threshold_mean": "inh_thr",
            "exc_effective_threshold_mean": "exc_eff_thr",
            "inh_effective_threshold_mean": "inh_eff_thr",
        },
        decimals_map={
            "exc_tau_mean": 3,
            "inh_tau_mean": 3,
            "exc_threshold_mean": 3,
            "inh_threshold_mean": 3,
            "exc_effective_threshold_mean": 3,
            "inh_effective_threshold_mean": 3,
        },
    )

    return postfix


# ============================================================
# Training / validation
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    num_classes: int,
    grad_clip_norm: Optional[float] = None,
    lambda_balance: float = 0.05,
    lambda_entropy: float = 0.001,
) -> Dict[str, float]:
    model.train()

    epoch_start = time.perf_counter()

    total_loss = 0.0
    total_samples = 0
    total_tokens = 0

    total_forward_time = 0.0
    total_backward_time = 0.0
    total_step_time = 0.0

    train_logits_cpu: List[torch.Tensor] = []
    train_labels_cpu: List[torch.Tensor] = []

    spike_metric_sums: Dict[str, float] = {}
    routing_metric_sums: Dict[str, float] = {}
    prototype_metric_sums: Dict[str, float] = {}
    diag_metric_sums: Dict[str, float] = {}

    spike_batches = 0
    routing_batches = 0
    prototype_batches = 0
    diag_batches = 0

    aux_metric_sums: Dict[str, float] = {
        "task_loss": 0.0,
        "balance_loss": 0.0,
        "entropy_loss": 0.0,
    }
    aux_batches = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=False,
        ascii=True,
    )

    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        batch_size = labels.size(0)
        seq_len = embeddings.size(1)

        optimizer.zero_grad(set_to_none=True)

        maybe_cuda_synchronize(device)
        t1 = time.perf_counter()

        outputs = model(
            embeddings=embeddings,
            attention_mask=attention_mask,
        )

        logits = outputs["logits"]
        routing_weights = outputs.get("routing_weights", None)

        task_loss = criterion(logits, labels)
        loss = task_loss

        balance_loss = torch.zeros((), device=device)
        entropy_loss = torch.zeros((), device=device)

        if routing_weights is not None:
            group_mean = routing_weights.mean(dim=(0, 1))
            target_group = torch.full_like(group_mean, 1.0 / group_mean.numel())
            balance_loss = torch.abs(group_mean - target_group).mean()

            eps = 1e-8
            routing_entropy = -(
                routing_weights * (routing_weights + eps).log()
            ).sum(dim=-1).mean()

            max_entropy = torch.log(
                torch.tensor(
                    routing_weights.size(-1),
                    device=device,
                    dtype=routing_weights.dtype,
                )
            )
            entropy_loss = torch.abs(max_entropy - routing_entropy)

            loss = (
                loss
                + lambda_balance * balance_loss
                + lambda_entropy * entropy_loss
            )

        maybe_cuda_synchronize(device)
        t2 = time.perf_counter()

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        maybe_cuda_synchronize(device)
        t3 = time.perf_counter()

        optimizer.step()

        maybe_cuda_synchronize(device)
        t4 = time.perf_counter()

        total_forward_time += (t2 - t1)
        total_backward_time += (t3 - t2)
        total_step_time += (t4 - t3)

        total_samples += batch_size
        total_tokens += batch_size * seq_len
        total_loss += loss.item() * batch_size

        append_cpu_tensors(train_logits_cpu, logits)
        append_cpu_tensors(train_labels_cpu, labels)

        spikes = outputs.get("spikes", None)
        group_assignments = outputs.get("group_assignments", None)

        if spikes is not None:
            spike_metrics = compute_spike_metrics(
                spikes=spikes,
                attention_mask=attention_mask,
                group_assignments=group_assignments,
            )
            accumulate_metric_dict(spike_metric_sums, spike_metrics)
            spike_batches += 1

        if routing_weights is not None:
            routing_metrics = compute_routing_metrics(
                routing_weights=routing_weights,
                attention_mask=attention_mask,
                routing_confidence=outputs.get("routing_confidence", None),
                agreement=outputs.get("agreement", None),
            )
            accumulate_metric_dict(routing_metric_sums, routing_metrics)
            routing_batches += 1

        group_prototypes = outputs.get("group_prototypes", None)
        if group_prototypes is not None:
            prototype_metrics = compute_prototype_metrics(group_prototypes)
            accumulate_metric_dict(prototype_metric_sums, prototype_metrics)
            prototype_batches += 1

        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_tau", "exc_tau")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_tau", "inh_tau")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_threshold", "exc_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_threshold", "inh_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_effective_threshold", "exc_effective_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_effective_threshold", "inh_effective_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_adaptive_threshold", "exc_adaptive_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_adaptive_threshold", "inh_adaptive_threshold")
        diag_batches += 1

        aux_metric_sums["task_loss"] += float(task_loss.item())
        aux_metric_sums["balance_loss"] += float(balance_loss.item())
        aux_metric_sums["entropy_loss"] += float(entropy_loss.item())
        aux_batches += 1

        postfix = build_postfix_train(
            total_loss=total_loss,
            total_samples=total_samples,
            logits_buffer=train_logits_cpu,
            labels_buffer=train_labels_cpu,
            num_classes=num_classes,
            spike_metric_sums=spike_metric_sums,
            routing_metric_sums=routing_metric_sums,
            prototype_metric_sums=prototype_metric_sums,
            diag_metric_sums=diag_metric_sums,
            aux_metric_sums=aux_metric_sums,
            spike_batches=spike_batches,
            routing_batches=routing_batches,
            prototype_batches=prototype_batches,
            diag_batches=diag_batches,
            aux_batches=aux_batches,
        )
        progress_bar.set_postfix(postfix)

    if total_samples == 0:
        raise ValueError("No samples processed during training epoch.")

    epoch_time = time.perf_counter() - epoch_start
    samples_per_sec = total_samples / max(epoch_time, 1e-8)
    tokens_per_sec = total_tokens / max(epoch_time, 1e-8)

    logits_all = torch.cat(train_logits_cpu, dim=0)
    labels_all = torch.cat(train_labels_cpu, dim=0)
    cls_metrics = compute_classification_metrics_from_logits(
        logits=logits_all,
        labels=labels_all,
        num_classes=num_classes,
    )

    metrics: Dict[str, float] = {
        "loss": total_loss / total_samples,
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "epoch_time_sec": epoch_time,
        "forward_time_sec": total_forward_time,
        "backward_time_sec": total_backward_time,
        "step_time_sec": total_step_time,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
    }

    if spike_batches > 0:
        for key, value in spike_metric_sums.items():
            metrics[f"train_{key}"] = value / spike_batches

    if routing_batches > 0:
        for key, value in routing_metric_sums.items():
            metrics[f"train_{key}"] = value / routing_batches

    if prototype_batches > 0:
        for key, value in prototype_metric_sums.items():
            metrics[f"train_{key}"] = value / prototype_batches

    if diag_batches > 0:
        for key, value in diag_metric_sums.items():
            metrics[f"train_{key}"] = value / diag_batches

    if aux_batches > 0:
        metrics["train_task_loss"] = aux_metric_sums["task_loss"] / aux_batches
        metrics["train_balance_loss"] = aux_metric_sums["balance_loss"] / aux_batches
        metrics["train_entropy_loss"] = aux_metric_sums["entropy_loss"] / aux_batches

    return metrics

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()

    epoch_start = time.perf_counter()

    total_loss = 0.0
    total_samples = 0
    total_tokens = 0

    val_logits_cpu: List[torch.Tensor] = []
    val_labels_cpu: List[torch.Tensor] = []

    spike_metric_sums: Dict[str, float] = {}
    routing_metric_sums: Dict[str, float] = {}
    prototype_metric_sums: Dict[str, float] = {}
    diag_metric_sums: Dict[str, float] = {}

    spike_batches = 0
    routing_batches = 0
    prototype_batches = 0
    diag_batches = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [val]",
        leave=False,
        dynamic_ncols=False,
        ascii=True,
    )

    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(
            embeddings=embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        seq_len = embeddings.size(1)

        total_samples += batch_size
        total_tokens += batch_size * seq_len
        total_loss += loss.item() * batch_size

        append_cpu_tensors(val_logits_cpu, logits)
        append_cpu_tensors(val_labels_cpu, labels)

        spikes = outputs.get("spikes", None)
        if spikes is not None:
            spike_metrics = compute_spike_metrics(
                spikes=spikes,
                attention_mask=attention_mask,
                group_assignments=outputs.get("group_assignments", None),
            )
            accumulate_metric_dict(spike_metric_sums, spike_metrics)
            spike_batches += 1

        routing_weights = outputs.get("routing_weights", None)
        if routing_weights is not None:
            routing_metrics = compute_routing_metrics(
                routing_weights=routing_weights,
                attention_mask=attention_mask,
                routing_confidence=outputs.get("routing_confidence", None),
                agreement=outputs.get("agreement", None),
            )
            accumulate_metric_dict(routing_metric_sums, routing_metrics)
            routing_batches += 1

        group_prototypes = outputs.get("group_prototypes", None)
        if group_prototypes is not None:
            prototype_metrics = compute_prototype_metrics(group_prototypes)
            accumulate_metric_dict(prototype_metric_sums, prototype_metrics)
            prototype_batches += 1

        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_tau", "exc_tau")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_tau", "inh_tau")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_threshold", "exc_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_threshold", "inh_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_effective_threshold", "exc_effective_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_effective_threshold", "inh_effective_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "exc_adaptive_threshold", "exc_adaptive_threshold")
        maybe_add_tensor_stats(diag_metric_sums, outputs, "inh_adaptive_threshold", "inh_adaptive_threshold")
        diag_batches += 1

        postfix = build_postfix_val(
            total_loss=total_loss,
            total_samples=total_samples,
            logits_buffer=val_logits_cpu,
            labels_buffer=val_labels_cpu,
            num_classes=num_classes,
            spike_metric_sums=spike_metric_sums,
            routing_metric_sums=routing_metric_sums,
            prototype_metric_sums=prototype_metric_sums,
            diag_metric_sums=diag_metric_sums,
            spike_batches=spike_batches,
            routing_batches=routing_batches,
            prototype_batches=prototype_batches,
            diag_batches=diag_batches,
        )
        progress_bar.set_postfix(postfix)

    if total_samples == 0:
        raise ValueError("No samples processed during validation epoch.")

    epoch_time = time.perf_counter() - epoch_start
    samples_per_sec = total_samples / max(epoch_time, 1e-8)
    tokens_per_sec = total_tokens / max(epoch_time, 1e-8)

    logits_all = torch.cat(val_logits_cpu, dim=0)
    labels_all = torch.cat(val_labels_cpu, dim=0)
    cls_metrics = compute_classification_metrics_from_logits(
        logits=logits_all,
        labels=labels_all,
        num_classes=num_classes,
    )

    metrics: Dict[str, float] = {
        "loss": total_loss / total_samples,
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "epoch_time_sec": epoch_time,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
    }

    if spike_batches > 0:
        for key, value in spike_metric_sums.items():
            metrics[f"val_{key}"] = value / spike_batches

    if routing_batches > 0:
        for key, value in routing_metric_sums.items():
            metrics[f"val_{key}"] = value / routing_batches

    if prototype_batches > 0:
        for key, value in prototype_metric_sums.items():
            metrics[f"val_{key}"] = value / prototype_batches

    if diag_batches > 0:
        for key, value in diag_metric_sums.items():
            metrics[f"val_{key}"] = value / diag_batches

    return metrics


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_f1: float,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_cfg = _get_required_section(config, "experiment")
    data_cfg = _get_required_section(config, "data")
    model_cfg = _get_required_section(config, "model")
    routing_cfg = _get_required_section(config, "routing")
    lif_exc_cfg = _get_required_section(config, "lif_exc")
    lif_inh_cfg = _get_required_section(config, "lif_inh")
    training_cfg = _get_required_section(config, "training")

    logging_cfg = config.get("logging", {})
    checkpoint_cfg = config.get("checkpoints", {})
    runtime_cfg = config.get("runtime", {})

    for key in ["dataset_name", "model_name", "routing_type", "run_name"]:
        if key not in experiment_cfg:
            raise KeyError(f"Missing required experiment key: '{key}'")

    dataset_name = experiment_cfg["dataset_name"]
    model_name = experiment_cfg["model_name"]
    routing_type = experiment_cfg["routing_type"]
    run_name = experiment_cfg["run_name"]

    seed = int(training_cfg.get("seed", 42))
    deterministic = bool(training_cfg.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device_name = runtime_cfg.get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    resolved_paths = resolve_data_paths(experiment_cfg, data_cfg)
    train_path = resolved_paths["train_path"]
    val_path = resolved_paths["val_path"]
    test_path = resolved_paths["test_path"]

    log_dir = resolve_log_dir(experiment_cfg, logging_cfg)
    ckpt_paths = resolve_checkpoint_paths(experiment_cfg, checkpoint_cfg)

    log_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(log_dir / "train_log.csv")
    save_json(config, log_dir / "config.json")

    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False)) and device.type == "cuda"
    max_cached_shards = int(training_cfg.get("max_cached_shards", 6))
    validate_on_load = bool(training_cfg.get("validate_on_load", False))
    grad_clip_norm = training_cfg.get("grad_clip_norm", None)
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    lambda_balance = float(training_cfg.get("lambda_balance", 0.05))
    lambda_entropy = float(training_cfg.get("lambda_entropy", 0.001))
    lambda_inh_usage = float(training_cfg.get("lambda_inh_usage", 0.1))
    target_inh_usage = float(training_cfg.get("target_inh_usage", 0.15))

    early_stopping_enabled = bool(training_cfg.get("early_stopping_enabled", True))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 3))
    early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))

    train_dataset, train_loader = build_dataloader(
        path=train_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        max_cached_shards=max_cached_shards,
        validate_on_load=validate_on_load,
    )

    val_dataset, val_loader = build_dataloader(
        path=val_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        max_cached_shards=max_cached_shards,
        validate_on_load=validate_on_load,
    )

    model = build_model(
        experiment_cfg=experiment_cfg,
        model_cfg=model_cfg,
        routing_cfg=routing_cfg,
        lif_exc_cfg=lif_exc_cfg,
        lif_inh_cfg=lif_inh_cfg,
    ).to(device)

    criterion_name = str(training_cfg.get("criterion", "cross_entropy")).lower()
    if criterion_name != "cross_entropy":
        raise ValueError(
            f"Only 'cross_entropy' is currently implemented, got '{criterion_name}'"
        )
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, training_cfg)

    epochs = int(training_cfg.get("epochs", 10))
    num_classes = int(model_cfg["num_classes"])

    dataset_summary = {
        "dataset_name": dataset_name,
        "train": train_dataset.summary(),
        "val": val_dataset.summary(),
        "test_path": test_path,
    }
    save_json(dataset_summary, log_dir / "dataset_summary.json")

    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "routing_type": routing_type,
        "run_name": run_name,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "best_checkpoint_path": str(ckpt_paths["best"]),
        "latest_checkpoint_path": str(ckpt_paths["latest"]),
        "regularization": {
            "lambda_balance": lambda_balance,
            "lambda_entropy": lambda_entropy,
            "lambda_inh_usage": lambda_inh_usage,
            "target_inh_usage": target_inh_usage,
        },
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
            "monitor": "val_f1",
            "mode": "max",
        },
    }
    save_json(run_info, log_dir / "run_info.json")

    print("=" * 100)
    print("General Training Script")
    print("=" * 100)
    print(f"Dataset        : {dataset_name}")
    print(f"Model          : {model_name}")
    print(f"Routing        : {routing_type}")
    print(f"Run name       : {run_name}")
    print(f"Device         : {device}")
    print(f"Train path     : {train_path}")
    print(f"Val path       : {val_path}")
    print(f"Log dir        : {log_dir}")
    print(f"Checkpoint dir : {ckpt_paths['dir']}")
    print(
        f"Regularization : balance={lambda_balance}, "
        f"entropy={lambda_entropy}, inh_usage={lambda_inh_usage}"
    )
    print(
        f"Early stopping : enabled={early_stopping_enabled}, "
        f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
    )
    print("=" * 100)

    best_val_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = ""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            num_classes=num_classes,
            grad_clip_norm=grad_clip_norm,
            lambda_balance=lambda_balance,
            lambda_entropy=lambda_entropy,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
            total_epochs=epochs,
        )

        gpu_peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        current_val_f1 = val_metrics["f1"]
        improved = current_val_f1 > (best_val_f1 + early_stopping_min_delta)

        if improved:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_f1=best_val_f1,
                metrics={
                    "train": train_metrics,
                    "val": val_metrics,
                    "gpu_peak_mem_mb": gpu_peak_mem_mb,
                },
                config=config,
                path=ckpt_paths["best"],
            )

            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "best_checkpoint_path": str(ckpt_paths["best"]),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "gpu_peak_mem_mb": gpu_peak_mem_mb,
                },
                log_dir / "best_metrics.json",
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_f1=best_val_f1,
            metrics={
                "train": train_metrics,
                "val": val_metrics,
                "gpu_peak_mem_mb": gpu_peak_mem_mb,
            },
            config=config,
            path=ckpt_paths["latest"],
        )

        row: Dict[str, Any] = {
            "epoch": epoch,
            "dataset": dataset_name,
            "model_name": model_name,
            "routing_type": routing_type,
            "run_name": run_name,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_epoch_time_sec": round(train_metrics["epoch_time_sec"], 4),
            "train_forward_time_sec": round(train_metrics["forward_time_sec"], 4),
            "train_backward_time_sec": round(train_metrics["backward_time_sec"], 4),
            "train_step_time_sec": round(train_metrics["step_time_sec"], 4),
            "train_samples_per_sec": round(train_metrics["samples_per_sec"], 4),
            "train_tokens_per_sec": round(train_metrics["tokens_per_sec"], 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_epoch_time_sec": round(val_metrics["epoch_time_sec"], 4),
            "val_samples_per_sec": round(val_metrics["samples_per_sec"], 4),
            "val_tokens_per_sec": round(val_metrics["tokens_per_sec"], 4),
            "gpu_peak_mem_mb": round(gpu_peak_mem_mb, 4),
            "best_val_f1_so_far": round(best_val_f1, 6),
            "is_best": int(improved),
            "epochs_without_improvement": epochs_without_improvement,
        }

        for key, value in train_metrics.items():
            if key.startswith("train_"):
                row[key] = round(value, 6)

        for key, value in val_metrics.items():
            if key.startswith("val_"):
                row[key] = round(value, 6)

        csv_logger.log(row)

        summary_parts = [
            f"Epoch {epoch:03d}/{epochs:03d}",
            f"train_loss={train_metrics['loss']:.4f}",
            f"train_acc={train_metrics['accuracy']:.4f}",
            f"train_f1={train_metrics['f1']:.4f}",
            f"val_loss={val_metrics['loss']:.4f}",
            f"val_acc={val_metrics['accuracy']:.4f}",
            f"val_f1={val_metrics['f1']:.4f}",
        ]

        optional_summary_keys = [
            ("train_firing_rate", "train_spk_rt"),
            ("train_exc_firing_rate", "train_exc_rt"),
            ("train_inh_firing_rate", "train_inh_rt"),
            ("train_exc_inh_diff", "train_exc_inh_diff"),
            ("train_spikes_per_token", "train_spk_tok"),
            ("train_routing_entropy", "train_H"),
            ("train_routing_conf_mean", "train_r_conf"),
            ("train_max_weight_mean", "train_max_w"),
            ("train_dominance_fraction", "train_dom"),
            ("train_agreement_mean", "train_agree"),
            ("train_prototype_cosine_distance", "train_proto_dist"),
            ("val_firing_rate", "val_spk_rt"),
            ("val_exc_firing_rate", "val_exc_rt"),
            ("val_inh_firing_rate", "val_inh_rt"),
            ("val_exc_inh_diff", "val_exc_inh_diff"),
            ("val_spikes_per_token", "val_spk_tok"),
            ("val_routing_entropy", "val_H"),
            ("val_routing_conf_mean", "val_r_conf"),
            ("val_max_weight_mean", "val_max_w"),
            ("val_dominance_fraction", "val_dom"),
            ("val_agreement_mean", "val_agree"),
            ("val_prototype_cosine_distance", "val_proto_dist"),
        ]

        for key, label in optional_summary_keys:
            if key.startswith("train_") and key in train_metrics:
                summary_parts.append(f"{label}={train_metrics[key]:.4f}")
            elif key.startswith("val_") and key in val_metrics:
                summary_parts.append(f"{label}={val_metrics[key]:.4f}")

        summary_parts.append(f"no_improve={epochs_without_improvement}")
        summary_parts.append(f"gpu_mem={gpu_peak_mem_mb:.1f}MB")

        if improved:
            summary_parts.append("<-- best")

        print(" | ".join(summary_parts))

        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            stop_reason = (
                f"Validation F1 did not improve by more than {early_stopping_min_delta} "
                f"for {early_stopping_patience} consecutive epoch(s)."
            )
            print("=" * 100)
            print("Early stopping triggered")
            print(stop_reason)
            print(f"Stopped at epoch   : {epoch}")
            print(f"Best epoch         : {best_epoch}")
            print(f"Best val_f1        : {best_val_f1:.6f}")
            print("=" * 100)
            break

    training_summary = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "routing_type": routing_type,
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_checkpoint_path": str(ckpt_paths["best"]),
        "latest_checkpoint_path": str(ckpt_paths["latest"]),
        "train_log_csv": str(log_dir / "train_log.csv"),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
    save_json(training_summary, log_dir / "training_summary.json")

    print("=" * 100)
    print("Training complete")
    print(f"Best epoch       : {best_epoch}")
    print(f"Best val_f1      : {best_val_f1:.6f}")
    print(f"Best checkpoint  : {ckpt_paths['best']}")
    print(f"Latest checkpoint: {ckpt_paths['latest']}")
    print(f"Train log        : {log_dir / 'train_log.csv'}")
    if stopped_early:
        print("Stopped early    : yes")
        print(f"Stop reason      : {stop_reason}")
    else:
        print("Stopped early    : no")
    print("=" * 100)


if __name__ == "__main__":
    main()