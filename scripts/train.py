from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cats.data.dataset import EmbeddingDataset
from cats.data.collate import collate_embeddings
from cats.utils.config import load_config
from cats.utils.metrics import accuracy_from_logits, binary_f1_from_logits
from cats.utils.seed import set_seed


# ============================================================
# Registries
# ============================================================
# عدلي المسارات هنا حسب أسماء الكلاسات الحقيقية عندك
MODEL_REGISTRY = {
    "baseline": "cats.model:CATSNoRoutingClassifier",
    # "carson": "cats.model:CARSONClassifier",
}

ROUTER_REGISTRY = {
    "no_routing": "cats.encoder.routing.identity:IdentityRouter",
    "identity": "cats.encoder.routing.identity:IdentityRouter",
    # "linear": "cats.encoder.routing.linear_router:LinearRouter",
    # "carson": "cats.encoder.routing.carson:CARSONRouter",
}


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


def import_from_string(path: str):
    """
    Expects:
        'package.module:ClassName'
    """
    if ":" not in path:
        raise ValueError(
            f"Invalid class path '{path}'. Expected format 'package.module:ClassName'"
        )
    module_name, class_name = path.split(":")
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' has no attribute '{class_name}'")
    return getattr(module, class_name)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class CSVLogger:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

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
    """
    Two supported modes:

    1) Automatic by dataset name:
        processed_root: data/processed
        dataset_name: sst2
        train_split: train
        val_split: val
        test_split: test

        -> data/processed/sst2/train
           data/processed/sst2/val
           data/processed/sst2/test

    2) Explicit override:
        train_path: ...
        val_path: ...
        test_path: ...
    """
    dataset_name = experiment_cfg["dataset_name"]

    processed_root = data_cfg.get("processed_root", "data/processed")
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "val")
    test_split = data_cfg.get("test_split", "test")
    has_test = bool(data_cfg.get("has_test", True))

    auto_train = str(Path(processed_root) / dataset_name / train_split)
    auto_val = str(Path(processed_root) / dataset_name / val_split)
    auto_test = str(Path(processed_root) / dataset_name / test_split) if has_test else None

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

    # logs/sst2/baseline/no_routing/run_001/
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

    # checkpoints/sst2/baseline/no_routing/best_run_001.pt
    ckpt_dir = base_dir / dataset_name / model_name / routing_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_name = checkpoint_cfg.get("best_name_template", "best_{run_name}.pt").format(
        run_name=run_name,
        dataset_name=dataset_name,
        model_name=model_name,
        routing_type=routing_type,
    )
    latest_name = checkpoint_cfg.get("latest_name_template", "latest_{run_name}.pt").format(
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
# Builders
# ============================================================
def build_router(routing_type: str, routing_cfg: Dict[str, Any]) -> nn.Module:
    """
    routing_cfg supports:
      routing:
        routing_type: no_routing
        kwargs: {}
        class_path: cats.encoder.routing.identity:IdentityRouter   # optional override
    """
    class_path = routing_cfg.get("class_path", None)
    if class_path is None:
        if routing_type not in ROUTER_REGISTRY:
            raise ValueError(
                f"Unknown routing_type='{routing_type}'. "
                f"Add it to ROUTER_REGISTRY or provide routing.class_path in config."
            )
        class_path = ROUTER_REGISTRY[routing_type]

    RouterClass = import_from_string(class_path)
    router_kwargs = routing_cfg.get("kwargs", {}) or {}
    return RouterClass(**router_kwargs)


def build_model(
    experiment_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    lif_cfg: Dict[str, Any],
    router: nn.Module,
) -> nn.Module:
    """
    model_cfg supports:
      model:
        model_name: baseline
        embedding_dim: 768
        hidden_dim: 256
        num_classes: 2
        excitatory_ratio: 0.5
        kwargs: {}
        class_path: cats.model:CATSNoRoutingClassifier   # optional override

    Important:
    - This assumes your model constructor accepts:
        embedding_dim, hidden_dim, num_classes, excitatory_ratio, router, lif_config, **kwargs
    - If one model differs, either adapt here or add model-specific branching.
    """
    model_name = experiment_cfg["model_name"]
    class_path = model_cfg.get("class_path", None)
    if class_path is None:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_name='{model_name}'. "
                f"Add it to MODEL_REGISTRY or provide model.class_path in config."
            )
        class_path = MODEL_REGISTRY[model_name]

    ModelClass = import_from_string(class_path)

    model_kwargs = model_cfg.get("kwargs", {}) or {}

    model = ModelClass(
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_classes=int(model_cfg["num_classes"]),
        excitatory_ratio=float(model_cfg.get("excitatory_ratio", 0.5)),
        router=router,
        lif_config=lif_cfg,
        **model_kwargs,
    )
    return model


def build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")


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
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(progress_bar):
        t0 = time.perf_counter()

        batch = move_batch_to_device(batch, device)
        t1 = time.perf_counter()

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            embeddings=embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        t2 = time.perf_counter()

        loss.backward()
        t3 = time.perf_counter()

        optimizer.step()
        t4 = time.perf_counter()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

        running_loss = total_loss / total_samples
        running_acc = total_correct / total_samples

        progress_bar.set_postfix(
            loss=f"{running_loss:.4f}",
            acc=f"{running_acc:.4f}",
        )

    if batch_idx < 5:
        print(
            f"[batch {batch_idx}] "
            f"to_device={t1-t0:.3f}s | "
            f"forward={t2-t1:.3f}s | "
            f"backward={t3-t2:.3f}s | "
            f"step={t4-t3:.3f}s"
        )

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

        running_loss = total_loss / total_samples
        running_acc = total_correct / total_samples

        progress_bar.set_postfix(
            loss=f"{running_loss:.4f}",
            acc=f"{running_acc:.4f}",
        )

    if total_samples == 0:
        raise ValueError("No samples processed during training epoch.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


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

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [val]",
        leave=False,
        dynamic_ncols=True,
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
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        running_loss = total_loss / total_samples
        progress_bar.set_postfix(loss=f"{running_loss:.4f}")

    if total_samples == 0:
        raise ValueError("No samples processed during validation epoch.")

    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    acc = accuracy_from_logits(logits_all, labels_all)
    f1 = binary_f1_from_logits(logits_all, labels_all) if num_classes == 2 else float("nan")

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(acc),
        "f1": float(f1),
    }


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
    lif_cfg = _get_required_section(config, "lif")
    training_cfg = _get_required_section(config, "training")

    logging_cfg = config.get("logging", {})
    checkpoint_cfg = config.get("checkpoints", {})
    runtime_cfg = config.get("runtime", {})

    # Required experiment keys
    for key in ["dataset_name", "model_name", "routing_type", "run_name"]:
        if key not in experiment_cfg:
            raise KeyError(f"Missing required experiment key: '{key}'")

    dataset_name = experiment_cfg["dataset_name"]
    model_name = experiment_cfg["model_name"]
    routing_type = experiment_cfg["routing_type"]
    run_name = experiment_cfg["run_name"]

    # Seed / device
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

    # Paths
    resolved_paths = resolve_data_paths(experiment_cfg, data_cfg)
    train_path = resolved_paths["train_path"]
    val_path = resolved_paths["val_path"]
    test_path = resolved_paths["test_path"]  # not used in training now

    log_dir = resolve_log_dir(experiment_cfg, logging_cfg)
    ckpt_paths = resolve_checkpoint_paths(experiment_cfg, checkpoint_cfg)

    log_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(log_dir / "train_log.csv")
    save_json(config, log_dir / "config.json")

    # Data
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False)) and device.type == "cuda"

    train_dataset, train_loader = build_dataloader(
        path=train_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataset, val_loader = build_dataloader(
        path=val_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Router / model
    router = build_router(routing_type=routing_type, routing_cfg=routing_cfg)

    model = build_model(
        experiment_cfg=experiment_cfg,
        model_cfg=model_cfg,
        lif_cfg=lif_cfg,
        router=router,
    ).to(device)

    # Loss / optimizer
    criterion_name = str(training_cfg.get("criterion", "cross_entropy")).lower()
    if criterion_name != "cross_entropy":
        raise ValueError(
            f"Only 'cross_entropy' is currently implemented, got '{criterion_name}'"
        )
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, training_cfg)

    epochs = int(training_cfg.get("epochs", 10))
    num_classes = int(model_cfg["num_classes"])

    # Metadata
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
    print("=" * 100)

    best_val_f1 = float("-inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
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

        current_val_f1 = val_metrics["f1"]
        improved = current_val_f1 > best_val_f1

        if improved:
            best_val_f1 = current_val_f1
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_f1=best_val_f1,
                metrics={
                    "train": train_metrics,
                    "val": val_metrics,
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
                },
                log_dir / "best_metrics.json",
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_f1=best_val_f1,
            metrics={
                "train": train_metrics,
                "val": val_metrics,
            },
            config=config,
            path=ckpt_paths["latest"],
        )

        row = {
            "epoch": epoch,
            "dataset": dataset_name,
            "model_name": model_name,
            "routing_type": routing_type,
            "run_name": run_name,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "best_val_f1_so_far": round(best_val_f1, 6),
            "is_best": int(improved),
        }
        csv_logger.log(row)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
            + ("  <-- best" if improved else "")
        )

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
    }
    save_json(training_summary, log_dir / "training_summary.json")

    print("=" * 100)
    print("Training complete")
    print(f"Best epoch       : {best_epoch}")
    print(f"Best val_f1      : {best_val_f1:.6f}")
    print(f"Best checkpoint  : {ckpt_paths['best']}")
    print(f"Latest checkpoint: {ckpt_paths['latest']}")
    print(f"Train log        : {log_dir / 'train_log.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()