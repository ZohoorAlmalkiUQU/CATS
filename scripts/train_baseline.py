from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from cats.data.dataset import EmbeddingDataset
from cats.data.collate import collate_embeddings
from cats.encoder.core import CATSEncoder
from cats.encoder.routing.identity import IdentityRouter
from cats.heads.classifier import ClassifierHead
from cats.utils.seed import set_seed


class BaselineModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, embeddings, attention_mask):
        features = self.encoder(embeddings, attention_mask)
        logits = self.head(features)
        return logits

    def get_spike_stats(self):
        if hasattr(self.encoder, "get_spike_stats"):
            return self.encoder.get_spike_stats()
        return 0.0, 0.0


def deep_update(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> dict:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parent_path = cfg.get("inherits_from")
    if parent_path is None:
        return cfg

    parent_path = Path(parent_path)
    if not parent_path.is_absolute():
        parent_path = Path.cwd() / parent_path

    with open(parent_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    cfg.pop("inherits_from", None)
    return deep_update(base_cfg, cfg)


def make_loader(path: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = EmbeddingDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_embeddings,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    total_spikes = 0.0
    total_elements = 0.0

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(embeddings, attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

        if hasattr(model, "get_spike_stats"):
            spikes, elements = model.get_spike_stats()
            total_spikes += float(spikes)
            total_elements += float(elements)

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    firing_rate = total_spikes / total_elements if total_elements > 0 else 0.0
    spikes_per_sample = total_spikes / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "firing_rate": firing_rate,
        "spikes_per_sample": spikes_per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/no_routing.yaml")
    parser.add_argument("--run_name", type=str, default="run_001")
    parser.add_argument("--dataset", type=str, default="sst2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    train_loader = make_loader(cfg["data"]["train_path"], cfg["training"]["batch_size"], True)
    val_loader = make_loader(cfg["data"]["val_path"], cfg["training"]["batch_size"], False)

    model = BaselineModel(
        encoder=CATSEncoder(
            routing_module=IdentityRouter(),
            pooling=cfg["model"].get("pooling", "mean"),
        ),
        head=ClassifierHead(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
        ),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    # === FINAL STRUCTURE ===
    dataset = args.dataset
    run_name = args.run_name

    log_dir = Path("logs") / dataset / run_name
    ckpt_dir = Path("checkpoints") / dataset / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = log_dir / "train_log.csv"
    txt_log_path = log_dir / "train_log.txt"
    ckpt_path = ckpt_dir / "baseline_best.pt"

    print(f"Dataset: {dataset}")
    print(f"Run: {run_name}")
    print(f"Logs: {log_dir}")
    print(f"Checkpoint: {ckpt_path}")

    # init logs
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_firing_rate",
            "train_spikes_per_sample",
            "val_loss",
            "val_accuracy",
            "val_firing_rate",
            "val_spikes_per_sample",
        ])

    with open(txt_log_path, "w", encoding="utf-8") as f:
        f.write(f"Training log - {dataset} - {run_name}\n")
        f.write("=" * 80 + "\n")

    best_val_acc = -1.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        log_line = (
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        print(log_line)

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["firing_rate"],
                train_metrics["spikes_per_sample"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["firing_rate"],
                val_metrics["spikes_per_sample"],
            ])

        with open(txt_log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "dataset": dataset,
                    "run_name": run_name,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()