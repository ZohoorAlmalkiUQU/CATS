from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cats.data.collate import collate_embeddings
from cats.data.dataset import EmbeddingDataset
from cats.encoder.core import CATSEncoder
from cats.encoder.routing.identity import IdentityRouter
from cats.heads.classifier import ClassifierHead


class BaselineModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder(embeddings, attention_mask)
        logits = self.head(features)
        return logits


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
    config_path = Path(path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parent = cfg.get("inherits_from")
    if parent is None:
        return cfg

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = PROJECT_ROOT / parent_path

    with open(parent_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    cfg.pop("inherits_from", None)
    return deep_update(base_cfg, cfg)


def make_loader(path: str, batch_size: int) -> DataLoader:
    dataset = EmbeddingDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_embeddings,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float | None, float | None, int]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(embeddings, attention_mask)

            if "labels" not in batch:
                continue

            labels = batch["labels"].to(device)

            valid_mask = labels >= 0
            if valid_mask.sum().item() == 0:
                continue

            logits = logits[valid_mask]
            labels = labels[valid_mask]

            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        return None, None, 0

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc, total_examples


def build_model(cfg: dict, device: torch.device) -> nn.Module:
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/no_routing.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_best.pt")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")

    cfg = load_config(args.config)

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = build_model(cfg, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint: {args.checkpoint}")
    if "best_val_acc" in checkpoint:
        print(f"Saved best validation accuracy: {checkpoint['best_val_acc']:.4f}")

    # 1) Try test split first
    test_path = cfg["data"]["test_path"]
    test_loader = make_loader(
        path=test_path,
        batch_size=cfg["training"]["batch_size"],
    )

    test_loss, test_acc, test_count = evaluate(model, test_loader, device)

    if test_loss is None:
        print("Test split has no valid labels (inference-only split).")
        print("Falling back to validation split for quantitative evaluation...")

        val_path = cfg["data"]["val_path"]
        val_loader = make_loader(
            path=val_path,
            batch_size=cfg["training"]["batch_size"],
        )

        val_loss, val_acc, val_count = evaluate(model, val_loader, device)

        if val_loss is None:
            raise ValueError("Validation split also has no valid labels. Cannot compute evaluation metrics.")

        print(f"Validation samples used: {val_count}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
    else:
        print(f"Test samples used: {test_count}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()