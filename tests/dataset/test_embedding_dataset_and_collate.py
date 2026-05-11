from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cats.data.dataset import EmbeddingDataset
from cats.data.collate import collate_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


DATASET_SPLITS = [
    ("sst2", "train"),
    ("ag_news", "train"),
    ("cifar10", "train"),
    ("speech_commands", "train"),
]


def _get_split_path(dataset_name: str, split_name: str) -> Path:
    return PROCESSED_DIR / dataset_name / split_name


def test_embedding_dataset_reads_sharded_split_directories() -> None:
    for dataset_name, split_name in DATASET_SPLITS:
        split_path = _get_split_path(dataset_name, split_name)
        ds = EmbeddingDataset(split_path)

        assert len(ds) > 0, f"{dataset_name}/{split_name} is empty"

        sample = ds[0]
        assert "embedding" in sample
        assert "attention_mask" in sample
        assert "index" in sample

        assert isinstance(sample["embedding"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)

        assert sample["embedding"].ndim == 2, (
            f"Expected [T, D] embedding for {dataset_name}/{split_name}, "
            f"got {tuple(sample['embedding'].shape)}"
        )
        assert sample["attention_mask"].ndim == 1, (
            f"Expected [T] mask for {dataset_name}/{split_name}, "
            f"got {tuple(sample['attention_mask'].shape)}"
        )

        t, d = sample["embedding"].shape
        assert t > 0
        assert d == 768

        assert sample["attention_mask"].shape[0] == t


def test_collate_embeddings_builds_valid_batch() -> None:
    for dataset_name, split_name in DATASET_SPLITS:
        split_path = _get_split_path(dataset_name, split_name)
        ds = EmbeddingDataset(split_path)

        samples = [ds[i] for i in range(min(4, len(ds)))]
        batch = collate_embeddings(samples)

        assert "embeddings" in batch
        assert "attention_mask" in batch
        assert "indices" in batch

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        indices = batch["indices"]

        assert isinstance(embeddings, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(indices, torch.Tensor)

        assert embeddings.ndim == 3
        assert attention_mask.ndim == 2
        assert indices.ndim == 1

        b, t, d = embeddings.shape
        assert b == len(samples)
        assert t > 0
        assert d == 768

        assert attention_mask.shape == (b, t)
        assert indices.shape == (b,)

        if "labels" in batch:
            assert batch["labels"].shape == (b,)


def test_dataloader_with_shuffle_runs_correctly() -> None:
    for dataset_name, split_name in DATASET_SPLITS:
        split_path = _get_split_path(dataset_name, split_name)
        ds = EmbeddingDataset(split_path)

        loader = DataLoader(
            ds,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_embeddings,
        )

        batch = next(iter(loader))

        assert "embeddings" in batch
        assert "attention_mask" in batch
        assert "indices" in batch

        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        indices = batch["indices"]

        assert embeddings.ndim == 3
        assert attention_mask.ndim == 2
        assert indices.ndim == 1

        b, t, d = embeddings.shape
        assert b <= 8
        assert t > 0
        assert d == 768
        assert attention_mask.shape == (b, t)
        assert indices.shape == (b,)

        if "labels" in batch:
            assert batch["labels"].shape == (b,)


def test_dataset_summary_exists_and_is_reasonable() -> None:
    for dataset_name, split_name in DATASET_SPLITS:
        split_path = _get_split_path(dataset_name, split_name)
        ds = EmbeddingDataset(split_path)

        summary = ds.summary()
        assert isinstance(summary, dict)

        assert "num_samples" in summary
        assert summary["num_samples"] == len(ds)

        if "dataset_name" in summary and summary["dataset_name"] is not None:
            assert summary["dataset_name"] == dataset_name