import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader

from src.cats.data.dataset import EmbeddingDataset
from src.cats.data.collate import collate_embeddings


def test_dataset_loading(path):
    dataset = EmbeddingDataset(path)

    summary = dataset.summary()
    print("\nDataset summary:", summary)

    assert len(dataset) > 0
    assert "embedding_shape" in summary


def test_single_sample(path):
    dataset = EmbeddingDataset(path)
    sample = dataset[0]

    print("\nSingle sample keys:", sample.keys())
    print("embedding shape:", sample["embedding"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)

    if "label" in sample:
        print("label:", sample["label"])

    if "sentence" in sample:
        print("sentence:", sample["sentence"])

    assert "embedding" in sample
    assert "attention_mask" in sample

    assert sample["embedding"].ndim == 2  # [T, D]
    assert sample["attention_mask"].ndim == 1


def test_dataloader_batch(path):
    dataset = EmbeddingDataset(path)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_embeddings,
    )

    batch = next(iter(loader))

    print("\nBatch shapes:")
    print("embeddings:", batch["embeddings"].shape)
    print("attention_mask:", batch["attention_mask"].shape)

    if "labels" in batch:
        print("labels:", batch["labels"].shape)

    print("indices:", batch["indices"].shape)

    assert batch["embeddings"].ndim == 3  # [B, T, D]
    assert batch["attention_mask"].ndim == 2
    assert batch["embeddings"].shape[0] == 8
    assert batch["attention_mask"].shape[0] == 8
    assert batch["embeddings"].shape[:2] == batch["attention_mask"].shape

    if "labels" in batch:
        assert batch["labels"].ndim == 1


if __name__ == "__main__":
    print("Project root:", project_root)

    print("Running dataset tests...\n")

    # 🔥 single place to define path
    data_path = project_root / "data" / "processed" / "sst2" / "train.pt"

    test_dataset_loading(data_path)
    print("test_dataset_loading passed")

    test_single_sample(data_path)
    print("test_single_sample passed")

    test_dataloader_batch(data_path)
    print("test_dataloader_batch passed")

    print("\nAll dataset tests passed successfully.")