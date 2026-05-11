from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASETS = ["sst2", "ag_news", "cifar10", "speech_commands"]
SPLITS = ["train", "validation", "test"]


EXPECTED = {
    "sst2": {
        "modality": "text",
        "sequence_type": "tokens",
        "num_classes": 2,
        "hidden_size": 768,
        "max_length": 64,
        "has_sentences": True,
        "has_paths": False,
    },
    "ag_news": {
        "modality": "text",
        "sequence_type": "tokens",
        "num_classes": 4,
        "hidden_size": 768,
        "max_length": 64,
        "has_sentences": True,
        "has_paths": False,
    },
    "cifar10": {
        "modality": "image",
        "sequence_type": "patches",
        "num_classes": 10,
        "hidden_size": 768,
        "max_length": None,
        "has_sentences": False,
        "has_paths": False,
    },
    "speech_commands": {
        "modality": "audio",
        "sequence_type": "frames",
        "num_classes": 35,
        "hidden_size": 768,
        "max_length": None,
        "has_sentences": False,
        "has_paths": True,
    },
}


def _split_dir(dataset_name: str, split_name: str) -> Path:
    return PROCESSED_DIR / dataset_name / split_name


def _list_pt_files(split_dir: Path) -> List[Path]:
    return sorted(split_dir.glob("*.pt"))


def _load_pt(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_tensor_shape(name: str, tensor: torch.Tensor, dims: int) -> None:
    assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor"
    assert tensor.ndim == dims, f"{name} must have {dims} dims, got {tensor.ndim}"


def _check_common_structure(dataset_name: str, split_name: str, obj: Dict, expected: Dict) -> Tuple[int, int, int]:
    assert "embeddings" in obj, f"Missing embeddings in {dataset_name}/{split_name}"
    assert "attention_mask" in obj, f"Missing attention_mask in {dataset_name}/{split_name}"
    assert "meta" in obj, f"Missing meta in {dataset_name}/{split_name}"

    embeddings = obj["embeddings"]
    attention_mask = obj["attention_mask"]
    meta = obj["meta"]

    _assert_tensor_shape("embeddings", embeddings, 3)
    _assert_tensor_shape("attention_mask", attention_mask, 2)

    n, t, h = embeddings.shape
    assert n > 0, f"Empty embeddings batch in {dataset_name}/{split_name}"
    assert t > 0, f"Empty sequence length in {dataset_name}/{split_name}"
    assert h == expected["hidden_size"], (
        f"Hidden size mismatch in {dataset_name}/{split_name}: expected {expected['hidden_size']}, got {h}"
    )
    assert attention_mask.shape == (n, t), (
        f"attention_mask shape mismatch in {dataset_name}/{split_name}: "
        f"expected {(n, t)}, got {tuple(attention_mask.shape)}"
    )

    assert meta["dataset_name"] == dataset_name
    assert meta["split"] == split_name
    assert meta["modality"] == expected["modality"]
    assert meta["sequence_type"] == expected["sequence_type"]
    assert meta["num_classes"] == expected["num_classes"]
    assert meta["num_samples"] == n

    if expected["max_length"] is None:
        assert meta["max_length"] is None
    else:
        assert meta["max_length"] == expected["max_length"]

    if "labels" in obj:
        labels = obj["labels"]
        _assert_tensor_shape("labels", labels, 1)
        assert labels.shape[0] == n, (
            f"labels length mismatch in {dataset_name}/{split_name}: expected {n}, got {labels.shape[0]}"
        )
        assert labels.dtype in (torch.int64, torch.long, torch.int32, torch.int16, torch.int8)

        # Some benchmark test splits use placeholder labels (for example SST-2 test uses -1).
        # Those files are still structurally valid for encoder readiness, but should not be
        # subjected to the normal class-range assertion.
        allow_placeholder_test_labels = (
            dataset_name == "sst2" and split_name == "test"
        )

        if allow_placeholder_test_labels:
            unique_labels = set(labels.unique().tolist())
            assert unique_labels == {-1}, (
                f"Expected only placeholder label -1 in {dataset_name}/{split_name}, got {unique_labels}"
            )
        else:
            assert int(labels.min()) >= 0
            assert int(labels.max()) < expected["num_classes"], (
                f"Label out of range in {dataset_name}/{split_name}: max={int(labels.max())}, "
                f"num_classes={expected['num_classes']}"
            )
    else:
        raise AssertionError(f"labels missing in {dataset_name}/{split_name}")

    if expected["has_sentences"]:
        assert "sentences" in obj, f"sentences missing in {dataset_name}/{split_name}"
        assert len(obj["sentences"]) == n
    else:
        assert "sentences" not in obj or len(obj["sentences"]) == 0

    if expected["has_paths"]:
        assert "paths" in obj, f"paths missing in {dataset_name}/{split_name}"
        assert len(obj["paths"]) == n
        assert "label_names" in obj and len(obj["label_names"]) == n
        assert "sample_rates" in obj and len(obj["sample_rates"]) == n
    
    return n, t, h


def test_processed_directories_exist() -> None:
    for dataset_name in DATASETS:
        dataset_dir = PROCESSED_DIR / dataset_name
        assert dataset_dir.exists(), f"Missing processed dataset directory: {dataset_dir}"
        for split_name in SPLITS:
            split_dir = dataset_dir / split_name
            assert split_dir.exists(), f"Missing split directory: {split_dir}"
            assert any(split_dir.glob("*.pt")), f"No .pt files found in {split_dir}"


def test_all_pt_files_match_expected_schema() -> None:
    for dataset_name in DATASETS:
        expected = EXPECTED[dataset_name]
        for split_name in SPLITS:
            split_dir = _split_dir(dataset_name, split_name)
            pt_files = _list_pt_files(split_dir)
            assert len(pt_files) > 0, f"No shard files in {split_dir}"

            seen_storage_modes = set()
            seen_part_idx = []
            total_samples = 0
            sequence_lengths = set()
            hidden_sizes = set()

            for pt_path in pt_files:
                obj = _load_pt(pt_path)
                n, t, h = _check_common_structure(dataset_name, split_name, obj, expected)
                total_samples += n
                sequence_lengths.add(t)
                hidden_sizes.add(h)

                meta = obj["meta"]
                seen_storage_modes.add(meta["storage_mode"])
                seen_part_idx.append(meta["part_idx"])

            assert hidden_sizes == {expected["hidden_size"]}, (
                f"Hidden sizes are inconsistent in {dataset_name}/{split_name}: {hidden_sizes}"
            )
            assert seen_storage_modes <= {"single_file", "sharded"}
            assert len(set(seen_part_idx)) == len(seen_part_idx), (
                f"Duplicate part_idx values found in {dataset_name}/{split_name}: {seen_part_idx}"
            )
            assert min(seen_part_idx) == 0, f"part_idx should start from 0 in {dataset_name}/{split_name}"
            assert total_samples > 0, f"Total samples must be > 0 in {dataset_name}/{split_name}"

            if dataset_name in {"sst2", "ag_news"}:
                assert sequence_lengths == {64}, (
                    f"Text sequence length should be 64 in {dataset_name}/{split_name}, got {sequence_lengths}"
                )
            elif dataset_name == "cifar10":
                assert sequence_lengths == {197}, (
                    f"ViT patch sequence length should be 197 in {dataset_name}/{split_name}, got {sequence_lengths}"
                )
            else:
                assert min(sequence_lengths) > 0, f"Audio sequence length invalid in {dataset_name}/{split_name}"


def test_embeddings_are_encoder_ready() -> None:
    expected_hidden = 768

    for dataset_name in DATASETS:
        for split_name in SPLITS:
            first_file = _list_pt_files(_split_dir(dataset_name, split_name))[0]
            obj = _load_pt(first_file)

            embeddings = obj["embeddings"]
            attention_mask = obj["attention_mask"]
            labels = obj["labels"]

            assert embeddings.ndim == 3
            assert attention_mask.ndim == 2
            assert labels.ndim == 1

            batch_size, seq_len, hidden = embeddings.shape
            assert attention_mask.shape == (batch_size, seq_len)
            assert labels.shape[0] == batch_size
            assert hidden == expected_hidden
            assert torch.isfinite(embeddings).all(), f"Non-finite values found in {dataset_name}/{split_name} embeddings"
            assert torch.isfinite(attention_mask).all(), f"Non-finite values found in {dataset_name}/{split_name} mask"
            assert torch.isfinite(labels).all(), f"Non-finite values found in {dataset_name}/{split_name} labels"

            # This is the exact contract your encoder/data loader will depend on.
            x = embeddings.float()
            m = attention_mask.long()
            y = labels.long()

            assert x.shape == (batch_size, seq_len, expected_hidden)
            assert m.shape == (batch_size, seq_len)
            assert y.shape == (batch_size,)


if __name__ == "__main__":
    raise SystemExit(
        "Run with: pytest -q tests/test_processed_embeddings.py"
    )
