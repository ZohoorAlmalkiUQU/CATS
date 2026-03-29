from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Unified dataset for precomputed embeddings saved as one or more .pt shard files.

    Supports:
    - split directory containing many shard_*.pt files
    - single .pt file path
    - legacy and new saved formats

    Returned sample format:
    {
        "embedding": Tensor[T, D],
        "attention_mask": Tensor[T],
        "label": Tensor[] or int,              # optional
        "sentence": str,                       # optional
        "path": str,                           # optional
        "label_name": str,                     # optional
        "speaker_id": str | None,              # optional
        "utterance_number": int | None,        # optional
        "sample_rate": int,                    # optional
        "index": int,                          # global sample index
        "shard_index": int,
        "local_index": int,
    }
    """

    def __init__(self, path: str | Path, cache_current_shard: bool = True):
        self.path = Path(path)
        self.cache_current_shard = cache_current_shard

        self.shard_paths = self._resolve_shard_paths(self.path)
        if len(self.shard_paths) == 0:
            raise FileNotFoundError(f"No .pt embedding files found under: {self.path}")

        self._shard_infos: List[Dict[str, Any]] = []
        self._cumulative_sizes: List[int] = []

        running_total = 0
        self.meta: Dict[str, Any] | None = None

        for shard_idx, shard_path in enumerate(self.shard_paths):
            obj = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._validate_required_keys(obj, shard_path)

            embeddings = obj["embeddings"]
            attention_mask = obj["attention_mask"]

            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(f"'embeddings' must be Tensor in {shard_path}")
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"'attention_mask' must be Tensor in {shard_path}")

            if embeddings.ndim == 2:
                num_samples = embeddings.shape[0]
            elif embeddings.ndim == 3:
                num_samples = embeddings.shape[0]
            else:
                raise ValueError(
                    f"Invalid embeddings shape in {shard_path}: {tuple(embeddings.shape)}"
                )

            info = {
                "path": shard_path,
                "num_samples": num_samples,
                "meta": self._extract_meta(obj),
            }
            self._shard_infos.append(info)

            running_total += num_samples
            self._cumulative_sizes.append(running_total)

            if self.meta is None:
                self.meta = info["meta"]

        self._current_shard_idx: Optional[int] = None
        self._current_shard_data: Optional[Dict[str, Any]] = None

    @staticmethod
    def _resolve_shard_paths(path: Path) -> List[Path]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            if path.suffix != ".pt":
                raise ValueError(f"Expected a .pt file, got: {path}")
            return [path]

        return sorted(path.glob("*.pt"))

    @staticmethod
    def _validate_required_keys(obj: Dict[str, Any], shard_path: Path) -> None:
        required_keys = ["embeddings", "attention_mask"]
        for key in required_keys:
            if key not in obj:
                raise KeyError(f"Missing required key '{key}' in {shard_path}")

    @staticmethod
    def _extract_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
        raw_meta = obj.get("meta", {})
        if raw_meta is not None and not isinstance(raw_meta, dict):
            raise TypeError(f"'meta' must be dict if present, got {type(raw_meta).__name__}")

        return {
            "dataset_name": raw_meta.get("dataset_name", "unknown"),
            "modality": raw_meta.get("modality", "unknown"),
            "backbone": raw_meta.get("backbone", obj.get("model_name", "unknown")),
            "num_classes": raw_meta.get("num_classes", None),
            "split": raw_meta.get("split", obj.get("split", "unknown")),
            "sequence_type": raw_meta.get("sequence_type", "unknown"),
            "max_length": raw_meta.get("max_length", obj.get("max_length", None)),
        }

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def _locate_index(self, global_idx: int) -> tuple[int, int]:
        if global_idx < 0:
            global_idx += len(self)

        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Index out of range: {global_idx}")

        shard_idx = bisect_right(self._cumulative_sizes, global_idx)
        shard_start = 0 if shard_idx == 0 else self._cumulative_sizes[shard_idx - 1]
        local_idx = global_idx - shard_start
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        if (
            self.cache_current_shard
            and self._current_shard_idx == shard_idx
            and self._current_shard_data is not None
        ):
            return self._current_shard_data

        shard_path = self._shard_infos[shard_idx]["path"]
        obj = torch.load(shard_path, map_location="cpu", weights_only=False)

        embeddings = obj["embeddings"]
        attention_mask = obj["attention_mask"]

        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(1)
        elif embeddings.ndim != 3:
            raise ValueError(
                f"Expected embeddings [N, T, D] or [N, D], got {tuple(embeddings.shape)} in {shard_path}"
            )

        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.ndim != 2:
            raise ValueError(
                f"Expected attention_mask [N, T] or [N], got {tuple(attention_mask.shape)} in {shard_path}"
            )

        if embeddings.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                f"Mismatch in sample count for {shard_path}: "
                f"{embeddings.shape[0]} vs {attention_mask.shape[0]}"
            )

        if embeddings.shape[1] != attention_mask.shape[1]:
            raise ValueError(
                f"Mismatch in sequence length for {shard_path}: "
                f"{embeddings.shape[1]} vs {attention_mask.shape[1]}"
            )

        obj["embeddings"] = embeddings
        obj["attention_mask"] = attention_mask

        if self.cache_current_shard:
            self._current_shard_idx = shard_idx
            self._current_shard_data = obj

        return obj

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, local_idx = self._locate_index(idx)
        shard = self._load_shard(shard_idx)

        item: Dict[str, Any] = {
            "embedding": shard["embeddings"][local_idx],             # [T, D]
            "attention_mask": shard["attention_mask"][local_idx],    # [T]
            "index": idx,
            "shard_index": shard_idx,
            "local_index": local_idx,
        }

        if "labels" in shard and shard["labels"] is not None:
            item["label"] = shard["labels"][local_idx]

        if "sentences" in shard and shard["sentences"] is not None:
            item["sentence"] = shard["sentences"][local_idx]

        if "paths" in shard and shard["paths"] is not None:
            item["path"] = shard["paths"][local_idx]

        if "label_names" in shard and shard["label_names"] is not None:
            item["label_name"] = shard["label_names"][local_idx]

        if "speaker_ids" in shard and shard["speaker_ids"] is not None:
            item["speaker_id"] = shard["speaker_ids"][local_idx]

        if "utterance_numbers" in shard and shard["utterance_numbers"] is not None:
            item["utterance_number"] = shard["utterance_numbers"][local_idx]

        if "sample_rates" in shard and shard["sample_rates"] is not None:
            item["sample_rate"] = shard["sample_rates"][local_idx]

        return item

    def summary(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.path),
            "num_shards": len(self.shard_paths),
            "num_samples": len(self),
            "dataset_name": None if self.meta is None else self.meta["dataset_name"],
            "modality": None if self.meta is None else self.meta["modality"],
            "backbone": None if self.meta is None else self.meta["backbone"],
            "num_classes": None if self.meta is None else self.meta["num_classes"],
            "split": None if self.meta is None else self.meta["split"],
            "sequence_type": None if self.meta is None else self.meta["sequence_type"],
            "max_length": None if self.meta is None else self.meta["max_length"],
        }