from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Dataset for loading precomputed embedding files saved as .pt objects.

    Expected saved format:
    {
        "split": str,
        "model_name": str,
        "max_length": int,
        "embeddings": Tensor[N, T, D],
        "attention_mask": Tensor[N, T],
        "labels": Tensor[N],              # optional (e.g., absent in SST-2 test)
        "sentences": list[str],           # optional
    }
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.data: Dict[str, Any] = torch.load(self.path)

        required_keys = ["embeddings", "attention_mask"]
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing required key '{key}' in {self.path}")

        self.embeddings = self.data["embeddings"]         # [N, T, D]
        self.attention_mask = self.data["attention_mask"] # [N, T]
        self.labels: Optional[torch.Tensor] = self.data.get("labels", None)
        self.sentences: Optional[list[str]] = self.data.get("sentences", None)

        self.split = self.data.get("split", "unknown")
        self.model_name = self.data.get("model_name", "unknown")
        self.max_length = self.data.get("max_length", None)

        num_samples = self.embeddings.shape[0]

        if self.attention_mask.shape[0] != num_samples:
            raise ValueError("Mismatch between embeddings and attention_mask sample counts.")

        if self.labels is not None and self.labels.shape[0] != num_samples:
            raise ValueError("Mismatch between embeddings and labels sample counts.")

        if self.sentences is not None and len(self.sentences) != num_samples:
            raise ValueError("Mismatch between embeddings and sentences sample counts.")

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "embedding": self.embeddings[idx],           # [T, D]
            "attention_mask": self.attention_mask[idx],  # [T]
            "index": idx,
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        if self.sentences is not None:
            item["sentence"] = self.sentences[idx]

        return item

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "split": self.split,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_samples": len(self),
            "embedding_shape": tuple(self.embeddings.shape),
            "attention_mask_shape": tuple(self.attention_mask.shape),
            "has_labels": self.labels is not None,
            "has_sentences": self.sentences is not None,
        }