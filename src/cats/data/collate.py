from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_embeddings(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for precomputed embedding samples.

    Input sample format:
    {
        "embedding": Tensor[T, D],
        "attention_mask": Tensor[T],
        "label": Tensor[] or int,               # optional
        "sentence": str,                        # optional
        "path": str,                            # optional
        "label_name": str,                      # optional
        "speaker_id": str | None,               # optional
        "utterance_number": int | None,         # optional
        "sample_rate": int,                     # optional
        "index": int,
        "shard_index": int,
        "local_index": int,
    }

    Output batch format:
    {
        "embeddings": Tensor[B, T, D],
        "attention_mask": Tensor[B, T],
        "labels": Tensor[B],                    # optional
        "sentences": list[str],                 # optional
        "paths": list[str],                     # optional
        "label_names": list[str],               # optional
        "speaker_ids": list[str | None],        # optional
        "utterance_numbers": list[int | None],  # optional
        "sample_rates": Tensor[B],              # optional
        "indices": Tensor[B],
        "shard_indices": Tensor[B],
        "local_indices": Tensor[B],
    }
    """
    if len(batch) == 0:
        raise ValueError("Received empty batch in collate_embeddings")

    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings [B, T, D], got {tuple(embeddings.shape)}")

    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention_mask [B, T], got {tuple(attention_mask.shape)}"
        )

    if attention_mask.shape[:2] != embeddings.shape[:2]:
        raise ValueError(
            "Mismatch between embeddings and attention_mask shapes: "
            f"{embeddings.shape[:2]} vs {attention_mask.shape[:2]}"
        )

    # Avoid redundant casts if dataset already normalized dtypes
    if embeddings.dtype != torch.float32:
        embeddings = embeddings.float()

    if attention_mask.dtype != torch.long:
        attention_mask = attention_mask.long()

    output: Dict[str, Any] = {
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "shard_indices": torch.tensor(
            [item["shard_index"] for item in batch], dtype=torch.long
        ),
        "local_indices": torch.tensor(
            [item["local_index"] for item in batch], dtype=torch.long
        ),
    }

    if all("label" in item for item in batch):
        labels = []
        for item in batch:
            label = item["label"]

            if isinstance(label, torch.Tensor):
                if label.numel() != 1:
                    raise ValueError(
                        f"Expected scalar label, got shape {tuple(label.shape)}"
                    )
                label = label.item()

            labels.append(int(label))

        output["labels"] = torch.tensor(labels, dtype=torch.long)

    if all("sentence" in item for item in batch):
        output["sentences"] = [item["sentence"] for item in batch]

    if all("path" in item for item in batch):
        output["paths"] = [item["path"] for item in batch]

    if all("label_name" in item for item in batch):
        output["label_names"] = [item["label_name"] for item in batch]

    if all("speaker_id" in item for item in batch):
        output["speaker_ids"] = [item["speaker_id"] for item in batch]

    if all("utterance_number" in item for item in batch):
        output["utterance_numbers"] = [item["utterance_number"] for item in batch]

    if all("sample_rate" in item for item in batch):
        output["sample_rates"] = torch.tensor(
            [int(item["sample_rate"]) for item in batch], dtype=torch.long
        )

    return output