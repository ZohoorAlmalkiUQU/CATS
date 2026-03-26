from typing import Any, Dict, List

import torch


def collate_embeddings(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for precomputed fixed-length embedding samples.

    Input sample format:
    {
        "embedding": Tensor[T, D],
        "attention_mask": Tensor[T],
        "label": Tensor[] or int,     # optional
        "sentence": str,              # optional
        "index": int,
    }

    Output batch format:
    {
        "embeddings": Tensor[B, T, D],
        "attention_mask": Tensor[B, T],
        "labels": Tensor[B],          # optional
        "sentences": list[str],       # optional
        "indices": Tensor[B],
    }
    """

    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)            # [B, T, D]
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)   # [B, T]
    indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)

    output = {
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "indices": indices,
    }

    if "label" in batch[0]:
        labels = [
            item["label"] if isinstance(item["label"], torch.Tensor)
            else torch.tensor(item["label"], dtype=torch.long)
            for item in batch
        ]
        output["labels"] = torch.stack(labels, dim=0)  # [B]

    if "sentence" in batch[0]:
        output["sentences"] = [item["sentence"] for item in batch]

    return output