from __future__ import annotations

import torch
import torch.nn as nn


class CATSEncoder(nn.Module):
    """
    Minimal encoder for Milestone 2:
    - apply routing module on token embeddings
    - pool token embeddings into a single vector per sample
    """

    def __init__(self, routing_module: nn.Module, pooling: str = "mean") -> None:
        super().__init__()
        self.routing_module = routing_module
        self.pooling = pooling

        if self.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

    def masked_mean_pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        attention_mask: [B, T]
        returns: [B, D]
        """
        mask = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
        x = x * mask
        summed = x.sum(dim=1)                         # [B, D]
        counts = mask.sum(dim=1).clamp(min=1e-6)      # [B, 1]
        return summed / counts

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.routing_module(x, attention_mask=attention_mask)  # still [B, T, D]

        if self.pooling == "cls":
            return x[:, 0, :]   # [B, D]

        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling")

        return self.masked_mean_pool(x, attention_mask)