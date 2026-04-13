from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseRouter


class LinearRouter(BaseRouter):
    """
    Simple token-wise learned routing baseline.

    Produces routing weights over neuron groups for each token.
    Example with 2 groups:
        group 0 = excitatory
        group 1 = inhibitory
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        num_groups: int = 2,
        dropout: float = 0.0,
        temperature: float = 1.0,
        use_residual: bool = True,
        use_layernorm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim) 

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.dropout_p = dropout
        self.temperature = temperature
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        self.norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.router = nn.Linear(embedding_dim, num_groups)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            attention_mask: [B, T] or None

        Returns:
            {
                "routed_x": [B, T, D],
                "routing_weights": [B, T, G],
                "routing_logits": [B, T, G]
            }
        """
        x = self.apply_positional_encoding(x, attention_mask)
        x_in = self.norm(x)
        x_in = self.dropout(x_in)

        logits = self.router(x_in)  # [B, T, G]

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        routing_weights = torch.softmax(logits / self.temperature, dim=-1)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # [B, T, 1]
            routing_weights = routing_weights * mask

        routed_x = x + x_in if self.use_residual else x_in

        return {
            "routed_x": routed_x,
            "routing_weights": routing_weights,
            "routing_logits": logits,
        }