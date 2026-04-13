from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import BaseRouter


class IdentityRouter(BaseRouter):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        num_groups: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        x = self.apply_positional_encoding(x, attention_mask)
        batch_size, seq_len, _ = x.shape
        routing_weights = x.new_full(
            (batch_size, seq_len, self.num_groups),
            1.0 / self.num_groups,
        )

        if attention_mask is not None:
            mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            routing_weights = routing_weights * mask

        return {
            "routed_x": x,
            "routing_weights": routing_weights,
        }