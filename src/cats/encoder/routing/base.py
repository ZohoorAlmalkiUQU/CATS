from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cats.encoder.position import PositionalEncoding


class BaseRouter(nn.Module, ABC):
    """
    Base interface for routing modules.

    Args:
        embedding_dim: Input embedding dimension D.
        hidden_dim: Optional hidden dimension used by some routers.
        num_groups: Number of routing groups.
        pos_enc: Optional positional encoding module (e.g., RoPE).
        use_pos_enc: Whether to apply positional encoding before routing.
        **kwargs: Extra router-specific arguments.

    Input:
        x: [B, T, D]
        attention_mask: [B, T] or None

    Output:
        dict containing at least:
            - "routed_x": [B, T, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        num_groups: int = 1,
        pos_enc: Optional[PositionalEncoding] = None,
        use_pos_enc: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups

        self.pos_enc = pos_enc
        self.use_pos_enc = use_pos_enc

        self.extra_kwargs = kwargs

    def apply_positional_encoding(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_pos_enc and self.pos_enc is not None:
            x = self.pos_enc(x, attention_mask)
        return x

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError