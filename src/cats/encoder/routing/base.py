from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseRouter(nn.Module, ABC):
    """
    Base interface for routing modules.

    Args:
        embedding_dim: Input embedding dimension D.
        hidden_dim: Optional hidden dimension used by some routers.
        num_groups: Number of routing groups.
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.extra_kwargs = kwargs

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError