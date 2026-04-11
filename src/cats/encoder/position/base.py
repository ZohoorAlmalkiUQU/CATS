from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attention_mask: [B, T] or None

        Returns:
            Tensor with the same shape [B, T, D]
        """
        raise NotImplementedError