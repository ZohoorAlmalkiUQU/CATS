from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseSpikingLayer(nn.Module, ABC):
    """
    Abstract base class for spiking layers.

    Expected input:
        x: [B, T, D]
        attention_mask: [B, T] or None

    Expected output:
        dict with at least:
            - "spikes": [B, T, D]
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError