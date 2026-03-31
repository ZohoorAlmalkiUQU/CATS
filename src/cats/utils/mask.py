"""
Mask Utilities

Helper functions for handling attention masks in sequence data.

Responsibilities:

* Apply masks to inputs or spikes
* Compute masked mean / sum
* Ensure padded tokens do not affect computation

Important:

* Prevents repeated mask logic across modules
* Critical for correct sequence modeling
  """
from __future__ import annotations

from typing import Optional

import torch


def masked_mean(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        x: [B, T, D]
        attention_mask: [B, T] or None

    Returns:
        [B, D]
    """
    if attention_mask is None:
        return x.mean(dim=1)

    mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # [B, T, 1]
    masked_x = x * mask
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return masked_x.sum(dim=1) / denom