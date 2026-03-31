from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import BaseRouter


class IdentityRouter(BaseRouter):
    """
    No-routing baseline.

    This router leaves the continuous embedding sequence unchanged.
    It is applied before spike conversion.
    """

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return {
            "routed_x": x,
        }