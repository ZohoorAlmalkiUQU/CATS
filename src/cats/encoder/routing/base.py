from __future__ import annotations

import torch.nn as nn


class BaseRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, attention_mask=None):
        raise NotImplementedError