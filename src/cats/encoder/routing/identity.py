from __future__ import annotations

from .base import BaseRouter


class IdentityRouter(BaseRouter):
    def forward(self, x, attention_mask=None):
        return x