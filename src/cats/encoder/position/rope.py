from __future__ import annotations

from typing import Optional

import torch

from .base import PositionalEncoding


class RotaryPositionalEncoding(PositionalEncoding):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even dimension, got dim={dim}")

        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        bsz, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected input dim={self.dim}, got dim={dim}")

        dtype = x.dtype
        device = x.device

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)   # [T]
        freqs = torch.outer(positions, self.inv_freq.to(device=device))         # [T, D/2]

        cos = torch.cos(freqs).to(dtype=dtype).unsqueeze(0)  # [1, T, D/2]
        sin = torch.sin(freqs).to(dtype=dtype).unsqueeze(0)  # [1, T, D/2]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        x_out = torch.empty_like(x)
        x_out[..., 0::2] = rotated_even
        x_out[..., 1::2] = rotated_odd

        return x_out