from __future__ import annotations

import torch
import torch.nn as nn


class ANNClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()

        layers = []

        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))

        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)