"""
Classifier Head

Simple classification module applied on top of encoder output.

Responsibilities:

* Take pooled representation [B, D]
* Output logits for classification

Design Philosophy:

* Keep simple (Linear or small MLP)
* Must remain identical across all experiments

Important:

* Do NOT include spiking or routing logic here
* Ensures fair comparison across models
  """
from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)