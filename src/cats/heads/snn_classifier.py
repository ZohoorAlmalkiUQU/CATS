from __future__ import annotations

import torch
import torch.nn as nn


class SNNClassifierHead(nn.Module):
    """
    Spike-based classifier head.

    Input:
        x: [B, T, D] spike or continuous sequence from the encoder

    Output:
        logits: [B, num_classes]

    Idea:
        Each time step is projected to class-level input current.
        A leaky membrane integrates this current over time.
        Final logits are taken from the mean membrane trace or final membrane state.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tau: float = 2.0,
        threshold: float = 1.0,
        dropout: float = 0.0,
        readout: str = "mean_membrane",  # "mean_membrane" or "final_membrane"
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.tau = tau
        self.threshold = threshold
        self.readout = readout

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(input_dim, num_classes)

        # Decay factor for membrane update
        self.alpha = torch.exp(torch.tensor(-1.0 / tau))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            [B, T, D]

        mask:
            [B, T], where 1 means valid token and 0 means padding
        """

        if x.dim() != 3:
            raise ValueError(f"SNNClassifierHead expects [B, T, D], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        alpha = self.alpha.to(device=device, dtype=dtype)

        membrane = torch.zeros(batch_size, self.num_classes, device=device, dtype=dtype)
        membrane_trace = []

        x = self.dropout(x)

        for t in range(seq_len):
            current = self.fc(x[:, t, :])

            if mask is not None:
                valid_t = mask[:, t].to(dtype=dtype).unsqueeze(-1)
                current = current * valid_t

            membrane = alpha * membrane + current

            spikes = (membrane >= self.threshold).to(dtype)
            membrane = membrane * (1.0 - spikes)

            membrane_trace.append(membrane)

        membrane_trace = torch.stack(membrane_trace, dim=1)  # [B, T, C]

        if self.readout == "final_membrane":
            logits = membrane_trace[:, -1, :]
        elif self.readout == "mean_membrane":
            if mask is not None:
                mask_f = mask.to(dtype=dtype).unsqueeze(-1)
                logits = (membrane_trace * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            else:
                logits = membrane_trace.mean(dim=1)
        else:
            raise ValueError(f"Unknown readout type: {self.readout}")

        return logits