from __future__ import annotations

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold):
        ctx.save_for_backward(membrane - threshold)
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (delta,) = ctx.saved_tensors
        surrogate = torch.clamp(1.0 - delta.abs(), min=0.0)
        return grad_output * surrogate, None

surrogate_spike = SurrogateSpike.apply


_VALID_READOUTS = frozenset({
    "mean_membrane",
    "last_membrane",
    "final_membrane",       # backward-compat alias for last_membrane
    "mean_spikes",
    "spike_count",
    "membrane_spike_hybrid",
    "temporal_pool",
})


class SNNClassifierHead(nn.Module):
    """
    Spike-based classifier head with multiple temporal readout strategies.

    Input:
        x: [B, T, D]  — spike or continuous sequence from the encoder
    Output:
        logits: [B, num_classes]

    Each time step is projected to class-level input current via self.fc.
    A leaky membrane (decay α = exp(-1/τ)) integrates this current over time.
    Binary spikes fire and reset the membrane whenever it crosses the threshold.
    Final logits are extracted from the accumulated membrane/spike traces according
    to the chosen readout mode.

    Readout modes
    -------------
    mean_membrane          Average of the membrane trace over valid timesteps.
                           Baseline SNN mode — equivalent to the original implementation.
    last_membrane          Final membrane state only (no averaging).
    mean_spikes            Average of the binary spike train over valid timesteps.
                           Tests whether spike rate carries discriminative information.
    spike_count            Total spike count per class over the full sequence.
                           Strict spike-based decision rule with no membrane information.
    membrane_spike_hybrid  Concatenates mean_membrane and spike_count, then projects
                           back to num_classes via a learned linear layer.
                           Tests whether membrane dynamics and spike events are complementary.
    temporal_pool          Soft attention over the membrane trace, with attention scores
                           learned from the membrane values themselves (nn.Linear → scalar).
                           Lightweight learnable temporal aggregation, no major arch change.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tau: float = 2.0,
        threshold: float = 1.0,
        dropout: float = 0.0,
        readout: str = "mean_membrane",
    ) -> None:
        super().__init__()

        if readout not in _VALID_READOUTS:
            raise ValueError(
                f"Unknown readout type: {readout!r}. "
                f"Must be one of: {sorted(_VALID_READOUTS)}"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.tau = tau
        self.threshold = threshold
        self.readout = readout

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(input_dim, num_classes)

        # Learnable decay factor stored as a buffer so it moves with the module
        self.register_buffer("alpha", torch.exp(torch.tensor(-1.0 / tau)))

        if readout == "membrane_spike_hybrid":
            self.hybrid_fc = nn.Linear(2 * num_classes, num_classes)

        if readout == "temporal_pool":
            # Scalar attention score over the class-level membrane at each step
            self.temporal_attn = nn.Linear(num_classes, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x    : [B, T, D]
        mask : [B, T]  — 1 for valid tokens, 0 for padding
        """
        if x.dim() != 3:
            raise ValueError(
                f"SNNClassifierHead expects [B, T, D], got {tuple(x.shape)}"
            )

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        alpha = self.alpha.to(device=device, dtype=dtype)

        membrane = torch.zeros(batch_size, self.num_classes, device=device, dtype=dtype)
        membrane_trace: list[torch.Tensor] = []
        spike_trace: list[torch.Tensor] = []

        x = self.dropout(x)

        for t in range(seq_len):
            current = self.fc(x[:, t, :])               # [B, C]

            if mask is not None:
                valid_t = mask[:, t].to(dtype=dtype).unsqueeze(-1)   # [B, 1]
                current = current * valid_t

            membrane = alpha * membrane + current
            spikes = surrogate_spike(membrane, self.threshold)
            membrane = membrane * (1.0 - spikes)        # reset on spike

            membrane_trace.append(membrane)
            spike_trace.append(spikes)

        membrane_trace = torch.stack(membrane_trace, dim=1)  # [B, T, C]
        spike_trace = torch.stack(spike_trace, dim=1)        # [B, T, C]

        # Shared mask tensor for masked reductions
        if mask is not None:
            mask_f = mask.to(dtype=dtype).unsqueeze(-1)           # [B, T, 1]
            valid_count = mask_f.sum(dim=1).clamp_min(1.0)        # [B, 1]

        readout = self.readout

        # ---- mean_membrane -----------------------------------------------
        if readout == "mean_membrane":
            if mask is not None:
                logits = (membrane_trace * mask_f).sum(dim=1) / valid_count
            else:
                logits = membrane_trace.mean(dim=1)

        # ---- last_membrane / final_membrane (alias) ----------------------
        elif readout in ("last_membrane", "final_membrane"):
            logits = membrane_trace[:, -1, :]

        # ---- mean_spikes -------------------------------------------------
        elif readout == "mean_spikes":
            if mask is not None:
                logits = (spike_trace * mask_f).sum(dim=1) / valid_count
            else:
                logits = spike_trace.mean(dim=1)

        # ---- spike_count -------------------------------------------------
        elif readout == "spike_count":
            if mask is not None:
                logits = (spike_trace * mask_f).sum(dim=1)
            else:
                logits = spike_trace.sum(dim=1)

        # ---- membrane_spike_hybrid ---------------------------------------
        elif readout == "membrane_spike_hybrid":
            if mask is not None:
                mean_mem = (membrane_trace * mask_f).sum(dim=1) / valid_count
                spike_cnt = (spike_trace * mask_f).sum(dim=1)
            else:
                mean_mem = membrane_trace.mean(dim=1)
                spike_cnt = spike_trace.sum(dim=1)
            logits = self.hybrid_fc(torch.cat([mean_mem, spike_cnt], dim=-1))

        # ---- temporal_pool (soft attention over membrane trace) ----------
        elif readout == "temporal_pool":
            attn_scores = self.temporal_attn(membrane_trace)       # [B, T, 1]
            if mask is not None:
                attn_scores = attn_scores + (1.0 - mask_f) * -1e9
            attn_weights = torch.softmax(attn_scores, dim=1)       # [B, T, 1]
            logits = (attn_weights * membrane_trace).sum(dim=1)    # [B, C]

        else:
            raise ValueError(f"Unknown readout type: {readout!r}")

        return logits
