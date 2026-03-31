from __future__ import annotations

from typing import Dict, Optional

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def binary_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Binary F1 for num_classes=2.
    """
    preds = logits.argmax(dim=-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def compute_spike_metrics(
    spikes: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    group_assignments: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Args:
        spikes: [B, T, D]
        attention_mask: [B, T]
        group_assignments: [D], 0=exc, 1=inh
    """
    if spikes.ndim != 3:
        raise ValueError("spikes must be [B, T, D]")

    bsz, seq_len, hidden_dim = spikes.shape

    if attention_mask is None:
        valid_steps = float(bsz * seq_len * hidden_dim)
        mask3 = torch.ones_like(spikes)
    else:
        mask = attention_mask.to(dtype=spikes.dtype, device=spikes.device).unsqueeze(-1)
        mask3 = mask.expand_as(spikes)
        valid_steps = float(mask3.sum().item())

    valid_steps = max(valid_steps, 1.0)
    masked_spikes = spikes * mask3

    total_spikes = masked_spikes.sum().item()
    firing_rate = total_spikes / valid_steps
    spikes_per_sample = masked_spikes.sum(dim=(1, 2)).mean().item()

    metrics = {
        "firing_rate": float(firing_rate),
        "spikes_per_sample": float(spikes_per_sample),
        "total_spikes": float(total_spikes),
    }

    if group_assignments is not None:
        group_assignments = group_assignments.to(spikes.device)
        exc_idx = group_assignments == 0
        inh_idx = group_assignments == 1

        if exc_idx.any():
            exc_spikes = masked_spikes[..., exc_idx].sum().item()
            exc_valid = max(float(mask3[..., exc_idx].sum().item()), 1.0)
            metrics["exc_firing_rate"] = float(exc_spikes / exc_valid)

        if inh_idx.any():
            inh_spikes = masked_spikes[..., inh_idx].sum().item()
            inh_valid = max(float(mask3[..., inh_idx].sum().item()), 1.0)
            metrics["inh_firing_rate"] = float(inh_spikes / inh_valid)

    return metrics