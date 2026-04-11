from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def binary_precision_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Binary precision for num_classes=2.
    """
    preds = logits.argmax(dim=-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    return float(precision)


def binary_recall_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Binary recall for num_classes=2.
    """
    preds = logits.argmax(dim=-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    recall = tp / (tp + fn + 1e-8)
    return float(recall)


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
    Compute spike-related metrics.

    Args:
        spikes: [B, T, D]
        attention_mask: [B, T] or None
        group_assignments: [D], where 0=exc and 1=inh

    Returns:
        Dictionary containing spike activity statistics.
    """
    if spikes.ndim != 3:
        raise ValueError(f"spikes must be [B, T, D], got shape {tuple(spikes.shape)}")

    bsz, seq_len, hidden_dim = spikes.shape

    if attention_mask is None:
        mask3 = torch.ones_like(spikes)
        valid_steps = float(bsz * seq_len * hidden_dim)
        valid_tokens = float(bsz * seq_len)
    else:
        if attention_mask.shape != (bsz, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(bsz, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        mask = attention_mask.to(dtype=spikes.dtype, device=spikes.device).unsqueeze(-1)
        mask3 = mask.expand_as(spikes)
        valid_steps = float(mask3.sum().item())
        valid_tokens = float(
            attention_mask.to(dtype=spikes.dtype, device=spikes.device).sum().item()
        )

    valid_steps = max(valid_steps, 1.0)
    valid_tokens = max(valid_tokens, 1.0)

    masked_spikes = spikes * mask3

    total_spikes = masked_spikes.sum().item()
    firing_rate = total_spikes / valid_steps
    spikes_per_sample = masked_spikes.sum(dim=(1, 2)).mean().item()
    spikes_per_token = total_spikes / valid_tokens

    metrics: Dict[str, float] = {
        "firing_rate": float(firing_rate),
        "spikes_per_sample": float(spikes_per_sample),
        "spikes_per_token": float(spikes_per_token),
        "total_spikes": float(total_spikes),
        "valid_steps": float(valid_steps),
        "num_samples": float(bsz),
    }

    if group_assignments is not None:
        if group_assignments.ndim != 1:
            raise ValueError(
                f"group_assignments must be [D], got shape {tuple(group_assignments.shape)}"
            )

        if group_assignments.shape[0] != hidden_dim:
            raise ValueError(
                f"group_assignments length must equal D={hidden_dim}, "
                f"got {group_assignments.shape[0]}"
            )

        group_assignments = group_assignments.to(
            device=spikes.device,
            dtype=torch.long,
        )

        exc_idx = group_assignments == 0
        inh_idx = group_assignments == 1

        if exc_idx.any():
            exc_spikes_tensor = masked_spikes[..., exc_idx]
            exc_spikes = exc_spikes_tensor.sum().item()
            exc_valid = max(float(mask3[..., exc_idx].sum().item()), 1.0)

            metrics["exc_firing_rate"] = float(exc_spikes / exc_valid)
            metrics["exc_total_spikes"] = float(exc_spikes)
            metrics["exc_spikes_per_token"] = float(exc_spikes / valid_tokens)
            metrics["exc_spikes_per_sample"] = float(
                exc_spikes_tensor.sum(dim=(1, 2)).mean().item()
            )

        if inh_idx.any():
            inh_spikes_tensor = masked_spikes[..., inh_idx]
            inh_spikes = inh_spikes_tensor.sum().item()
            inh_valid = max(float(mask3[..., inh_idx].sum().item()), 1.0)

            metrics["inh_firing_rate"] = float(inh_spikes / inh_valid)
            metrics["inh_total_spikes"] = float(inh_spikes)
            metrics["inh_spikes_per_token"] = float(inh_spikes / valid_tokens)
            metrics["inh_spikes_per_sample"] = float(
                inh_spikes_tensor.sum(dim=(1, 2)).mean().item()
            )

        if "exc_firing_rate" in metrics and "inh_firing_rate" in metrics:
            metrics["exc_inh_diff"] = float(
                metrics["exc_firing_rate"] - metrics["inh_firing_rate"]
            )

    return metrics


def compute_routing_metrics(
    routing_weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    routing_confidence: Optional[torch.Tensor] = None,
    agreement: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute routing-related metrics.

    Args:
        routing_weights: [B, T, G], expected G>=2 and typically G=2 for exc/inh routing
        attention_mask: [B, T] or None
        routing_confidence: [B, T] or None
        agreement: [B, T, G] or None

    Returns:
        Dictionary containing routing statistics.
    """
    if routing_weights.ndim != 3:
        raise ValueError(
            f"routing_weights must be [B, T, G], got shape {tuple(routing_weights.shape)}"
        )

    bsz, seq_len, num_groups = routing_weights.shape

    if attention_mask is None:
        mask = torch.ones(
            (bsz, seq_len),
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )
    else:
        if attention_mask.shape != (bsz, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(bsz, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        mask = attention_mask.to(
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )

    if routing_confidence is not None:
        if routing_confidence.shape != (bsz, seq_len):
            raise ValueError(
                f"routing_confidence must have shape {(bsz, seq_len)}, "
                f"got {tuple(routing_confidence.shape)}"
            )

    if agreement is not None:
        if agreement.shape != routing_weights.shape:
            raise ValueError(
                f"agreement must have shape {tuple(routing_weights.shape)}, "
                f"got {tuple(agreement.shape)}"
            )

    mask_bool = mask.bool()
    valid_count = max(float(mask.sum().item()), 1.0)

    mask_exp = mask.unsqueeze(-1)  # [B, T, 1]
    rw = routing_weights * mask_exp

    eps = 1e-8
    rw_safe = rw.clamp(min=eps)

    entropy = -(rw_safe * rw_safe.log()).sum(dim=-1)  # [B, T]
    entropy = (entropy * mask).sum().item() / valid_count

    metrics: Dict[str, float] = {
        "routing_entropy": float(entropy),
        "num_groups": float(num_groups),
    }

    valid_rw = rw[mask_bool]
    if valid_rw.numel() > 0:
        metrics["routing_variance"] = float(valid_rw.var(unbiased=False).item())

    if num_groups >= 2:
        exc = rw[..., 0]
        inh = rw[..., 1]

        exc_mean = exc.sum().item() / valid_count
        inh_mean = inh.sum().item() / valid_count

        metrics["exc_weight_mean"] = float(exc_mean)
        metrics["inh_weight_mean"] = float(inh_mean)

        metrics["exc_inh_balance"] = float(exc_mean - inh_mean)
        metrics["weight_gap"] = float(abs(exc_mean - inh_mean))

        exc_vals = exc[mask_bool]
        inh_vals = inh[mask_bool]

        if exc_vals.numel() > 0:
            metrics["exc_weight_std"] = float(exc_vals.std(unbiased=False).item())
        if inh_vals.numel() > 0:
            metrics["inh_weight_std"] = float(inh_vals.std(unbiased=False).item())

    if routing_confidence is not None:
        rc = routing_confidence.to(
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )
        rc = rc * mask

        metrics["routing_conf_mean"] = float(rc.sum().item() / valid_count)

        rc_vals = rc[mask_bool]
        if rc_vals.numel() > 0:
            metrics["routing_conf_std"] = float(rc_vals.std(unbiased=False).item())

    if agreement is not None:
        agreement = agreement.to(
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )
        agreement = agreement * mask_exp

        metrics["agreement_mean"] = float(
            agreement.sum().item() / (valid_count * num_groups)
        )

        if num_groups >= 2:
            exc_agreement = agreement[..., 0]
            inh_agreement = agreement[..., 1]

            metrics["exc_agreement_mean"] = float(
                exc_agreement.sum().item() / valid_count
            )
            metrics["inh_agreement_mean"] = float(
                inh_agreement.sum().item() / valid_count
            )

            exc_vals = exc_agreement[mask_bool]
            inh_vals = inh_agreement[mask_bool]

            if exc_vals.numel() > 0:
                metrics["exc_agreement_std"] = float(
                    exc_vals.std(unbiased=False).item()
                )
            if inh_vals.numel() > 0:
                metrics["inh_agreement_std"] = float(
                    inh_vals.std(unbiased=False).item()
                )

    dominant_group = rw.argmax(dim=-1)  # [B, T]

    if mask_bool.any():
        dominant_vals = dominant_group[mask_bool]

        if num_groups >= 1:
            metrics["exc_usage_fraction"] = float(
                (dominant_vals == 0).float().mean().item()
            )
        if num_groups >= 2:
            metrics["inh_usage_fraction"] = float(
                (dominant_vals == 1).float().mean().item()
            )

    max_weight = rw.max(dim=-1).values  # [B, T]
    metrics["max_weight_mean"] = float((max_weight * mask).sum().item() / valid_count)

    high_conf = (max_weight > 0.9).to(dtype=routing_weights.dtype)
    metrics["dominance_fraction"] = float(
        (high_conf * mask).sum().item() / valid_count
    )

    return metrics


def compute_prototype_metrics(
    group_prototypes: Optional[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute prototype separation metrics.

    Args:
        group_prototypes: [B, G, D]

    Returns:
        Dictionary containing prototype similarity / separation statistics.
    """
    if group_prototypes is None:
        return {}

    if group_prototypes.ndim != 3:
        raise ValueError(
            f"group_prototypes must be [B, G, D], got shape {tuple(group_prototypes.shape)}"
        )

    _, num_groups, _ = group_prototypes.shape

    if num_groups < 2:
        raise ValueError(
            f"group_prototypes must have at least 2 groups, got {num_groups}"
        )

    exc = group_prototypes[:, 0, :]
    inh = group_prototypes[:, 1, :]

    exc = F.normalize(exc, dim=-1)
    inh = F.normalize(inh, dim=-1)

    cos_sim = (exc * inh).sum(dim=-1)  # [B]

    return {
        "prototype_cosine_sim": float(cos_sim.mean().item()),
        "prototype_cosine_distance": float((1.0 - cos_sim).mean().item()),
    }