from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseSpikingLayer
from .params import (
    build_raw_parameter,
    build_spec_from_config,
    constrain_bounded,
    constrain_positive,
    expand_parameter,
)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad = torch.clamp(1.0 - x.abs(), min=0.0)
        return grad_output * grad


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


class LIFLayer(BaseSpikingLayer):
    """
    Standard LIF spike-conversion layer.

    This layer keeps a standard firing rule for all neurons:
        spike if membrane >= threshold

    Important:
    ----------
    Excitatory / inhibitory behavior should NOT be encoded here by using a
    negative threshold or a different spike condition.

    Instead, antagonistic behavior should be introduced upstream by constructing
    a signed input current before calling this layer, e.g.:
        x_total = x_exc - x_inh

    This layer is only responsible for:
        - membrane integration
        - spike generation
        - reset
        - configurable tau / threshold
          (shared / per_group / per_channel)
    """

    def __init__(
        self,
        hidden_dim: int,
        lif_config: Optional[dict] = None,
        reset_to_zero: bool = True,
        detach_reset: bool = False,
        num_groups: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.reset_to_zero = reset_to_zero
        self.detach_reset = detach_reset

        lif_config = lif_config or {}

        tau_cfg = lif_config.get("tau", {})
        threshold_cfg = lif_config.get("threshold", {})

        self.tau_spec = build_spec_from_config(
            cfg=tau_cfg,
            default_init=2.0,
            default_learnable=True,
            default_mode="per_group",
            default_min=0.5,
            default_max=20.0,
        )

        self.threshold_spec = build_spec_from_config(
            cfg=threshold_cfg,
            default_init=1.0,
            default_learnable=True,
            default_mode="per_group",
            default_min=0.1,
            default_max=5.0,
        )

        # Raw unconstrained parameters.
        # Their learnability is controlled by ParameterSpec.learnable.
        self.raw_tau = build_raw_parameter(
            spec=self.tau_spec,
            hidden_dim=self.hidden_dim,
            num_groups=self.num_groups,
        )

        self.raw_threshold = build_raw_parameter(
            spec=self.threshold_spec,
            hidden_dim=self.hidden_dim,
            num_groups=self.num_groups,
        )

    def _compute_tau(
        self,
        group_assignments: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.tau_spec.max_value is not None:
            tau_param = constrain_bounded(
                self.raw_tau,
                min_value=self.tau_spec.min_value,
                max_value=self.tau_spec.max_value,
            )
        else:
            tau_param = constrain_positive(
                self.raw_tau,
                min_value=self.tau_spec.min_value,
            )

        return expand_parameter(
            param=tau_param,
            mode=self.tau_spec.mode,
            hidden_dim=self.hidden_dim,
            group_assignments=group_assignments,
        )

    def _compute_threshold(
        self,
        group_assignments: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.threshold_spec.max_value is not None:
            threshold_param = constrain_bounded(
                self.raw_threshold,
                min_value=self.threshold_spec.min_value,
                max_value=self.threshold_spec.max_value,
            )
        else:
            threshold_param = constrain_positive(
                self.raw_threshold,
                min_value=self.threshold_spec.min_value,
            )

        return expand_parameter(
            param=threshold_param,
            mode=self.threshold_spec.mode,
            hidden_dim=self.hidden_dim,
            group_assignments=group_assignments,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        group_assignments: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:
                Continuous signed current of shape [B, T, D].
                If you want real excitatory/inhibitory antagonism, the sign should
                already be injected before calling this layer.
            attention_mask:
                Optional mask of shape [B, T].
            group_assignments:
                Optional tensor of shape [D] when using per_group parameters.
                Each channel is mapped to a group id in [0, num_groups-1].

        Returns:
            {
                "spikes": [B, T, D],
                "membrane": [B, T, D],
                "beta": [D],
                "tau": [D],
                "threshold": [D],
            }
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")

        bsz, seq_len, hidden_dim = x.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}"
            )

        device = x.device
        dtype = x.dtype

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=dtype)
        else:
            attention_mask = attention_mask.to(device=device, dtype=dtype)

        if self.tau_spec.mode == "per_group" or self.threshold_spec.mode == "per_group":
            if group_assignments is None:
                raise ValueError(
                    "group_assignments is required when tau/threshold mode is 'per_group'"
                )
            if group_assignments.ndim != 1 or group_assignments.numel() != self.hidden_dim:
                raise ValueError(
                    "group_assignments must have shape [D] and match hidden_dim"
                )
            group_assignments = group_assignments.to(device=device, dtype=torch.long)

        tau = self._compute_tau(group_assignments).to(device=device, dtype=dtype)
        threshold = self._compute_threshold(group_assignments).to(
            device=device,
            dtype=dtype,
        )

        # Standard LIF decay factor, assuming dt = 1
        beta = torch.exp(-1.0 / tau).clamp(min=0.0, max=0.9999)

        mem = torch.zeros((bsz, hidden_dim), device=device, dtype=dtype)

        spikes_over_time = []
        mem_over_time = []

        beta = beta.view(1, hidden_dim)
        threshold = threshold.view(1, hidden_dim)

        for t in range(seq_len):
            x_t = x[:, t, :]
            m_t = attention_mask[:, t].unsqueeze(-1)

            # Ignore padded tokens
            x_t = x_t * m_t

            # Standard membrane integration
            mem = beta * mem + x_t

            # Standard firing condition for all neurons
            s_t = spike_fn(mem - threshold)
            s_t = s_t * m_t

            # Reset
            reset_term = s_t.detach() if self.detach_reset else s_t
            if self.reset_to_zero:
                mem = mem * (1.0 - reset_term)
            else:
                mem = mem - reset_term * threshold

            # Keep padded states inactive
            mem = mem * m_t

            spikes_over_time.append(s_t)
            mem_over_time.append(mem)

        spikes = torch.stack(spikes_over_time, dim=1)
        membrane = torch.stack(mem_over_time, dim=1)

        return {
            "spikes": spikes,
            "membrane": membrane,
            "beta": beta.squeeze(0),
            "tau": tau,
            "threshold": threshold,
        }