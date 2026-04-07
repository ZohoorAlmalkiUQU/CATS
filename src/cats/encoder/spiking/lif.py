from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseSpikingLayer
from .params import (
    constrain_bounded,
    constrain_positive,
    expand_parameter,
    build_spec_from_config,
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
    LIF spike-conversion layer.

    Expects continuous input current [B, T, D] and converts it to spikes.
    Supports shared / per-group / per-channel tau and threshold.
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
            default_init=0.0,
            default_learnable=False,
            default_mode="shared",
            default_min=0.1,
            default_max=5.0,
        )

        # ---- Initialize raw tau ----
        tau_shape = self._param_shape(self.tau_spec.mode)
        tau_init = torch.full(
            tau_shape,
            float(self.tau_spec.init_value),
            dtype=torch.float32,
        )

        if self.tau_spec.learnable:
            self.raw_tau = nn.Parameter(tau_init)
        else:
            self.register_buffer("raw_tau", tau_init)

        # ---- Initialize raw threshold ----
        threshold_shape = self._param_shape(self.threshold_spec.mode)
        threshold_init = torch.full(
            threshold_shape,
            float(self.threshold_spec.init_value),
            dtype=torch.float32,
        )

        if self.threshold_spec.learnable:
            self.raw_threshold = nn.Parameter(threshold_init)
        else:
            self.register_buffer("raw_threshold", threshold_init)

    def _param_shape(self, mode: str) -> tuple[int, ...]:
        if mode == "shared":
            return (1,)
        if mode == "per_group":
            return (self.num_groups,)
        if mode == "per_channel":
            return (self.hidden_dim,)
        raise ValueError(f"Unsupported parameter mode: {mode}")

    def _compute_tau(self, group_assignments: Optional[torch.Tensor]) -> torch.Tensor:
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
            tau_param,
            mode=self.tau_spec.mode,
            hidden_dim=self.hidden_dim,
            group_assignments=group_assignments,
        )

    def _compute_threshold(self, group_assignments: Optional[torch.Tensor]) -> torch.Tensor:
        if self.threshold_spec.max_value is not None:
            thr_param = constrain_bounded(
                self.raw_threshold,
                min_value=self.threshold_spec.min_value,
                max_value=self.threshold_spec.max_value,
            )
        else:
            thr_param = constrain_positive(
                self.raw_threshold,
                min_value=self.threshold_spec.min_value,
            )

        return expand_parameter(
            thr_param,
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

        tau = self._compute_tau(group_assignments).to(device=device, dtype=dtype)
        threshold = self._compute_threshold(group_assignments).to(device=device, dtype=dtype)
        beta = torch.exp(-1.0 / tau).clamp(min=0.0, max=0.9999)

        mem = torch.zeros((bsz, hidden_dim), device=device, dtype=dtype)
        spikes_over_time = []
        mem_over_time = []

        beta = beta.view(1, hidden_dim)
        threshold = threshold.view(1, hidden_dim)

        for t in range(seq_len):
            x_t = x[:, t, :]
            m_t = attention_mask[:, t].unsqueeze(-1)

            # mask padded tokens
            x_t = x_t * m_t

            # membrane update
            mem = beta * mem + x_t

            # spike generation
            s_t = spike_fn(mem - threshold)
            s_t = s_t * m_t

            # reset
            reset_term = s_t.detach() if self.detach_reset else s_t
            if self.reset_to_zero:
                mem = mem * (1.0 - reset_term)
            else:
                mem = mem - reset_term * threshold

            # keep padded positions inactive
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
            "threshold": threshold.squeeze(0),
        }