from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseSpikingLayer
from .params import (
    build_adaptive_state_tensor,
    build_adaptive_threshold_config_from_config,
    build_raw_parameter,
    build_spec_from_config,
    clamp_tensor,
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
    Standard LIF spike-conversion layer with optional adaptive threshold.

    This layer keeps a standard firing rule for all neurons:
        spike if membrane >= effective_threshold

    Important:
    ----------
    Excitatory / inhibitory behavior should NOT be encoded here by using a
    negative threshold or a different spike condition.

    Instead, antagonistic behavior should be introduced upstream by constructing
    an appropriate population-specific current before calling this layer.

    This layer is only responsible for:
        - membrane integration
        - spike generation
        - reset
        - configurable tau / threshold
          (shared / per_group / per_channel)
        - optional adaptive threshold dynamics
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
        adaptive_threshold_cfg = lif_config.get("adaptive_threshold", {})

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

        self.adaptive_threshold_cfg = build_adaptive_threshold_config_from_config(
            cfg=adaptive_threshold_cfg,
            default_enabled=False,
            default_mode="per_group",
            default_init=0.0,
            default_decay=0.95,
            default_spike_increment=0.1,
            default_min=0.0,
            default_max=5.0,
            default_detach_spikes=True,
        )

        self._validate_configuration()

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

    def _validate_configuration(self) -> None:
        """
        Enforce meaningful configuration choices.

        In the new population-separated design, per_group is not meaningful
        when num_groups == 1. In that case, users should choose:
            - shared
            - per_channel
        """
        if self.num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {self.num_groups}")

        if self.num_groups == 1:
            if self.tau_spec.mode == "per_group":
                raise ValueError(
                    "Invalid LIF config: tau mode='per_group' is not meaningful when "
                    "num_groups=1. Use mode='shared' or mode='per_channel' instead."
                )

            if self.threshold_spec.mode == "per_group":
                raise ValueError(
                    "Invalid LIF config: threshold mode='per_group' is not meaningful "
                    "when num_groups=1. Use mode='shared' or mode='per_channel' instead."
                )

            if (
                self.adaptive_threshold_cfg.enabled
                and self.adaptive_threshold_cfg.mode == "per_group"
            ):
                raise ValueError(
                    "Invalid LIF config: adaptive_threshold mode='per_group' is not "
                    "meaningful when num_groups=1. Use mode='shared' or "
                    "mode='per_channel' instead."
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

    def _compute_base_threshold(
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

    def _expand_adaptive_value(
        self,
        value,
        mode: str,
        *,
        group_assignments: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build and expand an adaptive-threshold config value to shape [D].
        """
        cfg = type(self.adaptive_threshold_cfg)(
            enabled=self.adaptive_threshold_cfg.enabled,
            mode=mode,
            init=value,
            decay=self.adaptive_threshold_cfg.decay,
            spike_increment=self.adaptive_threshold_cfg.spike_increment,
            min_value=self.adaptive_threshold_cfg.min_value,
            max_value=self.adaptive_threshold_cfg.max_value,
            detach_spikes=self.adaptive_threshold_cfg.detach_spikes,
        )

        raw = build_adaptive_state_tensor(
            cfg,
            hidden_dim=self.hidden_dim,
            num_groups=self.num_groups,
            dtype=dtype,
            device=device,
        )

        return expand_parameter(
            param=raw,
            mode=mode,
            hidden_dim=self.hidden_dim,
            group_assignments=group_assignments,
        )

    def _compute_adaptive_components(
        self,
        group_assignments: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Prepare adaptive threshold tensors expanded to [D].
        """
        if not self.adaptive_threshold_cfg.enabled:
            return None

        mode = self.adaptive_threshold_cfg.mode

        init_state = self._expand_adaptive_value(
            self.adaptive_threshold_cfg.init,
            mode,
            group_assignments=group_assignments,
            device=device,
            dtype=dtype,
        )

        decay = self._expand_adaptive_value(
            self.adaptive_threshold_cfg.decay,
            mode,
            group_assignments=group_assignments,
            device=device,
            dtype=dtype,
        ).clamp(min=0.0, max=1.0)

        spike_increment = self._expand_adaptive_value(
            self.adaptive_threshold_cfg.spike_increment,
            mode,
            group_assignments=group_assignments,
            device=device,
            dtype=dtype,
        )

        min_value = self._expand_adaptive_value(
            self.adaptive_threshold_cfg.min_value,
            mode,
            group_assignments=group_assignments,
            device=device,
            dtype=dtype,
        )

        max_value = self._expand_adaptive_value(
            self.adaptive_threshold_cfg.max_value,
            mode,
            group_assignments=group_assignments,
            device=device,
            dtype=dtype,
        )

        return {
            "init_state": init_state,
            "decay": decay,
            "spike_increment": spike_increment,
            "min_value": min_value,
            "max_value": max_value,
        }

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        group_assignments: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:
                Continuous input current of shape [B, T, D].
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
                "threshold": [D],                  # base threshold
                "effective_threshold": [B, T, D], # threshold used at each timestep
                "adaptive_threshold": [B, T, D],  # adaptive state over time
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

        requires_group_assignments = (
            self.tau_spec.mode == "per_group"
            or self.threshold_spec.mode == "per_group"
            or (
                self.adaptive_threshold_cfg.enabled
                and self.adaptive_threshold_cfg.mode == "per_group"
            )
        )

        if requires_group_assignments:
            if group_assignments is None:
                raise ValueError(
                    "group_assignments is required when any LIF parameter uses "
                    "mode='per_group'."
                )
            if group_assignments.ndim != 1 or group_assignments.numel() != self.hidden_dim:
                raise ValueError(
                    "group_assignments must have shape [D] and match hidden_dim."
                )
            group_assignments = group_assignments.to(device=device, dtype=torch.long)

        tau = self._compute_tau(group_assignments).to(device=device, dtype=dtype)
        base_threshold = self._compute_base_threshold(group_assignments).to(
            device=device,
            dtype=dtype,
        )

        adaptive_components = self._compute_adaptive_components(
            group_assignments,
            device=device,
            dtype=dtype,
        )

        # Standard LIF decay factor, assuming dt = 1
        beta = torch.exp(-1.0 / tau).clamp(min=0.0, max=0.9999)

        mem = torch.zeros((bsz, hidden_dim), device=device, dtype=dtype)

        beta = beta.view(1, hidden_dim)
        base_threshold = base_threshold.view(1, hidden_dim)

        if adaptive_components is not None:
            adaptive_state = (
                adaptive_components["init_state"]
                .view(1, hidden_dim)
                .expand(bsz, -1)
                .clone()
            )
            adaptive_decay = adaptive_components["decay"].view(1, hidden_dim)
            adaptive_spike_increment = adaptive_components["spike_increment"].view(1, hidden_dim)
            adaptive_min = adaptive_components["min_value"].view(1, hidden_dim)
            adaptive_max = adaptive_components["max_value"].view(1, hidden_dim)
        else:
            adaptive_state = None
            adaptive_decay = None
            adaptive_spike_increment = None
            adaptive_min = None
            adaptive_max = None

        spikes_over_time = []
        mem_over_time = []
        effective_threshold_over_time = []
        adaptive_threshold_over_time = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            m_t = attention_mask[:, t].unsqueeze(-1)

            # Ignore padded tokens
            x_t = x_t * m_t

            # Standard membrane integration
            mem = beta * mem + x_t

            # Effective threshold = base + adaptive
            if adaptive_state is not None:
                effective_threshold = base_threshold + adaptive_state
            else:
                effective_threshold = base_threshold

            # Standard firing condition for all neurons
            s_t = spike_fn(mem - effective_threshold)
            s_t = s_t * m_t

            # Reset
            reset_term = s_t.detach() if self.detach_reset else s_t
            if self.reset_to_zero:
                mem = mem * (1.0 - reset_term)
            else:
                mem = mem - reset_term * effective_threshold

            # Adaptive threshold update
            if adaptive_state is not None:
                spikes_for_update = (
                    s_t.detach() if self.adaptive_threshold_cfg.detach_spikes else s_t
                )
                adaptive_state = (
                    adaptive_decay * adaptive_state
                    + adaptive_spike_increment * spikes_for_update
                )
                adaptive_state = clamp_tensor(
                    adaptive_state,
                    min_value=adaptive_min,
                    max_value=adaptive_max,
                )
                adaptive_state = adaptive_state * m_t

            # Keep padded states inactive
            mem = mem * m_t

            spikes_over_time.append(s_t)
            mem_over_time.append(mem)
            effective_threshold_over_time.append(effective_threshold)

            if adaptive_state is not None:
                adaptive_threshold_over_time.append(adaptive_state)
            else:
                adaptive_threshold_over_time.append(torch.zeros_like(mem))

        spikes = torch.stack(spikes_over_time, dim=1)
        membrane = torch.stack(mem_over_time, dim=1)
        effective_threshold_hist = torch.stack(effective_threshold_over_time, dim=1)
        adaptive_threshold_hist = torch.stack(adaptive_threshold_over_time, dim=1)

        return {
            "spikes": spikes,
            "membrane": membrane,
            "beta": beta.squeeze(0),
            "tau": tau,
            "threshold": base_threshold.squeeze(0),
            "effective_threshold": effective_threshold_hist,
            "adaptive_threshold": adaptive_threshold_hist,
        }