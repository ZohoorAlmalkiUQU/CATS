from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


ParamMode = Literal["shared", "per_channel", "per_group"]
ScalarOrSeq = Union[float, int, Sequence[float], torch.Tensor]


@dataclass
class ParameterSpec:
    """
    Defines a configurable static parameter.

    Attributes:
        init_value:
            Initial raw value(s) before constraint mapping.
            Can be:
                - scalar
                - sequence with shape matching mode
        learnable:
            Whether parameter is trainable.
        mode:
            shared / per_channel / per_group
        min_value:
            Lower bound after constraint mapping.
            Can be scalar or sequence matching mode.
        max_value:
            Optional upper bound after constraint mapping.
            Can be scalar or sequence matching mode.
    """
    init_value: ScalarOrSeq
    learnable: bool = True
    mode: ParamMode = "shared"
    min_value: ScalarOrSeq = 1e-4
    max_value: Optional[ScalarOrSeq] = None


@dataclass
class AdaptiveThresholdConfig:
    """
    Defines dynamic adaptive-threshold behavior.

    This is intentionally separate from ParameterSpec because it is a
    time-evolving state, not just a static learnable parameter.

    Attributes:
        enabled:
            Whether adaptive thresholding is active.
        mode:
            shared / per_channel / per_group
        init:
            Initial adaptive threshold state.
            Can be scalar or sequence matching mode.
        decay:
            Decay factor applied every timestep.
            Usually in [0, 1].
        spike_increment:
            Amount added to adaptive threshold after a spike.
            Can be scalar or sequence matching mode.
        min_value:
            Lower clamp bound for adaptive threshold state.
        max_value:
            Upper clamp bound for adaptive threshold state.
        detach_spikes:
            Whether spike tensor should be detached before updating the
            adaptive threshold state for stability.
    """
    enabled: bool = False
    mode: ParamMode = "per_group"
    init: ScalarOrSeq = 0.0
    decay: ScalarOrSeq = 0.95
    spike_increment: ScalarOrSeq = 0.1
    min_value: ScalarOrSeq = 0.0
    max_value: ScalarOrSeq = 5.0
    detach_spikes: bool = True


def _make_shape(mode: ParamMode, hidden_dim: int, num_groups: int) -> tuple[int, ...]:
    if mode == "shared":
        return (1,)
    if mode == "per_channel":
        return (hidden_dim,)
    if mode == "per_group":
        return (num_groups,)
    raise ValueError(f"Unsupported mode: {mode}")


def _numel_from_mode(mode: ParamMode, hidden_dim: int, num_groups: int) -> int:
    return _make_shape(mode, hidden_dim, num_groups)[0]


def _to_1d_tensor(
    value: ScalarOrSeq,
    *,
    expected_numel: Optional[int],
    name: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert scalar / sequence / tensor config value into a 1D tensor.

    Rules:
    - if expected_numel is provided:
        - scalar -> repeated to expected_numel
        - sequence/tensor of length expected_numel -> used directly
        - sequence/tensor of length 1 -> repeated to expected_numel
    - if expected_numel is None:
        - scalar -> tensor([value])
        - sequence/tensor -> flattened as-is
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().to(dtype=dtype).flatten()
    elif isinstance(value, (float, int)):
        if expected_numel is None:
            tensor = torch.tensor([float(value)], dtype=dtype)
        else:
            tensor = torch.full((expected_numel,), float(value), dtype=dtype)
    else:
        tensor = torch.tensor(list(value), dtype=dtype).flatten()

    if expected_numel is not None:
        if tensor.numel() == 1 and expected_numel != 1:
            tensor = tensor.expand(expected_numel).clone()

        if tensor.numel() != expected_numel:
            raise ValueError(
                f"{name} has {tensor.numel()} value(s), but expected {expected_numel} "
                f"for the selected mode."
            )

    return tensor

def build_raw_parameter(
    spec: ParameterSpec,
    hidden_dim: int,
    num_groups: int = 2,
) -> nn.Parameter:
    """
    Build an unconstrained raw parameter tensor.

    The raw parameter shape is determined by spec.mode:
        - shared      -> [1]
        - per_channel -> [hidden_dim]
        - per_group   -> [num_groups]

    spec.init_value may be:
        - scalar
        - sequence matching the mode size
    """
    expected_numel = _numel_from_mode(spec.mode, hidden_dim, num_groups)
    raw = _to_1d_tensor(
        spec.init_value,
        expected_numel=expected_numel,
        name="init_value",
        dtype=torch.float32,
    )
    return nn.Parameter(raw, requires_grad=spec.learnable)


def build_adaptive_state_tensor(
    cfg: AdaptiveThresholdConfig,
    hidden_dim: int,
    num_groups: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the initial adaptive-threshold state tensor before expansion.

    Shape depends on cfg.mode:
        - shared      -> [1]
        - per_channel -> [hidden_dim]
        - per_group   -> [num_groups]
    """
    expected_numel = _numel_from_mode(cfg.mode, hidden_dim, num_groups)
    state = _to_1d_tensor(
        cfg.init,
        expected_numel=expected_numel,
        name="adaptive_threshold.init",
        dtype=dtype,
    )
    if device is not None:
        state = state.to(device=device)
    return state


def _prepare_bound_tensor(
    bound: ScalarOrSeq,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """
    Convert bound into tensor broadcastable to reference.

    Supports:
    - scalar
    - full match
    - last-dimension match (per_channel)
    """
    bound_tensor = _to_1d_tensor(
        bound,
        expected_numel=1 if isinstance(bound, (int, float)) else None,
        name=name,
        dtype=reference.dtype,
    ).to(device=reference.device)

    # Case 1: scalar
    if bound_tensor.numel() == 1:
        return bound_tensor.view(*([1] * reference.ndim))

    # Case 2: exact full match
    if bound_tensor.numel() == reference.numel():
        return bound_tensor.view_as(reference)

    # Case 3: last-dim match (IMPORTANT FIX)
    if reference.ndim >= 1 and bound_tensor.numel() == reference.shape[-1]:
        shape = [1] * reference.ndim
        shape[-1] = reference.shape[-1]
        return bound_tensor.view(*shape)

    raise ValueError(
        f"{name} has {bound_tensor.numel()} value(s), but expected:\n"
        f"- 1 (scalar)\n"
        f"- {reference.shape[-1]} (last-dim match)\n"
        f"- {reference.numel()} (full match)\n"
        f"for tensor shape {tuple(reference.shape)}"
    )

def constrain_positive(
    raw: torch.Tensor,
    min_value: ScalarOrSeq = 1e-4,
) -> torch.Tensor:
    """
    Map raw parameter to strictly positive values.

    Supports scalar or elementwise min_value.
    """
    min_tensor = _prepare_bound_tensor(min_value, raw, name="min_value")
    return F.softplus(raw) + min_tensor


def constrain_bounded(
    raw: torch.Tensor,
    min_value: ScalarOrSeq,
    max_value: ScalarOrSeq,
) -> torch.Tensor:
    """
    Map raw parameter into [min_value, max_value].

    Supports scalar or elementwise min/max values.
    """
    min_tensor = _prepare_bound_tensor(min_value, raw, name="min_value")
    max_tensor = _prepare_bound_tensor(max_value, raw, name="max_value")

    if torch.any(max_tensor <= min_tensor):
        raise ValueError("All max_value entries must be > corresponding min_value entries.")

    return min_tensor + (max_tensor - min_tensor) * torch.sigmoid(raw)


def clamp_tensor(
    value: torch.Tensor,
    min_value: ScalarOrSeq,
    max_value: ScalarOrSeq,
) -> torch.Tensor:
    """
    Elementwise clamp for already-materialized tensors.

    Useful for adaptive threshold state updates.
    """
    min_tensor = _prepare_bound_tensor(min_value, value, name="min_value")
    max_tensor = _prepare_bound_tensor(max_value, value, name="max_value")

    if torch.any(max_tensor <= min_tensor):
        raise ValueError("All max_value entries must be > corresponding min_value entries.")

    return torch.max(torch.min(value, max_tensor), min_tensor)


def expand_parameter(
    param: torch.Tensor,
    mode: ParamMode,
    hidden_dim: int,
    group_assignments: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Expand parameter to shape [D].

    For per_group mode:
        group_assignments should be a tensor of shape [D]
        containing group ids in [0, num_groups-1].
    """
    if param.ndim != 1:
        raise ValueError(f"param must be 1D, got shape {tuple(param.shape)}")

    if mode == "shared":
        if param.numel() != 1:
            raise ValueError(
                f"shared parameter must have size 1, got {param.numel()}"
            )
        return param.expand(hidden_dim)

    if mode == "per_channel":
        if param.numel() != hidden_dim:
            raise ValueError(
                f"per_channel parameter has size {param.numel()} but hidden_dim={hidden_dim}"
            )
        return param

    if mode == "per_group":
        if group_assignments is None:
            raise ValueError("group_assignments is required for per_group mode")

        if group_assignments.ndim != 1 or group_assignments.numel() != hidden_dim:
            raise ValueError(
                "group_assignments must have shape [hidden_dim] for per_group mode"
            )

        if group_assignments.dtype != torch.long:
            group_assignments = group_assignments.to(dtype=torch.long)

        if group_assignments.numel() > 0:
            min_gid = int(group_assignments.min().item())
            max_gid = int(group_assignments.max().item())
            if min_gid < 0 or max_gid >= param.numel():
                raise ValueError(
                    f"group_assignments contains ids outside valid range [0, {param.numel() - 1}]"
                )

        return param[group_assignments]

    raise ValueError(f"Unsupported mode: {mode}")


def build_spec_from_config(
    cfg: dict | None,
    default_init: ScalarOrSeq,
    default_learnable: bool,
    default_mode: ParamMode,
    default_min: ScalarOrSeq,
    default_max: Optional[ScalarOrSeq],
) -> ParameterSpec:
    cfg = cfg or {}

    mode = cfg.get("mode", default_mode)
    if mode not in {"shared", "per_channel", "per_group"}:
        raise ValueError(
            f"Unsupported mode in config: {mode}. "
            f"Expected one of: shared, per_channel, per_group."
        )

    return ParameterSpec(
        init_value=cfg.get("init", default_init),
        learnable=cfg.get("learnable", default_learnable),
        mode=mode,
        min_value=cfg.get("min", default_min),
        max_value=cfg.get("max", default_max),
    )


def build_adaptive_threshold_config_from_config(
    cfg: dict | None,
    default_enabled: bool = False,
    default_mode: ParamMode = "per_group",
    default_init: ScalarOrSeq = 0.0,
    default_decay: ScalarOrSeq = 0.95,
    default_spike_increment: ScalarOrSeq = 0.1,
    default_min: ScalarOrSeq = 0.0,
    default_max: ScalarOrSeq = 5.0,
    default_detach_spikes: bool = True,
) -> AdaptiveThresholdConfig:
    cfg = cfg or {}

    mode = cfg.get("mode", default_mode)
    if mode not in {"shared", "per_channel", "per_group"}:
        raise ValueError(
            f"Unsupported adaptive threshold mode in config: {mode}. "
            f"Expected one of: shared, per_channel, per_group."
        )

    return AdaptiveThresholdConfig(
        enabled=cfg.get("enabled", default_enabled),
        mode=mode,
        init=cfg.get("init", default_init),
        decay=cfg.get("decay", default_decay),
        spike_increment=cfg.get("spike_increment", default_spike_increment),
        min_value=cfg.get("min", default_min),
        max_value=cfg.get("max", default_max),
        detach_spikes=cfg.get("detach_spikes", default_detach_spikes),
    )