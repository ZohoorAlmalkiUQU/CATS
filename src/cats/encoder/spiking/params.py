from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


ParamMode = Literal["shared", "per_channel", "per_group"]


@dataclass
class ParameterSpec:
    """
    Defines a configurable parameter.

    Attributes:
        init_value: Initial value before constraint mapping.
        learnable: Whether parameter is trainable.
        mode: shared / per_channel / per_group
        min_value: Lower bound after constraint mapping.
        max_value: Optional upper bound after constraint mapping.
    """
    init_value: float
    learnable: bool = True
    mode: ParamMode = "shared"
    min_value: float = 1e-4
    max_value: Optional[float] = None


def _make_shape(mode: ParamMode, hidden_dim: int, num_groups: int) -> tuple[int, ...]:
    if mode == "shared":
        return (1,)
    if mode == "per_channel":
        return (hidden_dim,)
    if mode == "per_group":
        return (num_groups,)
    raise ValueError(f"Unsupported mode: {mode}")


def build_raw_parameter(
    spec: ParameterSpec,
    hidden_dim: int,
    num_groups: int = 2,
) -> nn.Parameter:
    """
    Build an unconstrained raw parameter tensor.
    Constraint is applied later with constrain_positive/constrain_bounded.
    """
    shape = _make_shape(spec.mode, hidden_dim, num_groups)
    raw = torch.full(shape, float(spec.init_value), dtype=torch.float32)
    return nn.Parameter(raw, requires_grad=spec.learnable)


def constrain_positive(raw: torch.Tensor, min_value: float = 1e-4) -> torch.Tensor:
    """
    Map raw parameter to strictly positive values.
    """
    return torch.nn.functional.softplus(raw) + min_value


def constrain_bounded(
    raw: torch.Tensor,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    """
    Map raw parameter into [min_value, max_value].
    """
    if max_value <= min_value:
        raise ValueError("max_value must be > min_value")
    return min_value + (max_value - min_value) * torch.sigmoid(raw)


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
    if mode == "shared":
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
        return param[group_assignments]

    raise ValueError(f"Unsupported mode: {mode}")

def build_spec_from_config(
    cfg: dict | None,
    default_init: float,
    default_learnable: bool,
    default_mode: ParamMode,
    default_min: float,
    default_max: Optional[float],
) -> ParameterSpec:
    cfg = cfg or {}

    return ParameterSpec(
        init_value=cfg.get("init", default_init),
        learnable=cfg.get("learnable", default_learnable),
        mode=cfg.get("mode", default_mode),
        min_value=cfg.get("min", default_min),
        max_value=cfg.get("max", default_max),
    )