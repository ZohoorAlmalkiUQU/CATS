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
    Defines a configurable parameter.

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
    expected_numel: int,
    name: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert scalar / sequence / tensor config value into a 1D tensor
    with length expected_numel.

    Rules:
    - scalar -> repeated to expected_numel
    - sequence/tensor of length expected_numel -> used directly
    - sequence/tensor of length 1 -> repeated to expected_numel
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().to(dtype=dtype).flatten()
    elif isinstance(value, (float, int)):
        tensor = torch.full((expected_numel,), float(value), dtype=dtype)
    else:
        tensor = torch.tensor(list(value), dtype=dtype).flatten()

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


def _prepare_bound_tensor(
    bound: ScalarOrSeq,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """
    Convert a scalar / sequence bound into a tensor broadcastable to reference.
    """
    bound_tensor = _to_1d_tensor(
        bound,
        expected_numel=reference.numel(),
        name=name,
        dtype=reference.dtype,
    ).to(device=reference.device)
    return bound_tensor.view_as(reference)


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