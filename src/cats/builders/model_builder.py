from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

import torch.nn as nn

from cats.encoder.spiking.lif import LIFLayer
from cats.encoder.routing.identity import IdentityRouter
from cats.encoder.routing.linear import LinearRouter
from cats.encoder.routing.carson import CARSONRouter
from cats.builders.build_positional_encoding import build_positional_encoding


MODEL_REGISTRY = {
    "cats": "cats.model.CATSClassifier",
}

ROUTER_REGISTRY = {
    "identity": IdentityRouter,
    "linear": LinearRouter,
    "carson": CARSONRouter,
    # "carson_v1": CARSONRouterV1,
    # "carson_v2": CARSONRouterV2,
}

def import_from_string(path: str):
    """
    Import class/function from full dotted path.
    Example:
        cats.model.CATSClassifier
    """
    if "." not in path:
        raise ValueError(f"Invalid import path: {path}")

    module_name, attr_name = path.rsplit(".", 1)
    module = import_module(module_name)
    if not hasattr(module, attr_name):
        raise ImportError(f"Module '{module_name}' has no attribute '{attr_name}'")
    return getattr(module, attr_name)


def build_router(
    routing_type: str,
    routing_cfg: Dict[str, Any],
    position_cfg: Dict[str, Any] | None,
    *,
    embedding_dim: int,
    hidden_dim: int,
    num_groups: int,
) -> nn.Module:
    """
    Build router from routing_type + routing.kwargs.

    The builder injects common model-level dimensions automatically
    unless user overrides them explicitly in routing.kwargs.

    Positional encoding is also built once here and passed into the router
    so that all routers can use the same positional encoding mechanism
    under matched experimental settings.
    """
    if routing_type not in ROUTER_REGISTRY:
        raise ValueError(
            f"Unsupported routing_type='{routing_type}'. "
            f"Available: {list(ROUTER_REGISTRY.keys())}"
        )

    RouterClass = ROUTER_REGISTRY[routing_type]
    kwargs = dict(routing_cfg.get("kwargs", {}) or {})

    pos_enc, use_pos_enc = build_positional_encoding(
        embedding_dim=embedding_dim,
        position_cfg=position_cfg,
    )

    kwargs.setdefault("embedding_dim", embedding_dim)
    kwargs.setdefault("hidden_dim", hidden_dim)
    kwargs.setdefault("num_groups", num_groups)

    kwargs.setdefault("pos_enc", pos_enc)
    kwargs.setdefault("use_pos_enc", use_pos_enc)

    return RouterClass(**kwargs)


def build_lif_module(
    lif_cfg: Dict[str, Any],
    *,
    hidden_dim: int,
    num_groups: int,
) -> nn.Module:
    """
    Build one LIF module.

    Important design choice:
    - If a single LIF layer internally uses group assignments, pass num_groups > 1.
    - If you instantiate separate LIF layers for excitatory and inhibitory populations,
      num_groups=1 is the cleaner choice for each individual LIF layer.
    """
    return LIFLayer(
        hidden_dim=hidden_dim,
        lif_config=lif_cfg,
        num_groups=num_groups,
        reset_to_zero=bool(lif_cfg.get("reset_to_zero", True)),
        detach_reset=bool(lif_cfg.get("detach_reset", False)),
    )

def build_model(
    experiment_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    routing_cfg: Dict[str, Any],
    lif_exc_cfg: Dict[str, Any],
    lif_inh_cfg: Dict[str, Any],
    classifier_cfg: Dict[str, Any] | None = None,
    position_cfg: Dict[str, Any] | None = None,
) -> nn.Module:
    classifier_cfg = classifier_cfg or {}

    model_name = experiment_cfg["model_name"]
    routing_type = experiment_cfg["routing_type"]

    class_path = model_cfg.get("class_path")
    if class_path is None:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_name='{model_name}'. "
                f"Add it to MODEL_REGISTRY or provide model.class_path."
            )
        class_path = MODEL_REGISTRY[model_name]

    ModelClass = import_from_string(class_path)

    embedding_dim = int(model_cfg["embedding_dim"])
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    num_classes = int(model_cfg["num_classes"])
    excitatory_ratio = float(model_cfg.get("excitatory_ratio", 0.5))
    num_groups = int(model_cfg.get("num_groups", 2))

    model_kwargs = dict(model_cfg.get("kwargs", {}) or {})
    inhibition_cfg = dict(model_cfg.get("inhibition", {}) or {})
    spiking_cfg = dict(model_cfg.get("spiking", {}) or {})

    inhibition_enabled = bool(inhibition_cfg.get("enabled", True))
    spiking_enabled = bool(spiking_cfg.get("enabled", True))

    if inhibition_enabled:
        if not (0.0 < excitatory_ratio < 1.0):
            raise ValueError(
                f"excitatory_ratio must be in (0, 1), got {excitatory_ratio}"
            )

        exc_dim = max(1, int(round(hidden_dim * excitatory_ratio)))
        inh_dim = hidden_dim - exc_dim

        if inh_dim <= 0:
            raise ValueError(
                f"Invalid split: hidden_dim={hidden_dim}, "
                f"excitatory_ratio={excitatory_ratio}, "
                f"exc_dim={exc_dim}, inh_dim={inh_dim}"
            )
    else:
        exc_dim = hidden_dim
        inh_dim = 0

    router = build_router(
        routing_type=routing_type,
        routing_cfg=routing_cfg,
        position_cfg=position_cfg,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_groups=num_groups,
    )

    lif_exc = build_lif_module(
        lif_exc_cfg,
        hidden_dim=exc_dim,
        num_groups=1,
    )

    lif_inh = None

    if inhibition_enabled:
        lif_inh = build_lif_module(
            lif_inh_cfg,
            hidden_dim=inh_dim,
            num_groups=1,
        )

    model = ModelClass(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        excitatory_ratio=excitatory_ratio,
        num_groups=num_groups,
        router=router,
        lif_exc=lif_exc,
        lif_inh=lif_inh,
        classifier_cfg=classifier_cfg,
        inhibition_enabled=inhibition_enabled,
        spiking_enabled=spiking_enabled,
        **model_kwargs,
    )

    return model