from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from cats.utils.config import load_config
from cats.builders.model_builder import build_model


def _get_required_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name, None)
    if section is None:
        raise KeyError(f"Missing required config section: '{name}'")
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a dictionary")
    return section


def count_parameters(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print("=" * 100)
    print("Parameter count")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}")
    print("=" * 100)


def print_all_trainable_parameters(model: torch.nn.Module) -> None:
    print("=" * 100)
    print("All trainable parameters")
    print("=" * 100)

    found = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            found = True
            print(
                f"{name:90s} | "
                f"shape={tuple(param.shape)!s:20s} | "
                f"numel={param.numel():8d} | "
                f"mean={param.detach().float().mean().item(): .6f}"
            )

    if not found:
        print("No trainable parameters found.")

    print("=" * 100)


def print_lif_tau_threshold_parameters(model: torch.nn.Module) -> None:
    print("=" * 100)
    print("LIF tau / threshold parameter check")
    print("=" * 100)

    keywords = ["tau", "threshold", "theta"]
    found = False

    for name, param in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in keywords):
            found = True
            print(
                f"{name:90s} | "
                f"shape={tuple(param.shape)!s:20s} | "
                f"requires_grad={str(param.requires_grad):5s} | "
                f"numel={param.numel():8d} | "
                f"mean={param.detach().float().mean().item(): .6f} | "
                f"min={param.detach().float().min().item(): .6f} | "
                f"max={param.detach().float().max().item(): .6f}"
            )

    if not found:
        print("No tau / threshold / theta parameters found in model.named_parameters().")

    print("=" * 100)


def print_lif_modules(model: torch.nn.Module) -> None:
    print("=" * 100)
    print("Detected LIF-related modules")
    print("=" * 100)

    found = False
    for name, module in model.named_modules():
        lname = name.lower()
        class_name = module.__class__.__name__
        if "lif" in lname or "lif" in class_name.lower():
            found = True
            print(f"{name:60s} | class={class_name}")

            for attr in ["tau_spec", "threshold_spec", "adaptive_threshold_cfg"]:
                if hasattr(module, attr):
                    print(f"  - {attr}: {getattr(module, attr)}")

    if not found:
        print("No LIF-related modules found.")

    print("=" * 100)


def gradient_check(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    """
    Optional sanity check:
    Creates a dummy batch, runs forward/backward, then reports gradients
    for tau/threshold parameters.

    This only works if the model forward accepts:
        embeddings=...
        attention_mask=...
    and returns outputs['logits'].
    """

    model.train()
    model.zero_grad(set_to_none=True)

    model_cfg = _get_required_section(config, "model")

    embedding_dim = int(model_cfg["embedding_dim"])
    num_classes = int(model_cfg["num_classes"])

    batch_size = 2
    seq_len = 8

    embeddings = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    outputs = model(
        embeddings=embeddings,
        attention_mask=attention_mask,
    )

    if "logits" not in outputs:
        raise KeyError("Model output does not contain 'logits'.")

    logits = outputs["logits"]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    print("=" * 100)
    print("Gradient check for tau / threshold parameters")
    print(f"Dummy loss: {loss.item():.6f}")
    print("=" * 100)

    keywords = ["tau", "threshold", "theta"]
    found = False

    for name, param in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in keywords):
            found = True
            grad_exists = param.grad is not None
            grad_norm = (
                param.grad.detach().float().norm().item()
                if param.grad is not None
                else 0.0
            )
            grad_max_abs = (
                param.grad.detach().float().abs().max().item()
                if param.grad is not None
                else 0.0
            )

            print(
                f"{name:90s} | "
                f"requires_grad={str(param.requires_grad):5s} | "
                f"grad_exists={str(grad_exists):5s} | "
                f"grad_norm={grad_norm:.6e} | "
                f"grad_max_abs={grad_max_abs:.6e}"
            )

    if not found:
        print("No tau / threshold / theta parameters found for gradient check.")

    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check learnable parameters in a CATS model config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used to build/check the model.",
    )
    parser.add_argument(
        "--grad-check",
        action="store_true",
        help="Run a dummy forward/backward pass and print gradients.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(str(config_path))

    experiment_cfg = _get_required_section(config, "experiment")
    model_cfg = _get_required_section(config, "model")
    routing_cfg = _get_required_section(config, "routing")
    lif_exc_cfg = _get_required_section(config, "lif_exc")
    lif_inh_cfg = _get_required_section(config, "lif_inh")
    classifier_cfg = config.get("classifier", {})
    position_cfg = config.get("position", {})

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    print("=" * 100)
    print("Config")
    print(f"Path   : {config_path.resolve()}")
    print(f"Device : {device}")
    print("=" * 100)

    model = build_model(
        experiment_cfg=experiment_cfg,
        model_cfg=model_cfg,
        routing_cfg=routing_cfg,
        lif_exc_cfg=lif_exc_cfg,
        lif_inh_cfg=lif_inh_cfg,
        classifier_cfg=classifier_cfg,
        position_cfg=position_cfg,
    ).to(device)

    count_parameters(model)
    print_lif_modules(model)
    print_lif_tau_threshold_parameters(model)
    print_all_trainable_parameters(model)

    if args.grad_check:
        gradient_check(model=model, config=config, device=device)


if __name__ == "__main__":
    main()

# python tests/diagnostics/check_learnable_params.py --config configs/core_experiments/sst2/train_carson_sst2_run_001.yaml
# python tests/diagnostics/check_learnable_params.py --config configs/core_experiments/sst2/train_carson_sst2_run_001.yaml --grad-check
