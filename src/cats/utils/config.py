from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Returns:
        dict with top-level sections such as:
        - data
        - model
        - lif
        - training
        - runtime
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Top-level YAML structure must be a dictionary, got {type(config).__name__}"
        )

    return config