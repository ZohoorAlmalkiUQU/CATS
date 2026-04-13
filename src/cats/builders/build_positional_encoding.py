from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cats.encoder.position import PositionalEncoding, RotaryPositionalEncoding


def build_positional_encoding(
    embedding_dim: int,
    position_cfg: Optional[Dict[str, Any]],
) -> Tuple[Optional[PositionalEncoding], bool]:
    """
    Build positional encoding module from config.

    Returns:
        pos_enc: module or None
        use_pos_enc: bool flag
    """
    if not position_cfg:
        return None, False

    use_pos_enc = bool(position_cfg.get("use", False))
    pos_type = str(position_cfg.get("type", "none")).lower()
    kwargs = position_cfg.get("kwargs", {}) or {}

    if not use_pos_enc or pos_type == "none":
        return None, False

    if pos_type == "rope":
        pos_enc = RotaryPositionalEncoding(
            dim=embedding_dim,
            base=float(kwargs.get("base", 10000.0)),
        )
        return pos_enc, True

    raise ValueError(f"Unsupported positional encoding type: {pos_type}")