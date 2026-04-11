from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class CATSEncoder(nn.Module):
    """
    Context-Aware Token-to-Spike encoder.

    Pipeline:
        1) Normalize input embeddings
        2) Route tokens with the selected router
        3) Build shared / excitatory / inhibitory features
        4) Convert currents into spikes with separate LIF modules
        5) Pool excitatory / inhibitory representations
        6) Return a rich dictionary for classification + metrics/logging
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        excitatory_ratio: float,
        num_groups: int,
        router: nn.Module,
        lif_exc: nn.Module,
        lif_inh: nn.Module,
        use_input_layernorm: bool = True,
        use_shared_projection: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim <= 1:
            raise ValueError(f"hidden_dim must be > 1, got {hidden_dim}")

        if not (0.0 < excitatory_ratio < 1.0):
            raise ValueError(
                f"excitatory_ratio must be in (0, 1), got {excitatory_ratio}"
            )

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.excitatory_ratio = float(excitatory_ratio)

        self.num_groups = int(num_groups)
        if self.num_groups < 2:
            raise ValueError(
                f"CATSEncoder currently expects num_groups >= 2, got {self.num_groups}"
            )

        self.exc_dim = max(1, int(round(hidden_dim * excitatory_ratio)))
        self.inh_dim = hidden_dim - self.exc_dim

        if self.inh_dim <= 0:
            raise ValueError(
                f"Invalid split: exc_dim={self.exc_dim}, inh_dim={self.inh_dim}. "
                f"Adjust hidden_dim or excitatory_ratio."
            )

        self.router = router
        self.lif_exc = lif_exc
        self.lif_inh = lif_inh

        # Strong safety check: each branch LIF must match its branch dimension.
        if getattr(self.lif_exc, "hidden_dim", None) != self.exc_dim:
            raise ValueError(
                f"lif_exc.hidden_dim must equal exc_dim={self.exc_dim}, "
                f"got {getattr(self.lif_exc, 'hidden_dim', None)}"
            )

        if getattr(self.lif_inh, "hidden_dim", None) != self.inh_dim:
            raise ValueError(
                f"lif_inh.hidden_dim must equal inh_dim={self.inh_dim}, "
                f"got {getattr(self.lif_inh, 'hidden_dim', None)}"
            )

        self.input_norm = (
            nn.LayerNorm(embedding_dim) if use_input_layernorm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.use_shared_projection = bool(use_shared_projection)

        if self.use_shared_projection:
            self.shared_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.shared_proj = None

        self.exc_proj = nn.Linear(embedding_dim, self.exc_dim)
        self.inh_proj = nn.Linear(embedding_dim, self.inh_dim)

    def _build_mask(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns mask shaped [B, T, 1] with same dtype/device as x.
        """
        bsz, seq_len, _ = x.shape

        if attention_mask is None:
            return torch.ones(
                (bsz, seq_len, 1),
                dtype=x.dtype,
                device=x.device,
            )

        if attention_mask.shape != (bsz, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(bsz, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        return attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)

    def _masked_mean_pool(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Masked mean pooling over the time dimension.

        Args:
            x: [B, T, D]
            attention_mask: [B, T] or None

        Returns:
            pooled: [B, D]
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        if attention_mask is None:
            return x.mean(dim=1)

        mask = self._build_mask(x, attention_mask)  # [B, T, 1]
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _extract_routed_embeddings(
        self,
        routing_out: Dict[str, torch.Tensor],
        fallback_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Accept several possible router output names for routed embeddings.
        """
        for key in ("routed_embeddings", "routed_x", "output", "x"):
            value = routing_out.get(key, None)
            if value is not None:
                return value
        return fallback_x

    def _extract_routing_weights(
        self,
        routing_out: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Accept several possible router output names for routing weights.
        """
        for key in ("routing_weights", "weights", "scores"):
            value = routing_out.get(key, None)
            if value is not None:
                return value
        return None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, embedding_dim]
            attention_mask: [B, T] or None

        Returns:
            Rich dictionary used by:
            - model.py
            - train.py
            - metrics.py
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        bsz, seq_len, input_dim = x.shape
        if input_dim != self.embedding_dim:
            raise ValueError(
                f"Expected input embedding_dim={self.embedding_dim}, got {input_dim}"
            )

        x = self.input_norm(x)
        x = self.dropout(x)

        mask3 = self._build_mask(x, attention_mask)

        # --------------------------------------------------
        # Router
        # --------------------------------------------------
        routing_out = self.router(x, attention_mask=attention_mask)
        if not isinstance(routing_out, dict):
            raise TypeError(
                f"Router must return a dict, got {type(routing_out).__name__}"
            )

        routed_x = self._extract_routed_embeddings(routing_out, fallback_x=x)
        routing_weights = self._extract_routing_weights(routing_out)

        if routed_x.ndim != 3 or routed_x.shape[:2] != (bsz, seq_len):
            raise ValueError(
                "Router returned invalid routed embeddings shape. "
                f"Expected [B, T, D], got {tuple(routed_x.shape)}"
            )

        # --------------------------------------------------
        # Feature construction
        # --------------------------------------------------
        shared_features: Optional[torch.Tensor]
        if self.use_shared_projection:
            shared_features = self.shared_proj(routed_x)  # [B, T, H]
            shared_features = shared_features * mask3
        else:
            shared_features = None

        exc_features = self.exc_proj(routed_x) * mask3  # [B, T, D_exc]
        inh_features = self.inh_proj(routed_x) * mask3  # [B, T, D_inh]

        # --------------------------------------------------
        # Routing-driven current construction
        # --------------------------------------------------
        if routing_weights is not None:
            if routing_weights.ndim != 3 or routing_weights.shape[:2] != (bsz, seq_len):
                raise ValueError(
                    "routing_weights must have shape [B, T, G] matching the batch/time dims"
                )

            if routing_weights.size(-1) >= 2:
                exc_gate = routing_weights[..., 0:1]
                inh_gate = routing_weights[..., 1:2]
            else:
                exc_gate = torch.ones((bsz, seq_len, 1), dtype=x.dtype, device=x.device)
                inh_gate = torch.ones((bsz, seq_len, 1), dtype=x.dtype, device=x.device)
        else:
            exc_gate = torch.ones((bsz, seq_len, 1), dtype=x.dtype, device=x.device)
            inh_gate = torch.ones((bsz, seq_len, 1), dtype=x.dtype, device=x.device)

        if shared_features is not None:
            shared_exc = shared_features[..., : self.exc_dim]
            shared_inh = shared_features[..., self.exc_dim :]

            exc_current = (shared_exc + exc_features) * exc_gate
            inh_current = (shared_inh + inh_features) * inh_gate
        else:
            exc_current = exc_features * exc_gate
            inh_current = inh_features * inh_gate

        exc_current = exc_current * mask3
        inh_current = inh_current * mask3

        grouped_current = torch.cat([exc_current, inh_current], dim=-1)

        # --------------------------------------------------
        # Branch-specific group assignments
        # --------------------------------------------------
        # Each branch LIF sees only one population, so its per-group mapping
        # must match that branch dimensionality, not the concatenated hidden_dim.
        exc_group_assignments = torch.zeros(
            self.exc_dim,
            device=x.device,
            dtype=torch.long,
        )
        inh_group_assignments = torch.zeros(
            self.inh_dim,
            device=x.device,
            dtype=torch.long,
        )

        # Full assignments remain useful for logging/analysis after concatenation.
        group_assignments = torch.cat(
            [
                torch.zeros(self.exc_dim, device=x.device, dtype=torch.long),
                torch.ones(self.inh_dim, device=x.device, dtype=torch.long),
            ],
            dim=0,
        )

        # --------------------------------------------------
        # LIF populations
        # --------------------------------------------------
        exc_out = self.lif_exc(
            exc_current,
            attention_mask=attention_mask,
            group_assignments=exc_group_assignments,
        )
        inh_out = self.lif_inh(
            inh_current,
            attention_mask=attention_mask,
            group_assignments=inh_group_assignments,
        )

        if not isinstance(exc_out, dict):
            raise TypeError(
                f"lif_exc must return a dict, got {type(exc_out).__name__}"
            )
        if not isinstance(inh_out, dict):
            raise TypeError(
                f"lif_inh must return a dict, got {type(inh_out).__name__}"
            )

        required_lif_keys = ("spikes", "membrane", "beta", "tau", "threshold")
        for key in required_lif_keys:
            if key not in exc_out:
                raise KeyError(f"lif_exc output is missing required key: '{key}'")
            if key not in inh_out:
                raise KeyError(f"lif_inh output is missing required key: '{key}'")

        exc_spikes = exc_out["spikes"] * mask3
        inh_spikes = inh_out["spikes"] * mask3
        spikes = torch.cat([exc_spikes, inh_spikes], dim=-1)

        exc_membrane = exc_out["membrane"] * mask3
        inh_membrane = inh_out["membrane"] * mask3
        membrane = torch.cat([exc_membrane, inh_membrane], dim=-1)

        # --------------------------------------------------
        # Pooling
        # --------------------------------------------------
        pooled_exc = self._masked_mean_pool(exc_membrane, attention_mask)
        pooled_inh = self._masked_mean_pool(inh_membrane, attention_mask)
        pooled_concat = torch.cat([pooled_exc, pooled_inh], dim=-1)

        pooled_subtractive = pooled_exc.mean(dim=-1, keepdim=True) - pooled_inh.mean(
            dim=-1, keepdim=True
        )

        # --------------------------------------------------
        # Final return
        # --------------------------------------------------
        return {
            "routed_embeddings": routed_x,
            "shared_features": shared_features,
            "exc_features": exc_features,
            "inh_features": inh_features,

            "exc_current": exc_current,
            "inh_current": inh_current,
            "grouped_current": grouped_current,
            "group_assignments": group_assignments,
            "exc_group_assignments": exc_group_assignments,
            "inh_group_assignments": inh_group_assignments,

            "exc_spikes": exc_spikes,
            "inh_spikes": inh_spikes,
            "spikes": spikes,

            "exc_membrane": exc_membrane,
            "inh_membrane": inh_membrane,
            "membrane": membrane,

            "pooled_exc": pooled_exc,
            "pooled_inh": pooled_inh,
            "pooled_features": pooled_concat,
            "pooled_subtractive": pooled_subtractive,

            "exc_beta": exc_out["beta"],
            "inh_beta": inh_out["beta"],
            "beta": torch.cat([exc_out["beta"], inh_out["beta"]], dim=0),

            "exc_tau": exc_out["tau"],
            "inh_tau": inh_out["tau"],
            "tau": torch.cat([exc_out["tau"], inh_out["tau"]], dim=0),

            "exc_threshold": exc_out["threshold"],
            "inh_threshold": inh_out["threshold"],
            "threshold": torch.cat([exc_out["threshold"], inh_out["threshold"]], dim=0),

            "exc_effective_threshold": exc_out.get("effective_threshold", None),
            "inh_effective_threshold": inh_out.get("effective_threshold", None),

            "exc_adaptive_threshold": exc_out.get("adaptive_threshold", None),
            "inh_adaptive_threshold": inh_out.get("adaptive_threshold", None),

            "routing_weights": routing_weights,
            "routing_logits": routing_out.get("routing_logits", None),
            "routing_confidence": routing_out.get("routing_confidence", None),
            "context_vector": routing_out.get("context_vector", None),
            "agreement": routing_out.get("agreement", None),
            "group_prototypes": routing_out.get("group_prototypes", None),
        }