from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing.identity import IdentityRouter
from .spiking.lif import LIFLayer
from ..utils.mask import masked_mean


class CATSEncoder(nn.Module):
    """
    CATS encoder with routing-before-spiking design.

    Pipeline:
        continuous embeddings [B, T, D_in]
        -> LayerNorm
        -> router on token sequence
        -> token-to-group current construction
        -> first SNN layer (LIF spike conversion)
        -> masked readout
        -> pooled representation

    Updated design:
    - supports routers that only return routed_x
    - supports CARSON v2 routers that additionally return:
        * exc_features
        * inh_features
        * shared_features
    - builds excitatory and inhibitory currents from different feature paths
      instead of reusing the same representation for both groups
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        excitatory_ratio: float = 0.5,
        use_layernorm: bool = True,
        router: Optional[nn.Module] = None,
        spiking_layer: Optional[nn.Module] = None,
        lif_config: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if not (0.0 < excitatory_ratio < 1.0):
            raise ValueError("excitatory_ratio must be in (0, 1)")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.excitatory_ratio = excitatory_ratio

        self.norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()

        # ------------------------------------------------------------------
        # Split hidden neurons into excitatory and inhibitory groups
        # ------------------------------------------------------------------
        num_exc = int(round(hidden_dim * excitatory_ratio))
        num_exc = max(1, min(hidden_dim - 1, num_exc))
        num_inh = hidden_dim - num_exc

        self.num_exc = num_exc
        self.num_inh = num_inh

        # ------------------------------------------------------------------
        # Current projection heads
        #
        # Key update:
        # - excitatory current is projected from excitatory features
        # - inhibitory current is projected from inhibitory features
        # This avoids using the same routed representation for both groups.
        # ------------------------------------------------------------------
        self.exc_proj = nn.Linear(embedding_dim, num_exc)
        self.inh_proj = nn.Linear(embedding_dim, num_inh)

        # Optional shared projection for fallback / non-specialized routers
        self.shared_exc_proj = nn.Linear(embedding_dim, num_exc)
        self.shared_inh_proj = nn.Linear(embedding_dim, num_inh)

        # Learnable gains
        self.exc_gain = nn.Parameter(torch.ones(num_exc))
        self.inh_gain = nn.Parameter(torch.ones(num_inh))

        # Optional learned bias terms on current magnitudes
        self.exc_bias = nn.Parameter(torch.zeros(num_exc))
        self.inh_bias = nn.Parameter(torch.zeros(num_inh))

        # group assignments: 0 = excitatory, 1 = inhibitory
        group_assignments = torch.zeros(hidden_dim, dtype=torch.long)
        group_assignments[num_exc:] = 1
        self.register_buffer("group_assignments", group_assignments, persistent=False)

        self.router = router or IdentityRouter()
        self.spiking_layer = spiking_layer or LIFLayer(
            hidden_dim=hidden_dim,
            lif_config=lif_config,
            num_groups=2,
        )

    def _build_group_currents(
        self,
        routed_x: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        exc_features: Optional[torch.Tensor] = None,
        inh_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert continuous token embeddings into grouped neuron currents.

        Args:
            routed_x: [B, T, D_in]
                Backward-compatible routed representation.
            routing_weights: [B, T, 2] or None
                routing_weights[..., 0] = excitatory weight
                routing_weights[..., 1] = inhibitory weight
            exc_features: [B, T, D_in] or None
                Specialized excitatory features from router.
            inh_features: [B, T, D_in] or None
                Specialized inhibitory features from router.

        Returns:
            current: [B, T, H]
        """
        # --------------------------------------------------------------
        # Fallback behavior:
        # if router does not provide specialized features, use routed_x
        # through dedicated fallback projections.
        # --------------------------------------------------------------
        if exc_features is None:
            exc_drive = self.shared_exc_proj(routed_x)  # [B,T,E]
        else:
            exc_drive = self.exc_proj(exc_features)     # [B,T,E]

        if inh_features is None:
            inh_drive = self.shared_inh_proj(routed_x)  # [B,T,I]
        else:
            inh_drive = self.inh_proj(inh_features)     # [B,T,I]

        # --------------------------------------------------------------
        # Convert both branches to positive magnitudes first
        #
        # Why:
        # - excitatory branch should contribute positive drive
        # - inhibitory branch should learn useful magnitude, then we apply
        #   the negative sign explicitly afterward
        #
        # This is much more stable than letting inhibitory projection freely
        # produce arbitrary signed values and then forcing abs afterward.
        # --------------------------------------------------------------
        exc = F.softplus(exc_drive + self.exc_bias.view(1, 1, self.num_exc))
        inh = F.softplus(inh_drive + self.inh_bias.view(1, 1, self.num_inh))

        # Apply learnable gains
        exc = exc * self.exc_gain.view(1, 1, self.num_exc)
        inh = inh * self.inh_gain.view(1, 1, self.num_inh)

        # Make inhibitory neurons suppressive
        # inh = -inh

        # --------------------------------------------------------------
        # Apply routing weights if available
        # --------------------------------------------------------------
        if routing_weights is not None:
            if routing_weights.shape[-1] != 2:
                raise ValueError(
                    f"Expected routing_weights last dim = 2, got {routing_weights.shape[-1]}"
                )

            w_exc = routing_weights[..., 0].unsqueeze(-1)   # [B,T,1]
            w_inh = routing_weights[..., 1].unsqueeze(-1)   # [B,T,1]

            exc = exc * w_exc
            inh = inh * w_inh

        current = torch.cat([exc, inh], dim=-1)  # [B,T,H]
        return current

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, D_in]
            attention_mask: [B, T] or None

        Returns:
            dict with encoder outputs and diagnostics
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        x = self.norm(x)

        # --------------------------------------------------------------
        # Routing on continuous embeddings
        # --------------------------------------------------------------
        routing_out = self.router(x, attention_mask=attention_mask)

        routed_x = routing_out["routed_x"]  # [B, T, D_in]
        routing_weights = routing_out.get("routing_weights", None)

        # New CARSON v2-specific features, if available
        exc_features = routing_out.get("exc_features", None)
        inh_features = routing_out.get("inh_features", None)
        shared_features = routing_out.get("shared_features", None)

        # --------------------------------------------------------------
        # Build token-to-group current for first SNN layer
        # --------------------------------------------------------------
        grouped_current = self._build_group_currents(
            routed_x=routed_x,
            routing_weights=routing_weights,
            exc_features=exc_features,
            inh_features=inh_features,
        )

        # --------------------------------------------------------------
        # First SNN layer converts routed embeddings to spikes
        # --------------------------------------------------------------
        spk_out = self.spiking_layer(
            grouped_current,
            attention_mask=attention_mask,
            group_assignments=self.group_assignments,
        )

        spikes = spk_out["spikes"]                     # [B, T, H]
        pooled = masked_mean(spikes, attention_mask)  # [B, H]

        return {
            "routed_embeddings": routed_x,
            "shared_features": shared_features,
            "exc_features": exc_features,
            "inh_features": inh_features,
            "grouped_current": grouped_current,
            "spikes": spikes,
            "membrane": spk_out["membrane"],
            "pooled_features": pooled,
            "beta": spk_out["beta"],
            "tau": spk_out["tau"],
            "threshold": spk_out["threshold"],
            "group_assignments": self.group_assignments,
            "routing_weights": routing_weights,
            "routing_logits": routing_out.get("routing_logits", None),
            "routing_confidence": routing_out.get("routing_confidence", None),
            "context_vector": routing_out.get("context_vector", None),
        }