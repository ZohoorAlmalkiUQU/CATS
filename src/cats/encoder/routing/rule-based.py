from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseRouter


class RuleBasedRouter(BaseRouter):
    """
    Rule-Based: deterministic mean-based routing.

    Core idea:
    - For each sample, compute a global mean over all valid token values.
    - For each token, compute its token mean over embedding dimension D.
    - If token_mean >= sample_global_mean  -> excitatory
    - Else                                 -> inhibitory

    This router keeps the same general output contract used by CARSON v2:
        {
            "routed_x": [B, T, D],
            "routing_weights": [B, T, 2],
            "routing_logits": [B, T, 2],
            "context_vector": [B, D],
            "routing_confidence": [B, T],
            "exc_features": [B, T, D],
            "inh_features": [B, T, D],
            "shared_features": [B, T, D],
        }

    Notes:
    - This is rule-based, not learned routing.
    - routing_weights are hard one-hot assignments.
    - routing_logits are constructed from signed distance to the sample mean.
    - context_vector is the masked mean token embedding, for compatibility.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_groups: int = 2,
        use_residual: bool = True,
        use_layernorm: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if num_groups != 2:
            raise ValueError(
                f"CARSONRouterV3 expects num_groups=2 for excitatory/inhibitory routing, "
                f"got {num_groups}."
            )

        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.use_residual = use_residual
        self.eps = eps

        self.input_norm = (
            nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()
        )
        self.output_norm = (
            nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()
        )

    @staticmethod
    def _build_mask(
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build float mask [B, T] from optional attention mask.
        """
        bsz, seq_len, _ = x.shape

        if attention_mask is None:
            return torch.ones((bsz, seq_len), device=x.device, dtype=x.dtype)

        if attention_mask.shape != (bsz, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(bsz, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        return attention_mask.to(device=x.device, dtype=x.dtype)

    def _masked_token_embedding_mean(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked mean over sequence dimension.

        Args:
            x: [B, T, D]
            mask: [B, T]

        Returns:
            [B, D]
        """
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        denom = mask_expanded.sum(dim=1).clamp_min(1.0)  # [B, 1]
        return (x * mask_expanded).sum(dim=1) / denom

    def _sample_global_mean(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute one scalar global mean per sample across all valid tokens and embedding dims.

        Args:
            x: [B, T, D]
            mask: [B, T]

        Returns:
            [B, 1]
        """
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        valid_count = (mask_expanded.sum(dim=(1, 2)) * x.size(-1)).clamp_min(1.0)  # [B]
        total_sum = (x * mask_expanded).sum(dim=(1, 2))  # [B]
        return (total_sum / valid_count).unsqueeze(-1)  # [B, 1]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            attention_mask: [B, T] or None

        Returns:
            Dict[str, torch.Tensor]
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        bsz, seq_len, dim = x.shape
        if dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {dim}"
            )

        mask = self._build_mask(x, attention_mask=attention_mask)  # [B, T]
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]

        # --------------------------------------------------------------
        # Step 1: normalize input
        # --------------------------------------------------------------
        x_norm = self.input_norm(x)  # [B, T, D]
        x_norm = x_norm * mask_expanded

        # --------------------------------------------------------------
        # Step 2: compute compatibility context vector
        # same idea as masked sample representation
        # --------------------------------------------------------------
        context_vector = self._masked_token_embedding_mean(x_norm, mask)  # [B, D]

        # --------------------------------------------------------------
        # Step 3: compute deterministic routing criterion
        # token_mean: mean of each token across embedding dimension
        # sample_mean: one scalar mean per sample over all valid token values
        # --------------------------------------------------------------
        token_means = x_norm.mean(dim=-1)  # [B, T]
        sample_mean = self._sample_global_mean(x_norm, mask)  # [B, 1]

        # signed distance from sample-level global mean
        distance = token_means - sample_mean  # [B, T]

        # excitatory if token_mean >= sample_mean, else inhibitory
        exc_assign = (distance >= 0).to(dtype=x.dtype) * mask  # [B, T]
        inh_assign = (distance < 0).to(dtype=x.dtype) * mask   # [B, T]

        # hard one-hot routing weights [exc, inh] order
        routing_weights = torch.stack([exc_assign, inh_assign], dim=-1)  # [B, T, 2]

        # for padded tokens keep zero
        routing_weights = routing_weights * mask_expanded

        # --------------------------------------------------------------
        # Step 4: construct logits for diagnostics/compatibility
        #
        # We define:
        #   excitatory logit = +distance
        #   inhibitory logit = -distance
        #
        # so whichever side is larger determines the route.
        # --------------------------------------------------------------
        routing_logits = torch.stack([distance, -distance], dim=-1)  # [B, T, 2]
        routing_logits = routing_logits.masked_fill(mask_expanded == 0, 0.0)

        # --------------------------------------------------------------
        # Step 5: confidence = normalized absolute distance
        # This is only for monitoring; not used for routing decision.
        # --------------------------------------------------------------
        abs_distance = distance.abs() * mask  # [B, T]
        max_abs_distance = abs_distance.max(dim=1, keepdim=True).values.clamp_min(self.eps)
        routing_confidence = abs_distance / max_abs_distance  # [B, T]
        routing_confidence = routing_confidence * mask

        # --------------------------------------------------------------
        # Step 6: group-specific features
        # These remain positive/normal features; routing decides usage.
        # --------------------------------------------------------------
        shared_features = x_norm * mask_expanded              # [B, T, D]
        exc_features = x_norm * exc_assign.unsqueeze(-1)      # [B, T, D]
        inh_features = x_norm * inh_assign.unsqueeze(-1)      # [B, T, D]

        # --------------------------------------------------------------
        # Step 7: routed_x
        # For compatibility with existing pipeline, we keep a routed representation.
        # Here the routing itself is assignment-based, so routed_x can remain x plus
        # a simple group-aware modulation.
        #
        # Since:
        #   exc token -> +x_norm
        #   inh token -> -x_norm
        #
        # this creates explicit polarity separation in routed_x.
        # --------------------------------------------------------------
        signed_routed = (exc_assign.unsqueeze(-1) - inh_assign.unsqueeze(-1)) * x_norm

        if self.use_residual:
            routed_x = x + signed_routed
        else:
            routed_x = signed_routed

        routed_x = self.output_norm(routed_x)
        routed_x = routed_x * mask_expanded

        return {
            "routed_x": routed_x,
            "routing_weights": routing_weights,
            "routing_logits": routing_logits,
            "context_vector": context_vector,
            "routing_confidence": routing_confidence,
            "exc_features": exc_features,
            "inh_features": inh_features,
            "shared_features": shared_features,
        }