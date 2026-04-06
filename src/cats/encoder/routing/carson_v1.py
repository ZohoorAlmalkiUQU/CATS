from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseRouter


class CARSONRouterV1(BaseRouter):
    """
    Context-Aware Routing for Spiking encoders (CARSON).

    This router operates on continuous token embeddings before spike conversion.
    It performs iterative context-aware routing to produce:

        1) routed_x: context-modulated token embeddings [B, T, D]
        2) routing_weights: token-wise group weights [B, T, 2]

    Design goals for this implementation:
    - keep the same input/output contract as existing routers
    - remain compatible with CATSEncoder._build_group_currents(...)
    - support masked sequences
    - provide a stronger routing mechanism than Identity / LinearRouter
      without changing downstream encoder logic

    Expected input:
        x: [B, T, D]
        attention_mask: [B, T] or None

    Returned dictionary:
        {
            "routed_x": [B, T, D],
            "routing_weights": [B, T, 2],
            "routing_logits": [B, T, 2],
            "context_vector": [B, D],
            "routing_confidence": [B, T],
        }
    """

    def __init__(
        self,
        embedding_dim: int,
        num_groups: int = 2,
        num_iterations: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_residual: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if num_groups != 2:
            raise ValueError(
                f"CARSONRouter currently expects num_groups=2 for "
                f"excitatory/inhibitory routing, got {num_groups}."
            )
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim or embedding_dim
        self.temperature = temperature
        self.use_residual = use_residual

        self.input_norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()
        self.output_norm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Token projection into routing space
        self.token_proj = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        # Context update network:
        # combines token representation with current global context estimate
        self.context_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        # Token modulation conditioned on global context
        self.token_context_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        # Final routing head for excitatory/inhibitory groups
        self.group_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_groups),
        )

    @staticmethod
    def _masked_mean(
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked mean over sequence dimension.

        Args:
            x: [B, T, D]
            attention_mask: [B, T] or None

        Returns:
            [B, D]
        """
        if attention_mask is None:
            return x.mean(dim=1)

        mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # [B,T,1]
        denom = mask.sum(dim=1).clamp_min(1.0)  # [B,1]
        return (x * mask).sum(dim=1) / denom

    def _iterative_context_routing(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively refine a global context vector and token importance.

        Args:
            tokens: [B, T, D]
            attention_mask: [B, T] or None

        Returns:
            context: [B, D]
            token_scores: [B, T]
        """
        # Initial context = masked mean of token representations
        context = self._masked_mean(tokens, attention_mask=attention_mask)  # [B,D]

        if attention_mask is None:
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=tokens.dtype)
        else:
            mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype)

        token_scores = torch.zeros(
            tokens.size(0),
            tokens.size(1),
            device=tokens.device,
            dtype=tokens.dtype,
        )

        for _ in range(self.num_iterations):
            context_expanded = context.unsqueeze(1).expand(-1, tokens.size(1), -1)  # [B,T,D]
            fusion_input = torch.cat([tokens, context_expanded], dim=-1)  # [B,T,2D]

            # Unnormalized token relevance to current context
            logits = self.context_gate(fusion_input).squeeze(-1)  # [B,T]

            # Mask padding tokens before softmax
            logits = logits.masked_fill(mask == 0, -1e9)

            token_scores = torch.softmax(logits / self.temperature, dim=1)  # [B,T]
            token_scores = token_scores * mask

            # Renormalize after masking for numerical cleanliness
            token_scores = token_scores / token_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # Weighted context update
            context = torch.sum(tokens * token_scores.unsqueeze(-1), dim=1)  # [B,D]

        return context, token_scores

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
            {
                "routed_x": [B, T, D],
                "routing_weights": [B, T, 2],
                "routing_logits": [B, T, 2],
                "context_vector": [B, D],
                "routing_confidence": [B, T],
            }
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        bsz, seq_len, dim = x.shape
        if dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {dim}"
            )

        if attention_mask is None:
            mask = torch.ones((bsz, seq_len), device=x.device, dtype=x.dtype)
        else:
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"attention_mask must have shape {(bsz, seq_len)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            mask = attention_mask.to(device=x.device, dtype=x.dtype)

        # Step 1: normalize + token projection
        x_norm = self.input_norm(x)
        token_features = self.token_proj(x_norm)  # [B,T,D]
        token_features = self.dropout(token_features)

        # Step 2: iterative context-aware routing
        context_vector, routing_confidence = self._iterative_context_routing(
            token_features,
            attention_mask=mask,
        )  # [B,D], [B,T]

        # Step 3: fuse token with routed global context
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)  # [B,T,D]
        fusion_input = torch.cat([token_features, context_expanded], dim=-1)  # [B,T,2D]

        routed_delta = self.token_context_fusion(fusion_input)  # [B,T,D]
        routed_delta = self.dropout(routed_delta)

        if self.use_residual:
            routed_x = x + routed_delta
        else:
            routed_x = routed_delta

        routed_x = self.output_norm(routed_x)

        # Zero padded tokens cleanly
        routed_x = routed_x * mask.unsqueeze(-1)

        # Step 4: produce group routing weights [B,T,2]
        routing_logits = self.group_head(
            torch.cat([routed_x, context_expanded], dim=-1)
        )  # [B,T,2]

        routing_weights = torch.softmax(routing_logits / self.temperature, dim=-1)

        # zero-out padded tokens
        routing_weights = routing_weights * mask.unsqueeze(-1)

        return {
            "routed_x": routed_x,
            "routing_weights": routing_weights,
            "routing_logits": routing_logits,
            "context_vector": context_vector,
            "routing_confidence": routing_confidence,
        }