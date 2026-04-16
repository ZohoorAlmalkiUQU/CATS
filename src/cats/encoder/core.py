from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignedLinear(nn.Module):
    """
    Linear layer with sign-constrained weights.

    - sign="positive" -> weights >= 0
    - sign="negative" -> weights <= 0

    The constraint is applied to weights only.
    Bias remains unconstrained by design.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sign: str,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if sign not in {"positive", "negative"}:
            raise ValueError(
                f"sign must be 'positive' or 'negative', got {sign}"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sign = sign

        self.raw_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.raw_weight)

    @property
    def weight(self) -> torch.Tensor:
        magnitude = F.softplus(self.raw_weight)
        if self.sign == "positive":
            return magnitude
        return -magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class CATSEncoder(nn.Module):
    """
    Context-Aware Token-to-Spike encoder.

    Supports:
    - inhibitory branch on/off
    - spiking branch on/off
    - sign-constrained projections:
        * excitatory weights >= 0
        * inhibitory weights <= 0
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        excitatory_ratio: float,
        num_groups: int,
        router: nn.Module,
        lif_exc: nn.Module,
        lif_inh: nn.Module | None,
        inhibition_enabled: bool = True,
        spiking_enabled: bool = True,
        use_input_layernorm: bool = True,
        use_shared_projection: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.excitatory_ratio = float(excitatory_ratio)
        self.inhibition_enabled = bool(inhibition_enabled)
        self.spiking_enabled = bool(spiking_enabled)

        self.num_groups = int(num_groups)
        if self.num_groups < 1:
            raise ValueError(
                f"num_groups must be >= 1, got {self.num_groups}"
            )

        if self.inhibition_enabled:
            if not (0.0 < excitatory_ratio < 1.0):
                raise ValueError(
                    f"excitatory_ratio must be in (0, 1), got {excitatory_ratio}"
                )

            self.exc_dim = max(1, int(round(hidden_dim * excitatory_ratio)))
            self.inh_dim = hidden_dim - self.exc_dim

            if self.inh_dim <= 0:
                raise ValueError(
                    f"Invalid split: exc_dim={self.exc_dim}, inh_dim={self.inh_dim}. "
                    f"Adjust hidden_dim or excitatory_ratio."
                )
        else:
            self.exc_dim = hidden_dim
            self.inh_dim = 0

        self.router = router
        self.lif_exc = lif_exc
        self.lif_inh = lif_inh

        if getattr(self.lif_exc, "hidden_dim", None) != self.exc_dim:
            raise ValueError(
                f"lif_exc.hidden_dim must equal exc_dim={self.exc_dim}, "
                f"got {getattr(self.lif_exc, 'hidden_dim', None)}"
            )

        if self.inhibition_enabled:
            if self.lif_inh is None:
                raise ValueError(
                    "lif_inh must be provided when inhibition is enabled"
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

        self.exc_proj = SignedLinear(
            embedding_dim,
            self.exc_dim,
            sign="positive",
            bias=True,
        )

        if self.inhibition_enabled:
            self.inh_proj = SignedLinear(
                embedding_dim,
                self.inh_dim,
                sign="negative",
                bias=True,
            )
        else:
            self.inh_proj = None

    def _build_mask(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got shape {tuple(x.shape)}")

        if attention_mask is None:
            return x.mean(dim=1)

        mask = self._build_mask(x, attention_mask)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _extract_routed_embeddings(
        self,
        routing_out: Dict[str, torch.Tensor],
        fallback_x: torch.Tensor,
    ) -> torch.Tensor:
        for key in ("routed_embeddings", "routed_x", "output", "x"):
            value = routing_out.get(key, None)
            if value is not None:
                return value
        return fallback_x

    def _extract_routing_weights(
        self,
        routing_out: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        for key in ("routing_weights", "weights", "scores"):
            value = routing_out.get(key, None)
            if value is not None:
                return value
        return None

    def _validate_lif_output(
        self,
        lif_out: Dict[str, torch.Tensor],
        name: str,
    ) -> None:
        if not isinstance(lif_out, dict):
            raise TypeError(
                f"{name} must return a dict, got {type(lif_out).__name__}"
            )

        required_lif_keys = ("spikes", "membrane", "beta", "tau", "threshold")
        for key in required_lif_keys:
            if key not in lif_out:
                raise KeyError(f"{name} output is missing required key: '{key}'")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
            shared_features = self.shared_proj(routed_x)
            shared_features = shared_features * mask3
        else:
            shared_features = None

        exc_features = self.exc_proj(routed_x) * mask3

        if self.inhibition_enabled:
            inh_features = self.inh_proj(routed_x) * mask3
        else:
            inh_features = None

        # --------------------------------------------------
        # Routing-driven current construction
        # --------------------------------------------------
        if routing_weights is not None:
            if routing_weights.ndim != 3 or routing_weights.shape[:2] != (bsz, seq_len):
                raise ValueError(
                    "routing_weights must have shape [B, T, G] matching the batch/time dims"
                )

            exc_gate = routing_weights[..., 0:1]

            if self.inhibition_enabled:
                if routing_weights.size(-1) >= 2:
                    inh_gate = routing_weights[..., 1:2]
                else:
                    inh_gate = torch.ones(
                        (bsz, seq_len, 1), dtype=x.dtype, device=x.device
                    )
            else:
                inh_gate = None
        else:
            exc_gate = torch.ones((bsz, seq_len, 1), dtype=x.dtype, device=x.device)
            inh_gate = (
                torch.ones((bsz, seq_len, 1), dtype=x.dtype, device=x.device)
                if self.inhibition_enabled else None
            )

        if shared_features is not None:
            if self.inhibition_enabled:
                shared_exc = shared_features[..., : self.exc_dim]
                shared_inh = shared_features[..., self.exc_dim:]

                exc_current = (shared_exc + exc_features) * exc_gate
                inh_current = (shared_inh + inh_features) * inh_gate
            else:
                exc_current = (shared_features + exc_features) * exc_gate
                inh_current = None
        else:
            exc_current = exc_features * exc_gate
            inh_current = inh_features * inh_gate if self.inhibition_enabled else None

        exc_current = exc_current * mask3
        if self.inhibition_enabled:
            inh_current = inh_current * mask3

        grouped_current = (
            torch.cat([exc_current, inh_current], dim=-1)
            if self.inhibition_enabled else exc_current
        )

        # --------------------------------------------------
        # Group assignments
        # --------------------------------------------------
        exc_group_assignments = torch.zeros(
            self.exc_dim,
            device=x.device,
            dtype=torch.long,
        )

        if self.inhibition_enabled:
            inh_group_assignments = torch.zeros(
                self.inh_dim,
                device=x.device,
                dtype=torch.long,
            )
            group_assignments = torch.cat(
                [
                    torch.zeros(self.exc_dim, device=x.device, dtype=torch.long),
                    torch.ones(self.inh_dim, device=x.device, dtype=torch.long),
                ],
                dim=0,
            )
        else:
            inh_group_assignments = None
            group_assignments = torch.zeros(
                self.exc_dim,
                device=x.device,
                dtype=torch.long,
            )

        # --------------------------------------------------
        # Spiking or non-spiking branch
        # --------------------------------------------------
        if self.spiking_enabled:
            exc_out = self.lif_exc(
                exc_current,
                attention_mask=attention_mask,
                group_assignments=exc_group_assignments,
            )
            self._validate_lif_output(exc_out, "lif_exc")

            exc_spikes = exc_out["spikes"] * mask3
            exc_membrane = exc_out["membrane"] * mask3
            pooled_exc = self._masked_mean_pool(exc_membrane, attention_mask)

            if self.inhibition_enabled:
                inh_out = self.lif_inh(
                    inh_current,
                    attention_mask=attention_mask,
                    group_assignments=inh_group_assignments,
                )
                self._validate_lif_output(inh_out, "lif_inh")

                inh_spikes = inh_out["spikes"] * mask3
                inh_membrane = inh_out["membrane"] * mask3
                pooled_inh = self._masked_mean_pool(inh_membrane, attention_mask)

                spikes = torch.cat([exc_spikes, inh_spikes], dim=-1)
                membrane = torch.cat([exc_membrane, inh_membrane], dim=-1)
                pooled_features = torch.cat([pooled_exc, pooled_inh], dim=-1)
                pooled_subtractive = (
                    pooled_exc.mean(dim=-1, keepdim=True)
                    - pooled_inh.mean(dim=-1, keepdim=True)
                )

                beta = torch.cat([exc_out["beta"], inh_out["beta"]], dim=0)
                tau = torch.cat([exc_out["tau"], inh_out["tau"]], dim=0)
                threshold = torch.cat(
                    [exc_out["threshold"], inh_out["threshold"]],
                    dim=0,
                )
            else:
                inh_out = None
                inh_spikes = None
                inh_membrane = None
                pooled_inh = None

                spikes = exc_spikes
                membrane = exc_membrane
                pooled_features = pooled_exc
                pooled_subtractive = None

                beta = exc_out["beta"]
                tau = exc_out["tau"]
                threshold = exc_out["threshold"]

        else:
            # --------------------------------------------------
            # No-spiking mode: bypass LIF and use currents directly
            # --------------------------------------------------
            exc_out = None
            exc_spikes = None
            exc_membrane = exc_current
            pooled_exc = self._masked_mean_pool(exc_membrane, attention_mask)

            if self.inhibition_enabled:
                inh_out = None
                inh_spikes = None
                inh_membrane = inh_current
                pooled_inh = self._masked_mean_pool(inh_membrane, attention_mask)

                spikes = None
                membrane = torch.cat([exc_membrane, inh_membrane], dim=-1)
                pooled_features = torch.cat([pooled_exc, pooled_inh], dim=-1)
                pooled_subtractive = (
                    pooled_exc.mean(dim=-1, keepdim=True)
                    - pooled_inh.mean(dim=-1, keepdim=True)
                )
            else:
                inh_out = None
                inh_spikes = None
                inh_membrane = None
                pooled_inh = None

                spikes = None
                membrane = exc_membrane
                pooled_features = pooled_exc
                pooled_subtractive = None

            beta = None
            tau = None
            threshold = None

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
            "pooled_features": pooled_features,
            "pooled_subtractive": pooled_subtractive,

            "exc_beta": exc_out["beta"] if exc_out is not None else None,
            "inh_beta": inh_out["beta"] if inh_out is not None else None,
            "beta": beta,

            "exc_tau": exc_out["tau"] if exc_out is not None else None,
            "inh_tau": inh_out["tau"] if inh_out is not None else None,
            "tau": tau,

            "exc_threshold": exc_out["threshold"] if exc_out is not None else None,
            "inh_threshold": (
                inh_out["threshold"] if inh_out is not None else None
            ),
            "threshold": threshold,

            "exc_effective_threshold": (
                exc_out.get("effective_threshold", None)
                if exc_out is not None else None
            ),
            "inh_effective_threshold": (
                inh_out.get("effective_threshold", None)
                if inh_out is not None else None
            ),

            "exc_adaptive_threshold": (
                exc_out.get("adaptive_threshold", None)
                if exc_out is not None else None
            ),
            "inh_adaptive_threshold": (
                inh_out.get("adaptive_threshold", None)
                if inh_out is not None else None
            ),

            "routing_weights": routing_weights,
            "routing_logits": routing_out.get("routing_logits", None),
            "routing_confidence": routing_out.get("routing_confidence", None),
            "context_vector": routing_out.get("context_vector", None),
            "agreement": routing_out.get("agreement", None),
            "group_prototypes": routing_out.get("group_prototypes", None),
        }