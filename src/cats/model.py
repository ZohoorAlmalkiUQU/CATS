from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder.core import CATSEncoder
from .heads.classifier import ClassifierHead


class CATSNoRoutingClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        excitatory_ratio: float = 0.5,
        router: Optional[nn.Module] = None,
        lif_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.encoder = CATSEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            excitatory_ratio=excitatory_ratio,
            router=router,
            lif_config=lif_config,
        )
        self.head = ClassifierHead(
            input_dim=hidden_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(embeddings, attention_mask=attention_mask)
        logits = self.head(enc["pooled_features"])
        enc["logits"] = logits
        return enc