from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder.core import CATSEncoder
from .heads.classifier import ClassifierHead


class CATSClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        excitatory_ratio: float,
        num_groups: int,
        router: nn.Module,
        lif_exc: nn.Module,
        lif_inh: nn.Module,
        classifier_cfg: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        classifier_cfg = classifier_cfg or {}
        classifier_kwargs = dict(classifier_cfg.get("kwargs", {}) or {})

        self.encoder = CATSEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            excitatory_ratio=excitatory_ratio,
            num_groups=num_groups,
            router=router,
            lif_exc=lif_exc,
            lif_inh=lif_inh,
            **kwargs,
        )

        classifier_input_dim = int(
            classifier_cfg.get("input_dim", hidden_dim)
        )

        self.classifier = ClassifierHead(
            input_dim=classifier_input_dim,
            num_classes=num_classes,
            **classifier_kwargs,
        )

    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [B, T, D] input embeddings
            attention_mask: [B, T]
            labels: [B]
            x: optional alias for embeddings, for compatibility

        Returns:
            Dict containing logits, probs, predictions, and encoder outputs.
        """
        if embeddings is None:
            embeddings = x

        if embeddings is None:
            raise ValueError(
                "CATSClassifier.forward requires `embeddings` "
                "or alias `x`."
            )

        encoder_outputs = self.encoder(
            x=embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

        pooled_features = encoder_outputs["pooled_features"]
        logits = self.classifier(pooled_features)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "pooled_features": pooled_features,
            **encoder_outputs,
        }

        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= 0.5).long()
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        outputs["probs"] = probs
        outputs["preds"] = preds

        if labels is not None:
            outputs["labels"] = labels

        return outputs