from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder.core import CATSEncoder
from .heads.ann_classifier import ANNClassifierHead
from .heads.snn_classifier import SNNClassifierHead


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
        lif_inh: nn.Module | None,
        classifier_cfg: Optional[Dict] = None,
        inhibition_enabled: bool = True,
        spiking_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        classifier_cfg = classifier_cfg or {}
        classifier_kwargs = dict(classifier_cfg.get("kwargs", {}) or {})

        self.classifier_type = classifier_cfg.get("type", "ann").lower()

        self.encoder = CATSEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            excitatory_ratio=excitatory_ratio,
            num_groups=num_groups,
            router=router,
            lif_exc=lif_exc,
            lif_inh=lif_inh,
            inhibition_enabled=inhibition_enabled,
            spiking_enabled=spiking_enabled,
            **kwargs,
        )

        classifier_input_dim = int(classifier_cfg.get("input_dim", hidden_dim))

        if self.classifier_type == "ann":
            self.classifier = ANNClassifierHead(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                **classifier_kwargs,
            )
        elif self.classifier_type == "snn":
            self.classifier = SNNClassifierHead(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                **classifier_kwargs,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        if embeddings is None:
            embeddings = x

        if embeddings is None:
            raise ValueError(
                "CATSClassifier.forward requires `embeddings` or alias `x`."
            )

        encoder_outputs = self.encoder(
            x=embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

        # if not hasattr(self, "_printed_encoder_keys"):
        #     print("encoder_outputs keys:", encoder_outputs.keys())
        #     for k, v in encoder_outputs.items():
        #         if torch.is_tensor(v):
        #             print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        #     self._printed_encoder_keys = True

        pooled_features = encoder_outputs["pooled_features"]

        if self.classifier_type == "snn":
            if "spikes" in encoder_outputs:
                sequence_features = encoder_outputs["spikes"]
            elif "sequence_features" in encoder_outputs:
                sequence_features = encoder_outputs["sequence_features"]
            elif "encoded_sequence" in encoder_outputs:
                sequence_features = encoder_outputs["encoded_sequence"]
            else:
                raise KeyError(
                    "SNN classifier requires sequence output from encoder_outputs. "
                    f"Available keys: {list(encoder_outputs.keys())}"
                )

            logits = self.classifier(sequence_features, mask=attention_mask)
        else:
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