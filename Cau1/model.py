"""
model.py
========
MultiTaskPhoBERT — one shared PhoBERT backbone with three task-specific heads.

Heads
-----
1. **ner_head**       : Linear(hidden→9) + CRF   (token-level)
2. **sentiment_head** : mean-pool → Dropout → Linear(hidden→3)
3. **emotion_head**   : [CLS] → Dropout → Linear(hidden→6)   (ABSA — 2 aspects × 3 polarities packed)

Author : NLP Pipeline
"""

from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# ---------------------------------------------------------------------------
# CRF import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from torchcrf import CRF
except ImportError as _crf_err:
    print(
        "[model.py] torchcrf is not installed. "
        "Install it with:  pip install pytorch-crf"
    )
    raise _crf_err

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "phobert-base": "vinai/phobert-base",
    "phobert-large": "vinai/phobert-large",
}


def _resolve_model_name() -> str:
    """Return the HuggingFace model identifier from the ``MODEL_NAME`` env var
    or default to ``vinai/phobert-base``."""
    env = os.environ.get("MODEL_NAME", "vinai/phobert-base")
    return MODEL_REGISTRY.get(env, env)


# ---------------------------------------------------------------------------
# NER constants (must match data_loader.py)
# ---------------------------------------------------------------------------

NER_NUM_LABELS: int = 9  # O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC
SENTIMENT_NUM_LABELS: int = 3
ABSA_NUM_LABELS: int = 6  # 4 aspects encoded as a flat 6-dim vector is wrong
# ↑ We actually output per-aspect predictions.  The head outputs [B, 4, 3]
# reshaped from a Linear(hidden→12).  We keep the constant for clarity.
ABSA_NUM_ASPECTS: int = 4
ABSA_NUM_POLARITIES: int = 3


# ---------------------------------------------------------------------------
# MultiTaskPhoBERT
# ---------------------------------------------------------------------------

class MultiTaskPhoBERT(nn.Module):
    """Shared PhoBERT backbone with three task-specific classification heads.

    Parameters
    ----------
    model_name : str
        A HuggingFace model identifier (default resolved via ``MODEL_NAME``
        env-var or ``vinai/phobert-base``).
    ner_num_labels : int
        Number of NER IOB2 labels (default 9).
    sentiment_num_labels : int
        Number of sentiment classes (default 3).
    absa_num_aspects : int
        Number of ABSA aspects (default 4).
    absa_num_polarities : int
        Number of ABSA polarity classes (default 3).
    dropout : float
        Dropout probability for classification heads (default 0.3).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ner_num_labels: int = NER_NUM_LABELS,
        sentiment_num_labels: int = SENTIMENT_NUM_LABELS,
        absa_num_aspects: int = ABSA_NUM_ASPECTS,
        absa_num_polarities: int = ABSA_NUM_POLARITIES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if model_name is None:
            model_name = _resolve_model_name()

        self.model_name: str = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone: nn.Module = AutoModel.from_pretrained(model_name, config=self.config)
        hidden: int = self.config.hidden_size  # 768 for base, 1024 for large

        # ---- NER head: Linear + CRF ----
        self.ner_classifier = nn.Linear(hidden, ner_num_labels)
        self.crf = CRF(ner_num_labels, batch_first=True)

        # ---- Sentiment head: mean-pool → Dropout → Linear ----
        self.sentiment_dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Linear(hidden, sentiment_num_labels)

        # ---- ABSA (emotion) head: CLS → Dropout → Linear(hidden→aspects*pols) ----
        self.absa_num_aspects: int = absa_num_aspects
        self.absa_num_polarities: int = absa_num_polarities
        self.emotion_dropout = nn.Dropout(dropout)
        self.emotion_classifier = nn.Linear(
            hidden, absa_num_aspects * absa_num_polarities
        )

        # Store label counts for external use
        self.ner_num_labels: int = ner_num_labels
        self.sentiment_num_labels: int = sentiment_num_labels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Route the input through the shared backbone and the appropriate head.

        Parameters
        ----------
        input_ids      : LongTensor [B, L]
        attention_mask  : LongTensor [B, L]
        task            : ``"ner"`` | ``"sentiment"`` | ``"absa"``
        labels          : Optional target tensor (shape depends on task).

        Returns
        -------
        (logits, loss | None)
            *logits* shape depends on the task:
            - ner       : [B, L, ner_num_labels]  (emissions)
            - sentiment : [B, sentiment_num_labels]
            - absa      : [B, num_aspects, num_polarities]
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states: torch.Tensor = outputs.last_hidden_state  # [B, L, H]

        if task == "ner":
            return self._forward_ner(hidden_states, attention_mask, labels)
        elif task == "sentiment":
            return self._forward_sentiment(hidden_states, attention_mask, labels)
        elif task == "absa":
            return self._forward_absa(hidden_states, labels)
        else:
            raise ValueError(f"Unknown task: {task!r}")

    # ------------------------------------------------------------------
    # Task-specific forward methods
    # ------------------------------------------------------------------

    def _forward_ner(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emissions: torch.Tensor = self.ner_classifier(hidden_states)  # [B, L, C]
        loss: Optional[torch.Tensor] = None

        if labels is not None:
            loss = self.compute_loss("ner", emissions, labels, attention_mask)

        return emissions, loss

    def _forward_sentiment(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Mean-pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (hidden_states * mask_expanded).sum(dim=1)  # [B, H]
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)    # [B, 1]
        pooled = summed / counts                              # [B, H]

        pooled = self.sentiment_dropout(pooled)
        logits: torch.Tensor = self.sentiment_classifier(pooled)  # [B, C]
        loss: Optional[torch.Tensor] = None

        if labels is not None:
            loss = self.compute_loss("sentiment", logits, labels)

        return logits, loss

    def _forward_absa(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Use [CLS] token (index 0)
        cls_hidden: torch.Tensor = hidden_states[:, 0, :]  # [B, H]
        cls_hidden = self.emotion_dropout(cls_hidden)
        raw: torch.Tensor = self.emotion_classifier(cls_hidden)  # [B, A*P]
        logits = raw.view(
            -1, self.absa_num_aspects, self.absa_num_polarities
        )  # [B, A, P]
        loss: Optional[torch.Tensor] = None

        if labels is not None:
            loss = self.compute_loss("absa", logits, labels)

        return logits, loss

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the task-appropriate loss.

        Parameters
        ----------
        task            : ``"ner"`` | ``"sentiment"`` | ``"absa"``
        logits          : Model emissions / logits.
        labels          : Ground-truth labels.
        attention_mask  : Required for NER (CRF masking).
        class_weights   : Optional weight tensor for CrossEntropy tasks.

        Returns
        -------
        Scalar loss tensor.
        """
        if task == "ner":
            return self._ner_loss(logits, labels, attention_mask)
        elif task == "sentiment":
            return self._ce_loss(logits, labels, class_weights)
        elif task == "absa":
            return self._absa_loss(logits, labels, class_weights)
        else:
            raise ValueError(f"Unknown task: {task!r}")

    def _ner_loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CRF negative log-likelihood for NER."""
        # Replace PAD_LABEL_ID (-100) with 0 for CRF (it ignores via mask)
        crf_labels = labels.clone()
        crf_labels[crf_labels < 0] = 0

        # Build a byte mask: True where we have real tokens
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = (labels != -100)

        # CRF returns log-likelihood; negate for NLL loss
        log_likelihood: torch.Tensor = self.crf(
            emissions, crf_labels, mask=mask, reduction="mean"
        )
        return -log_likelihood

    @staticmethod
    def _ce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted cross-entropy for sentiment."""
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        return nn.functional.cross_entropy(logits, labels, weight=class_weights)

    def _absa_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-entropy averaged over aspects.

        logits : [B, A, P]
        labels : [B, A]   with values in {0, 1, 2}
        """
        B, A, P = logits.shape
        logits_flat = logits.reshape(B * A, P)    # [B*A, P]
        labels_flat = labels.reshape(B * A)       # [B*A]
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        return nn.functional.cross_entropy(
            logits_flat, labels_flat, weight=class_weights
        )

    # ------------------------------------------------------------------
    # CRF decoding (for inference)
    # ------------------------------------------------------------------

    def crf_decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi-decode the best NER tag sequence."""
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)

    # ------------------------------------------------------------------
    # Backbone freezing
    # ------------------------------------------------------------------

    def freeze_backbone(self, n_layers: int) -> None:
        """
        Freeze the embeddings and the bottom *n_layers* transformer encoder
        layers of the backbone.

        Parameters
        ----------
        n_layers : int
            Number of encoder layers to freeze (0 = only embeddings,
            12 = entire base model).
        """
        # Freeze embeddings
        if hasattr(self.backbone, "embeddings"):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False

        # Freeze encoder layers
        encoder_layers: Optional[nn.ModuleList] = None
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layers"):
            encoder_layers = self.backbone.layers

        if encoder_layers is not None:
            for idx in range(min(n_layers, len(encoder_layers))):
                for param in encoder_layers[idx].parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def model_size_mb(self) -> float:
        """Approximate model size in megabytes."""
        param_bytes = sum(
            p.nelement() * p.element_size() for p in self.parameters()
        )
        buffer_bytes = sum(
            b.nelement() * b.element_size() for b in self.buffers()
        )
        return (param_bytes + buffer_bytes) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("model.py — standalone check")
    print(f"MODEL_REGISTRY: {MODEL_REGISTRY}")
    print(f"Resolved model name: {_resolve_model_name()}")
    print("Instantiating MultiTaskPhoBERT (this will download the model if needed)…")
    model = MultiTaskPhoBERT()
    print(f"  Total parameters : {model.count_parameters(trainable_only=False):,}")
    print(f"  Trainable params : {model.count_parameters(trainable_only=True):,}")
    print(f"  Model size       : {model.model_size_mb():.1f} MB")
    print("Done.")
