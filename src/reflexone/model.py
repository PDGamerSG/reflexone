"""Shared encoder with parallel intent and overlapping extractive slot heads."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DOMAINS = {"banking": 0, "hotels": 1}


class DecisionNetwork(nn.Module):
    """Predict all intent marginals and all slot tags in one encoder pass."""

    def __init__(
        self, encoder: nn.Module, num_intents: int, num_slots: int, dropout: float = 0.15
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.dropout_rate = dropout
        self.domain_embedding = nn.Embedding(2, 16)
        hidden = encoder.config.hidden_size
        self.intent_head = nn.Sequential(
            nn.Linear(hidden + 16, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_intents),
        )
        self.slot_dropout = nn.Dropout(dropout)
        self.slot_head = nn.Linear(hidden + 16, num_slots * 3)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, domains: Tensor
    ) -> dict[str, Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        domain = self.domain_embedding(domains)
        intent_logits = self.intent_head(torch.cat([pooled, domain], dim=-1))
        domain_tokens = domain[:, None, :].expand(-1, hidden.shape[1], -1)
        slot_logits = self.slot_head(self.slot_dropout(torch.cat([hidden, domain_tokens], -1)))
        return {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits.reshape(*hidden.shape[:2], self.num_slots, 3),
        }


def masked_multitask_loss(
    outputs: dict[str, Tensor],
    intent_targets: Tensor,
    slot_targets: Tensor,
    intent_mask: Tensor,
    slot_mask: Tensor,
    pos_weight: Tensor | None = None,
    slot_weight: float = 1.0,
) -> Tensor:
    """Exclude inapplicable labels, padding and special tokens from both losses."""
    intent_loss = F.binary_cross_entropy_with_logits(
        outputs["intent_logits"],
        intent_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    intent_loss = (intent_loss * intent_mask).sum() / intent_mask.sum().clamp_min(1)
    valid = (slot_targets != -100) & slot_mask[:, None, :]
    if valid.any():
        # B/I classes are rare. These fixed weights are configuration, not fitted evidence.
        class_weights = outputs["slot_logits"].new_tensor([1.0, 4.0, 4.0])
        slot_loss = F.cross_entropy(
            outputs["slot_logits"][valid], slot_targets[valid], weight=class_weights
        )
    else:
        slot_loss = outputs["slot_logits"].sum() * 0.0
    return intent_loss + slot_weight * slot_loss


def align_slot_targets(
    offsets: Sequence[Sequence[int]], slots: Sequence[Any], slot_labels: Sequence[str]
) -> np.ndarray:
    """Map gold character spans to independent BIO streams; never silently drop a span.

    Subword tokens intersecting a span receive B then I. Special/pad positions are
    ignored. Gold boundaries must be exactly representable by token boundaries.
    """
    targets = np.zeros((len(offsets), len(slot_labels)), dtype=np.int64)
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            targets[i, :] = -100
    label_to_id = {name: i for i, name in enumerate(slot_labels)}
    max_end = max((end for start, end in offsets if end > start), default=0)
    for slot in slots:
        if slot.label not in label_to_id:
            raise ValueError(f"Unknown slot label: {slot.label}")
        indices = [
            i
            for i, (start, end) in enumerate(offsets)
            if end > start and start < slot.end and end > slot.start
        ]
        if not indices or max_end < slot.end:
            raise ValueError(f"Tokenization cannot cover slot {slot.label}; possible truncation")
        if offsets[indices[0]][0] != slot.start or offsets[indices[-1]][1] != slot.end:
            raise ValueError(
                f"Slot {slot.label} boundaries are not representable by tokenizer offsets"
            )
        column = label_to_id[slot.label]
        targets[indices[0], column] = 1
        targets[indices[1:], column] = 2
    return targets


def decode_slot_spans(
    text: str,
    offsets: Sequence[Sequence[int]],
    probabilities: np.ndarray,
    slot_labels: Sequence[str],
    slot_mask: Sequence[bool],
) -> list[dict[str, Any]]:
    """Decode every independent BIO stream and copy values from the original input.

    Orphan I tags begin a span. Span probability is the minimum chosen tag
    probability, a conservative summary of token marginals, not a calibrated
    joint probability that the extracted span is correct.
    """
    spans: list[dict[str, Any]] = []
    tags = probabilities.argmax(-1)
    for column, name in enumerate(slot_labels):
        if not slot_mask[column]:
            continue
        active_start: int | None = None
        active_end = 0
        active_probs: list[float] = []

        def finish() -> None:
            nonlocal active_start, active_probs
            if active_start is not None:
                spans.append(
                    {
                        "label": name,
                        "start": int(active_start),
                        "end": int(active_end),
                        "text": text[active_start:active_end],
                        "probability": float(min(active_probs)),
                    }
                )
            active_start = None
            active_probs = []

        for i, (start, end) in enumerate(offsets):
            if end <= start:
                finish()
                continue
            tag = int(tags[i, column])
            if tag == 0:
                finish()
            else:
                if tag == 1:
                    finish()
                if active_start is None:
                    active_start = int(start)
                active_end = int(end)
                active_probs.append(float(probabilities[i, column, tag]))
        finish()
    return sorted(spans, key=lambda s: (s["start"], s["end"], s["label"]))
