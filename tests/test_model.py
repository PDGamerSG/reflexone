from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel

from reflexone.model import (
    DecisionNetwork,
    align_slot_targets,
    decode_slot_spans,
    masked_multitask_loss,
)


def test_independent_slot_heads_preserve_overlapping_dates():
    slots = [
        SimpleNamespace(label="date", start=0, end=10, text="only today"),
        SimpleNamespace(label="period", start=5, end=10, text="today"),
    ]
    offsets = [(0, 0), (0, 4), (5, 10), (0, 0)]
    targets = align_slot_targets(offsets, slots, ("date", "period"))
    np.testing.assert_array_equal(targets, [[-100, -100], [1, 0], [2, 1], [-100, -100]])


def test_slot_alignment_refuses_a_truncated_annotation():
    slot = SimpleNamespace(label="date", start=5, end=10, text="today")
    with pytest.raises(ValueError, match="truncat|cover"):
        align_slot_targets([(0, 0), (0, 4), (0, 0)], [slot], ("date",))


def test_slot_alignment_refuses_gold_boundary_inside_a_token():
    slot = SimpleNamespace(label="date", start=1, end=4, text="oda")
    with pytest.raises(ValueError, match="boundar|represent"):
        align_slot_targets([(0, 0), (0, 5), (0, 0)], [slot], ("date",))


def test_slot_decoder_copies_exact_input_and_keeps_cross_type_overlap():
    probs = np.full((4, 2, 3), 0.01)
    probs[:, :, 0] = 0.98
    probs[1, 0] = [0.01, 0.98, 0.01]
    probs[2, 0] = [0.01, 0.01, 0.98]
    probs[2, 1] = [0.01, 0.98, 0.01]
    spans = decode_slot_spans(
        "only today", [(0, 0), (0, 4), (5, 10), (0, 0)], probs, ("date", "period"),
        [True, True],
    )
    assert {(s["label"], s["start"], s["end"], s["text"]) for s in spans} == {
        ("date", 0, 10, "only today"), ("period", 5, 10, "today")
    }


def test_slot_decoder_masks_labels_and_repairs_orphan_i_tags():
    probs = np.array([[[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]])
    spans = decode_slot_spans("May", [(0, 3)], probs, ("date", "period"), [True, False])
    assert len(spans) == 1
    assert spans[0]["label"] == "date"
    assert spans[0]["text"] == "May"


def test_neural_heads_have_gradients_and_inapplicable_labels_have_no_loss():
    torch.manual_seed(1)
    encoder = BertModel(BertConfig(
        vocab_size=30, hidden_size=24, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
    ))
    model = DecisionNetwork(encoder, num_intents=3, num_slots=2, dropout=0)
    inputs = dict(input_ids=torch.tensor([[2, 5, 6, 3]]),
                  attention_mask=torch.ones((1, 4), dtype=torch.long),
                  domains=torch.tensor([0]))
    outputs = model(**inputs)
    assert outputs["intent_logits"].shape == (1, 3)
    assert outputs["slot_logits"].shape == (1, 4, 2, 3)
    intent_mask = torch.tensor([[True, True, False]])
    slot_mask = torch.tensor([[True, False]])
    targets = torch.tensor([[[0, 0], [1, 1], [2, 2], [-100, -100]]])
    loss = masked_multitask_loss(outputs, torch.tensor([[1., 0., 1.]]), targets,
                                 intent_mask, slot_mask)
    changed = {k: v.clone() for k, v in outputs.items()}
    changed["intent_logits"][:, 2] += 100
    changed["slot_logits"][:, :, 1, :] += 100 * torch.randn(3)
    assert loss.item() == pytest.approx(masked_multitask_loss(
        changed, torch.tensor([[1., 0., 0.]]), targets, intent_mask, slot_mask).item())
    loss.backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert model.intent_head[-1].weight.grad.abs().sum() > 0
    assert model.slot_head.weight.grad.abs().sum() > 0


def test_loss_all_special_tokens_is_finite_and_differentiable():
    outputs = {"intent_logits": torch.zeros(1, 2, requires_grad=True),
               "slot_logits": torch.zeros(1, 2, 1, 3, requires_grad=True)}
    loss = masked_multitask_loss(outputs, torch.zeros(1, 2),
                                 torch.full((1, 2, 1), -100),
                                 torch.ones(1, 2, dtype=torch.bool),
                                 torch.ones(1, 1, dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
