import hashlib
import json

import pytest
import torch
from pydantic import ValidationError
from transformers import BertConfig, BertModel, BertTokenizerFast

from reflexone.data import load_ontology
from reflexone.inference import SupportPredictor
from reflexone.model import DecisionNetwork
from reflexone.schema import ExtractedSpan, PredictRequest
from reflexone.training import SupportConfig, save_network


@pytest.fixture
def local_artifact(tmp_path):
    ontology = load_ontology()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "card", "lost"]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    encoder = BertModel(
        BertConfig(
            vocab_size=7,
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    model = DecisionNetwork(encoder, len(ontology.intents), len(ontology.slot_types), 0)
    with torch.no_grad():
        model.intent_head[-1].weight.zero_()
        model.intent_head[-1].bias.fill_(-10)
        model.intent_head[-1].bias[ontology.intents.index("card")] = 10
        model.slot_head.weight.zero_()
        model.slot_head.bias.reshape(-1, 3)[:, 0] = 10
    artifact = tmp_path / "artifact"
    save_network(model, tokenizer, artifact, SupportConfig(dropout=0), ontology)
    from importlib.resources import files

    (artifact / "ontology.json").write_text(
        files("reflexone").joinpath("resources/ontology.json").read_text()
    )
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": "local-test",
                "model_sha256": hashlib.sha256(
                    (artifact / "model.safetensors").read_bytes()
                ).hexdigest(),
            }
        )
    )
    (artifact / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": 1.0,
                "raw_threshold": 0.5,
                "calibrated_threshold": 0.5,
                "review_cutoff": 0.1,
            }
        )
    )
    return artifact


def test_saved_network_predicts_typed_labels_with_domain_masks(local_artifact):
    predictor = SupportPredictor(local_artifact)
    result = predictor.predict("card lost", "banking")
    assert result.intents["card"].value
    assert "booking" not in result.intents
    assert result.model_id == "local-test"
    assert not result.review_required
    hotel = predictor.predict("card lost", "hotels")
    assert "card" not in hotel.intents
    assert hotel.review_required
    assert "no_intent_selected" in hotel.review_reasons


def test_prediction_rejects_invalid_domain_and_blank_text(local_artifact):
    predictor = SupportPredictor(local_artifact)
    with pytest.raises(ValueError):
        predictor.predict(" ", "banking")
    with pytest.raises(ValueError):
        predictor.predict("card", "unknown")


def test_support_refuses_inconsistent_saved_calibration(local_artifact):
    path = local_artifact / "calibration.json"
    calibration = json.loads(path.read_text())
    calibration["calibrated_threshold"] = 0.9
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="threshold|calibration"):
        SupportPredictor(local_artifact)


def test_support_verifies_weights_before_loading(local_artifact):
    with (local_artifact / "model.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum|identity"):
        SupportPredictor(local_artifact)


def test_profile_contract_requires_domain_for_support_only():
    with pytest.raises(ValidationError):
        PredictRequest(profile="support", text="hello")
    with pytest.raises(ValidationError):
        PredictRequest(profile="spam", domain="banking", text="hello")
    with pytest.raises(ValidationError):
        PredictRequest(profile="commands", text="  ")
    assert PredictRequest(profile="spam", text="hello").text == "hello"


def test_extraction_schema_refuses_impossible_span_bounds():
    with pytest.raises(ValidationError):
        ExtractedSpan(label="date", start=8, end=3, text="May", token_score=0.9)
    with pytest.raises(ValidationError):
        ExtractedSpan(label="made_up_slot", start=0, end=3, text="May", token_score=0.9)
