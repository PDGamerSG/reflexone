
import hashlib
import json

import pytest
from transformers import BertTokenizerFast

from reflexone.training import SupportConfig, encode_records, span_metrics, verify_prepared_data


def tiny_tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "only", "today", "card"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def test_encoded_records_preserve_overlapping_gold_slots(tmp_path):
    from reflexone.data import Record, Slot, load_ontology
    ontology = load_ontology()
    record = Record("fixture", "only today", "hotels", (),
                    (Slot("date", 5, 10, "today"), Slot("date_period", 0, 10, "only today")),
                    "only today")
    encoded = encode_records([record], tiny_tokenizer(tmp_path), ontology, max_length=8)
    targets = encoded.tensors["slot_targets"].numpy()
    date, period = ontology.slot_types.index("date"), ontology.slot_types.index("date_period")
    assert targets[0, 2, date] == 1
    assert targets[0, 1, period] == 1
    assert targets[0, 2, period] == 2
    assert encoded.alignment_audit["gold_spans"] == 2
    assert encoded.alignment_audit["exact_token_boundaries"] == 2


def test_encode_refuses_silent_loss_of_a_truncated_gold_span(tmp_path):
    from reflexone.data import Record, Slot, load_ontology
    record = Record("fixture", "card card card today", "hotels", (),
                    (Slot("date", 15, 20, "today"),), "group")
    with pytest.raises(ValueError, match="truncat|cover"):
        encode_records([record], tiny_tokenizer(tmp_path), load_ontology(), max_length=4)


def test_span_metrics_require_both_type_and_exact_character_offsets():
    gold = [[{"label": "date", "start": 0, "end": 5}]]
    predicted = [[{"label": "date", "start": 0, "end": 4},
                  {"label": "time", "start": 0, "end": 5}]]
    result = span_metrics(gold, predicted)
    assert result["f1"] == 0
    assert result["true_positive"] == 0
    assert result["false_positive"] == 2
    assert result["false_negative"] == 1


def test_config_rejects_non_positive_training_settings():
    with pytest.raises(ValueError):
        SupportConfig(epochs=0)
    with pytest.raises(ValueError):
        SupportConfig(max_length=2)


def test_prepared_verification_detects_modified_test_data_and_identity(tmp_path):
    payload = b'{"id": "test"}\n'
    (tmp_path / "test.jsonl").write_bytes(payload)
    manifest = {"output_sha256": {"test.jsonl": hashlib.sha256(payload).hexdigest()}}
    raw = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    fingerprint = hashlib.sha256(raw).hexdigest()
    assert verify_prepared_data(tmp_path, fingerprint) == fingerprint
    with pytest.raises(ValueError, match="identity|manifest"):
        verify_prepared_data(tmp_path, "wrong-dataset")
    (tmp_path / "test.jsonl").write_text('{"id":"different"}\n')
    with pytest.raises(ValueError, match="checksum|hash"):
        verify_prepared_data(tmp_path, fingerprint)
