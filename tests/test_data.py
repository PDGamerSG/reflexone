"""Regression tests for source fidelity and the support dataset's leakage controls."""

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest


def raw_dataset(root: Path, records: dict[str, list[dict]]) -> None:
    """Make complete local source fixtures so preparation never needs the network."""
    ontology = {
        "intents": {
            "affirm": {"domain": ["general"]},
            "greet": {"domain": ["general"]},
            "new": {"domain": ["general"]},
            "card": {"domain": ["banking"]},
            "booking": {"domain": ["hotels"]},
        },
        "slots": {
            "date": {"domain": ["general"]},
            "time_period": {"domain": ["general"]},
            "number": {"domain": ["general"]},
            "rooms": {"domain": ["hotels"]},
        },
    }
    raw = root / "raw"
    raw.mkdir(parents=True)
    (raw / "ontology.json").write_text(json.dumps(ontology))
    for domain in ("banking", "hotels"):
        (raw / domain).mkdir()
        for fold in range(20):
            name = f"{domain}/fold{fold}.json"
            (raw / name).write_text(json.dumps(records.get(name, [])))


def prepared_records(root: Path) -> list:
    from reflexone.data import load_records

    return [
        record
        for split in ("train", "validation", "calibration", "test")
        for record in load_records(root / f"{split}.jsonl")
    ]


def test_builtin_ontology_masks_shared_and_domain_specific_labels():
    from reflexone.data import load_ontology

    ontology = load_ontology()
    assert len(ontology.intents) == 62
    assert len(ontology.slot_types) == 17
    assert sum(ontology.intent_mask("banking")) == 48
    assert sum(ontology.intent_mask("hotels")) == 40
    assert sum(ontology.slot_mask("banking")) == 13
    assert sum(ontology.slot_mask("hotels")) == 14
    for domain in ("banking", "hotels"):
        assert ontology.intent_mask(domain)[ontology.intents.index("refund")]
    assert not ontology.intent_mask("hotels")[ontology.intents.index("card")]
    with pytest.raises(ValueError, match="domain"):
        ontology.slot_mask("retail")


def test_preparation_preserves_original_text_half_open_offsets_and_overlaps(tmp_path):
    from reflexone.data import prepare_dataset

    raw_dataset(
        tmp_path,
        {
            "hotels/fold0.json": [
                {
                    "text": "only today",
                    "slots": {
                        "date": {"text": "today", "span": [5, 10], "value": {"day": 15}},
                        "time_period": {
                            "text": "only today",
                            "span": [0, 10],
                            "values": {"days": 1},
                        },
                    },
                }
            ]
        },
    )
    audit = prepare_dataset(tmp_path)
    (record,) = prepared_records(tmp_path)
    assert record.text == "only today"
    assert record.intents == ()
    assert {(s.label, s.start, s.end, s.text) for s in record.slots} == {
        ("date", 5, 10, "today"),
        ("time_period", 0, 10, "only today"),
    }
    assert audit["overlap_pairs"] == 1
    assert audit["invalid_spans"] == 0
    assert all(record.text[s.start : s.end] == s.text for s in record.slots)
    with pytest.raises(FrozenInstanceError):
        record.text = "changed"
    with pytest.raises(FrozenInstanceError):
        record.slots[0].end = 8
    serialized = json.loads((tmp_path / "train.jsonl").read_text())
    assert serialized == json.loads(json.dumps(asdict(record)))


@pytest.mark.parametrize("span", [[5, 9], [5, 11], [-1, 10], [10, 5]])
def test_preparation_rejects_inexact_or_invalid_source_spans(tmp_path, span):
    from reflexone.data import prepare_dataset

    raw_dataset(
        tmp_path,
        {
            "banking/fold0.json": [
                {"text": "only today", "slots": {"date": {"text": "today", "span": span}}}
            ]
        },
    )
    with pytest.raises(ValueError, match="span"):
        prepare_dataset(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_conflicting_intent_and_extractive_slot_groups_are_quarantined(tmp_path):
    from reflexone.data import prepare_dataset

    raw_dataset(
        tmp_path,
        {
            "banking/fold0.json": [
                {"text": "yes", "intents": ["affirm"]},
                {"text": "only today", "slots": {"date": {"text": "today", "span": [5, 10]}}},
                {"text": "keep", "intents": ["greet"]},
            ],
            "banking/fold1.json": [
                {"text": "YES"},
                {"text": "only today", "slots": {"date": {"text": "only today", "span": [0, 10]}}},
            ],
        },
    )
    audit = prepare_dataset(tmp_path)
    assert [r.text for r in prepared_records(tmp_path)] == ["keep"]
    assert audit["conflicting_intent_groups"] == 1
    assert audit["conflicting_slot_groups"] == 1
    assert audit["quarantined_records"] == 4
    assert {r["id"] for group in audit["quarantine"] for r in group["records"]} == {
        "banking/fold0.json:0",
        "banking/fold0.json:1",
        "banking/fold1.json:0",
        "banking/fold1.json:1",
    }


def test_deduplication_uses_numeric_folds_and_ignores_calendar_normalization(tmp_path):
    from reflexone.data import prepare_dataset

    raw_dataset(
        tmp_path,
        {
            "banking/fold2.json": [
                {
                    "text": "today",
                    "intents": ["affirm", "affirm"],
                    "slots": {"date": {"text": "today", "span": [0, 5], "value": {"day": 15}}},
                }
            ],
            "banking/fold10.json": [
                {
                    "text": "today",
                    "intents": ["affirm"],
                    "slots": {"date": {"text": "today", "span": [0, 5], "value": {"day": 16}}},
                }
            ],
        },
    )
    audit = prepare_dataset(tmp_path)
    (record,) = prepared_records(tmp_path)
    assert record.id == "banking/fold2.json:0"
    assert record.intents == ("affirm",)
    assert audit["canonical_records"] == 1
    assert audit["quarantined_records"] == 0
    assert audit["repeated_intent_labels_removed"] == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["records"][0]["source_ids"] == ["banking/fold2.json:0", "banking/fold10.json:0"]


def test_grouped_splits_are_frozen_repeatable_and_isolate_cross_domain_text(tmp_path):
    from reflexone.data import load_records, prepare_dataset

    raw_dataset(
        tmp_path,
        {
            "banking/fold0.json": [
                {"text": "  ＢＲＡＮＤ   NEW  ", "intents": ["new"]},
                *[{"text": f"bank question {i}", "intents": ["card"]} for i in range(100)],
            ],
            "hotels/fold0.json": [
                {"text": "brand new", "intents": ["new"]},
                *[{"text": f"hotel question {i}", "intents": ["booking"]} for i in range(100)],
            ],
        },
    )
    audit = prepare_dataset(tmp_path, seed=17)
    snapshot = {p.name: p.read_bytes() for p in tmp_path.glob("*.json*")}
    assert prepare_dataset(tmp_path, seed=17) == audit
    assert snapshot == {p.name: p.read_bytes() for p in tmp_path.glob("*.json*")}
    groups_by_split = {}
    shared_records = []
    for split in ("train", "validation", "calibration", "test"):
        records = load_records(tmp_path / f"{split}.jsonl")
        assert records
        groups_by_split[split] = {r.group for r in records}
        shared_records += [(split, r) for r in records if "NEW" in r.text or r.text == "brand new"]
    assert len(shared_records) == 2
    assert len({split for split, _ in shared_records}) == 1
    assert len({r.group for _, r in shared_records}) == 1
    assert {r.text for _, r in shared_records} == {"  ＢＲＡＮＤ   NEW  ", "brand new"}
    for split, groups in groups_by_split.items():
        assert all(not groups & other for name, other in groups_by_split.items() if name != split)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for filename, expected in manifest["output_sha256"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == expected
    assert audit["supports"]["train"]["intent_positives"]["card"] > 0
    with pytest.raises(ValueError, match="frozen"):
        prepare_dataset(tmp_path, seed=42)


def test_domain_incompatible_annotations_fail_before_publishing_splits(tmp_path):
    from reflexone.data import prepare_dataset

    raw_dataset(tmp_path, {"hotels/fold0.json": [{"text": "my card", "intents": ["card"]}]})
    with pytest.raises(ValueError, match="domain"):
        prepare_dataset(tmp_path)
    assert not (tmp_path / "manifest.json").exists()
