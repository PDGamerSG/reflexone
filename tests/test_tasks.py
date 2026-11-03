"""Behavior tests for locally trained classical profiles; no network is required."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from reflexone import tasks


def write_commands(path: Path) -> None:
    raw = path / "raw"
    raw.mkdir(parents=True)
    payload = {}
    for split, count in (("train", 24), ("val", 16), ("test", 8)):
        payload[split] = [
            [f"{verb} {noun} {split} request {index}", label]
            for label, verb, noun in (
                ("lights_on", "illuminate", "lights"),
                ("play_music", "play", "music"),
            )
            for index in range(count)
        ]
    payload["oos_train"] = [["galaxy chocolate training", "oos"]]
    payload["oos_val"] = [[f"quantum banana asteroid validation {i}", "oos"] for i in range(8)]
    payload["oos_test"] = [[f"quantum banana asteroid final {i}", "oos"] for i in range(8)]
    # Duplicate official evaluation texts must never reach fitting or calibration.
    payload["train"].append(["  PLAY music duplicate  ", "play_music"])
    payload["test"].append(["play MUSIC duplicate", "play_music"])
    (raw / "data_full.json").write_text(json.dumps(payload))


def write_spam(path: Path) -> None:
    raw = path / "raw"
    raw.mkdir(parents=True)
    lines = []
    for i in range(40):
        lines.append(f"ham\tHey friend lunch meeting confirmed number {i}")
        lines.append(f"spam\tWIN cash prize free lottery claim now offer {i}")
    lines.extend(
        [
            "ham\tHey friend lunch meeting confirmed number 1",
            "spam\tConflicting duplicate",
            "ham\t  conflicting DUPLICATE ",
        ]
    )
    with zipfile.ZipFile(raw / "sms.zip", "w") as archive:
        archive.writestr("SMSSpamCollection", "\n".join(lines))


def records(path: Path) -> dict[str, list[dict]]:
    return {
        p.stem: [json.loads(line) for line in p.read_text().splitlines()]
        for p in path.glob("*.jsonl")
    }


def test_commands_preserve_official_test_and_isolate_calibration(tmp_path):
    write_commands(tmp_path)
    manifest = tasks.prepare_task("commands", tmp_path)
    splits = records(tmp_path)
    groups = {split: {r["group"] for r in rows} for split, rows in splits.items()}
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert groups[left].isdisjoint(groups[right]), (left, right)
    assert len(splits["test"]) == 25
    assert any(r["text"] == "play MUSIC duplicate" for r in splits["test"])
    assert all("duplicate" not in r["text"].lower() for r in splits["train"])
    assert all(r["source_split"] == "val" for r in splits["calibration"])
    assert all(r["source_split"] == "oos_val" for r in splits["oos_validation"])
    assert manifest["audit"]["cross_split_duplicates_removed"] == 1
    assert tasks.prepare_task("commands", tmp_path) == manifest


def test_spam_group_split_is_deterministic_and_quarantines_label_conflicts(tmp_path):
    write_spam(tmp_path)
    first = tasks.prepare_task("spam", tmp_path, seed=17)
    first_text = {p.name: p.read_text() for p in tmp_path.glob("*.jsonl")}
    tasks.prepare_task("spam", tmp_path, seed=17)
    assert first_text == {p.name: p.read_text() for p in tmp_path.glob("*.jsonl")}
    assert first["labels"] == ["ham", "spam"]
    assert first["audit"]["conflicting_groups_quarantined"] == 1
    assert first["audit"]["within_split_duplicates_removed"] == 1
    sets = [set(r["group"] for r in rs) for rs in records(tmp_path).values()]
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            assert left.isdisjoint(right)
    assert all({r["label"] for r in rs} == {"ham", "spam"} for rs in records(tmp_path).values())


def test_spam_training_round_trip_preserves_positive_label_and_evaluation(tmp_path):
    data, artifact = tmp_path / "data", tmp_path / "artifact"
    write_spam(data)
    tasks.prepare_task("spam", data)
    report = tasks.train_task("spam", data, artifact)
    predictor = tasks.TaskPredictor(artifact)
    spam = predictor.predict("WIN cash prize free lottery claim now")
    ham = tasks.predict_task(artifact, "Hey friend lunch meeting confirmed")
    assert spam["label"] == "spam"
    assert ham["label"] == "ham"
    assert spam["probabilities"]["spam"] > spam["probabilities"]["ham"]
    assert sum(spam["probabilities"].values()) == pytest.approx(1.0)
    assert report["test"]["accuracy"] == 1.0
    assert report["test"]["positive_label"] == "spam"
    assert len(report["baselines"]) == 2
    assert all(baseline["test"]["n"] == 8 for baseline in report["baselines"].values())
    assert tasks.evaluate_task(artifact)["test"] == report["test"]
    json.dumps(spam, allow_nan=False)
    assert predictor.predict_batch(["free lottery cash", "hey friend lunch"])[0]["label"] == "spam"
    benchmark = tasks.benchmark_task(artifact, repeats=5)
    assert benchmark["repeats"] == 5
    assert 0 <= benchmark["latency_ms"]["p50"] <= benchmark["latency_ms"]["p95"]
    with pytest.raises(ValueError, match="non-empty"):
        predictor.predict(" \t")
    with pytest.raises(ValueError):
        tasks.benchmark_task(artifact, repeats=0)


def test_command_rejection_is_learned_without_touching_final_oos_labels(tmp_path):
    data, artifact = tmp_path / "data", tmp_path / "artifact"
    write_commands(data)
    tasks.prepare_task("commands", data)
    report = tasks.train_task("commands", data, artifact)
    known = tasks.predict_task(artifact, "illuminate lights request")
    unknown = tasks.predict_task(artifact, "quantum banana asteroid")
    assert known["label"] == "lights_on"
    assert known["abstain"] is False
    assert unknown["label"] == "oos"
    assert unknown["abstain"] is True
    assert report["test"]["oos_recall"] == 1.0
    assert report["selection"]["partition"] == "validation"
    assert report["calibration"]["partition"] == "calibration"
    assert report["rejection"]["oos_partition"] == "oos_validation"

    # Official re-evaluation rejects changed data. Explicit perturbation is separately identified.
    test_path = artifact / "evaluation.jsonl"
    heldout = [json.loads(line) for line in test_path.read_text().splitlines()]
    heldout[0]["label"] = "oos" if heldout[0]["label"] != "oos" else "lights_on"
    test_path.write_text("\n".join(json.dumps(row) for row in heldout) + "\n")
    with pytest.raises(ValueError, match="evaluation.*checksum|checksum.*evaluation"):
        tasks.evaluate_task(artifact)
    perturbed = tasks.evaluate_task(artifact, allow_modified_evaluation=True)
    assert perturbed["test"]["accuracy"] != report["test"]["accuracy"]
    assert perturbed["evaluation_modified"] is True


def test_temperature_is_fitted_on_disjoint_data_and_normalizes_probabilities():
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
    targets = np.array([0, 1, 1, 0])
    temperature = tasks._fit_temperature(logits, targets)
    assert temperature > 1
    probabilities = tasks._softmax(logits / temperature)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[0, 0] < 0.99


def test_cached_artifact_never_claims_an_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="commands.*spam"):
        tasks.prepare_task("invented", tmp_path)
    with pytest.raises(FileNotFoundError):
        tasks.TaskPredictor(tmp_path)


def test_selective_curve_never_splits_tied_confidence_groups():
    curve = tasks._coverage_curve(np.array([0, 1]), np.array([[0.9, 0.1], [0.9, 0.1]]))
    assert all(point["accepted"] == 2 for point in curve)
    assert all(point["risk"] == 0.5 for point in curve)
    assert all(point["score_threshold"] == 0.9 for point in curve)


def test_model_artifact_checksum_prevents_silent_identity_swap(tmp_path):
    data, artifact = tmp_path / "data", tmp_path / "artifact"
    write_spam(data)
    tasks.prepare_task("spam", data)
    tasks.train_task("spam", data, artifact)
    with (artifact / "model.joblib").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="model.*checksum|checksum.*model"):
        tasks.TaskPredictor(artifact)
