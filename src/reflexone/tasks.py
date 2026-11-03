"""Locally trained command routing and SMS spam profiles.

Models are classical discriminative classifiers, not neural language models.  The
artifact format uses joblib and must only be loaded from trusted local sources.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/data/data_full.json"
SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MAX_TEXT_CHARS = 20_000
SPLITS = ("train", "validation", "calibration", "rejection_validation", "oos_validation", "test")


def _check_task(task: str) -> None:
    if task not in {"commands", "spam"}:
        raise ValueError("task must be one of commands, spam")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _group(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return _sha256(normalized.encode())


def _download(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "ReflexOne/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _deduplicate(rows: list[dict], priority: dict[str, int]) -> tuple[list[dict], dict, list]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    clean, quarantine = [], []
    audit = {
        "input_rows": len(rows),
        "within_split_duplicates_removed": 0,
        "cross_split_duplicates_removed": 0,
        "conflicting_groups_quarantined": 0,
        "conflicting_rows_quarantined": 0,
    }
    for group in sorted(grouped):
        members = grouped[group]
        if len({r["label"] for r in members}) > 1:
            quarantine.extend(members)
            audit["conflicting_groups_quarantined"] += 1
            audit["conflicting_rows_quarantined"] += len(members)
            continue
        ordered = sorted(members, key=lambda r: (priority[r["split"]], r["id"]))
        chosen = ordered[0]
        clean.append(chosen)
        for duplicate in ordered[1:]:
            key = (
                "within_split_duplicates_removed"
                if duplicate["split"] == chosen["split"]
                else "cross_split_duplicates_removed"
            )
            audit[key] += 1
    audit["retained_rows"] = len(clean)
    return clean, audit, quarantine


def _stratified_split(
    rows: list[dict], names: tuple[str, ...], fractions: tuple[float, ...], seed: int
) -> dict[str, list[dict]]:
    """Stratify independent normalized-text groups; inputs are already deduplicated."""
    output: dict[str, list[dict]] = {name: [] for name in names}
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = np.random.default_rng(seed)
    for label in sorted(by_label):
        members = sorted(by_label[label], key=lambda r: r["group"])
        indices = rng.permutation(len(members))
        desired = np.asarray(fractions) * len(members)
        counts = np.floor(desired).astype(int)
        for index in np.argsort(-(desired - counts), kind="stable")[: len(members) - counts.sum()]:
            counts[index] += 1
        if len(members) >= len(names):
            for index in np.where(counts == 0)[0]:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[index] += 1
        offset = 0
        for name, count in zip(names, counts, strict=True):
            for i in indices[offset : offset + count]:
                output[name].append({**members[int(i)], "split": name})
            offset += int(count)
    return {name: sorted(records, key=lambda r: r["id"]) for name, records in output.items()}


def prepare_task(task: str, data_dir: Path, seed: int = 17) -> dict:
    """Download/audit real data and persist frozen train/select/calibrate/test groups."""
    _check_task(task)
    data_dir = Path(data_dir)
    rows = []
    if task == "commands":
        raw = _download(CLINC_URL, data_dir / "raw" / "data_full.json")
        payload = json.loads(raw)
        source_map = {
            "train": "train",
            "val": "validation_pool",
            "test": "test",
            "oos_val": "oos_validation",
            "oos_test": "test",
        }
        for source_split, split in source_map.items():
            for index, (text, label) in enumerate(payload[source_split]):
                rows.append(
                    {
                        "id": f"clinc:{source_split}:{index:05d}",
                        "text": text,
                        "label": label,
                        "source_split": source_split,
                        "split": split,
                        "group": _group(text),
                    }
                )
        clean, audit, quarantine = _deduplicate(
            rows, {"test": 0, "oos_validation": 1, "validation_pool": 2, "train": 3}
        )
        splits = {name: [r for r in clean if r["split"] == name] for name in SPLITS}
        splits.update(
            _stratified_split(
                [r for r in clean if r["split"] == "validation_pool"],
                ("validation", "calibration", "rejection_validation"),
                (0.5, 0.25, 0.25),
                seed,
            )
        )
        labels = sorted({row["label"] for row in splits["train"]})
        audit["unused_official_oos_train"] = len(payload.get("oos_train", []))
        source = {
            "dataset": "CLINC150 full",
            "url": CLINC_URL,
            "revision": CLINC_REVISION,
            "license": "CC-BY-3.0",
            "license_url": f"https://github.com/clinc/oos-eval/blob/{CLINC_REVISION}/LICENSE",
            "citation": "Larson et al. (2019), https://aclanthology.org/D19-1131/",
        }
        policy = (
            "Official test + oos_test retained. Duplicate priority: test > oos_val > val > train. "
            "Conflicting labels are quarantined globally. Official val split per class into "
            "50% selection, 25% probability calibration, 25% ID rejection validation. "
            "Official oos_val is used only to tune rejection; oos_train is unused."
        )
    else:
        raw = _download(SMS_URL, data_dir / "raw" / "sms.zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            contents = archive.read("SMSSpamCollection").decode("utf-8-sig")
        for index, line in enumerate(contents.splitlines()):
            if not line.strip():
                continue
            label, text = line.split("\t", 1)
            if label not in {"ham", "spam"}:
                raise ValueError(f"unexpected SMS label: {label!r}")
            rows.append(
                {
                    "id": f"sms:{index:05d}",
                    "text": text,
                    "label": label,
                    "source_split": "collection",
                    "split": "pool",
                    "group": _group(text),
                }
            )
        clean, audit, quarantine = _deduplicate(rows, {"pool": 0})
        splits = _stratified_split(
            clean, ("train", "validation", "calibration", "test"), (0.65, 0.15, 0.1, 0.1), seed
        )
        labels = ["ham", "spam"]
        source = {
            "dataset": "UCI SMS Spam Collection",
            "url": SMS_URL,
            "revision": None,
            "license": "CC-BY-4.0",
            "license_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
            "citation": "Almeida & Hidalgo (2011), https://doi.org/10.24432/C5CC84",
        }
        policy = (
            "Custom stratified normalized-text-group 65/15/10/10 train/validation/calibration/test "
            "split. Conflicting labels quarantined; same-label duplicates collapsed. "
            "No official chronological split exists. ham=0, spam=1; spam is positive."
        )
    source["sha256"] = _sha256(raw)
    all_groups: set[str] = set()
    for name, records in splits.items():
        groups = {row["group"] for row in records}
        if groups & all_groups:
            raise ValueError("normalized-text group leakage between partitions")
        all_groups.update(groups)
        _write_rows(data_dir / f"{name}.jsonl", records)
    for name in set(SPLITS) - set(splits):
        (data_dir / f"{name}.jsonl").unlink(missing_ok=True)
    manifest = {
        "format_version": 1,
        "task": task,
        "seed": seed,
        "labels": labels,
        "source": source,
        "normalization": "NFKC + casefold + collapsed whitespace; SHA-256 key",
        "split_policy": policy,
        "audit": audit,
        "splits": {
            name: {"rows": len(rs), "labels": dict(sorted(Counter(r["label"] for r in rs).items()))}
            for name, rs in splits.items()
        },
        "partition_sha256": {
            name: _sha256((data_dir / f"{name}.jsonl").read_bytes()) for name in splits
        },
    }
    _write_json(data_dir / "audit" / "quarantine.json", quarantine)
    _write_json(data_dir / "manifest.json", manifest)
    return manifest


def _softmax(logits: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=float), axis=1)


def _logits(model: Pipeline, texts: list[str]) -> np.ndarray:
    # A common probability-derived logit representation works for both algorithms.
    return np.log(np.clip(model.predict_proba(texts), 1e-15, 1.0))


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits, targets = np.asarray(logits, float), np.asarray(targets, int)
    if not len(targets):
        raise ValueError("calibration requires non-empty held-out data")

    def objective(log_temperature: float) -> float:
        scores = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(targets)), targets]))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def _metrics(targets: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(labels))[targets]
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    bins, ece = [], 0.0
    edges = np.linspace(0, 1, 11)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            accuracy, mean_confidence = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - mean_confidence)
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "accuracy": accuracy,
                    "confidence": mean_confidence,
                }
            )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(len(labels)), zero_division=0
    )
    result = {
        "n": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=np.arange(len(labels)),
                average="macro",
                zero_division=0,
            )
        ),
        "nll": float(log_loss(targets, probabilities, labels=np.arange(len(labels)))),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": ece,
        "reliability_bins": bins,
        "per_label": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }
    if labels == ["ham", "spam"]:
        result.update(
            {
                "positive_label": "spam",
                "spam_precision": float(precision[1]),
                "spam_recall": float(recall[1]),
                "spam_f1": float(f1[1]),
                "spam_average_precision": float(
                    average_precision_score(targets, probabilities[:, 1])
                ),
                "confusion_matrix_labels": labels,
                "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
            }
        )
    return result


def _select_rejection_threshold(known: np.ndarray, unknown: np.ndarray) -> tuple[float, dict]:
    # Equal weighting of the ID and OOS classes avoids optimizing CLINC's unequal prevalence.
    values = np.unique(np.concatenate([known, unknown]))
    candidates = np.concatenate([[0.0], values, np.nextafter(values, np.inf), [1.0000001]])
    scored = [
        (0.5 * (float((known >= t).mean()) + float((unknown < t).mean())), float(t))
        for t in candidates
    ]
    score, threshold = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return threshold, {
        "criterion": "balanced in-scope acceptance / OOS rejection accuracy",
        "validation_balanced_accuracy": score,
        "validation_in_scope_acceptance": float((known >= threshold).mean()),
        "validation_oos_recall": float((unknown < threshold).mean()),
        "known_n": len(known),
        "oos_n": len(unknown),
    }


def _model(name: str, seed: int) -> Pipeline:
    classifier = (
        LogisticRegression(C=8.0, max_iter=1000, solver="lbfgs", random_state=seed)
        if name == "tfidf_logistic_regression"
        else ComplementNB(alpha=0.1)
    )
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    max_features=60_000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def train_task(task: str, data_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Fit two algorithms; select on validation; calibrate on separate data; freeze."""
    _check_task(task)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if not (data_dir / "manifest.json").exists():
        prepare_task(task, data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["task"] != task:
        raise ValueError("prepared data belongs to a different task")
    splits = {name: _read_rows(data_dir / f"{name}.jsonl") for name in manifest["splits"]}
    for name, expected in manifest["partition_sha256"].items():
        if _sha256((data_dir / f"{name}.jsonl").read_bytes()) != expected:
            raise ValueError(f"prepared partition was modified after audit: {name}")
    labels = manifest["labels"]
    index = {label: i for i, label in enumerate(labels)}
    targets = {
        name: np.array([index[r["label"]] for r in splits[name]], dtype=int)
        for name in ("train", "validation", "calibration")
    }
    train_texts = [r["text"] for r in splits["train"]]
    validation_texts = [r["text"] for r in splits["validation"]]
    selection_metric = "spam_f1" if task == "spam" else "macro_f1"
    candidates, baseline_reports = {}, {}
    started = time.perf_counter()
    for name in ("tfidf_logistic_regression", "tfidf_complement_nb"):
        fit_started = time.perf_counter()
        model = _model(name, seed)
        model.fit(train_texts, targets["train"])
        probabilities = model.predict_proba(validation_texts)
        metrics = _metrics(targets["validation"], probabilities, labels)
        baseline_reports[name] = {
            "validation": metrics,
            "fit_seconds": time.perf_counter() - fit_started,
        }
        candidates[name] = model
    selected = max(
        candidates, key=lambda name: baseline_reports[name]["validation"][selection_metric]
    )
    model = candidates[selected]
    calibration_logits = _logits(model, [r["text"] for r in splits["calibration"]])
    temperature = _fit_temperature(calibration_logits, targets["calibration"])
    threshold, rejection = None, None
    if task == "commands":
        known_probabilities = _softmax(
            _logits(model, [r["text"] for r in splits["rejection_validation"]]) / temperature
        )
        oos_probabilities = _softmax(
            _logits(model, [r["text"] for r in splits["oos_validation"]]) / temperature
        )
        threshold, rejection = _select_rejection_threshold(
            known_probabilities.max(axis=1), oos_probabilities.max(axis=1)
        )
        rejection.update(
            {
                "threshold": threshold,
                "id_partition": "rejection_validation",
                "oos_partition": "oos_validation",
                "score": "temperature-scaled maximum class probability",
                "probability_scope": "class probabilities condition on membership in supported intents",
            }
        )
    # The selection and calibration decisions above are already frozen. Test
    # scores below are comparison evidence and never feed model selection.
    baseline_test_rows = [r for r in splits["test"] if r["label"] != "oos"]
    baseline_test_targets = np.array([index[r["label"]] for r in baseline_test_rows])
    for name, candidate in candidates.items():
        baseline_reports[name]["test"] = _metrics(
            baseline_test_targets,
            candidate.predict_proba([r["text"] for r in baseline_test_rows]),
            labels,
        )
        baseline_reports[name]["test_scope"] = (
            "uncalibrated supported-intent classification; excludes OOS"
            if task == "commands"
            else "uncalibrated full spam/ham classification"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "task": task,
        "model": model,
        "labels": labels,
        "temperature": temperature,
        "rejection_threshold": threshold,
        "selected_model": selected,
    }
    joblib.dump(checkpoint, output_dir / "model.joblib", compress=3)
    _write_rows(output_dir / "evaluation.jsonl", splits["test"])
    metadata = {
        "format_version": 1,
        "task": task,
        "seed": seed,
        "dataset": manifest,
        "selected_model": selected,
        "selection": {
            "partition": "validation",
            "metric": selection_metric,
            "refit_on_validation": False,
        },
        "baselines": baseline_reports,
        "training_config": {
            "vectorizer": {
                "name": "TfidfVectorizer",
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "strip_accents": "unicode",
                "max_features": 60_000,
                "fit_partition": "train",
            },
            "logistic_regression": {"C": 8.0, "max_iter": 1000, "solver": "lbfgs"},
            "complement_naive_bayes": {"alpha": 0.1},
            "spam_decision_threshold": 0.5 if task == "spam" else None,
            "temperature_bounds": [float(np.exp(-3)), float(np.exp(3))],
        },
        "calibration": {
            "method": "scalar temperature scaling of clipped log probabilities",
            "partition": "calibration",
            "temperature": temperature,
            "n": len(targets["calibration"]),
            "before": _metrics(targets["calibration"], _softmax(calibration_logits), labels),
            "after": _metrics(
                targets["calibration"], _softmax(calibration_logits / temperature), labels
            ),
        },
        "rejection": rejection,
        "training_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "hashes": {
            "model_sha256": _sha256((output_dir / "model.joblib").read_bytes()),
            "evaluation_sha256": _sha256((output_dir / "evaluation.jsonl").read_bytes()),
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    report = evaluate_task(output_dir)
    _write_json(output_dir / "report.json", report)
    return report


class TaskPredictor:
    """Load a trusted local artifact once for CLI, HTTP and batch inference."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        metadata = json.loads((self.artifact_dir / "metadata.json").read_text())
        actual_hash = _sha256((self.artifact_dir / "model.joblib").read_bytes())
        if actual_hash != metadata["hashes"]["model_sha256"]:
            raise ValueError("model artifact checksum does not match recorded identity")
        checkpoint = joblib.load(self.artifact_dir / "model.joblib")
        if (
            checkpoint["task"] != metadata["task"]
            or checkpoint["selected_model"] != metadata["selected_model"]
            or checkpoint["labels"] != metadata["dataset"]["labels"]
            or checkpoint["temperature"] != metadata["calibration"]["temperature"]
        ):
            raise ValueError("Model checkpoint metadata does not match its recorded identity")
        saved_threshold = metadata["rejection"]["threshold"] if metadata["rejection"] else None
        if checkpoint["rejection_threshold"] != saved_threshold:
            raise ValueError("Model rejection threshold differs from recorded metadata")
        self.task = checkpoint["task"]
        _check_task(self.task)
        self.model = checkpoint["model"]
        self.labels = checkpoint["labels"]
        self.temperature = checkpoint["temperature"]
        self.rejection_threshold = checkpoint["rejection_threshold"]
        self.selected_model = checkpoint["selected_model"]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"text must contain at most {MAX_TEXT_CHARS} characters")
        probabilities = _softmax(_logits(self.model, texts) / self.temperature)
        output = []
        for probs in probabilities:
            index = int(probs.argmax())
            confidence = float(probs[index])
            abstain = self.task == "commands" and confidence < self.rejection_threshold
            output.append(
                {
                    "task": self.task,
                    "label": "oos" if abstain else self.labels[index],
                    "candidate_label": self.labels[index],
                    "confidence": confidence,
                    "probabilities": dict(zip(self.labels, map(float, probs), strict=True)),
                    "abstain": bool(abstain),
                    "reason": "below_validation_selected_in_scope_threshold" if abstain else None,
                    "model": self.selected_model,
                }
            )
        return output


def predict_task(artifact_dir: Path, text: str) -> dict:
    return TaskPredictor(artifact_dir).predict(text)


def _coverage_curve(targets: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence, kind="stable")
    thresholds = {
        float(confidence[order[max(1, int(np.ceil(fraction * len(targets)))) - 1]])
        for fraction in (0.1, 0.25, 0.5, 0.75, 1.0)
    }
    result = []
    for threshold in sorted(thresholds, reverse=True):
        accepted = confidence >= threshold
        accuracy = float(correct[accepted].mean())
        result.append(
            {
                "score_threshold": threshold,
                "coverage": float(accepted.mean()),
                "accepted": int(accepted.sum()),
                "accuracy": accuracy,
                "risk": 1 - accuracy,
            }
        )
    return result


def evaluate_task(artifact_dir: Path, *, allow_modified_evaluation: bool = False) -> dict:
    """Evaluate the frozen artifact on its copied held-out examples, without fitting."""
    artifact_dir = Path(artifact_dir)
    predictor = TaskPredictor(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    evaluation_hash = _sha256((artifact_dir / "evaluation.jsonl").read_bytes())
    evaluation_modified = evaluation_hash != metadata["hashes"]["evaluation_sha256"]
    if evaluation_modified and not allow_modified_evaluation:
        raise ValueError("evaluation dataset checksum differs from the frozen artifact")
    rows = _read_rows(artifact_dir / "evaluation.jsonl")
    texts = [row["text"] for row in rows]
    raw_logits = _logits(predictor.model, texts)
    raw_probabilities = _softmax(raw_logits)
    probabilities = _softmax(raw_logits / predictor.temperature)
    labels = predictor.labels
    index = {label: i for i, label in enumerate(labels)}
    known_mask = np.array([r["label"] != "oos" for r in rows])
    known_targets = np.array([index[r["label"]] for r in rows if r["label"] != "oos"])
    raw_metrics = _metrics(known_targets, raw_probabilities[known_mask], labels)
    calibrated_metrics = _metrics(known_targets, probabilities[known_mask], labels)
    test = dict(calibrated_metrics)
    if predictor.task == "commands":
        confidence = probabilities.max(axis=1)
        accepted = confidence >= predictor.rejection_threshold
        predictions = [
            labels[int(probs.argmax())] if keep else "oos"
            for probs, keep in zip(probabilities, accepted, strict=True)
        ]
        truth = [row["label"] for row in rows]
        test = {
            "n": len(rows),
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1_including_oos": float(
                f1_score(
                    truth,
                    predictions,
                    labels=labels + ["oos"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "in_scope": calibrated_metrics,
            "oos_n": int((~known_mask).sum()),
            "oos_recall": float((~accepted[~known_mask]).mean()),
            "in_scope_acceptance": float(accepted[known_mask].mean()),
            "oos_detection_auroc": float(roc_auc_score(~known_mask, 1 - confidence)),
            "accepted_coverage": float(accepted.mean()),
            "accepted_accuracy": float(
                np.mean(
                    np.array(predictions, dtype=object)[accepted]
                    == np.array(truth, dtype=object)[accepted]
                )
            )
            if accepted.any()
            else None,
            "conditional_probability_warning": (
                "Class probabilities are conditional on supported intents; confidence is not P(in-scope)."
            ),
        }
    return {
        **metadata,
        "test": test,
        "calibration_ablation": {
            "uncalibrated_in_scope": raw_metrics,
            "calibrated_in_scope": calibrated_metrics,
        },
        "selective_curve_in_scope": _coverage_curve(known_targets, probabilities[known_mask]),
        "evaluation_sha256_current": evaluation_hash,
        "evaluation_modified": evaluation_modified,
    }


def benchmark_task(artifact_dir: Path, repeats: int = 100) -> dict:
    """Warm batch-one CPU latency including vectorization, calibration and JSON dict decoding."""
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    artifact_dir = Path(artifact_dir)
    started = time.perf_counter()
    predictor = TaskPredictor(artifact_dir)
    load_ms = (time.perf_counter() - started) * 1000
    examples = [row["text"] for row in _read_rows(artifact_dir / "evaluation.jsonl")][:100]
    for text in examples[:5]:
        predictor.predict(text)
    timings = []
    for index in range(repeats):
        started = time.perf_counter()
        predictor.predict(examples[index % len(examples)])
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "task": predictor.task,
        "repeats": repeats,
        "batch_size": 1,
        "load_ms": load_ms,
        "latency_ms": {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "mean": float(np.mean(timings)),
        },
        "scope": "warm CPU; text vectorization + classifier + temperature + typed dict; no network or HTTP",
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
    }
