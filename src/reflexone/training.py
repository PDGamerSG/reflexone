"""Reproducible fine-tuning, classical baseline, and held-out support evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoConfig, AutoModel, AutoTokenizer

from reflexone.data import Ontology, Record, load_ontology, load_records
from reflexone.model import (
    DOMAINS,
    ENCODER_NAME,
    ENCODER_REVISION,
    DecisionNetwork,
    align_slot_targets,
    decode_slot_spans,
    masked_multitask_loss,
)
from reflexone.reliability import (
    apply_temperature,
    calibrated_threshold,
    fit_review_threshold,
    fit_temperature,
    multilabel_metrics,
    review_scores,
    selective_curve,
    tune_threshold,
)


@dataclass
class SupportConfig:
    encoder_name: str = ENCODER_NAME
    encoder_revision: str | None = ENCODER_REVISION
    seed: int = 42
    epochs: int = 14
    batch_size: int = 32
    max_length: int = 128
    encoder_lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    dropout: float = 0.15
    slot_weight: float = 1.0
    patience: int = 4
    device: str = "auto"
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.max_length < 4:
            raise ValueError("epochs/batch_size must be positive and max_length must be >=4")
        if self.encoder_lr <= 0 or self.head_lr <= 0 or self.cpu_threads < 1:
            raise ValueError("Learning rates and cpu_threads must be positive")
        if not 0 <= self.dropout < 1 or self.patience < 1:
            raise ValueError("Invalid dropout or patience")


class EncodedSupport(Dataset):
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        offsets: list,
        records: Sequence[Record],
        alignment_audit: dict,
    ) -> None:
        self.tensors = tensors
        self.offsets = offsets
        self.records = list(records)
        self.alignment_audit = alignment_audit

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{name: tensor[index] for name, tensor in self.tensors.items()},
            "index": torch.tensor(index),
        }


def encode_records(
    records: Sequence[Record], tokenizer: Any, ontology: Ontology, max_length: int = 128
) -> EncodedSupport:
    encoded = tokenizer(
        [r.text for r in records],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping").tolist()
    intent_ids = {name: i for i, name in enumerate(ontology.intents)}
    targets = np.zeros((len(records), len(ontology.intents)), dtype=np.float32)
    slots = []
    gold_spans, exact_boundaries, truncated = 0, 0, 0
    for row, (record, token_offsets) in enumerate(zip(records, offsets, strict=True)):
        for label in record.intents:
            targets[row, intent_ids[label]] = 1
        slots.append(align_slot_targets(token_offsets, record.slots, ontology.slot_types))
        visible_end = max(end for _, end in token_offsets)
        if record.text[visible_end:].strip():
            truncated += 1
        for slot in record.slots:
            gold_spans += 1
            intersect = [
                (s, e) for s, e in token_offsets if e > s and s < slot.end and e > slot.start
            ]
            if intersect and intersect[0][0] == slot.start and intersect[-1][1] == slot.end:
                exact_boundaries += 1
    tensors = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "domains": torch.tensor([DOMAINS[r.domain] for r in records], dtype=torch.long),
        "intent_targets": torch.from_numpy(targets),
        "slot_targets": torch.from_numpy(np.asarray(slots)),
        "intent_mask": torch.tensor([ontology.intent_mask(r.domain) for r in records]),
        "slot_mask": torch.tensor([ontology.slot_mask(r.domain) for r in records]),
    }
    return EncodedSupport(
        tensors,
        offsets,
        records,
        {
            "records": len(records),
            "gold_spans": gold_spans,
            "exact_token_boundaries": exact_boundaries,
            "non_exact_token_boundaries": gold_spans - exact_boundaries,
            "truncated_texts": truncated,
        },
    )


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = default_collate(rows)
    length = int(batch["attention_mask"].sum(1).max())
    for name in ("input_ids", "attention_mask", "slot_targets"):
        batch[name] = batch[name][:, :length]
    return batch


def span_metrics(gold: Sequence[Sequence[dict]], predicted: Sequence[Sequence[dict]]) -> dict:
    tp = fp = fn = 0
    exact = []
    per_type: dict[str, dict[str, int]] = {}
    for actual, guess in zip(gold, predicted, strict=True):
        a = {(s["label"], s["start"], s["end"]) for s in actual}
        b = {(s["label"], s["start"], s["end"]) for s in guess}
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
        exact.append(a == b)
        for label in {x[0] for x in a | b}:
            counts = per_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            counts["tp"] += sum(x[0] == label for x in a & b)
            counts["fp"] += sum(x[0] == label for x in b - a)
            counts["fn"] += sum(x[0] == label for x in a - b)
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_match": float(np.mean(exact)) if exact else 0.0,
        "per_type": per_type,
    }


def _gold_spans(records: Sequence[Record]) -> list[list[dict]]:
    return [[asdict(slot) for slot in r.slots] for r in records]


def collect_predictions(
    model: DecisionNetwork,
    dataset: EncodedSupport,
    ontology: Ontology,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, list[list[dict]]]:
    logits = np.zeros((len(dataset), len(ontology.intents)), dtype=np.float32)
    spans: list[list[dict]] = [[] for _ in dataset.records]
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            indices = batch["index"].tolist()
            result = model(
                **{k: batch[k].to(device) for k in ("input_ids", "attention_mask", "domains")}
            )
            logits[indices] = result["intent_logits"].cpu().numpy()
            probabilities = result["slot_logits"].softmax(-1).cpu().numpy()
            for row, index in enumerate(indices):
                length = probabilities.shape[1]
                spans[index] = decode_slot_spans(
                    dataset.records[index].text,
                    dataset.offsets[index][:length],
                    probabilities[row],
                    ontology.slot_types,
                    ontology.slot_mask(dataset.records[index].domain),
                )
    return logits, spans


def save_network(
    model: DecisionNetwork,
    tokenizer: Any,
    output_dir: Path,
    config: SupportConfig,
    ontology: Ontology,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(output_dir / "model.safetensors"),
    )
    _write_json(output_dir / "config.json", asdict(config))
    _write_json(
        output_dir / "labels.json",
        {
            "intents": list(ontology.intents),
            "slots": list(ontology.slot_types),
        },
    )


def load_network(artifact_dir: Path, device: str = "cpu") -> tuple[DecisionNetwork, Any, dict]:
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    actual_hash = hashlib.sha256((artifact_dir / "model.safetensors").read_bytes()).hexdigest()
    if actual_hash != metadata.get("model_sha256"):
        raise ValueError("Support model checksum differs from recorded artifact identity")
    config = json.loads((artifact_dir / "config.json").read_text())
    labels = json.loads((artifact_dir / "labels.json").read_text())
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(artifact_dir / "encoder"))
    model = DecisionNetwork(
        encoder, len(labels["intents"]), len(labels["slots"]), dropout=config["dropout"]
    )
    model.load_state_dict(load_file(str(artifact_dir / "model.safetensors")), strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer", local_files_only=True)
    return model, tokenizer, config


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify_prepared_data(data_dir: Path, expected_manifest_hash: str | None = None) -> str:
    """Reject changed partitions or a swapped dataset before trusting recorded results."""
    raw = (data_dir / "manifest.json").read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if expected_manifest_hash is not None and fingerprint != expected_manifest_hash:
        raise ValueError("Prepared dataset manifest identity does not match the trained artifact")
    manifest = json.loads(raw)
    for name, expected in manifest["output_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Manifest output names must be plain filenames")
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Prepared dataset checksum mismatch for {name}")
    return fingerprint


def fit_support_review(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float, mask: np.ndarray
) -> float:
    """Fit review only among nonempty decisions eligible for deployed acceptance."""
    eligible = ((probabilities >= threshold) & mask).any(axis=1)
    return fit_review_threshold(
        targets[eligible], probabilities[eligible], threshold, mask[eligible]
    )


def _baseline(datasets: dict[str, EncodedSupport], output_dir: Path) -> dict:
    train = datasets["train"]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", max_features=40000
    )
    features = vectorizer.fit_transform([r.text for r in train.records])
    classifier = OneVsRestClassifier(LogisticRegression(C=4, max_iter=500, random_state=42))
    start = time.perf_counter()
    classifier.fit(features, train.tensors["intent_targets"].numpy())
    duration = time.perf_counter() - start
    probs = {
        split: classifier.predict_proba(vectorizer.transform([r.text for r in ds.records]))
        for split, ds in datasets.items()
        if split != "train"
    }
    targets = {k: ds.tensors["intent_targets"].numpy() for k, ds in datasets.items()}
    masks = {k: ds.tensors["intent_mask"].numpy() for k, ds in datasets.items()}
    threshold = tune_threshold(probs["validation"], targets["validation"], masks["validation"])
    temperature = fit_temperature(
        logit(np.clip(probs["calibration"], 1e-7, 1 - 1e-7)),
        targets["calibration"],
        masks["calibration"],
    )
    test_calibrated = apply_temperature(logit(np.clip(probs["test"], 1e-7, 1 - 1e-7)), temperature)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "temperature": temperature,
            "raw_threshold": threshold,
        },
        output_dir / "baseline.joblib",
    )
    return {
        "algorithm": "TF-IDF word (1,2)-grams + one-vs-rest logistic regression",
        "training_seconds": duration,
        "raw_threshold": threshold,
        "temperature": temperature,
        "raw": multilabel_metrics(targets["test"], probs["test"], threshold, masks["test"]),
        "calibrated": multilabel_metrics(
            targets["test"],
            test_calibrated,
            calibrated_threshold(threshold, temperature),
            masks["test"],
        ),
    }


def train_support(data_dir: Path, output_dir: Path, config: SupportConfig) -> dict:
    """Select epochs on validation, calibrate separately, then evaluate frozen test data."""
    verify_prepared_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "model.safetensors").exists():
        raise ValueError(
            f"Refusing to overwrite trained weights: {output_dir}; use a new directory"
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.seed)
    device = (
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        if config.device == "auto"
        else config.device
    )
    ontology = load_ontology(data_dir / "ontology.json")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name, revision=config.encoder_revision)
    datasets = {
        split: encode_records(
            load_records(data_dir / f"{split}.jsonl"), tokenizer, ontology, config.max_length
        )
        for split in ("train", "validation", "calibration", "test")
    }
    _write_json(
        output_dir / "alignment.json", {k: ds.alignment_audit for k, ds in datasets.items()}
    )
    encoder = AutoModel.from_pretrained(config.encoder_name, revision=config.encoder_revision)
    model = DecisionNetwork(
        encoder, len(ontology.intents), len(ontology.slot_types), config.dropout
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.encoder_lr},
            {
                "params": [
                    p for name, p in model.named_parameters() if not name.startswith("encoder.")
                ],
                "lr": config.head_lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
    )
    total_steps = len(loader) * config.epochs
    warmup = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(0.05, step / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    train_targets = datasets["train"].tensors["intent_targets"]
    valid_count = datasets["train"].tensors["intent_mask"].sum(0)
    positives = train_targets.sum(0).clamp_min(1)
    pos_weight = ((valid_count - positives) / positives).sqrt().clamp(1, 8).to(device)
    history = []
    best_score, best_epoch, stale = -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items() if name != "index"}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "domains")})
            loss = masked_multitask_loss(
                outputs,
                batch["intent_targets"],
                batch["slot_targets"],
                batch["intent_mask"],
                batch["slot_mask"],
                pos_weight,
                config.slot_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in epoch {epoch}; checkpoint not saved")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu()) * len(batch["input_ids"])
            count += len(batch["input_ids"])
        val_logits, val_spans = collect_predictions(
            model, datasets["validation"], ontology, config.batch_size, device
        )
        val_probs = expit(val_logits)
        val_y = datasets["validation"].tensors["intent_targets"].numpy()
        val_mask = datasets["validation"].tensors["intent_mask"].numpy()
        threshold = tune_threshold(val_probs, val_y, val_mask)
        val_intent = multilabel_metrics(val_y, val_probs, threshold, val_mask)
        val_slot = span_metrics(_gold_spans(datasets["validation"].records), val_spans)
        score = (val_intent["micro_f1"] + val_slot["f1"]) / 2
        row = {
            "epoch": epoch,
            "loss": total_loss / count,
            "validation_intent_micro_f1": val_intent["micro_f1"],
            "validation_slot_f1": val_slot["f1"],
            "selection_score": score,
            "threshold": threshold,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            save_network(model, tokenizer, output_dir, config, ontology)
        else:
            stale += 1
        _write_json(output_dir / "history.json", history)
        if stale >= config.patience:
            break
    duration = time.perf_counter() - start
    # Restore the validation-selected checkpoint. Everything below uses fixed weights.
    model.load_state_dict(load_file(str(output_dir / "model.safetensors")))
    val_logits, _ = collect_predictions(
        model, datasets["validation"], ontology, config.batch_size, device
    )
    cal_logits, _ = collect_predictions(
        model, datasets["calibration"], ontology, config.batch_size, device
    )
    val = datasets["validation"]
    cal = datasets["calibration"]
    threshold = tune_threshold(
        expit(val_logits), val.tensors["intent_targets"].numpy(), val.tensors["intent_mask"].numpy()
    )
    temperature = fit_temperature(
        cal_logits, cal.tensors["intent_targets"].numpy(), cal.tensors["intent_mask"].numpy()
    )
    review_cutoff = fit_support_review(
        val.tensors["intent_targets"].numpy(),
        expit(val_logits),
        threshold,
        val.tensors["intent_mask"].numpy(),
    )
    calibration = {
        "temperature": temperature,
        "raw_threshold": threshold,
        "calibrated_threshold": calibrated_threshold(threshold, temperature),
        "review_cutoff": review_cutoff,
        "review_score": "minimum raw probability distance to raw intent threshold",
        "scope": "empirical in-distribution intent-set review; empty/truncated requests reviewed; no guarantee",
        "slot_scores": "uncalibrated minimum selected token probability",
    }
    _write_json(output_dir / "calibration.json", calibration)
    manifest = (data_dir / "manifest.json").read_bytes()
    metadata = {
        "model_id": f"reflexone-support-seed{config.seed}",
        "model_sha256": hashlib.sha256((output_dir / "model.safetensors").read_bytes()).hexdigest(),
        "encoder": config.encoder_name,
        "encoder_revision": config.encoder_revision,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch,
        "selection": "mean validation intent micro-F1 and slot F1",
        "training_seconds": duration,
        "training_device": device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dataset_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "counts": {k: len(ds) for k, ds in datasets.items()},
    }
    _write_json(output_dir / "metadata.json", metadata)
    (output_dir / "ontology.json").write_bytes((data_dir / "ontology.json").read_bytes())
    (output_dir / "data_manifest.json").write_bytes(manifest)
    report = evaluate_support(output_dir, data_dir=data_dir, device=device)
    report["baseline"] = _baseline(datasets, output_dir)
    _write_json(output_dir / "evaluation.json", report)
    return report


def evaluate_support(artifact_dir: Path, data_dir: Path, device: str = "cpu") -> dict:
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    verify_prepared_data(data_dir, metadata["dataset_manifest_sha256"])
    model, tokenizer, config = load_network(artifact_dir, device)
    torch.set_num_threads(config["cpu_threads"])
    ontology = load_ontology(artifact_dir / "ontology.json")
    labels = json.loads((artifact_dir / "labels.json").read_text())
    if list(ontology.intents) != labels["intents"] or list(ontology.slot_types) != labels["slots"]:
        raise ValueError("Artifact label ordering differs from ontology")
    dataset = encode_records(
        load_records(data_dir / "test.jsonl"), tokenizer, ontology, config["max_length"]
    )
    logits, spans = collect_predictions(model, dataset, ontology, config["batch_size"], device)
    y, mask = dataset.tensors["intent_targets"].numpy(), dataset.tensors["intent_mask"].numpy()
    calibration = json.loads((artifact_dir / "calibration.json").read_text())
    raw = expit(logits)
    probs = apply_temperature(logits, calibration["temperature"])
    threshold = calibration["calibrated_threshold"]
    guessed = (probs >= threshold) & mask
    intent_exact = ((guessed == y) | ~mask).all(1)
    gold = _gold_spans(dataset.records)
    slot_exact = [
        {(s["label"], s["start"], s["end"]) for s in a}
        == {(s["label"], s["start"], s["end"]) for s in b}
        for a, b in zip(gold, spans, strict=True)
    ]
    scores = review_scores(raw, calibration["raw_threshold"], mask)
    truncated = np.array(
        [
            bool(r.text[max(e for _, e in offsets) :].strip())
            for r, offsets in zip(dataset.records, dataset.offsets, strict=True)
        ]
    )
    accepted = (scores >= calibration["review_cutoff"]) & guessed.any(axis=1) & ~truncated
    report = {
        "metadata": metadata,
        "calibration": calibration,
        "raw": multilabel_metrics(y, raw, calibration["raw_threshold"], mask),
        "calibrated": multilabel_metrics(y, probs, threshold, mask),
        "slots": span_metrics(gold, spans),
        "complete_frame_accuracy": float(np.mean(intent_exact & np.array(slot_exact))),
        "review": {
            "accepted_count": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "intent_set_risk": float(1 - intent_exact[accepted].mean()) if accepted.any() else None,
            "scope": "full deployed rule: margin + nonempty intent set + no truncation",
            "complete_frame_risk": float(1 - (intent_exact & np.array(slot_exact))[accepted].mean())
            if accepted.any()
            else None,
        },
        "selective_curve": selective_curve(y, raw, calibration["raw_threshold"], mask),
        "alignment": dataset.alignment_audit,
        "by_domain": {},
        "label_order": list(ontology.intents),
    }
    for domain in DOMAINS:
        rows = np.array([r.domain == domain for r in dataset.records])
        report["by_domain"][domain] = {
            "count": int(rows.sum()),
            "intent": multilabel_metrics(y[rows], probs[rows], threshold, mask[rows]),
            "slots": span_metrics(
                [g for g, use in zip(gold, rows, strict=True) if use],
                [s for s, use in zip(spans, rows, strict=True) if use],
            ),
        }
    errors = []
    for i, record in enumerate(dataset.records):
        if not (intent_exact[i] and slot_exact[i]):
            errors.append(
                {
                    "id": record.id,
                    "text": record.text,
                    "domain": record.domain,
                    "gold_intents": list(record.intents),
                    "predicted_intents": [
                        name for name, yes in zip(ontology.intents, guessed[i], strict=True) if yes
                    ],
                    "gold_slots": gold[i],
                    "predicted_slots": spans[i],
                    "review": not bool(accepted[i]),
                }
            )
    _write_json(artifact_dir / "errors.json", errors)
    return report
