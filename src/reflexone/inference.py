"""Local CPU prediction, validated structured output, and scoped latency measurements."""

from __future__ import annotations

import json
import math
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from reflexone.data import load_ontology
from reflexone.model import DOMAINS, decode_slot_spans
from reflexone.reliability import apply_temperature, calibrated_threshold, review_scores
from reflexone.schema import IntentPrediction, PredictRequest, SupportDecision
from reflexone.training import load_network


class SupportPredictor:
    def __init__(self, artifact_dir: Path) -> None:
        start = time.perf_counter()
        self.artifact_dir = Path(artifact_dir)
        self.model, self.tokenizer, self.config = load_network(self.artifact_dir, "cpu")
        self.ontology = load_ontology(self.artifact_dir / "ontology.json")
        labels = json.loads((self.artifact_dir / "labels.json").read_text())
        if labels != {
            "intents": list(self.ontology.intents),
            "slots": list(self.ontology.slot_types),
        }:
            raise ValueError("Model and ontology label ordering differ")
        torch.set_num_threads(self.config["cpu_threads"])
        self.calibration = json.loads((self.artifact_dir / "calibration.json").read_text())
        expected_threshold = calibrated_threshold(
            self.calibration["raw_threshold"], self.calibration["temperature"]
        )
        if not math.isclose(
            expected_threshold, self.calibration["calibrated_threshold"], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                "Saved calibration threshold is inconsistent with temperature and raw threshold"
            )
        cutoff = self.calibration["review_cutoff"]
        if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("Review cutoff must be a finite score between zero and one")
        self.metadata = json.loads((self.artifact_dir / "metadata.json").read_text())
        self._lock = threading.Lock()
        self.load_seconds = time.perf_counter() - start

    def predict(self, text: str, domain: str) -> SupportDecision:
        request = PredictRequest(profile="support", text=text, domain=domain)
        with self._lock:
            tokens = self.tokenizer(
                request.text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config["max_length"],
                return_offsets_mapping=True,
            )
            offsets = tokens.pop("offset_mapping")[0].tolist()
            visible_end = max(end for _, end in offsets)
            truncated = bool(text[visible_end:].strip())
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    domains=torch.tensor([DOMAINS[domain]]),
                )
            logits = outputs["intent_logits"].numpy()
            raw = expit(logits)
            probs = apply_temperature(logits, self.calibration["temperature"])[0]
            mask = np.array(self.ontology.intent_mask(domain), dtype=bool)
            selected = probs >= self.calibration["calibrated_threshold"]
            predictions = {
                name: IntentPrediction(value=bool(selected[i]), probability=float(probs[i]))
                for i, name in enumerate(self.ontology.intents)
                if mask[i]
            }
            decoded = decode_slot_spans(
                text,
                offsets,
                outputs["slot_logits"][0].softmax(-1).numpy(),
                self.ontology.slot_types,
                self.ontology.slot_mask(domain),
            )
            spans = [
                {
                    **{k: v for k, v in span.items() if k != "probability"},
                    "token_score": span["probability"],
                }
                for span in decoded
            ]
            score = float(review_scores(raw, self.calibration["raw_threshold"], mask[None])[0])
            reasons = []
            if score < self.calibration["review_cutoff"]:
                reasons.append("low_intent_margin")
            if not any(p.value for p in predictions.values()):
                reasons.append("no_intent_selected")
            if truncated:
                reasons.append("input_truncated")
            return SupportDecision(
                model_id=self.metadata["model_id"],
                domain=domain,
                text=text,
                intents=predictions,
                slots=spans,
                review_required=bool(reasons),
                review_reasons=reasons,
                truncated=truncated,
            )


BENCHMARK_CASES = [
    ("My card was stolen last night. Please block it.", "banking"),
    ("Why is my transfer still pending?", "banking"),
    ("How much did I spend on Amazon this month?", "banking"),
    ("Can I book two rooms for three adults next Friday?", "hotels"),
    ("I cannot connect to the hotel wifi.", "hotels"),
    ("What time does the swimming pool close?", "hotels"),
]


def benchmark_support(artifact_dir: Path, repeats: int = 120) -> dict:
    if repeats < 10:
        raise ValueError("Use at least 10 timed repetitions")
    predictor = SupportPredictor(artifact_dir)
    for index in range(12):
        predictor.predict(*BENCHMARK_CASES[index % len(BENCHMARK_CASES)])
    timings = []
    for index in range(repeats):
        text, domain = BENCHMARK_CASES[index % len(BENCHMARK_CASES)]
        start = time.perf_counter_ns()
        predictor.predict(text, domain)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    # This second measurement excludes tokenization, decoding and schema validation.
    tokens = predictor.tokenizer(BENCHMARK_CASES[0][0], return_tensors="pt")
    model_times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter_ns()
            predictor.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                domains=torch.tensor([0]),
            )
            model_times.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "scope": "warmed batch-one CPU prediction: tokenization + encoder/heads + decode + validation",
        "excludes": "process startup, artifact loading, HTTP transport, external service latency",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cpu_threads": predictor.config["cpu_threads"],
        "repeats": repeats,
        "warmup": 12,
        "case_count": len(BENCHMARK_CASES),
        "load_seconds": predictor.load_seconds,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "mean_ms": float(np.mean(timings)),
        "sequential_requests_per_second": 1000 / float(np.mean(timings)),
        "model_only_single_case_p50_ms": float(np.percentile(model_times, 50)),
        "cases": [{"text": t, "domain": d} for t, d in BENCHMARK_CASES],
    }
