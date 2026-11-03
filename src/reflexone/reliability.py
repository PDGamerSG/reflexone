"""Masked multilabel calibration, evaluation, and empirical selective review.

Intent probabilities are sigmoid marginals, not a joint confidence estimate.
Review scores measure distance from a decision boundary; see docs/reliability.md.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def _temperature(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be positive and finite")
    return value


def _threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    return value


def _masked_values(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask)
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype.kind != "b":
        raise ValueError("values and a boolean mask must have the same [N, C] shape")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError("unmasked entries must be finite")
    return values, mask


def _data(
    targets: np.ndarray, values: np.ndarray, mask: np.ndarray, *, probabilities: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, mask = _masked_values(values, mask)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != values.shape:
        raise ValueError("targets and values must have the same [N, C] shape")
    if not np.all((targets[mask] == 0.0) | (targets[mask] == 1.0)):
        raise ValueError("unmasked targets must be binary (0 or 1)")
    if probabilities and not np.all((values[mask] >= 0.0) & (values[mask] <= 1.0)):
        raise ValueError("unmasked probabilities must be between 0 and 1")
    # Invalid-domain annotations have no meaning and must not enter any operation.
    targets = np.where(mask, targets, 0.0).astype(bool)
    return targets, values, mask


def fit_temperature(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Fit one temperature in [0.1, 10] by masked binary NLL.

    Call only with the calibration partition. Identity is retained when the
    optimum is numerically indistinguishable from doing no calibration.
    """
    targets, logits, mask = _data(targets, logits, mask, probabilities=False)
    if not mask.any():
        raise ValueError("temperature fitting needs at least one valid entry")
    signed_logits = np.where(targets[mask], -logits[mask], logits[mask])

    def objective(log_temperature: float) -> float:
        return float(np.logaddexp(0.0, signed_logits / math.exp(log_temperature)).mean())

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.1), math.log(10.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    candidates = [1.0, 0.1, 10.0]
    if result.success and math.isfinite(result.fun):
        candidates.append(float(np.clip(math.exp(result.x), 0.1, 10.0)))
    best = min(candidates, key=lambda temperature: objective(math.log(temperature)))
    if objective(0.0) <= objective(math.log(best)) + 1e-12:
        return 1.0
    return float(best)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return stable float64 sigmoid(logits / temperature), without normalization."""
    temperature = _temperature(temperature)
    logits = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    return expit(logits / temperature)


def calibrated_threshold(raw_threshold: float, temperature: float) -> float:
    """Transform a raw threshold together with its logits to preserve decisions."""
    return float(expit(logit(_threshold(raw_threshold)) / _temperature(temperature)))


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def tune_threshold(probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Select validation micro-F1 cutoff on the 0.10..0.80 grid, step 0.01.

    Ties prefer the cutoff nearest 0.5, then the lower cutoff. Use raw validation
    probabilities and transform the selected cutoff after fitting temperature.
    """
    targets, probabilities, mask = _data(targets, probabilities, mask, probabilities=True)
    if not mask.any():
        raise ValueError("threshold tuning needs at least one valid entry")
    targets = targets[mask]
    probabilities = probabilities[mask]
    best_threshold, best_f1 = 0.5, -1.0
    candidates = sorted(np.arange(10, 81) / 100.0, key=lambda value: (abs(value - 0.5), value))
    for threshold in candidates:
        predictions = probabilities >= threshold
        true_positive = int(np.count_nonzero(predictions & targets))
        false_positive = int(np.count_nonzero(predictions & ~targets))
        false_negative = int(np.count_nonzero(~predictions & targets))
        score = _f1(true_positive, false_positive, false_negative)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold


def _average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float | None:
    support = int(targets.sum())
    if not support:
        return None
    order = np.argsort(-probabilities, kind="stable")
    scores, labels = probabilities[order], targets[order]
    cumulative_positives = np.cumsum(labels)
    # A threshold includes a complete group of tied scores, never part of one.
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    positives = cumulative_positives[ends]
    increments = np.diff(np.r_[0, positives])
    return float(np.sum(increments * (positives / (ends + 1))) / support)


def _nll(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(np.where(targets, np.log(clipped), np.log1p(-clipped))))


def _ece(targets: np.ndarray, probabilities: np.ndarray) -> float:
    # Ten equal-width bins: [0,.1), ..., [.9,1], weighted by pair count.
    bins = np.minimum((probabilities * 10).astype(np.int64), 9)
    differences = np.bincount(bins, weights=probabilities - targets, minlength=10)
    return float(np.abs(differences).sum() / len(targets))


def multilabel_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float, mask: np.ndarray
) -> dict:
    """Evaluate valid entries; return JSON values, with None for missing support.

    Macro F1 includes every label valid for at least one example. Macro AP and
    supported_label_ece include only labels with at least one positive target.
    Correctly predicted empty sets receive samples-F1 1; undefined label/micro
    F1 is 0. All-masked rows are excluded rather than counted as correct.
    """
    threshold = _threshold(threshold)
    targets, probabilities, mask = _data(targets, probabilities, mask, probabilities=True)
    predictions = np.zeros_like(mask)
    predictions[mask] = probabilities[mask] >= threshold
    valid_rows = mask.any(axis=1)
    per_label = []
    for index in range(targets.shape[1]):
        valid = mask[:, index]
        labels, scores, predicted = (
            targets[valid, index],
            probabilities[valid, index],
            predictions[valid, index],
        )
        count = int(valid.sum())
        support = int(labels.sum())
        true_positive = int(np.count_nonzero(predicted & labels))
        false_positive = int(np.count_nonzero(predicted & ~labels))
        false_negative = int(np.count_nonzero(~predicted & labels))
        per_label.append(
            {
                "index": index,
                "valid_count": count,
                "support": support,
                "predicted_positive": int(predicted.sum()),
                "f1": _f1(true_positive, false_positive, false_negative) if count else None,
                "average_precision": _average_precision(labels, scores) if count else None,
                "brier": float(np.mean((scores - labels) ** 2)) if count else None,
                "nll": _nll(labels, scores) if count else None,
                "ece": _ece(labels, scores) if count else None,
            }
        )
    metrics = {
        "n_samples": int(valid_rows.sum()),
        "n_valid_entries": int(mask.sum()),
        "n_supported_labels": sum(item["support"] > 0 for item in per_label),
        "micro_f1": None,
        "macro_f1": None,
        "samples_f1": None,
        "exact_match": None,
        "micro_average_precision": None,
        "macro_average_precision": None,
        "brier": None,
        "nll": None,
        "ece": None,
        "supported_label_ece": None,
        "per_label": per_label,
    }
    if not mask.any():
        return metrics

    true_positive = int(np.count_nonzero(predictions & targets))
    false_positive = int(np.count_nonzero(predictions & ~targets))
    false_negative = int(np.count_nonzero(~predictions & targets & mask))
    sample_true_positive = np.count_nonzero(predictions & targets, axis=1)
    sample_denominator = np.count_nonzero(predictions, axis=1) + np.count_nonzero(targets, axis=1)
    sample_f1 = np.divide(
        2.0 * sample_true_positive,
        sample_denominator,
        out=np.ones(len(targets), dtype=np.float64),
        where=sample_denominator > 0,
    )
    supported = [item for item in per_label if item["support"] > 0]
    labels, scores = targets[mask], probabilities[mask]
    metrics.update(
        micro_f1=_f1(true_positive, false_positive, false_negative),
        macro_f1=float(np.mean([item["f1"] for item in per_label if item["valid_count"]])),
        samples_f1=float(sample_f1[valid_rows].mean()),
        exact_match=float(np.all((predictions == targets) | ~mask, axis=1)[valid_rows].mean()),
        micro_average_precision=_average_precision(labels, scores),
        macro_average_precision=(
            float(np.mean([item["average_precision"] for item in supported])) if supported else None
        ),
        brier=float(np.mean((scores - labels) ** 2)),
        nll=_nll(labels, scores),
        ece=_ece(labels, scores),
        supported_label_ece=(
            float(np.mean([item["ece"] for item in supported])) if supported else None
        ),
    )
    return metrics


def review_scores(probabilities: np.ndarray, threshold: float, mask: np.ndarray) -> np.ndarray:
    """Return min valid |p-threshold| for each row; no valid label gives -1.

    This is a boundary margin, not a probability of an exact-set decision being
    correct. Fit and deploy using the same probability scale (normally raw).
    """
    threshold = _threshold(threshold)
    probabilities, mask = _masked_values(probabilities, mask)
    if not np.all((probabilities[mask] >= 0.0) & (probabilities[mask] <= 1.0)):
        raise ValueError("unmasked probabilities must be between 0 and 1")
    scores = np.full(len(probabilities), -1.0)
    if mask.shape[1]:
        valid_rows = mask.any(axis=1)
        margins = np.where(mask, np.abs(probabilities - threshold), np.inf)
        scores[valid_rows] = np.min(margins[valid_rows], axis=1)
    return scores


def _review_groups(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold = _threshold(threshold)
    targets, probabilities, mask = _data(targets, probabilities, mask, probabilities=True)
    scores = review_scores(probabilities, threshold, mask)
    valid_rows = mask.any(axis=1)
    scores = scores[valid_rows]
    errors = np.any(((probabilities >= threshold) != targets) & mask, axis=1)[valid_rows]
    if not len(scores):
        return np.array([]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    return scores[ends], ends + 1, np.cumsum(errors[order])[ends]


def selective_curve(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float, mask: np.ndarray
) -> list[dict]:
    """Report empirical exact-set risk at approximately 10%, ..., 100% coverage.

    Each point includes all scores tied at its cutoff. Coverage is over rows
    having a valid label; the exact denominator is returned in each point.
    """
    cutoffs, counts, errors = _review_groups(targets, probabilities, threshold, mask)
    if not len(counts):
        return []
    total = int(counts[-1])
    requested = (np.arange(1, 11) * total + 9) // 10
    indices = np.unique(np.searchsorted(counts, requested))
    return [
        {
            "score_threshold": float(cutoffs[index]),
            "accepted": int(counts[index]),
            "eligible": total,
            "coverage": float(counts[index] / total),
            "errors": int(errors[index]),
            "risk": float(errors[index] / counts[index]),
        }
        for index in indices
    ]


def fit_review_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    target_risk: float = 0.1,
    min_accepted: int = 20,
) -> float:
    """Maximize validation coverage subject to empirical exact-set error rate.

    Accept rows with score >= returned cutoff, or review when score < cutoff.
    With no qualifying group, 1.0 reviews all rows. Reusing validation data for
    selection makes this an empirical heuristic, not a finite-sample guarantee.
    """
    if not math.isfinite(target_risk) or not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be between 0 and 1")
    if isinstance(min_accepted, bool) or int(min_accepted) != min_accepted or min_accepted < 1:
        raise ValueError("min_accepted must be a positive integer")
    cutoffs, counts, errors = _review_groups(targets, probabilities, threshold, mask)
    eligible = np.flatnonzero((counts >= min_accepted) & (errors / counts <= target_risk))
    return float(cutoffs[eligible[-1]]) if len(eligible) else 1.0
