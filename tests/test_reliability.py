"""Numerical contracts for masked multilabel calibration and selective review."""

import json
import math

import numpy as np
import pytest


def test_temperature_recovers_known_binary_frequency_without_masked_labels():
    from reflexone import reliability

    # logit(0.9) / 2 = logit(0.75), and the observed rate is exactly 0.75.
    logits = np.full((400, 2), math.log(9.0))
    logits[:, 1] = 1000.0
    targets = np.zeros((400, 2))
    targets[:300, 0] = 1.0
    mask = np.zeros_like(targets, dtype=bool)
    mask[:, 0] = True

    temperature = reliability.fit_temperature(logits, targets, mask)

    assert temperature == pytest.approx(2.0, abs=0.001)
    probabilities = reliability.apply_temperature(logits, temperature)
    assert probabilities[0, 0] == pytest.approx(0.75, abs=0.0001)


def test_temperature_is_bounded_for_overconfident_wrong_predictions():
    from reflexone import reliability

    temperature = reliability.fit_temperature(
        np.array([[-8.0, 8.0]]),
        np.array([[1.0, 0.0]]),
        np.ones((1, 2), dtype=bool),
    )

    assert temperature == pytest.approx(10.0, abs=0.001)


def test_temperature_leaves_uninformative_logits_at_identity():
    from reflexone import reliability

    assert (
        reliability.fit_temperature(
            np.zeros((2, 1)), np.array([[0.0], [1.0]]), np.ones((2, 1), dtype=bool)
        )
        == 1.0
    )


def test_temperature_ignores_nonfinite_values_outside_domain_mask():
    from reflexone import reliability

    temperature = reliability.fit_temperature(
        np.array([[0.0, np.nan], [0.0, np.inf]]),
        np.array([[0.0, np.nan], [1.0, -99.0]]),
        np.array([[True, False], [True, False]]),
    )

    assert temperature == 1.0


def test_stable_sigmoid_handles_large_magnitudes_without_overflow():
    from reflexone import reliability

    with np.errstate(over="raise", invalid="raise"):
        probabilities = reliability.apply_temperature(np.array([-10000.0, 0.0, 10000.0]), 1.0)

    np.testing.assert_array_equal(probabilities, [0.0, 0.5, 1.0])


def test_transformed_threshold_preserves_labels_after_temperature_scaling():
    from reflexone import reliability

    logits = np.array([[-4.0, -1.0, 0.0, 0.5, 2.0, 6.0]])
    probabilities = reliability.apply_temperature(logits, 1.0)
    for raw_threshold in (0.1, 0.35, 0.5, 0.8):
        for temperature in (0.1, 0.5, 2.0, 10.0):
            calibrated = reliability.apply_temperature(logits, temperature)
            cutoff = reliability.calibrated_threshold(raw_threshold, temperature)
            np.testing.assert_array_equal(calibrated >= cutoff, probabilities >= raw_threshold)


def test_validation_threshold_optimizes_micro_f1_ignoring_masked_labels():
    from reflexone import reliability

    threshold = reliability.tune_threshold(
        np.array([[0.25, 0.99], [0.25, 0.99], [0.15, 0.99], [0.15, 0.99]]),
        np.array([[1, 0], [1, 0], [0, 0], [0, 0]]),
        np.array([[True, False]] * 4),
    )

    assert 0.15 < threshold <= 0.25


def test_validation_threshold_uses_identity_threshold_when_all_choices_tie():
    from reflexone import reliability

    assert (
        reliability.tune_threshold(np.zeros((3, 1)), np.zeros((3, 1)), np.ones((3, 1), dtype=bool))
        == 0.5
    )


def test_multilabel_metrics_match_hand_checked_masked_example():
    from reflexone import reliability

    targets = np.array([[1, 0], [0, 0], [0, 1]])
    probabilities = np.array([[0.8, 0.6], [0.2, 0.3], [0.4, 0.7]])
    mask = np.array([[True, True], [True, False], [True, True]])

    metrics = reliability.multilabel_metrics(targets, probabilities, 0.5, mask)

    assert metrics["micro_f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx(5 / 6)
    assert metrics["samples_f1"] == pytest.approx(8 / 9)
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert metrics["micro_average_precision"] == 1.0
    assert metrics["macro_average_precision"] == 1.0
    assert metrics["brier"] == pytest.approx(0.138)
    assert metrics["nll"] == pytest.approx(
        -(math.log(0.8) * 2 + math.log(0.4) + math.log(0.6) + math.log(0.7)) / 5
    )
    assert metrics["ece"] == pytest.approx(0.34)
    assert metrics["supported_label_ece"] == pytest.approx(0.35833333333333334)
    assert metrics["per_label"][0]["support"] == 1
    assert metrics["per_label"][0]["valid_count"] == 3
    assert metrics["per_label"][1]["valid_count"] == 2
    assert metrics["n_samples"] == 3
    assert metrics["n_valid_entries"] == 5
    json.dumps(metrics, allow_nan=False)


def test_average_precision_respects_tied_scores_and_excludes_unsupported_labels():
    from reflexone import reliability

    metrics = reliability.multilabel_metrics(
        np.array([[1, 0, 1], [0, 0, 0]]),
        np.array([[0.8, 0.9, 0.1], [0.8, 0.9, 0.9]]),
        0.5,
        np.array([[True, True, False], [True, True, False]]),
    )

    assert metrics["micro_average_precision"] == 0.25
    assert metrics["macro_average_precision"] == 0.5
    assert metrics["per_label"][0]["average_precision"] == 0.5
    assert metrics["per_label"][1]["average_precision"] is None
    assert metrics["per_label"][2]["average_precision"] is None
    assert metrics["per_label"][2]["valid_count"] == 0
    assert metrics["macro_f1"] == pytest.approx(1 / 3)


def test_empty_intent_sets_count_as_exact_matches_without_calibration_claims():
    from reflexone import reliability

    metrics = reliability.multilabel_metrics(
        np.zeros((2, 2)), np.full((2, 2), 0.1), 0.5, np.ones((2, 2), dtype=bool)
    )

    assert metrics["micro_f1"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["samples_f1"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["micro_average_precision"] is None
    assert metrics["macro_average_precision"] is None
    assert metrics["supported_label_ece"] is None
    assert metrics["ece"] == pytest.approx(0.1)
    json.dumps(metrics, allow_nan=False)


def test_empty_masks_return_missing_metrics_and_never_accept_rows():
    from reflexone import reliability

    targets = np.array([[1, 0]])
    probabilities = np.array([[np.nan, np.nan]])
    mask = np.zeros((1, 2), dtype=bool)

    metrics = reliability.multilabel_metrics(targets, probabilities, 0.5, mask)

    assert metrics["n_samples"] == 0
    assert metrics["micro_f1"] is None
    assert metrics["exact_match"] is None
    np.testing.assert_array_equal(reliability.review_scores(probabilities, 0.5, mask), [-1.0])
    assert reliability.selective_curve(targets, probabilities, 0.5, mask) == []
    assert reliability.fit_review_threshold(targets, probabilities, 0.5, mask) == 1.0
    json.dumps(metrics, allow_nan=False)


def test_probability_endpoints_have_finite_nll_and_correct_ece_bins():
    from reflexone import reliability

    metrics = reliability.multilabel_metrics(
        np.array([[1, 0]]), np.array([[0.0, 1.0]]), 0.5, np.ones((1, 2), dtype=bool)
    )

    assert math.isfinite(metrics["nll"])
    assert metrics["nll"] > 30.0
    assert metrics["brier"] == 1.0
    assert metrics["ece"] == 1.0


def test_ece_compares_bin_frequency_instead_of_individual_absolute_errors():
    from reflexone import reliability

    metrics = reliability.multilabel_metrics(
        np.array([[1], [0]]), np.array([[0.5], [0.5]]), 0.5, np.ones((2, 1), dtype=bool)
    )

    assert metrics["ece"] == 0.0
    assert metrics["supported_label_ece"] == 0.0
    assert metrics["brier"] == 0.25
    assert metrics["nll"] == pytest.approx(math.log(2.0))


def test_review_score_is_weakest_valid_decision_margin():
    from reflexone import reliability

    probabilities = np.array([[0.9, 0.51, 0.5], [0.1, 0.8, 0.5], [0.5, 0.5, 0.5]])
    mask = np.array([[True, True, False], [True, True, False], [False, False, False]])

    scores = reliability.review_scores(probabilities, 0.5, mask)

    np.testing.assert_allclose(scores, [0.01, 0.3, -1.0])


def test_selective_curve_uses_exact_set_risk_and_accepts_ties_together():
    from reflexone import reliability

    targets = np.array([[1, 0], [1, 0], [0, 0], [1, 0], [1, 1]])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.8, 0.2], [0.55, 0.45], [0.9, 0.1]])
    mask = np.array([[True, True]] * 4 + [[False, False]])

    curve = reliability.selective_curve(targets, probabilities, 0.5, mask)

    assert curve
    first = curve[0]
    assert first["accepted"] == 1
    assert first["coverage"] == 0.25
    assert first["risk"] == 0.0
    last = curve[-1]
    assert last["accepted"] == 4
    assert last["coverage"] == 1.0
    assert last["risk"] == 0.25
    assert all(point["accepted"] != 2 for point in curve)
    assert all(a["coverage"] < b["coverage"] for a, b in zip(curve, curve[1:]))
    for point in curve:
        scores = reliability.review_scores(probabilities, 0.5, mask)
        assert np.count_nonzero(scores >= point["score_threshold"]) == point["accepted"]
    json.dumps(curve, allow_nan=False)


def test_review_rule_maximizes_coverage_even_when_risk_is_nonmonotone():
    from reflexone import reliability

    targets = np.array([[0], [1], [1], [1], [0]])
    probabilities = np.array([[0.9], [0.8], [0.7], [0.6], [0.55]])
    mask = np.ones((5, 1), dtype=bool)

    cutoff = reliability.fit_review_threshold(
        targets, probabilities, 0.5, mask, target_risk=0.25, min_accepted=1
    )

    assert cutoff == pytest.approx(0.1)
    assert np.count_nonzero(reliability.review_scores(probabilities, 0.5, mask) >= cutoff) == 4


def test_review_rule_does_not_split_ties_to_satisfy_empirical_risk():
    from reflexone import reliability

    cutoff = reliability.fit_review_threshold(
        np.array([[1], [0]]),
        np.array([[0.9], [0.9]]),
        0.5,
        np.ones((2, 1), dtype=bool),
        target_risk=0.0,
        min_accepted=1,
    )

    assert cutoff == 1.0


def test_review_rule_requires_minimum_validation_support():
    from reflexone import reliability

    cutoff = reliability.fit_review_threshold(
        np.ones((5, 1)), np.full((5, 1), 0.9), 0.5, np.ones((5, 1), dtype=bool)
    )

    assert cutoff == 1.0


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    from reflexone import reliability

    with pytest.raises(ValueError):
        reliability.apply_temperature(np.array([0.0]), temperature)
    with pytest.raises(ValueError):
        reliability.calibrated_threshold(0.5, temperature)


def test_invalid_valid_entries_and_shapes_are_rejected():
    from reflexone import reliability

    with pytest.raises(ValueError):
        reliability.fit_temperature(np.array([[1.0]]), np.array([[0]]), np.array([[False]]))
    with pytest.raises(ValueError):
        reliability.multilabel_metrics(np.array([[1]]), np.array([[1.1]]), 0.5, np.array([[True]]))
    with pytest.raises(ValueError):
        reliability.multilabel_metrics(
            np.array([[0.5]]), np.array([[0.5]]), 0.5, np.array([[True]])
        )
    with pytest.raises(ValueError):
        reliability.review_scores(np.array([[0.5]]), 0.5, np.array([True]))
    with pytest.raises(ValueError):
        reliability.calibrated_threshold(0.0, 1.0)
