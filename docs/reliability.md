# Calibration, metrics, and review

ReflexOne predicts multiple intent modules at once. Each sigmoid output is a marginal
probability that one module applies. The outputs do not sum to one, and their product
is not a justified probability that the complete intent set is correct. Slot predictions
use separate token heads and are evaluated separately; the functions in
`src/reflexone/reliability.py` evaluate intent modules only.

## Split responsibilities

1. Train model weights on the training partition.
2. Select the checkpoint and raw intent threshold on validation data. The threshold
   search maximizes masked micro F1 over 0.10 through 0.80, inclusive, in steps of 0.01.
   Ties prefer the threshold nearest 0.50, then the lower value.
3. Fit one temperature on the separate calibration partition by minimizing binary
   negative log likelihood over valid entries. The bounded optimizer searches
   temperatures from 0.1 through 10, evaluates the endpoints and identity too, and
   retains identity when an improvement is numerically negligible.
4. Transform the selected threshold along with the logits:

   ```text
   calibrated_probability = sigmoid(logit / temperature)
   calibrated_threshold = sigmoid(logit(raw_threshold) / temperature)
   ```

   This monotone transformation preserves intent decisions in real arithmetic.
   Finite precision can produce ties for saturated probabilities; output probabilities
   and labels should use the same saved temperature and threshold.
5. Fit the review cutoff from raw validation probabilities and the raw threshold.
   Apply that same score definition and cutoff at inference. A transformed probability
   distance can change the ranking, even though the intent decisions stay the same.
6. Report frozen-model results on untouched test data. Temperature fitting can improve
   calibration-set NLL while worsening test NLL or other test metrics. Report either
   outcome, including uncalibrated and calibrated results.

Validation is reused for checkpoint, decision-threshold, and review-cutoff selection.
Its selected results are optimistic estimates. The independent test partition measures
the final combined procedure. Calibration data is not used to select the model.

## Array contract and masking

Functions that combine predictions and targets take arrays of shape `[N, C]` and a
boolean mask of exactly that shape. A true entry denotes a label valid for that row's
known domain. Valid targets must be binary; valid probabilities must lie in `[0, 1]`.
Masked entries are ignored, including nonfinite or otherwise irrelevant placeholder
values. Input arrays are not modified. `apply_temperature` accepts any array shape
of finite logits and returns float64 probabilities using SciPy's stable sigmoid.

Rows with no valid labels are excluded from evaluation and review-curve denominators.
Their review score is `-1`, so they are reviewed for every cutoff fitted by this module.
An entirely empty mask produces `null` aggregate metrics and an empty curve; it does
not produce apparent perfect performance. Fitting a temperature or decision threshold
without any valid entries raises `ValueError`.

## Reported metrics

Decisions use `probability >= threshold`. Returned dictionaries contain ordinary JSON
numbers and `null` for undefined quantities; they do not contain NaN or infinity.

| Metric | Definition and support policy |
| --- | --- |
| `micro_f1` | Pool true positives, false positives, and false negatives across valid pairs. Return 0 when its denominator is zero. |
| `macro_f1` | Mean label F1 across labels valid on at least one row. A valid label with no true or predicted positives contributes 0. |
| `samples_f1` | Mean row F1 over rows with valid labels. A correct empty intent set contributes 1; an incorrect empty set contributes 0. This explicit convention differs from a zero-division setting of 0. |
| `exact_match` | Fraction of eligible rows whose complete set of valid intent modules is correct, including correct empty sets. |
| `micro_average_precision` | Average precision over all valid label–row pairs. Complete tied-score groups enter together. `null` if no positive target exists. |
| `macro_average_precision` | Mean label average precision over labels with at least one positive target. Labels without positive support are excluded and explicitly reported. |
| `brier` | Mean squared error of each valid Bernoulli probability, without summing over labels per example. |
| `nll` | Mean binary negative log likelihood over valid pairs. For reporting only, probabilities are clipped to `[1e-15, 1 − 1e-15]` to keep endpoint scores finite. Temperature fitting instead uses stable logit-space NLL. |
| `ece` | Pooled binary expected calibration error over valid pairs, with ten equal-width bins: `[0, .1)`, …, `[.9, 1]`. Each bin compares its mean probability with its observed positive frequency and is weighted by its pair count. |
| `supported_label_ece` | Macro mean of label-specific ECEs for labels with at least one observed positive. Every such label's ECE uses **all** its valid rows, including negative rows. This is not ECE conditioned on positive outcomes. |

Pooled ECE can hide weak rare-label performance when most pairs are negative. The
supported-label statistic and per-label reports make that limitation more visible,
but rare-label estimates remain noisy. ECE depends on binning and sample size; use
it alongside NLL and Brier score rather than as the sole calibration measure.

The report also includes `n_samples`, `n_valid_entries`, `n_supported_labels`, and a
`per_label` list in model output order. Each entry includes its `index`, `valid_count`,
positive `support`, `predicted_positive` count, F1, average precision, Brier score,
NLL, and ECE. The caller may attach ontology names to those indices. A wholly invalid
label has counts of zero and `null` metrics.

## Empirical review rule

For an example with valid intent labels, the score is:

```text
score = min(abs(raw_probability[label] − raw_threshold)) over valid labels
```

A small score means at least one intent decision is close to its boundary. It is a
ranking heuristic: a large score is not a calibrated probability of correctness or
proof that all decisions are certain. Errors can be far from a decision boundary.
The score does not assess slot correctness, unseen domains, or distribution shift.

`fit_review_threshold(targets, probabilities, threshold, mask, target_risk=0.1,
min_accepted=20)` checks all observed score cutoffs on validation data. It chooses
the largest accepted count whose empirical complete-intent-set error rate is no more
than `target_risk` and whose accepted count is at least `min_accepted`. Every tied
score is accepted or reviewed together. The scan permits nonmonotone error curves;
it does not stop at the first failed cutoff.

The rule accepts when `score >= cutoff` and reviews when `score < cutoff`. If no
cutoff qualifies, the function returns `1.0`, which reviews every row because valid
scores are strictly below 1 for the supported decision thresholds. A fitted cutoff
of `0.0` accepts all eligible rows. Always persist the cutoff and the raw intent
threshold together.

`selective_curve` reports up to ten points at requested coverages of 10%, 20%, …,
100%. Each requested count rounds up to a complete tied-score group, and duplicate
groups are returned once. A point contains:

- `score_threshold`: the minimum score accepted at that point.
- `accepted` and `eligible`: accepted and total evaluable row counts.
- `coverage`: `accepted / eligible`.
- `errors`: accepted rows with at least one wrong valid intent label.
- `risk`: `errors / accepted`.

The curve therefore measures **complete intent-set error**, not per-label error or
complete-frame error including slots. Report the untouched test curve and performance
at the frozen validation-selected cutoff. No accepted examples means no measurable
accepted-set risk, rather than a demonstrated risk of zero.

The empirical target has no confidence bound, finite-sample guarantee, conformal
guarantee, or guarantee under distribution shift. A target of 10% does not establish
that deployed risk is below 10%. Support-profile review is not supervised OOD
detection. The command-routing profile's labeled out-of-scope evaluation is a
separate experiment.

## Research basis

- [Guo et al., 2017, *On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599)
  motivates temperature scaling. The multilabel implementation here uses binary
  likelihood over sigmoid marginals rather than a mutually exclusive softmax.
- [Ovadia et al., 2019, *Can You Trust Your Model's Uncertainty?*](https://arxiv.org/abs/1906.02530)
  shows why post-hoc calibration should not be treated as protection against shift.
- [TypeSafe's confidence documentation](https://docs.typesafe.ai/confidence.md)
  distinguishes a probability distribution from a statistic summarizing it. ReflexOne
  uses an empirically evaluated boundary margin and does not claim to reproduce Jev's
  architecture, training method, or proprietary RLCD.
