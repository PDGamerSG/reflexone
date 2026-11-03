# Locally trained command and spam profiles

ReflexOne includes two trained classical classifiers alongside its neural support model. They demonstrate where a supervised lexical model gives fast, inspectable decisions. Neither profile calls an external model, generates text, or executes its predicted action.

## Data and frozen partitions

**Commands:** [CLINC150 full](https://github.com/clinc/oos-eval), revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`, contains 150 supported intents plus separately labeled out-of-scope (OOS) requests. The [dataset license](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE) is CC BY 3.0. Cite [Larson et al. (2019)](https://aclanthology.org/D19-1131/).

The training examples come exclusively from official `train`. Official `val` is split per label into 50% model selection, 25% temperature calibration, and 25% in-scope rejection validation. Official `oos_val` is used only to select the rejection threshold. Official `test` and `oos_test` form the final test partition. The 100 official `oos_train` examples are deliberately unused.

We normalize grouping keys using NFKC, Unicode casefold, and collapsed whitespace, while preserving the original input text. Duplicate priority is **test, then OOS validation, then official validation, then training**. A repeated evaluation message therefore cannot enter the training vocabulary or classifier fitting. Any group with conflicting class labels is quarantined everywhere. This removes four conflicting groups/eight rows and one additional cross-split duplicate. The resulting test is a documented decontaminated variant of the official split, not a directly interchangeable leaderboard result.

**Spam:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). UCI's current dataset page explicitly licenses the data under CC BY 4.0. The downloaded collection has 5,574 rows. Collapsing 415 same-label normalized duplicates leaves 5,159 independent text groups, with no conflicting groups. There is no official chronological partition. We freeze a custom per-class stratified 65/15/10/10 split. **`ham=0`, `spam=1`; all spam precision, recall and average-precision metrics treat spam as positive.**

| Profile | Training | Selection | Calibration | ID rejection validation | OOS validation | Final test |
|---|---:|---:|---:|---:|---:|---:|
| Commands | 14,995 | 1,498 | 750 | 750 | 100 | 5,498: 4,498 ID + 1,000 OOS |
| Spam | 3,354 | 773 | 516 | — | — | 516: 452 ham + 64 spam |

Split seed is 17; model seed is 42. These are different controls: changing the model seed does not alter the prepared partitions. All partitions have disjoint normalized-text groups. Each prepared record retains its source ID and source partition. `data/<task>/manifest.json` records label supports, raw-data hash, partition hashes, source revision, license and audit counts. `audit/quarantine.json` records conflicting examples. Training refuses a partition whose content differs from its manifest hash.

Raw input hashes for the downloaded files:

```text
CLINC150 JSON  36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0
UCI SMS ZIP    1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3
```

The UCI download is unversioned, so its hash identifies the exact bytes used here. Raw files are cached locally; a manually supplied replacement is recorded with its own content hash and must not be described as the original experiment.

## Algorithms and calibration

Both algorithms fit a TF-IDF vocabulary on training text only. Features are word unigrams and bigrams, with sublinear term frequency, Unicode accent stripping, and at most 60,000 features.

- **Logistic regression:** discriminative linear class scores with L2 regularization (`C=8`, `lbfgs`, maximum 1,000 iterations). These hyperparameters were fixed before evaluation.
- **Complement Naive Bayes:** class-complement feature counts with smoothing `alpha=0.1`. It is a fast baseline well suited to sparse text features and uneven class counts.

Model selection uses command macro-F1 or spam-class F1 on the selection partition. No model is refit on selection examples. After selection is frozen, a positive scalar temperature is fitted on a separate calibration partition by minimizing multiclass negative log likelihood. The implementation scales clipped log probabilities by `1/T` and renormalizes with softmax; `T` is bounded to `[exp(-3), exp(3)]`. The calibration method follows [Guo et al. (2017)](https://arxiv.org/abs/1706.04599). It preserves the highest-scoring class, so any accuracy change must come from a separate decision rule, not temperature scaling itself.

For commands, a threshold on the calibrated maximum class probability maximizes the equally weighted mean of in-scope acceptance and OOS rejection on **rejection validation only**. Supported-class probabilities are conditional on the request belonging to the supported classes; maximum probability is **not** an estimate of `P(in-scope)`. Low-confidence requests receive `label="oos"`, `abstain=true`, and a retained `candidate_label`. This is an empirically tested OOS rejection rule, not a guarantee that a request is unsupported.

Spam uses the calibrated argmax, equivalent to a 0.5 spam-probability threshold. No OOS or review rule is claimed for spam. The two baseline algorithms are also evaluated on test after all selection decisions are frozen. Their baseline test metrics are uncalibrated; command baseline comparisons cover supported intents only. The selected model has a separate calibration ablation and, for commands, a full OOS evaluation.

## Measured results

The saved reports are [commands.json](../reports/commands.json) and [spam.json](../reports/spam.json). They include per-label support, precision/recall/F1, NLL, multiclass Brier score, ten-bin top-confidence ECE, reliability bins, a coverage curve, dependencies, timing scope and artifact hashes.

| Profile / algorithm | Selection accuracy | Selection criterion | Final supported-class accuracy |
|---|---:|---:|---:|
| Commands / logistic regression **selected** | 90.45% | macro-F1 90.41% | 90.86% |
| Commands / ComplementNB | 82.24% | macro-F1 81.72% | 83.35% |
| Spam / logistic regression | 98.71% | spam F1 94.57% | 98.45% |
| Spam / ComplementNB **selected** | 98.71% | spam F1 94.85% | 98.64% |

The selected command model has 77.86% overall accuracy across 150 intent labels plus OOS, 92.30% OOS recall, 76.01% in-scope acceptance, and OOS-detection AUROC 0.9128. Its accepted predictions have 96.05% accuracy at 63.59% coverage of the complete final test. The validation-selected threshold is 0.96016. The in-scope class accuracy before applying rejection is 90.86%. Reporting only accepted accuracy would hide the coverage cost.

The selected spam model has 98.64% accuracy, 94.57% spam F1, 93.85% spam precision, and 95.31% spam recall. Its final confusion matrix is `[[448, 4], [3, 61]]`, with rows/columns ordered `ham, spam`. The small 64-example spam holdout limits precision of these estimates.

| Calibration ablation on held-out test | Raw NLL → calibrated NLL | Raw ECE → calibrated ECE |
|---|---:|---:|
| Commands, supported intents | 0.5742 → 0.3785 | 0.1674 → 0.0211 |
| Spam | 0.0671 → 0.0547 | 0.0293 → 0.0142 |

The temperatures are 0.52258 for commands and 0.75082 for spam. These improvements are observed on these particular held-out examples; calibration can worsen on another sample or under distribution shift. The saved coverage curves are descriptive test summaries, not thresholds selected using test labels.

Warm batch-one CPU benchmarks used 100 predictions on this Apple arm64 Mac. Latest measured command p50/p95 was 0.271/0.373 ms; spam was 0.366/0.397 ms. Artifact loading took approximately 199 ms/93 ms respectively. Each timed prediction includes TF-IDF vectorization, classification, temperature scaling and output-dictionary construction. It excludes loading, HTTP, network and JSON serialization. These local measurements are not service-level guarantees or comparisons with Jev. Exact environment/thread settings appear in the reports; concurrent machine load may change timing.

## Reproduce and inspect

After `uv sync --extra dev`, run the module APIs from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
uv run python - <<'PY'
import json
from pathlib import Path
from reflexone.tasks import prepare_task, train_task, benchmark_task

for task in ("commands", "spam"):
    prepare_task(task, Path("data") / task, seed=17)
    report = train_task(task, Path("data") / task, Path("artifacts") / task, seed=42)
    report["benchmark"] = benchmark_task(Path("artifacts") / task, repeats=100)
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / f"{task}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
PY
```

An artifact directory contains a compressed `model.joblib`, `metadata.json`, `report.json`, and copied `evaluation.jsonl`. Weights and data remain outside Git. The command artifact is about 33 MB; spam is under 1 MB. Joblib can execute code when loading: use trusted local artifacts. Independent evaluation uses the copied test examples and never fits a model; it also reports their current hash so altered evaluation data is visible.

```python
from pathlib import Path
from reflexone.tasks import TaskPredictor, evaluate_task

router = TaskPredictor(Path("artifacts/commands"))  # Load once.
print(router.predict("turn the bedroom lights on"))
print(router.predict_batch(["play some music", "explain quantum gravity"]))
print(evaluate_task(Path("artifacts/spam"))["test"])
```

`predict_task` is the one-shot convenience function; `TaskPredictor` avoids repeated model loading in an API. Empty or over-20,000-character input is rejected. Tests use tiny local fixtures and cover duplicate isolation, deterministic splits, conflicting labels, explicit spam direction, artifact round-trip, calibration, empirical OOS rejection, re-evaluation and latency return structure.

## Practical limits

CLINC is an English, single-intent, crowdsourced benchmark with a fixed label set. Its OOS examples do not establish performance on adversarial requests, overlapping intents or every future domain. Only 100 OOS validation examples set the threshold. SMS messages are older English examples from mixed sources; this split is not a contemporary, chronological deployment evaluation. Neither profile reasons generally or learns new classes during inference. A fixed schema ensures valid labels and numbers, while semantic predictions can still be wrong.
