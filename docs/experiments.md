# Experiments and measured results

All numbers below come from saved local training runs, not estimates or vendor claims. The primary neural seed was fixed at 42 before running the additional seeds. Additional runs are reported together; their test scores did not select the deployed checkpoint. Each run selects its epoch on validation data.

## Protocol

| Profile | Training and model selection | Calibration / rejection | Final evaluation |
|---|---|---|---|
| Support | Custom normalized-text-grouped NLU++ holdout, 1,968 train / 455 validation | 302 separate calibration; review fitted on validation | 302 frozen test requests |
| Commands | Official supported-intent train; half of official validation selects LR vs NB | Separate quarters of official validation for temperature and ID rejection; official OOS validation for rejection | 4,498 supported + 1,000 OOS, after duplicate audit |
| Spam | Deduplicated grouped stratified 65/15/10/10 split; validation selects LR vs NB | Separate calibration partition | 516 messages: 452 ham, 64 spam |

Normalized duplicate text never crosses partitions. Conflicting duplicate annotations are quarantined. Exact normalized text control does not remove semantic paraphrases. Manifests, hashes and per-label supports are available in prepared data and model metadata. The support protocol is custom and cannot be directly compared with NLU++ paper tables.

## Support: neural model versus a meaningful baseline

| Metric | TF-IDF + OVR logistic regression | Fine-tuned MiniLM, primary seed42 |
|---|---:|---:|
| Intent micro-F1 | 0.7618 | 0.8686 |
| Intent macro-F1 | 0.7001 | 0.8517 |
| Exact intent-set match | 0.4139 | 0.6490 |
| Calibrated binary NLL | 0.07370 | 0.03671 |
| Exact entity-span F1 | Not implemented in the baseline | 0.6145 |
| Complete-frame accuracy | Not comparable: baseline has no extractor | 0.4669 |

The neural intent gain is 10.69 percentage points of micro-F1 on this split. It is not evidence of a universal gain or SOTA. Banking test intent micro-F1 is 0.8884; hotels is 0.8094. Banking/hotel exact-span F1 is 0.6404 / 0.5936.

Extraction is the weakest measured component. Primary exact-span precision is approximately 0.518 and recall 0.755; the network produces extra spans as well as missing or mistyping boundaries. The complete-frame score requires every intent and every entity to be correct and is much lower than intent micro-F1. Inspect `artifacts/support/errors.json` for all test failures. These examples are for analysis, not further fitting on the test set.

### Seed variation

| Seed | Validation-selected epoch | Intent micro-F1 | Exact-span F1 | Complete-frame accuracy |
|---|---:|---:|---:|---:|
| 42, primary | 14 | 0.8686 | 0.6145 | 0.4669 |
| 7 | 13 | 0.8488 | 0.5957 | 0.4238 |
| 123 | 14 | 0.8604 | 0.6351 | 0.4768 |
| Mean ± sample SD | — | 0.8593 ± 0.0099 | 0.6151 ± 0.0197 | 0.4558 ± 0.0282 |

These are initialization/data-order variations on the same test set, not three independent population tests or confidence intervals. Runtime differed with concurrent local workload: 65.5, 70.5 and 155.4 seconds of neural fine-tuning. See `reports/support-seeds.json` and the individual complete reports.

### Probability calibration

The primary calibration temperature is approximately 0.85384. The raw threshold 0.66 becomes approximately 0.684998 after the same monotone temperature transform, so intent decisions remain unchanged.

| Held-out probability metric | Raw | Temperature-scaled |
|---|---:|---:|
| Binary NLL | 0.037484 | 0.036706 |
| Brier score | 0.009799 | 0.009699 |
| Ten-bin pooled ECE | 0.008765 | 0.006592 |
| Mean ECE over labels with positive support | 0.016164 | 0.014247 |

Negative labels dominate pooled scores, so per-label support and label-averaged diagnostics matter. Calibration has no positive `business` or `accesibility` examples. Improvement on this split does not prove calibration under future shift. Token scores are not calibrated span-correctness probabilities.

### Deployed review policy

Validation fits a minimum raw probability-distance threshold among nonempty intent decisions. Deployment also reviews empty predicted sets and truncated inputs. Evaluation applies all three conditions.

The primary policy accepts 63/302 test requests (20.86%). Accepted intent-set error is 8/63 (12.70%), above the validation fitting target of 10%. Accepted complete-frame error is 17/63 (26.98%). The policy does not control entity error. This is a useful negative result about relying on confidence margins, not a demonstrated risk guarantee.

## Commands: supervised unsupported-request rejection

Validation selected TF-IDF multiclass logistic regression over Complement Naive Bayes. Probability temperature is fitted separately from the OOS threshold. Final supported-intent accuracy before rejection is 90.86%.

| Final metric over supported + OOS mixture | Result |
|---|---:|
| Overall accuracy including rejection | 77.86% |
| Accepted coverage | 63.59% |
| Accepted accuracy | 96.05% |
| OOS recall | 92.30% |
| In-scope acceptance | 76.01% |
| OOS detection AUROC | 0.9128 |

The cost of higher accepted accuracy is substantial rejection of supported requests. Maximum class probability is conditional on the 150 known intents, not P(in-scope). OOS validity is limited to this labeled benchmark. Baseline validation and frozen supported-test comparisons are in `reports/commands.json`.

## Spam: simple algorithms can be competitive

Validation selected TF-IDF Complement Naive Bayes over logistic regression. Held-out accuracy is 98.64%, macro-F1 96.90%, spam F1 94.57%, spam precision 93.85%, and spam recall 95.31%. Spam is the positive class.

| Actual / predicted | Ham | Spam |
|---|---:|---:|
| Ham | 448 | 4 |
| Spam | 3 | 61 |

Accuracy alone would conceal class imbalance. This old SMS dataset does not validate modern phishing or messaging-app scams. Argmax resolves exact probability ties to ham. Both algorithms' validation and test results are in `reports/spam.json`.

## Latency scope

Primary support warmed batch-one CPU inference measured p50 5.34 ms, p95 6.71 ms, p99 7.07 ms over 120 repetitions of six illustrative short requests. Four PyTorch CPU threads were used on this arm64 macOS machine. Tokenization, encoder/heads, span decoding and Pydantic output validation are included; process startup, artifact loading, HTTP transport and queuing are excluded. Single-case model-only median was 4.79 ms.

Classical warm dict-output measurements were approximately 0.271/0.373 ms p50/p95 for commands and 0.366/0.397 ms for spam. They include vectorization, classifier, calibration and dict decoding, but omit API schema validation. Because tasks and measurement scopes differ, these numbers are not a controlled algorithm speed comparison. No remote Jev/LLM benchmark was run.

## Reproduce and inspect

```bash
sh scripts/reproduce.sh
uv run reflexone train --seed 7 --artifacts runs/seed7 --output runs/seed7.json
uv run reflexone train --seed 123 --artifacts runs/seed123 --output runs/seed123.json
uv run reflexone evaluate --artifacts artifacts/support --output runs/primary-reevaluation.json
uv run reflexone evaluate-task commands --output runs/commands-reevaluation.json
uv run reflexone evaluate-task spam --output runs/spam-reevaluation.json
```

No frozen-encoder ablation, separate-encoder efficiency comparison, quantization experiment, multilingual evaluation or real traffic test has been performed. These remain possible extensions, not shipped performance claims.
