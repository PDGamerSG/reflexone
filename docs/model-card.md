# ReflexOne model card

## Intended use and model family

ReflexOne is a support semantic model plus two classical profiles for portfolio demonstration, supervised NLP study and structured inference experiments. Inputs are English text and an explicitly chosen profile. Support also requires a known banking/hotels domain. It does not answer arbitrary questions or infer dynamic output schemas.

The neural model starts from Apache-2.0 all-MiniLM-L6-v2, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Six transformer layers produce width-384 token states. A 16-dimensional domain embedding conditions an intent MLP (62 sigmoid outputs) and independent slot streams (17 types × O/B/I). The saved architecture has 22,822,657 parameters marked trainable; the unused original BERT pooler accounts for 147,840 of those and receives no gradient. The active forward path includes jointly fine-tuned encoder layers, domain embedding and heads.

## Training

NLU++ has 3,080 source requests, reduced to 3,027 canonical requests after duplicate auditing and conflict quarantine. The custom grouped split is 1,968 train, 455 validation, 302 calibration and 302 test. Original text/offsets remain unchanged. Validation selects epochs and raw thresholds; calibration fits temperature. Final evaluation uses frozen examples and recorded artifact identity.

Primary seed42 uses14 epochs, batch32, max128 wordpieces, encoder LR3e-5, head LR1e-3, AdamW weight decay0.01, dropout0.15, 5% warmup, cosine decay and gradient clipping1. Intent BCE uses square-root prevalence-odds positive weights clipped1–8; token CE weights are `[1,4,4]` for O/B/I. Checkpoint selection maximizes mean validation intent micro-F1 and exact span F1. These are fixed choices, not a broad hyperparameter search. Seeds7 and123 repeat the protocol; see `reports/support-seeds.json`.

Commands use CLINC150 supported-intent training data. Official test+OOS test is preserved except duplicate auditing. Separate official-validation subsets support algorithm selection, probability calibration and in-scope rejection fitting. OOS validation tunes rejection; OOS training examples are unused. Spam uses deduplicated, stratified grouped UCI SMS data. See `classical-profiles.md` for counts and configuration.

## Primary support measurements

| Metric | Seed42 |
|---|---:|
| Intent micro / macro F1 | 0.8686 / 0.8517 |
| Exact intent-set match | 0.6490 |
| Exact character-span F1 | 0.6145 |
| Complete-frame accuracy | 0.4669 |
| Deployed policy accepted coverage | 0.2086 |
| Accepted intent-set / complete-frame error | 0.1270 / 0.2698 |

Commands accepted accuracy is0.9605 at0.6359 coverage over supported+OOS requests; OOS recall0.9230. Spam accuracy0.9864 and F1≈0.9457. These are dataset/protocol-specific. The experiment report includes baselines, calibration and seed variation.

## Output interpretation

- Intent probabilities are marginals. They need not sum to one and must not be multiplied into a joint correctness probability.
- Intent modules describe topics/components. `deny` and `lost_stolen` can coexist in “not lost.” They do not authorize actions.
- Entities are original input slices. `token_score` is the minimum selected tag probability, uncalibrated and not a joint probability that a span is correct.
- Review uses raw intent-boundary margin, nonempty predicted intents and truncation status. The test error exceeds its validation fitting target. It does not validate entities or detect all OOD inputs.
- Command probabilities condition on supported intents. Maximum class probability is not P(in-scope); rejection is separately tuned on labeled examples.
- Spam uses argmax. Exactly0.5/0.5 ties select ham, equivalent to a strict `p(spam)>0.5` threshold.

## Limits

Small support data and rare labels restrict conclusions. Calibration has no positive `accesibility` or `business` examples; reliable per-label probabilities for them are unsupported. A302-request test set gives unstable subgroup estimates. Production language, ASR mistakes and adversarial messages differ from benchmarks. Extraction is weaker than intent decisions. No speech, temporal normalization, external workflow execution or production load testing is included.

Only normalized exact duplicates are controlled; semantic paraphrases can span splits. Some NLU++ requests need dialogue history. CLINC OOS is a limited sample of unsupported requests. UCI SMS is old and does not establish performance on modern scams.

## Artifacts

Primary artifacts are `artifacts/support`, `artifacts/commands`, `artifacts/spam`; extra neural seeds remain under `artifacts/support-seed7` and `artifacts/support-seed123`. The release packages primary weights, tokenizers, metadata and evaluation records. Support weights and prepared partitions are checksum-verified; classical weights and evaluation checksums are verified by default. Source revisions, manifests, configs and model identity accompany reports. Checksums detect accidental changes; they are not signatures.

Original code is MIT; NLU++/SMS are CC BY4.0, CLINC150 CC BY3.0 and the pretrained encoder Apache2.0. See `THIRD_PARTY.md` for authors, sources and modifications.
