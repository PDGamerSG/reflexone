# ReflexOne implementation plan

> For agentic workers: execute independent modules in parallel with explicit ownership; run tests before integration and review the final repository. This session owns implementation and completion.

**Goal:** Ship four demonstrated use cases through a locally trained neural decision model and two classical decision profiles.

**Architecture:** Shared typed prediction concepts and reliability evaluation; one jointly fine-tuned MiniLM support model, with intent and overlapping entity heads. Additional command-routing/OOS and spam profiles use locally trained classical classifiers with calibration. A CLI/API exposes all profiles.

**Tech Stack:** Python 3.12, uv, PyTorch, Transformers, scikit-learn, SciPy, Pydantic, FastAPI, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-04-reflexone-design.md`

## Global constraints

- NLU++ revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Original text and half-open offsets remain unchanged; global normalized-text groups never cross splits.
- Split seed 17 is frozen; all model seeds reuse that manifest.
- No test-based selection, invented supervision, RLCD claims, fabricated performance or synthetic commit dates.
- Data/licenses retain attribution, generated weights stay outside ordinary git history.

## File ownership and interfaces

- `data.py`, `resources/ontology.json`, `tests/test_data.py`: immutable `Slot(label,start,end,text)` and `Record(id,text,domain,intents,slots,group)`; `Ontology` with `intents`, `slot_types`, `intent_mask(domain)`, `slot_mask(domain)`; `prepare_dataset(path,seed=17)`, `load_records(path)`, `load_ontology(path=None)`.
- `reliability.py`, `tests/test_reliability.py`: `fit_temperature(logits,targets,mask)`, `apply_temperature(logits,temperature)`, `tune_threshold(probabilities,targets,mask)`, `calibrated_threshold(raw_threshold,temperature)`, `multilabel_metrics(targets,probabilities,threshold,mask)`, `selective_curve(targets,probabilities,threshold,mask)`; arrays are NumPy `[N,C]`, masks boolean `[N,C]`.
- `tasks.py`, `tests/test_tasks.py`: `prepare_task(task,data_dir,seed=17)`, `train_task(task,data_dir,output_dir,seed=42)`, `evaluate_task(artifact_dir)`, `predict_task(artifact_dir,text)`, `benchmark_task(artifact_dir,repeats=100)` for task names `commands` and `spam`. Task artifacts include their data/config provenance and evaluation metadata; predictions are plain JSON-serializable dicts, later validated by API wrapper.
- `model.py`, `training.py`, `schema.py`, `inference.py`, `cli.py`, `api.py`: primary agent. Encoder receives token offsets and domain masks; output intent logits `[B,62]`, slot logits `[B,L,17,3]`. Training saves model/config/tokenizer/history; inference returns validated JSON from an artifact directory.

## Task 1: Reproducible grouped support data

- [ ] Write meaningful failing tests: conflicting annotations quarantine, numeric fold ordering, exact spans, cross-domain group isolation and seed reproducibility.
- [ ] Download pinned ontology and 40 folds with hashes; validate original annotation bounds and ontology compatibility.
- [ ] Implement canonicalization/quarantine and frozen 65/15/10/10 group splitting; persist audit and supports.
- [ ] Run data tests and real preparation; confirm canonical audit counts and no overlap.
- [ ] Commit tested data pipeline and attribution.

## Task 2: Calibration and evaluation

- [ ] Write failing numerical tests: Brier/NLL hand examples, temperature on overconfident wrong logits, decision-threshold equivalence, masked labels, risk/coverage edges.
- [ ] Implement scalar temperature optimization bounded to a positive finite interval, multilabel metrics and reliability bins; preserve threshold semantics.
- [ ] Implement validation-tuned threshold and report selective curves without guarantee language.
- [ ] Run tests and commit reliability module.

## Task 3: Additional real-data profiles

- [ ] Write failing tests using tiny local files that prove grouped split isolation, classifier artifact round-trip, spam label direction, and command OOS rejection.
- [ ] Retrieve pinned CLINC150 and UCI SMS data with hashes, licenses and source manifest; custom split policy is recorded.
- [ ] Train TF-IDF logistic regression and Complement Naive Bayes, select on validation, calibrate on a separate holdout; tune OOS using official OOS validation only.
- [ ] Save profiles, reports and repeatable commands; run tests and actual experiments before milestone commit.

## Task 4: Fine-tuned parallel neural model

- [ ] Write failing tests with a tiny local randomly initialized BERT config: output shapes, domain masks, overlapping BIO targets and backpropagation.
- [ ] Build tokenization/alignment with offset mapping; reject unmappable gold spans, detect truncation and never train silently dropped entities.
- [ ] Implement mean-pooled intent head and 17 independent BIO token heads with masked losses; trainable encoder and safe save/load.
- [ ] Implement deterministic AdamW training, gradient clipping, validation-selected checkpoint, configurable CPU/MPS device, and logged epoch timing.
- [ ] Train a TF-IDF OVR baseline, neural model, and calibration; evaluate only frozen artifacts.
- [ ] Run real training and save weights, support test metrics, per-label errors, complete-frame accuracy and slot F1.
- [ ] Commit model/training after tests and actual training evidence.

## Task 5: Shared CLI/API and benchmarks

- [ ] Write failing contract tests for unknown profile/domain, blank text, invalid spans, and artifact round-trip.
- [ ] Provide `prepare`, `train`, `evaluate`, `predict`, `benchmark`, `train-task`, `predict-task`, `evaluate-task`, `benchmark-task`, `serve` commands with clear error messages.
- [ ] Add Pydantic-validated outputs and FastAPI `/health`, `/profiles`, `/predict`, `/predict/batch`; load artifacts once and serialize CPU access safely.
- [ ] Benchmark warmed batch-one end-to-end CPU inference including tokenization and decoding; separate model-only and load timing.
- [ ] Verify CLI predictions and TestClient API calls using real saved weights, then commit.

## Task 6: Portfolio delivery

- [ ] Run independent code and ML-methodology review; fix material issues with regression tests.
- [ ] Run tests, Ruff, build/install and real smoke commands; ensure git has no data/private material.
- [ ] Write algorithm/training guide, data/model cards, experiment report, architecture, research evidence and 60-second/5-minute interview walkthrough.
- [ ] Create repository, push genuine milestone history and attach model archive to a versioned release if feasible.
- [ ] Document exact installed/reproduced commands, measured outcomes, limitations and artifact checksums; commit results.
