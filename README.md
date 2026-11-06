# ReflexOne

**Trainable decision models that turn text into typed predictions in one pass.**

ReflexOne explores the interface behind [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): software often needs decisions, probabilities and extracted values rather than a generated chat response. This independent project implements that idea with domain-specific supervised ML, real training data, held-out evaluation, a CLI and an API.

Four use cases share the same serving interface:

| Use case | Learned behavior | Dataset |
|---|---|---|
| Banking support | Multiple intent modules and extracted money/date/name fields | NLU++ |
| Hotel support | Booking, facilities and service modules plus extracted details | NLU++ |
| Assistant command routing | 150 intents and rejection of unsupported requests | CLINC150 |
| Spam screening | Spam/legitimate message classification | UCI SMS Spam Collection |

## Try the trained models

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The repository and release are public, so the release assets can be downloaded from GitHub without authentication.

```bash
git clone https://github.com/PDGamerSG/reflexone.git
cd reflexone
uv sync --frozen --extra dev

# Download and verify the actual trained checkpoints without authentication.
uv run reflexone download-models

uv run reflexone predict --domain banking --text 'My card was stolen last night. Please block it.' --selected-only
uv run reflexone predict --domain hotels --text 'Can I book two rooms for three adults next Friday?' --selected-only
uv run reflexone predict-task commands --text 'Please play jazz music'
uv run reflexone predict-task spam --text 'WIN a FREE cash prize, claim now'
```

Support predictions contain ontology-defined intent booleans and probabilities, exact input spans with token scores, and review reasons. Intent modules are semantic topics and request components: `lost_stolen` together with `deny` does not assert that something was stolen. No model output executes an operation.

## What is trained

- **Support:** a six-layer MiniLM transformer jointly fine-tuned with 62 sigmoid intent outputs and 17 independent O/B/I entity streams. Domain masks constrain banking and hotel outputs. Independent streams preserve overlapping entity types. AdamW trains the shared encoder and heads with masked binary cross-entropy and weighted token cross-entropy.
- **Support baseline:** TF-IDF plus one-vs-rest logistic regression.
- **Commands:** TF-IDF plus multiclass logistic regression, compared with Complement Naive Bayes. Separate calibration and rejection-validation partitions fit temperature and an OOS threshold.
- **Spam:** TF-IDF plus Complement Naive Bayes, selected against logistic regression using validation spam F1; temperature scaling fits a separate calibration partition.

All checkpoints are trained locally. MiniLM starts from pretrained weights; support heads are newly initialized. This project does not reproduce TypeSafe's proprietary architecture/RLCD, arbitrary dynamic schemas or frontier-level general reasoning.

## Measured results

Fixed primary seed **42**, with additional neural seeds 7 and 123 reported rather than selected by test score. NLU++ uses a custom grouped holdout, not the paper's cross-validation protocol.

| Measurement | Result |
|---|---:|
| Support intent micro-F1 | **86.86%** |
| Support TF-IDF baseline intent micro-F1 | 76.18% |
| Support exact entity-span F1 | 61.45% |
| Support complete-frame accuracy: all intents and spans correct | 46.69% |
| Three-seed support intent micro-F1, mean ± sample SD | 85.93% ± 0.99 percentage points |
| Commands supported-intent accuracy before rejection | 90.86% |
| Commands accepted accuracy / overall coverage | 96.05% / 63.59% |
| Commands labeled OOS recall | 92.30% |
| Spam accuracy / spam F1 | 98.64% / 94.57% |
| Support warm CPU p50 / p95 | 5.34 / 6.71 ms |

The support test set has 302 requests. Complete-frame performance and extraction remain limitations. The deployed support review policy accepts 63/302 requests (20.86%); its accepted intent-set error is 12.70%, exceeding its 10% validation fitting target. Probability calibration and review are empirically evaluated, with no per-request, OOD or deployed risk guarantee. Only the commands profile has labeled OOS evidence.

CPU timings are local warmed batch-one measurements, including tokenization, model, decoding and output validation. They exclude startup, model loading and HTTP transport; they are not comparisons to remote Jev/LLM services. See the [experiment report](docs/experiments.md), [machine-readable results](reports/) and [interactive local report](reports/dashboard.html).

## Train and reproduce

```bash
# Pinned support data, audit, grouped splits, fine-tuning and held-out evaluation.
uv run reflexone prepare
uv run reflexone train --artifacts runs/my-run/support --output runs/my-run/support.json
uv run reflexone evaluate --artifacts runs/my-run/support --output runs/my-run/reevaluation.json

# Classical profiles: compare algorithms, calibrate and test.
uv run reflexone prepare-task commands
uv run reflexone train-task commands --artifacts runs/my-run/commands --output runs/my-run/commands.json
uv run reflexone prepare-task spam
uv run reflexone train-task spam --artifacts runs/my-run/spam --output runs/my-run/spam.json

uv run reflexone benchmark --output runs/my-run/latency.json
uv run pytest -q -W error
uv run ruff check src tests
```

`sh scripts/reproduce.sh` runs primary experiments into a fresh `runs/reproduction/` directory. Set `REFLEXONE_RUN_DIR` for another directory. First-time neural training downloads the pinned ~91 MB encoder. MPS/CUDA is selected when available; CPU training is supported and slower. Primary fine-tuning took 65.5 seconds on this Apple silicon machine; downloads and evaluation add time. Exact reproducibility across hardware is not promised.

## API

```bash
uv run reflexone serve --artifacts-root artifacts --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"profile":"support","domain":"banking","text":"Why is my transfer pending?"}'
```

`GET /health`, `GET /ready`, `GET /profiles`, `POST /predict`, `POST /predict/batch` and generated `/docs` are available. The root opens the interactive API documentation. Readiness returns 503 unless all three profiles are loaded. A batch accepts up to 32 mixed requests; API text is limited to 4,000 characters. Support requires the known `banking` or `hotels` domain. Truncated input requires review. Models load once; misplaced classical profile artifacts are rejected at startup.

For a CPU container or a public Hugging Face Docker Space, follow the [deployment guide](docs/deployment.md). The image includes checksum-verified models, runs as a non-root user, and supports all four use cases through the same API. Verify a running instance with `uv run python scripts/check_api.py http://127.0.0.1:7860`.

## Read and explain it

1. [Learning guide](docs/learning-guide.md): concepts, algorithms, tensors, losses, training, calibration and exercises.
2. [Architecture](docs/architecture.md): module boundaries and data flow.
3. [Experiment report](docs/experiments.md) and [model card](docs/model-card.md): results, failure cases and limits.
4. [Interview guide](docs/interview-guide.md): 60-second explanation, five-minute walkthrough, questions and defensible resume bullets.
5. [Data audit](docs/data-audit.md), [classical profiles](docs/classical-profiles.md), [reliability](docs/reliability.md) and [research](docs/research.md): reproducibility and evidence.

Source code is MIT. Datasets and the encoder retain their licenses; see [third-party attribution](THIRD_PARTY.md). Neural weights use safetensors; classical joblib checkpoints are for trusted local loading.
