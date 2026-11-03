# ReflexOne design

## Objective

Build a reproducible ML portfolio project with a real fine-tuned neural model, additional locally trained classifiers, honest uncertainty evaluation, and a small CLI/API. No frontend is required. The user's Jev reference motivates parallel typed decisions, not chat generation. The user explicitly authorized starting after a five-minute review window; that window elapsed before implementation. A subsequent request expanded the project to multiple use cases.

## Use cases

1. Banking support: simultaneous intent modules and extractive entities.
2. Hotel support: the same model architecture and ontology, conditioned on the known domain.
3. Assistant command routing: CLINC150's 150 intents and labeled out-of-scope requests.
4. SMS spam screening: UCI SMS Spam Collection binary labels.

No predictions execute banking operations, bookings, or external messages. Workflow recommendations are authored policy, not fabricated ground-truth labels.

## Neural support model

Use the pinned Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` encoder, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Fine-tune the encoder jointly with 62 sigmoid intent outputs and 17 independent three-class BIO token heads. Independent slot heads preserve the two cross-type overlapping spans in NLU++. The known domain masks labels using the official ontology (48 banking / 40 hotel intent modules; 13 / 14 slot types). The domain is input configuration, not a learned label. Intent probabilities are marginals, not a joint distribution. Slot values are copied from exact input character offsets; normalized calendar values are outside scope.

## Data and leakage controls

NLU++ source revision is `57ec275d8078af65b7731c2a98be812d844a6d6b`. Preserve original text and half-open span offsets. Normalize a grouping key with NFKC, casefold, and collapsed whitespace. Quarantine conflicting duplicate annotations. Collapse identical within-domain annotations; assign cross-domain identical text to one global split. Freeze a custom 65/15/10/10 train/validation/calibration/test grouped split, seed 17, across model seeds. Publish checksums, source IDs, quarantined rows, split manifest, and per-label support. This is not the paper's original cross-validation protocol. Empty intent sets are valid examples, not OOS labels.

CLINC150 and SMS ingestion likewise freeze grouped partitions and separate training, validation, calibration, and test data. Retain official CLINC test separation while auditing/removing cross-split duplicate text. Record dataset citations, licenses, source revisions and hashes. Each profile must explain any custom split and any subsampling.

## Training and comparison

Support training uses AdamW, binary cross-entropy for intent modules, masked token cross-entropy for slots, gradient clipping, deterministic seed/config recording, and validation-selected checkpoints. The baseline is TF-IDF plus one-vs-rest logistic regression. Command and spam profiles compare interpretable classical algorithms (TF-IDF logistic regression and Complement Naive Bayes); algorithm choice is selected on validation data, not test. Calibration fits on a separate calibration partition. Do not claim empirical improvements before measuring them.

## Reliability

Tune an intent threshold on raw validation probabilities. Fit scalar temperature on calibration logits with binary NLL. Convert the raw decision threshold through the same monotone transformation so calibration does not silently change selected labels. Fit an empirical review rule on validation data and report the risk/coverage tradeoff on test; it is not a per-case guarantee or a conformal guarantee. Command routing alone has supervised OOS test evidence. Support review is not validated OOD detection. Probability calibration can worsen held-out metrics and must be reported honestly. Do not multiply marginal probabilities into a joint confidence.

## Interfaces

Python 3.12+, PyTorch, Hugging Face Transformers, NumPy, SciPy, scikit-learn, Pydantic, FastAPI. `uv` lockfile makes installation reproducible. Modules own data, metrics/calibration, neural model, training, inference, classical task profiles, CLI and API. Typed outputs validate domain, known labels, probabilities, exact offsets and text slices. Reject empty/oversized requests; flag token truncation for review. API loads configured artifacts once; CLI supports preparation, training, evaluation, prediction, benchmarking, and suite execution. Trusted local model files only.

## Acceptance

- Execute real training for support, commands, and spam; save deployable artifacts and configs locally.
- Run baseline comparisons and untouched holdout evaluation; record algorithm family, seed, dataset fingerprint, supports, metrics, timing scope and hardware.
- Unit tests prove duplicate isolation, span fidelity/overlaps, masking, loss/gradient behavior, calibration threshold equivalence, metrics, artifact round-trip, review behavior and API validation.
- Run CLI and API integration checks with the trained artifacts.
- Deliver research sources, architecture, data/model cards, experiment report, algorithm/training guide and interview explanation.
- Commit actual completed milestones with genuine timestamps; never fabricate timing or performance.
- Create a GitHub repository; keep generated bulky data and weights outside git with clear reproduction commands and an accessible trained-model download if practical.

## Limits

This is specialized supervised ML, not a general reasoning model or proprietary RLCD reproduction. Fixed schemas prevent malformed outputs, not wrong decisions. Small data and rare labels limit conclusions. English text only, no speech/front-end, no production operations. A single machine latency result is not a production SLO or a comparison to remote Jev/LLM services.
