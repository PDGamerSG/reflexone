# Running and hosting ReflexOne

ReflexOne serves the three released profiles from one CPU process. The same image
runs locally and on a free Hugging Face Docker Space. The model archive is downloaded
and verified during image build; startup and inference do not contact a model service.

## Install the released models

From an installed checkout, run:

```bash
uv sync --frozen --extra dev
uv run reflexone download-models
uv run reflexone serve
```

The installer pins the v0.1.0 archive SHA256, verifies its per-file manifest, stages
the three profiles, and refuses to replace conflicting existing models. Repeating
it leaves matching profiles unchanged. It does not require GitHub authentication.
Training new models remains available through [the learning guide](learning-guide.md).

## Build and exercise the container

```bash
docker build -t reflexone:0.2.0 .
docker run --rm --name reflexone -p 127.0.0.1:7860:7860 reflexone:0.2.0
```

In another shell:

```bash
uv run python scripts/check_api.py http://127.0.0.1:7860
```

Open `http://127.0.0.1:7860/` for interactive API documentation. The check waits for
all three profiles, exercises individual and mixed predictions across all four
use cases, and verifies rejection of a request missing its required domain.

Application v0.2.0 uses the unchanged v0.1.0 trained model release. The image uses
the original lockfile's dependency versions with the matching
PyTorch CPU build. It avoids the CUDA dependency bundle used by the general training
environment. It runs as user 1000 with one worker and a concurrency limit of 16.
Increasing worker count duplicates loaded model memory. Docker health checks use
`/ready`; `/health` remains a lightweight process/profile check, including for local
installations intentionally serving only some profiles. `/ready` returns 503 until
support, commands, and spam are all loaded.

## Publish a free Docker Space

A Hugging Face account with permission to create and write a Space is required.
The deployment script creates a **public CPU Basic Space** in your namespace. It
does not select paid hardware or persistence. Authenticate with `hf auth login`,
then review and publish:

```bash
uv run python scripts/deploy_space.py --dry-run
uv run python scripts/deploy_space.py --space YOUR_ACCOUNT/reflexone
```

The script requires a clean committed checkout and uploads an explicit source
allowlist from the selected Git commit. Ignored local credentials, raw data, artifacts,
test caches, and `.git` are never uploaded. The managed source and license trees are
mirrored on updates so deleted modules do not survive a redeployment. The Space card
records the source commit. An existing Space must be public,
use Docker, and identify this source project before the script will update it.

Watch the build on the returned Space page, then run:

```bash
uv run python scripts/check_api.py https://YOUR_SPACE_HOST.hf.space --wait-seconds 180
```

Use the actual `api_url` returned by deployment. The root opens `/docs`, which can
submit requests directly; a separate frontend is optional. Free CPU Spaces may
sleep while idle and have cold starts. Local latency measurements do not describe
internet latency or hosted availability. This portfolio demo does not provide an
availability guarantee. Use public sample inputs on a publicly hosted demo.

## Deployment verification

GitHub CI checks installation, lint, tests, and package builds. Its container job
builds the same image on Linux, waits for complete readiness, and tests the HTTP
contract. A successful image build also proves the pinned release archive can be
downloaded and verified without credentials.

After deployment, retain the Space URL, deployment commit, source commit, and
`scripts/check_api.py` output as evidence. Publishing source and release assets on
GitHub is distinct from a running hosted service; only a successful public HTTP
check establishes the latter.
