# CPU inference image. The root lockfile remains suitable for GPU training.
FROM ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 AS uv
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

COPY --from=uv /uv /usr/local/bin/uv
RUN useradd --create-home --uid 1000 app && mkdir /app && chown app:app /app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/reflexone-hf \
    TOKENIZERS_PARALLELISM=false \
    PATH=/app/.venv/bin:$PATH

COPY --chown=app:app pyproject.toml uv.lock README.md LICENSE THIRD_PARTY.md ./
COPY --chown=app:app licenses/ licenses/
COPY --chown=app:app src/ src/

# Constraints keep the trained classifiers' dependency versions. Selecting the
# CPU backend avoids installing the lockfile's CUDA runtime in the serving image.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --output-file /tmp/constraints.txt >/dev/null \
    && uv venv --python /usr/local/bin/python .venv \
    && uv pip install --python .venv/bin/python --torch-backend cpu \
       --constraint /tmp/constraints.txt . \
    && rm /tmp/constraints.txt

USER app
RUN reflexone download-models --artifacts-root /app/artifacts

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/ready', timeout=5)"
CMD ["uvicorn", "reflexone.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
