# Base image pinned by digest (multi-arch index of python:3.12-slim, Python
# 3.12.14, resolved 2026-09-29). Both stages and the Cloud Build test step use
# it; bump all three together. To refresh:
#   docker buildx imagetools inspect python:3.12-slim   # "Digest:" line
# uv (first stage only, never in the image) is pinned the same way, with the
# same digest as cloudbuild.yaml's export step; bump both together:
#   docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.12.6

# Stage 1: uv.lock -> hash-pinned requirements file. --no-dev leaves pytest
# and ruff out; --frozen exports the lock as is (never re-resolves).
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS lock
COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /usr/local/bin/uv
WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --quiet --no-dev --no-emit-project --format requirements.txt -o /requirements.txt

# Stage 2: the image.
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

# Prevent Python from writing .pyc and enable unbuffered output.
# HOME=/tmp: appuser has no home directory, and some libraries write caches
# under $HOME (/tmp is writable on Cloud Run).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app

# Every file sha256-checked and nothing resolved at build time (the export is
# complete, so --no-deps). Edit pyproject.toml and re-lock instead of touching
# uv.lock (see DEPLOY.md, "依賴鎖定與本機驗證").
COPY --from=lock /requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos "" --no-create-home appuser
USER appuser

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
