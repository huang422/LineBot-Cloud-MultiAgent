# Base image pinned by digest (multi-arch index of python:3.12-slim, Python
# 3.12.14, resolved 2026-09-29). The same digest is used by the Cloud Build
# test step in cloudbuild.yaml; bump both together. To refresh:
#   docker buildx imagetools inspect python:3.12-slim   # "Digest:" line
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

# Prevent Python from writing .pyc and enable unbuffered output.
# HOME=/tmp: appuser has no home directory, and some libraries write caches
# under $HOME (/tmp is writable on Cloud Run).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app

# Install dependencies from the hash-pinned lock (complete, so --no-deps).
# Edit requirements.in and re-lock instead of touching the lock (see DEPLOY.md).
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes --no-deps -r requirements.lock

# Copy application
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos "" --no-create-home appuser
USER appuser

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
