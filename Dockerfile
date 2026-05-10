# syntax=docker/dockerfile:1
# Multi-stage build: builder installs deps, runtime stage is slim.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System deps for psycopg etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Install dependencies into a venv we can copy across stages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -e .

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime deps: libpq for psycopg, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --uid 1001 mao
WORKDIR /home/mao/app

# Bring the venv from the builder
COPY --from=builder /opt/venv /opt/venv

# App source
COPY --chown=mao:mao app ./app
COPY --chown=mao:mao scripts ./scripts
COPY --chown=mao:mao data ./data
COPY --chown=mao:mao pyproject.toml ./

# Persistent dirs for Chroma + corpus
RUN mkdir -p /data/chroma /data/arxiv_corpus && chown -R mao:mao /data

USER mao

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
