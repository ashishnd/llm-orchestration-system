# Maintainer handoff

Short checklist for the next person running or extending this repo.

## Prerequisites

- Python **3.11+** (Dockerfile uses 3.12; both work if deps resolve).
- **Docker** + Docker Compose v2 for the full stack.
- **OpenAI API key** for LLM + embeddings in production paths.

## First-time local setup

1. `cp .env.example .env` — set **`OPENAI_API_KEY`**. For **Docker Compose**, set a non-empty **`POSTGRES_PASSWORD`**. For **Postgres.app** on the host with no password, leave **`POSTGRES_PASSWORD`** empty and set **`POSTGRES_HOST=localhost`** (and user/db to match your cluster).
2. `python -m venv .venv && source .venv/bin/activate` (or equivalent on Windows).
3. `pip install -e ".[dev]"` — editable install must succeed (see `pyproject.toml` `[tool.setuptools.packages.find]`; only package **`app`** is shipped). SQLAlchemy **async** requires **`greenlet`** (listed in dependencies); if you see `No module named 'greenlet'`, reinstall from the lockfile/pyproject.
4. **Corpus (needed for meaningful RAG):** First run `python -m scripts.ingest_corpus` (network) to create `data/arxiv_corpus/papers.jsonl` and Chroma. After that, `python -m scripts.ingest_corpus --snapshot-only` is offline. Without the JSONL file, Compose **skips** ingest and Chroma stays **empty**.

## Verify

- **Tests:** `pytest` — hermetic suite (default excludes `@pytest.mark.live_llm`). Live OpenAI smoke: `RUN_LIVE_LLM=1 pytest -m live_llm tests/integration -v`.
- **API (Compose):** `docker compose up --build` → `http://localhost:8000/healthz` and `/docs`.
- **Eval (Compose):** `docker compose run --rm worker python -m scripts.run_eval` (uses real LLM; costs apply).

## Architecture pointers

| Area | Entry points |
|------|----------------|
| HTTP API | `app/api/main.py` |
| Orchestrator | `app/orchestrator/orchestrator.py`, `routing.py`, `tool_dispatch.py` |
| Agents | `app/agents/*.py`, prompts `app/agents/prompts.py` |
| Schema / budget | `app/context/schema.py`, `app/context/budget.py` |
| RAG | `app/rag/`, `scripts/ingest_corpus.py` |
| Persistence | `app/persistence/models.py`, `repository.py`, `db.py` |
| Eval / meta | `app/eval/`, `scripts/run_eval.py` |
| SSE | `app/streaming/` |

## Known intentional gaps (see README)

- Worker container is mostly a **placeholder**; per-query pipeline runs **in-process** with SSE.
- **`create_all`** on startup — no Alembic migrations.
- **`code_exec`** is **not** a security sandbox.

## Packaging gotcha (fixed)

If editable install errors with **multiple top-level packages `app` and `data`**, ensure `pyproject.toml` restricts discovery to **`app`** only (`data/` is not a Python package).

## AI attestation

Assessment AI-use log: **`AI_COLLABORATION.md`**.

---

## Pre-submission checklist (human / non-sandbox)

These were **not** automated in Cursor’s sandbox (no Docker, no GitHub push, no paid LLM eval). Do them on your machine before the deadline:

1. **Git history (optional rubric story):** Review `scripts/initial_commits.sh` commit-by-commit; only run it on a **fresh clone** or backup branch if you want that narrative—it will create **23 commits** from staged groups. Then **`git push`** to your public repo.
2. **Corpus snapshot:** Run `python -m scripts.ingest_corpus` (network) to create `data/arxiv_corpus/papers.jsonl`, or fetch then `python -m scripts.ingest_corpus --snapshot-only` if JSONL already exists. **Commit `papers.jsonl`** so graders reproduce the same RAG index.
3. **Real eval + DB rows:** `docker compose up --build`, then `docker compose run --rm worker python -m scripts.run_eval` so **`eval_runs` / `eval_scores`** are populated (requires **`OPENAI_API_KEY`** in `.env`).
4. **Compose smoke test:** At least one full **`docker compose up`** to confirm ingest → API healthcheck → optional eval (volume paths, `POSTGRES_PASSWORD`, etc.).
5. **Lint:** `ruff check .` and `ruff format .` — repo is configured to **ignore B008** (FastAPI `Depends` defaults). Run before every push.
