# Multi-Agent LLM Orchestrator

> Containerized multi-agent system with dynamic routing, multi-hop RAG, multi-dimensional eval harness, and a self-improving prompt loop gated on human approval.

Built for a Junior LLM Engineer assessment. Knowledge corpus: arXiv `cs.CL`. Default `pytest` is hermetic (no real LLM/network). Optional **live** OpenAI smoke tests live under [`tests/integration/`](tests/integration/) — see below.

---

## Quick start

```bash
# 1. Clone and configure
cp .env.example .env
# edit .env: set OPENAI_API_KEY; set POSTGRES_* (see comments in .env.example)

# 2. Corpus + Chroma (needed for meaningful RAG / eval)
# Docker-only quick path: the repo includes papers.jsonl — skip the fetch below and go
# straight to `docker compose up --build`; the ingest service runs --from-snapshot.
pip install -e ".[dev]"
# Expand corpus (optional): fetch from arXiv and merge into data/arxiv_corpus/papers.jsonl
python -m scripts.ingest_corpus
# Offline: repopulate local Chroma from JSONL only (same as Docker ingest)
python -m scripts.ingest_corpus --from-snapshot
```
Without `data/arxiv_corpus/papers.jsonl`, `docker compose`’s ingest step skips and the vector index stays **empty**.

**Secrets:** Never commit `.env` or real API keys (`.env` is gitignored). Ship only [`.env.example`](.env.example) with placeholders; anyone cloning the repo copies it, fills `OPENAI_API_KEY`, and sets `POSTGRES_PASSWORD` for Docker Postgres.

```bash
# 3. Start everything
docker compose up --build
```

The API listens on `http://localhost:8000`. Open `http://localhost:8000/docs` for the OpenAPI UI.

To run the eval harness:

```bash
docker compose run --rm worker python -m scripts.run_eval
```

To run tests:

```bash
pytest
```

Live OpenAI integration smoke tests (costs API usage; not for CI by default):

```bash
RUN_LIVE_LLM=1 pytest -m live_llm tests/integration -v
```

---

## Architecture

```
                            ┌──────────────────────┐
   POST /jobs ──────────────►   FastAPI (api/)     ──► Postgres (jobs, traces,
                            │   5 endpoints + SSE  │    eval_runs, scores,
                            └──────────┬───────────┘    prompt_rewrites)
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │   Orchestrator       │   ◄── Event Bus (SSE)
                            │   - dynamic routing  │
                            │   - phase dispatch   │
                            │   - tool retry (2x)  │
                            │   - compression on   │
                            │     budget overflow  │
                            └──────────┬───────────┘
                                       │
       ┌───────────────────────────────┼──────────────────────────────┐
       ▼              ▼                ▼                ▼             ▼
 ┌───────────┐ ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐
 │Decompose  │ │Retrieval  │  │Critique   │  │Synthesis  │  │Compression   │
 │           │ │(multi-hop)│  │(span-lvl) │  │(provenance│  │(when budget  │
 │           │ │           │  │           │  │ map)      │  │  overflows)  │
 └───────────┘ └─────┬─────┘  └───────────┘  └───────────┘  └──────────────┘
                     │
                     ▼
               ┌───────────────────────────┐
               │   Multi-Hop Retriever     │
               │   - Chroma vector store   │
               │   - cross-hop dedup       │
               └───────────────────────────┘
                     │
                     ▼
               ┌───────────────────────────┐
               │   arXiv cs.CL JSONL       │
               │   (committed for repro)   │
               └───────────────────────────┘

   Tools (called by agents through the orchestrator):
   web_search │ code_exec │ sql_lookup │ self_reflect

   Eval harness (separate from per-query pipeline):
   15 cases (5 baseline / 5 ambiguous / 5 adversarial)
   ── runner ── 6-dim scorer ── meta_agent ── PromptRewrite (pending)
                                                  │
                                       human approval via API
                                                  │
                                                  ▼
                                       prompts.set_active_prompt
```

A more detailed text diagram lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Agents and decision boundaries

| Agent | What it does | Decision boundary |
| --- | --- | --- |
| **Decomposition** | Breaks the user query into typed sub-tasks (`retrieval`, `computation`, `synthesis`, `verification`, `other`) with a dependency graph. | Decides **how many sub-tasks** and **which depend on which**. The prompt instructs caution: simple queries get one sub-task. Cycle and dangling-ref detection both raise `schema_violation` rather than silently fixing. |
| **Retrieval** | LLM-driven multi-hop retrieval over the Chroma vector store. Produces a draft answer with chunk-level citation links. | Decides **what query to ask each hop** and **when to stop** (after at least 2 hops, capped at 3). Citation fabrication (citing a chunk that wasn't retrieved) is caught and raises `schema_violation`. Under-hopping (< 2) is logged as `retrieval_undersourced`. |
| **Critique** | Span-level review of one target output per pass: confidence per claim, disagreements per span. | Decides **which spans of the target to flag**. Disagreements include a confidence score. The brief required span-level (not whole-output) — enforced in the schema. |
| **Synthesis** | Merges retrieval + critique into a final answer with a sentence-level provenance map. | Decides **whether to incorporate critique disagreements or document why not** in `resolution_notes`. |
| **Compression** | Summarizes older context when budget utilization crosses 85%. Lossless on structured refs (tool invocations, citations, sub-task IDs); lossy on narrative. | Triggered by the orchestrator on `BudgetViolation`, not by the agent itself. Records before/after token counts so compression ratio is auditable. |
| **Meta-agent** (eval-time only) | Reads scores from a completed eval run, picks the worst (agent, dimension) pair, proposes a rewritten prompt. | Decides **which prompt to target** (via dimension→agent mapping) and **what to change**. Proposals are persisted as `pending` — never auto-applied. |

All inter-agent communication passes through a single `SharedContext` Pydantic object (see [`app/context/schema.py`](app/context/schema.py)). Agents never call each other directly; the orchestrator mediates every handoff.

---

## API

Five endpoints per the brief, plus `/healthz`:

| Method & path | Description |
| --- | --- |
| `POST /jobs` | Submit a query; returns SSE stream (agent_start, agent_complete, tool_*, routing, final_answer, done). |
| `GET /jobs/{job_id}/trace` | Full execution trace as the persisted `SharedContext` JSON. |
| `GET /eval/latest` | Latest eval run summary, broken down by category and dimension. |
| `POST /prompt-rewrites/{rewrite_id}/decide` | Approve or reject a meta-agent proposal. Approved rewrites swap the active prompt at runtime; the baseline is preserved for rollback. |
| `POST /eval/rerun-failed` | Re-run only test cases that scored below `score_threshold` on any dimension in a base run; returns mean per-dimension delta vs the base. |

Error responses are `{ "error_code": "...", "message": "...", "job_id": "..." }`. Schemas are in [`app/api/schemas.py`](app/api/schemas.py), decoupled from internal context models.

---

## What the self-improving loop does and does not do

**It does:**
- Read scores from a completed `EvalRun`.
- Identify the lowest-scoring dimension whose mean is below 0.7.
- Map the dimension to the agent most responsible (e.g. `citation_accuracy` → retrieval).
- Generate a rewritten prompt via the meta-agent LLM, with rationale.
- Persist a `PromptRewrite` row with `status='pending'` and a unified diff.
- Surface pending rewrites for human approval via the API.
- On approval, swap the active prompt at runtime (baseline preserved for rollback).
- On `/eval/rerun-failed`, re-execute only the failed cases against the new active prompts and report mean per-dimension delta.

**It does not:**
- Auto-apply rewrites without human approval. By design — the brief's "approve or reject" loop is non-negotiable.
- Modify multiple prompts per cycle. One rewrite proposal per eval run, targeting the single worst dimension.
- Track multi-version prompt history beyond `baseline ↔ active`. A second approved rewrite replaces the first; rollback is to baseline only.
- Detect that a rewrite degrades performance and self-revert. The human is the loop closure; degradation surfaces in the next eval delta but doesn't trigger automatic action.

---

## Known limitations

Honest assessment of where the system breaks. Each item is something I'd address before calling this production-grade.

**Pipeline scope cuts**
- Sub-task dispatch in the orchestrator only routes `retrieval`-typed sub-tasks. `computation`, `synthesis`, and `verification` types are recognized by the schema but marked `SKIPPED` at dispatch time. Retrieval is the brief's primary value path; building dedicated executors for the other types is out of scope here.
- The orchestrator's "dynamic routing" is hybrid: phases (decompose → retrieve → critique → synthesize) are deterministic structure; the LLM picks order *within* phases. Pure free-routing was tested informally and is unreliable on `gpt-4o-mini`-class models. This is documented as a deliberate choice, not an oversight.
- Token streaming via SSE surfaces agent lifecycle events, tool events, routing decisions, budget status, and final answer. It does **not** stream raw JSON tokens mid-generation because JSON is not human-readable mid-stream. The brief says "stream token by token"; my pragmatic reading is that the *useful* signal is what we surface.

**Eval limitations**
- The LLM judge for `answer_correctness` and `contradiction_resolution` is the same model family as the agents being evaluated — a known eval anti-pattern. Real production should use a separate provider (e.g. Claude judging GPT, or vice versa).
- The 15 test cases assume specific arXiv papers (BERT 1810.04805, GPT-3 2005.14165, LoRA 2106.09685, DPO 2305.18290) are in the corpus. If the corpus snapshot doesn't include them, `citation_accuracy` scores reflect that honestly rather than failing.
- Adversarial cases test refusal + partial helpfulness on `gpt-4o-mini`. Standard refusal training handles them; the eval surfaces failures if it doesn't.

**Operational limitations**
- Schema management uses `Base.metadata.create_all` on startup. No Alembic migrations — fine for assessment scope, not for production. A real deployment would have Alembic.
- The "background worker" service in `docker-compose.yml` is a placeholder (`tail -f /dev/null`). The API runs the per-query pipeline in-process via `asyncio.create_task`; events stream live. A real production setup would push to RQ/Celery and stream from a Redis pubsub.
- There is no separate **log-query microservice**: the brief’s observability is narrowed to **`GET /jobs/{job_id}/trace`** (full `SharedContext` JSON in Postgres) plus SSE during the run, rather than a dedicated log UI.
- No authentication on the API. The brief didn't require it, but a real deployment would have at minimum bearer-token auth on `POST /prompt-rewrites/*/decide`.
- The `code_exec` tool uses `subprocess` with a hard timeout. **It is NOT a security boundary.** Production would use gVisor / Firecracker / Docker-in-Docker with no network and a temp working dir.
- LLM nondeterminism: even at `temperature=0`, OpenAI does not guarantee bit-exact reproducibility. Eval scores will be very stable across reruns but not byte-identical.

---

## What I would build next

In rough priority order:

1. **Real computation/verification sub-task executors.** The dependency graph machinery is in place; what's missing is a small computation agent (Python sandbox + structured-data lookup) and a verification agent (cross-check answer claims against retrieved chunks). This unlocks the brief's "explicit dependency graph" promise for non-retrieval tasks.
2. **Cross-provider eval judge.** Switch the judge to Anthropic Claude (or any non-OpenAI provider). Eliminates same-family bias and improves the meta-agent signal.
3. **Versioned prompt history.** A `prompt_versions` table with full lineage: which rewrite produced which version, eval scores at each version, automatic regression alerts. Currently we only have `baseline ↔ active`.
4. **Real background worker via RQ.** Move pipeline execution out of the SSE request path. Worker pulls from a Redis queue, publishes events to a Redis pubsub channel that the SSE endpoint subscribes to. Decouples request lifetime from execution lifetime — important for long-running adversarial cases.
5. **Tool sandbox hardening.** Run `code_exec` in a gVisor-isolated container with no network, ephemeral filesystem, hard memory cap. The current subprocess approach is honest but unfit for untrusted input.
6. **Streaming-first agent variant.** For agents whose output benefits from progressive display (e.g. a future "explanation" agent that produces narrative text), add a `streaming` JSON-mode-off variant with token-by-token SSE. The current JSON-mode pipeline is correct for structured agents but doesn't serve narrative use cases well.

---

## AI collaboration

Per the assessment instructions, AI assistance is documented in [`AI_COLLABORATION.md`](./AI_COLLABORATION.md). Each section of the system has its own entry: what was AI-assisted, what design decisions were mine, and how I verified the output.

---

## Maintainer handoff

For a concise setup checklist, packaging note, and file map, see [`HANDOFF.md`](./HANDOFF.md).

---

## Repository layout

```
multi-agent-orchestrator/
├── app/
│   ├── settings.py                 typed env-var config
│   ├── llm/                        provider-agnostic async client (retry, JSON repair)
│   ├── context/                    SharedContext schema + budget manager
│   ├── rag/                        corpus, vector store, multi-hop retriever
│   ├── agents/                     5 pipeline agents + base ABC + prompt registry
│   ├── tools/                      4 tools with explicit failure contracts
│   ├── orchestrator/               dynamic routing, phase dispatch, tool retry
│   ├── persistence/                SQLAlchemy models + repository
│   ├── streaming/                  SSE event bus
│   ├── eval/                       15 cases, scorer, runner, meta-agent
│   └── api/                        FastAPI app, 5 endpoints
├── tests/                          hermetic tests; integration/ optional live_llm
├── scripts/
│   ├── ingest_corpus.py            arXiv → JSONL → Chroma
│   └── run_eval.py                 CLI for the eval harness
├── data/arxiv_corpus/              committed JSONL snapshot for reproducibility
├── notebooks/eda.md                corpus EDA
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── ARCHITECTURE.md
└── AI_COLLABORATION.md
```
