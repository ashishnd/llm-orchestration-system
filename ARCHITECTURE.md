# Architecture

## Per-query pipeline (one job)

```
User query
    │
    ▼
[POST /jobs] ─────► creates Job row, registers SSE queue, fires
                    asyncio.create_task(pipeline_run)
    │                        │
    ▼                        ▼
SSE stream         ┌─────────────────────────────────────────────┐
opens              │ Orchestrator.run(SharedContext)             │
                   │                                             │
                   │  ┌────────────────────────────────────┐    │
                   │  │ phase 1: decompose                 │    │
                   │  │  - LLM router picks "decomposition"│    │
                   │  │  - DecompositionAgent.run()        │    │
                   │  │  - sub_tasks populated, slug IDs   │    │
                   │  │    re-stamped to canonical UUIDs   │    │
                   │  └────────────────────────────────────┘    │
                   │                  │                          │
                   │                  ▼                          │
                   │  ┌────────────────────────────────────┐    │
                   │  │ phase 2: retrieve                  │    │
                   │  │  - walk dep graph, dispatch ready  │    │
                   │  │    retrieval-typed sub-tasks       │    │
                   │  │  - RetrievalAgent.run() does the   │    │
                   │  │    multi-hop loop (LLM picks query │    │
                   │  │    each hop, hard cap at 3 hops)   │    │
                   │  │  - Citation fabrication caught;    │    │
                   │  │    under-hopping logged            │    │
                   │  └────────────────────────────────────┘    │
                   │                  │                          │
                   │                  ▼                          │
                   │  ┌────────────────────────────────────┐    │
                   │  │ phase 3: critique                  │    │
                   │  │  - one pass per latest retrieval   │    │
                   │  │    output                          │    │
                   │  │  - span-level claim_scores +       │    │
                   │  │    disagreements                   │    │
                   │  └────────────────────────────────────┘    │
                   │                  │                          │
                   │                  ▼                          │
                   │  ┌────────────────────────────────────┐    │
                   │  │ phase 4: synthesize                │    │
                   │  │  - merge retrieval + critique      │    │
                   │  │  - sentence-level provenance map   │    │
                   │  │  - resolution_notes for any        │    │
                   │  │    contradictions                  │    │
                   │  └────────────────────────────────────┘    │
                   │                                             │
                   │  Compression-on-overflow:                   │
                   │  any agent.run() raising BudgetViolation    │
                   │  is caught, CompressionAgent runs on        │
                   │  prior outputs, agent retried once.         │
                   │  Second overflow → AgentExecutionError,     │
                   │  phase aborted but pipeline continues.      │
                   └─────────────────────────────────────────────┘
                                       │
                                       ▼
                          save_trace + mark_job_completed
                                       │
                                       ▼
                          emit_final_answer + emit_done
                                       │
                                       ▼
                          SSE stream closes; client gets
                          final_answer event before done
```

## Shared context

Every agent reads from and writes to **one** `SharedContext` object per job. The orchestrator mediates handoffs; agents never call each other directly.

Key fields:
- `agent_outputs` — append-only list of every agent turn, each with a discriminated-union `structured_output` (one of five typed pipeline outputs)
- `sub_tasks` — typed sub-tasks with dependency graph
- `tool_invocations` — full input/output/latency log, plus `retry_of` chain for retries
- `routing_log` — every routing decision with justification
- `policy_violations` — budget overflows, schema violations, retrieval undersourcing, etc. — logged, never silently fixed
- `final_answer` + `latest_synthesis()` — ground truth for the answer + provenance map

Project-wide convention: half-open `[start, end)` char offsets everywhere, enforced by `Span.model_validator`.

## Eval pipeline

```
scripts/run_eval.py
    │
    ▼
EvalRun row created with prompt_snapshot
    │
    ▼
For each test case:
    job_id = create_job
    ctx = SharedContext(...)
    Orchestrator.run(ctx)              ← same pipeline as live queries
    save_trace(ctx)
    score_case(case, ctx, judge_llm)   ← 6 dimensions
    add_eval_score row per dimension   ← carries job_id for traceability
    │
    ▼
EvalRun marked complete

────── self-improving loop (optional) ──────

propose_rewrite(eval_run_id, judge_llm)
    │
    ▼
Identify worst (agent, dimension) below threshold
    │
    ▼
LLM generates rewrite + rationale
    │
    ▼
PromptRewrite row with status='pending'
    │
    ▼
Human approves via POST /prompt-rewrites/{id}/decide
    │
    ▼
prompts.set_active_prompt(key, new_text)
    │
    ▼
POST /eval/rerun-failed → re-runs only failed cases against new prompts,
                          returns mean per-dimension delta vs base run
```

## Conventions and invariants

- **Half-open intervals** `[start, end)` for every char span. Enforced.
- **All confidences in [0.0, 1.0]**. Enforced by Pydantic `Field(ge=0, le=1)`.
- **Discriminator field `type`** on every in-pipeline structured output. Used by Pydantic v2 tagged union.
- **Logged-violation-not-silent-fix posture.** Schema mismatches, budget overflows, dependency cycles, citation fabrication — all become `PolicyViolation` records, never silent corrections. The system's audit trail is the source of truth.
- **Reproducibility**: every `AgentOutput` carries a SHA-256 `prompt_hash` so trace replays detect when a prompt changed; every `EvalRun` stores the active prompt snapshot.
