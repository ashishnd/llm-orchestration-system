#!/usr/bin/env bash
# Suggested git commit sequence for the foundation work.
#
# Run this from inside the project root AFTER you have already:
#   1. Created an empty GitHub repo
#   2. Cloned it locally
#   3. Copied the contents of this snapshot into that clone
#   4. Set git user.name and user.email locally
#
# The schema landed as ONE clean commit per the amend strategy you picked:
# we got the schema right before building on it, so the commit history
# reflects that rather than showing an iteration.
#
# Each commit is small enough to review independently.

set -e

if [ ! -d ".git" ]; then
  echo "ERROR: not in a git repo. Run 'git init' first or clone an empty GitHub repo."
  exit 1
fi

echo "Commit 1: project scaffolding"
git add pyproject.toml .gitignore .env.example README.md AI_COLLABORATION.md
git commit -m "chore: project scaffolding, deps, and AI collaboration log"

echo "Commit 2: typed settings"
git add app/__init__.py app/settings.py
git commit -m "feat(config): typed settings loaded from env vars

- All configuration flows through pydantic-settings
- No credentials hardcoded anywhere
- compression_threshold exposed as a tunable env var"

echo "Commit 3: LLM client"
git add app/llm/
git commit -m "feat(llm): provider-agnostic async client with retry and JSON repair

- Wraps OpenAI chat completions; OpenAI-compatible base_url is configurable
  so swapping to Groq, a local server, or another provider is one env change
- Tenacity owns retry policy entirely (max_retries=0 on the SDK to prevent
  retry layer composition that would multiply attempt counts and hide
  behavior from observability)
- Retries on RateLimitError / APIConnectionError / APITimeoutError only;
  4xx user errors are NOT retried (those indicate bugs)
- complete_json runs ONE constrained repair retry on malformed output:
  first attempt at caller's temperature; on parse failure, retry at
  temperature=0.0 with an explicit JSON-only instruction. On second failure,
  raises MalformedJSONError carrying both raw outputs so the orchestrator
  can log a schema_violation PolicyViolation with raw_preview populated.
- tiktoken encoder is lazy-loaded — construction is hermetic, no network."

echo "Commit 4: shared schema (typed envelope, discriminated union, provenance)"
git add app/context/schema.py app/context/__init__.py
git commit -m "feat(context): hardened shared schema for inter-agent communication

The brief requires that all inter-agent comms pass through one typed
context object. This is that object, with all schema-level invariants
defined up front so we don't refactor mid-build.

Conventions:
- Half-open [start, end) char offsets project-wide, enforced by a Span
  validator (not just documented).
- Confidence values clamped to [0.0, 1.0] via Field constraints.
- Every in-pipeline structured output carries a 'type' discriminator.

Hybrid envelope:
- AgentOutput is the common envelope (id, agent_name, content, tokens,
  prompt_hash, timestamp).
- structured_output is a Pydantic v2 discriminated union over five typed
  outputs: DecompositionOutput | RetrievalOutput | CritiqueOutput |
  SynthesisOutput | CompressionOutput.
- MetaRewriteOutput is intentionally NOT in this union — it runs after
  eval, persists to its own table, and shouldn't share an envelope with
  per-job pipeline outputs.

Compliance posture:
- Brief-mandated invariants (e.g. retrieval >= 2 hops) are enforced at
  the orchestrator level via helper methods like is_compliant(), NOT as
  schema constraints. Non-compliance becomes a logged PolicyViolation
  with an audit trail, not a silent parse error.
- PolicyViolation gains raw_preview (capped at 500 chars) so the
  meta-agent can read what failed when proposing prompt rewrites.

Provenance:
- ProvenanceEntry maps each sentence in the final answer to source
  agent + chunks via half-open spans.
- list[ProvenanceEntry] (sorted by sentence_index) over dict[str, dict]
  for stable JSON diffs in regression-visible eval reruns.
- Sourced from the latest SynthesisOutput; not duplicated on
  SharedContext.

Sub-tasks:
- task_type Literal includes 'other' with a model_validator requiring
  task_type_detail. Unanticipated decomposition is absorbed gracefully
  but still flagged."

echo "Commit 5: budget manager with policy-violation logging"
git add app/context/budget.py
git commit -m "feat(context): per-agent token budget manager

- Each agent declares its budget at the start of its turn.
- check() raises BudgetViolation rather than silently truncating; the
  orchestrator decides whether to compress or refuse.
- needs_compression() flags utilization >= threshold (default 0.85).
- record_violation() appends a PolicyViolation to the shared context, per
  the brief's requirement that overflow is logged not silently fixed.
- TokenCounter Protocol decouples this module from the concrete LLM
  client — keeps tests fast and the dependency graph clean."

echo "Commit 6: foundation tests (46 cases across schema, budget, LLM client)"
git add tests/test_context_budget.py tests/test_schema.py tests/test_llm_client.py
git commit -m "test: cover schema validators, budget overflow, and LLM client retry

46 cases:

Schema (26): Span bounds (negative start, end<start, empty allowed),
SubTask conditional task_type_detail, PolicyViolation raw_preview
truncation at 500 chars, discriminated union dispatch (one test per
variant + unknown-type + missing-type rejection), RetrievalOutput
compliance helper, AgentOutput envelope (confidence clamping, structured
output dispatch).

Budget (9): declaration, within-budget check, overflow as
BudgetViolation, consumption tracking, compression threshold trigger
at 85%, per-turn redeclaration semantics, missing-agent KeyError,
policy violation logging, full-utilization clamping.

LLM client (11): hermetic via respx HTTP mocking. Token counting
determinism via stub encoder, message overhead, retry on 429, no-retry
on 400/401, 3-attempt cap, JSON valid-first-try, malformed-then-repair
success, malformed-twice raises with raw_preview, non-object array
triggers repair.

The LLM client tests caught a real bug: the OpenAI SDK's internal retry
layer was composing with tenacity, multiplying attempt counts. Fixed by
setting max_retries=0 on the SDK so tenacity owns retry policy."

echo "Commit 7: corpus types and JSONL IO"
git add app/rag/corpus.py
git commit -m "feat(rag): arXiv paper type and JSONL corpus IO

The corpus is committed as JSONL (one paper per line) for reproducibility.
The grader runs against the same papers we evaluated against rather than
whatever arXiv returns on the day. Re-running ingestion merges new papers
into the snapshot without replacing existing entries.

Paper.chunk_text concatenates title and abstract because titles often
carry domain keywords (benchmark names, architectures) that the abstract
body doesn't repeat. arxiv_id is the stable chunk identifier — re-ingest
is idempotent (upsert by chunk_id)."

echo "Commit 8: vector store wrapper with explicit embedder injection"
git add app/rag/vector_store.py
git commit -m "feat(rag): Chroma-backed vector store with explicit embedder

Wraps Chroma rather than exposing it directly so the rest of the system
depends on a small, stable interface (ingest, query, count). Embeddings
are computed explicitly via an injected Embedder callable and passed to
Chroma as 'embeddings=...' rather than registering a Chroma embedding
function on the collection.

Why bypass Chroma's EF system:
- Chroma 1.1.13+ persists embedding functions server-side and silently
  falls back to its default ONNX model if a custom EF doesn't conform
  to the persistence protocol. Tests appeared to pass while actually
  using the wrong embeddings.
- Decouples our embedding choice from Chroma's lifecycle. Swapping to
  local sentence-transformers later is a one-line change.
- Tests inject a deterministic stub embedder without conforming to
  Chroma's evolving EF protocol.

VectorStore.collection_name is parameterizable for test isolation
because chromadb.EphemeralClient() instances share an in-memory
collection registry across the process — true isolation requires
unique names, not separate clients."

echo "Commit 9: multi-hop retriever primitive"
git add app/rag/retriever.py app/rag/__init__.py
git commit -m "feat(rag): multi-hop retriever with cross-hop deduplication

This module is the deterministic primitive: execute a sequence of
queries with state, deduplicate chunks across hops, return aggregated
results. The LLM-driven decision about what query to ask next lives in
the retrieval agent — splitting these makes the agent's prompt logic
and the retrieval mechanics independently testable.

Cross-hop dedup via seen_chunk_ids is non-negotiable: without it every
hop re-retrieves the most-similar chunk and 'multi-hop' is a lie."

echo "Commit 10: arXiv ingestion script + RAG dependency"
git add scripts/ingest_corpus.py pyproject.toml
git commit -m "feat(rag): arXiv ingestion script (fetch + JSONL + Chroma)

Two run modes:
  python -m scripts.ingest_corpus              # fetch from arXiv + Chroma
  python -m scripts.ingest_corpus --from-snapshot  # offline reseed

The second mode is what 'docker compose up' calls, so reviewers don't
need network access to arXiv to run the system."

echo "Commit 11: RAG tests (21 cases) and AI collaboration log update"
git add tests/test_rag.py AI_COLLABORATION.md
git commit -m "test(rag): corpus IO, vector store, multi-hop retriever (21 cases)

Corpus IO (8): JSONL roundtrip, malformed-line error with location,
blank-line skipping, count without parsing, missing-file safety,
chunk_text/chunk_id behavior.

Vector store (7): ingest count, upsert idempotency, empty-list safety,
top-k results, metadata propagation, relevance score in [0,1],
exclude_ids filtering.

Multi-hop retriever (6): single-hop result shape, cross-hop exclusion,
first-hop has no exclusions, all_chunks dedup defense-in-depth,
chunk_to_hop mapping, hop ordering preserved.

The vector store tests caught two Chroma 1.5+ behaviors that would
have shipped silently broken: (1) custom embedding functions are
ignored unless they conform to a server-side persistence protocol, and
(2) EphemeralClient instances share state across the process. Both
are documented in the module docstring."

echo "Commit 12: prompt registry + base agent"
git add app/agents/prompts.py app/agents/base.py
git commit -m "feat(agents): prompt registry and base agent ABC

Prompt registry:
- Stable keys like 'decomposition.system' that the meta-agent uses to
  identify worst-performing prompts and propose rewrites
- Decentralized definition (each agent registers its own at import time)
  with centralized lookup
- Baseline preserved separately from active so approved rewrites can be
  diffed against the original and rolled back
- Append-only: re-registering a key with different text raises, since
  that indicates a bug

Base agent (BaseAgent ABC):
- Common run loop: declare budget, build messages, call LLM, validate
  structured output against the agent's declared output_type, log
  PolicyViolation on schema/budget failures, return AgentOutput envelope
- Subclasses set name, output_type, default_budget; implement
  build_messages(); optionally override _validate_output for agent-
  specific invariants (e.g. dependency-graph integrity)
- json_mode: ClassVar toggle for narrative-only agents
- AgentExecutionError raised on any non-recoverable failure; orchestrator
  catches and decides whether to retry, skip, or abort"

echo "Commit 13: decomposition agent"
git add app/agents/decomposition.py app/agents/__init__.py
git commit -m "feat(agents): decomposition agent with typed sub-tasks and dep graph

Breaks ambiguous queries into typed sub-tasks (retrieval / computation /
synthesis / verification / other) with an explicit dependency graph the
orchestrator topologically orders.

Notable:
- LLM emits readable slug IDs (t1, t2, ...) which we re-stamp to
  canonical uuid hex IDs at validation time. Slugs in prompts, canonical
  IDs in the rest of the system. Preserves graph references via lockstep
  rewrite of depends_on.
- Cycle detection via DFS BEFORE re-stamping. Cycles are logical bugs in
  the LLM output (Pydantic doesn't know about the graph), so we detect
  explicitly and log schema_violation with raw_preview.
- Dangling depends_on (reference to a non-existent task) caught the same
  way.
- task_type='other' requires task_type_detail; SubTask validator enforces
  this and the prompt explicitly mentions the rule.
- System prompt is conservative — simple queries get one task, not five.
  Over-decomposition is penalized in eval scoring (unnecessary tool calls)."

echo "Commit 14: tests for prompt registry and decomposition (20 cases)"
git add tests/test_prompts.py tests/test_decomposition.py
git commit -m "test(agents): prompt registry semantics and decomposition agent

Prompt registry (10): roundtrip, idempotency, conflict rejection, get
unknown key, set/reset semantics, sorted keys, snapshot independence,
auto-registration on agent import.

Decomposition agent (10) with stub LLM:
- Happy path: simple query single-task, complex query 3-task dep graph
  with verified slug-to-canonical re-stamp, task_type='other' with detail
- Failure modes: dangling dependency, dependency cycle, invalid task
  type, 'other' without detail, malformed JSON twice
  (each logs schema_violation with raw_preview when applicable)
- Budget integration: declared on run, custom override

Tests confirm slug ids never leak past validation — assertions verify
canonical ids are 12 hex chars and depends_on refs use canonical ids."

echo "Commit 15: retrieval agent with multi-hop loop and citation linking"
git add app/agents/retrieval.py app/agents/__init__.py tests/test_retrieval_agent.py
git commit -m "feat(agents): retrieval agent with LLM-driven multi-hop loop

Wraps the MultiHopRetriever primitive with an LLM-driven loop that
decides what to query at each hop and when to stop.

- Hop 1 always uses the user's query verbatim
- Hop 2..N: LLM hop-planner picks the next query or signals stop
- Hard cap at MAX_HOPS=3 to control cost
- Synthesis step produces answer_draft + chunk_links over the union of
  retrieved chunks; orchestrator overwrites .hops with actual hop sequence
- Citation fabrication (chunk_link to a non-retrieved chunk_id) raises
  AgentExecutionError and logs schema_violation
- Hops < 2 logged as retrieval_undersourced policy violation, not a
  parse error — output is preserved for audit

Tests (3): happy path 2-hop with citations, fabricated citation caught,
undersourced flagged. Schema-level concerns covered by test_schema.py."

echo "Commit 16: critique, synthesis, and compression agents"
git add app/agents/critique.py app/agents/synthesis.py app/agents/compression.py app/agents/__init__.py tests/test_other_agents.py
git commit -m "feat(agents): critique, synthesis, and compression

Three remaining pipeline agents, all using the standard BaseAgent.run()
path:

- Critique reviews ONE target output per pass, emits per-claim
  confidence scores with spans plus span-level disagreements (per the
  brief: span-level, not whole-output). Span char offsets are half-open
  [start, end) into the target output.
- Synthesis pulls the latest retrieval and critique outputs from the
  shared context, resolves disagreements, produces final answer with
  sentence-level provenance map (text_span -> source_agent +
  source_output_id + citations).
- Compression triggers when an agent's context utilization crosses the
  threshold (BudgetManager.needs_compression). Lossless on tool
  outputs, citations, sub-task IDs, scores, policy violations; lossy
  only on narrative filler. Reports before/after token counts so the
  trace can audit compression ratio.

One happy-path test per agent — schema validation, budget tracking, and
policy violation logging are already covered by base + decomposition
tests, so we don't re-cover that surface here."

echo "Commit 17: four tools with explicit failure contracts"
git add app/tools/ tests/test_tools.py
git commit -m "feat(tools): web_search, code_exec, sql_lookup, self_reflect

Per the brief, every tool has a defined failure contract: ok / timeout /
empty / malformed_input / error. Tools never raise — Tool.call() wraps
the implementation, times the call, contains exceptions, and always
returns a structured ToolResult.

- web_search: stub returning canned structured results with URLs and
  relevance scores. Real production would swap in Tavily/Serper/Bing
  behind the same contract.
- code_exec: subprocess with hard timeout. Returns stdout/stderr/exit
  code. Timeouts surface as ToolStatus.TIMEOUT (retryable). NOT a
  security boundary — documented in Known Limitations.
- sql_lookup: in-memory SQLite with a small fixed schema (papers,
  benchmarks, paper_benchmark_results). Read-only enforcement: first
  token must be SELECT. Mutating queries return MALFORMED_INPUT.
- self_reflect: scans SharedContext.agent_outputs via an LLM call to
  find pairs of contradicting claims. Returns EMPTY when fewer than 2
  outputs exist (not an error — graceful degradation).

ToolResult.is_retryable is True for TIMEOUT and EMPTY; the orchestrator
uses this to decide whether re-calling with modified input makes sense.
Malformed input is NOT retryable — the agent must fix its call.

Tests (9): happy path + key failure mode per tool."

echo "Commit 18: orchestrator with dynamic routing, tool retry, compression"
git add app/orchestrator/ tests/test_orchestrator.py
git commit -m "feat(orchestrator): phase-based pipeline with dynamic routing

Owns the per-job pipeline: decompose -> retrieve (walk dependency graph)
-> critique -> synthesize. Phases are deterministic structure; the LLM
router picks agent/order within each phase, so different valid runs
over the same query produce different (but valid) sequences.

Routing (app/orchestrator/routing.py):
- pick_next_agent() asks the LLM to choose from a candidates list and
  emit a structured justification. Decision logged to ctx.routing_log.
- Hallucinated agent name -> fallback to first candidate, fallback
  recorded in justification (graceful degradation, no crash).
- Malformed-JSON router output -> same fallback path.

Tool dispatch (app/orchestrator/tool_dispatch.py):
- Up to 2 retries (per the brief) on retryable failures (TIMEOUT/EMPTY).
- Each invocation logged to ctx.tool_invocations with full input,
  output, latency, and retry_of link to the prior invocation.
- mark_invocation_accepted() records the agent's accept/reject decision.
- Retry planner is a Callable injected by the agent; planners modify
  input between attempts. Default behavior is no retry if no planner.

Compression-on-overflow:
- _run_with_compression catches BudgetViolation, invokes the
  compression agent on prior outputs, retries the failed agent ONCE.
- Second overflow -> AgentExecutionError (caller skips the phase).

Tests (6): routing decision logged with justification, fallback on
unknown agent name, tool retries with modified input, retry cap at
MAX_RETRIES, accept/reject recording, compression-on-overflow flow."

echo "Commit 19: persistence (Postgres) and SSE event bus"
git add app/persistence/ app/streaming/ tests/test_persistence.py tests/test_streaming.py pyproject.toml
git commit -m "feat(persistence,streaming): postgres models, repository, SSE event bus

Persistence (SQLAlchemy 2.x async):
- jobs: one row per submitted query
- job_traces: full SharedContext as JSON for replay
- eval_runs + eval_scores: reproducible eval persistence
- prompt_rewrites: meta-agent proposals with approval state
- JSONB on Postgres, JSON elsewhere via SQLAlchemy variant typing
- Schema management via Base.metadata.create_all on startup (no Alembic
  for assessment scope; documented in Known Limitations)

Repository functions are thin async wrappers over the models so the
orchestrator and API don't depend on the ORM directly.

SSE event bus:
- Per-job asyncio.Queue keyed by job_id
- Event types: agent_start/complete, tool_start/complete, routing,
  budget_status, final_answer, policy_violation, done, error
- Convenience publishers (emit_*) used by orchestrator
- consume() async-iterates until DONE/ERROR and cleans up
- No persistence here — JobTrace is source of truth; events are
  ephemeral for live observability

Orchestrator wired with optional event_bus: agent lifecycle, routing
decisions, final answer, and policy violations all surface through SSE.
JSON-mode agents do NOT stream tokens (raw JSON is not human-readable
mid-generation; documented in AI_COLLABORATION.md).

Tests (7): trace roundtrip, eval run lifecycle, prompt rewrite decision
flow, SSE publish/consume, drop-on-no-subscriber, DONE termination,
SSE data is valid JSON."

echo "Commit 20: FastAPI app with the five required endpoints"
git add app/api/ tests/test_api.py
git commit -m "feat(api): five required endpoints with documented error contract

POST /jobs                          submit query, return SSE stream
GET  /jobs/{job_id}/trace           full execution trace by job_id
GET  /eval/latest                   latest eval run summary
POST /prompt-rewrites/{id}/decide   human approval/rejection
POST /eval/rerun-failed             targeted re-eval on failed cases

Plus GET /healthz for ops.

Per the brief, error responses are { error_code, message, job_id? }.
ErrorResponse and request/response models live in app/api/schemas.py
so the API surface is decoupled from internal Pydantic models — a
refactor of an internal model can't silently change the API contract.

The pipeline runs in the foreground of the SSE request so events stream
as they happen (asyncio.create_task wrapping the orchestrator). A real
production setup would push to RQ/Celery and stream from a Redis pubsub;
in-process is honest for assessment scope.

Approving a prompt rewrite swaps the active prompt at runtime via
prompts.set_active_prompt(). Rejected rewrites leave the registry
unchanged. The baseline is preserved separately so we can roll back.

The /eval/rerun-failed endpoint correctly identifies cases below the
threshold; the actual re-evaluation execution is stubbed pending the
eval runner (next chunk).

Tests (10): healthz, trace roundtrip + 404, eval summary grouped by
category/dimension + 404, rewrite approve-swaps-active + 404 + 400 on
double-decide, rerun-failed case identification + 404."

echo "Commit 21: 15-case eval harness with multi-dimensional scoring"
git add app/eval/ scripts/run_eval.py tests/test_eval_scorer.py tests/test_api.py app/api/main.py
git commit -m "feat(eval): 15 test cases, hand-rolled multi-dimensional scoring, runner

15 test cases as Pydantic data (5 baseline / 5 ambiguous / 5 adversarial)
in app/eval/cases.py. Each case carries:
- query, expected_facts (for LLM-judge correctness scoring)
- expected_chunk_ids (for citation_accuracy)
- rationale (notes for the human reviewer)

Six scoring dimensions per the brief:
1. answer_correctness        LLM judge vs expected_facts (temp=0)
2. citation_accuracy         chunk_id overlap pure function
3. contradiction_resolution  LLM judge on synthesis resolution_notes
4. tool_selection_efficiency penalize rejected/excess calls
5. context_budget_compliance count of budget_overflow violations
6. critique_agreement_rate   mean claim confidence minus strong disagreements

Hand-rolled per the brief — no third-party eval framework. Mechanical
dimensions are pure functions of SharedContext; LLM-judge dimensions
use temperature=0 for reproducibility. Each score row carries the
job_id of the run that produced it for traceability.

Reproducibility:
- Each EvalRun stores prompt_snapshot at run time
- Re-running with the same prompts yields the same scores (modulo
  small LLM nondeterminism at temp=0)
- Re-running on the same inputs produces diff-able output via the
  /eval/latest summary endpoint

Targeted re-eval (rerun_failed_cases): identifies cases below
score_threshold on any dimension, re-runs only those, computes
delta_summary as mean per-dimension delta vs the base run.

/eval/rerun-failed endpoint wired to the runner; the prior stub
return is replaced. The earlier stub-era test was updated to mock the
runner since real execution requires a live LLM.

scripts/run_eval.py is the CLI entry point for full or subset runs.

Tests (11): citation full/partial/no-synthesis, tool efficiency
no-calls/rejected/excess, budget compliance no-violation/overflow,
critique agreement neutral/high-confidence, full case scoring smoke
test confirms all six dimensions return."

echo "Commit 22: meta-agent for prompt rewrite proposals"
git add app/eval/meta_agent.py app/eval/__init__.py tests/test_meta_agent.py
git commit -m "feat(eval): meta-agent proposes rewrites for worst-dimension prompts

Reads scores from a completed EvalRun, identifies the lowest-scoring
dimension whose mean is below 0.7, maps the dimension to the agent
most responsible (citation_accuracy -> retrieval, etc.), and proposes
a rewrite via an LLM call.

Proposals are persisted as PromptRewrite rows with status='pending'.
The approval flow is wired in the API: POST /prompt-rewrites/{id}/decide
calls prompts.set_active_prompt() on approval. Baseline is preserved
for rollback.

If the rewrite is identical to the baseline, or empty, or generated
malformed JSON twice — the proposal is rejected and no row is created.

The meta-agent is intentionally NOT in the StructuredPipelineOutput
discriminated union. It runs after eval, has a different lifetime,
and persists to its own table.

Tests (2): worst-dimension selection persists a rewrite proposal,
no-rewrite-when-all-dimensions-above-threshold returns None."

echo "Commit 23: Docker compose + README + Architecture + AI_COLLABORATION"
git add Dockerfile docker-compose.yml .dockerignore README.md ARCHITECTURE.md AI_COLLABORATION.md
git commit -m "docs: full README, architecture diagram, AI collaboration log; Docker

Dockerfile: multi-stage build (builder installs deps, runtime is slim).
Non-root user, healthcheck on /healthz.

docker-compose.yml: postgres + redis + chroma + api + worker + ingest.
Env-var-only configuration; no hardcoded credentials anywhere. ingest
is a one-shot service that populates Chroma from the JSONL snapshot;
api waits for service_completed_successfully so reviewers don't see
stale Chroma state on first boot.

README:
- 5-minute setup
- architecture text diagram
- per-agent decision-boundary table
- API endpoints with the brief's required error contract
- explicit 'what the self-improving loop does and does not do'
- detailed Known Limitations covering scope cuts (sub-task dispatch,
  hybrid routing, JSON-mode SSE), eval limitations (single-model
  judge bias, corpus assumptions), operational limitations (no
  Alembic, no auth, code_exec is not a security boundary, LLM
  nondeterminism)
- 'what I would build next' with prioritized roadmap

ARCHITECTURE.md: detailed pipeline + eval flow diagrams plus
project-wide invariants.

AI_COLLABORATION.md: rewritten as one coherent narrative in 9
sections (planning + 8 build phases). Each section: AI-assisted
vs. decisions I made vs. verification. Includes the bugs the test
suite caught (SDK-tenacity retry stacking, Chroma EF protocol,
EphemeralClient state sharing, my own confused first-draft comment
on slug remapping)."

echo ""
echo "All 23 commits laid down. Push with:"
echo "  git push -u origin main"
echo ""
echo "System complete. ~138 hermetic tests passing."
