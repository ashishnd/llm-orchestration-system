"""Shared context schema.

The brief requires that all inter-agent communication pass through a single
typed context object. Agents read from and write to this object via the
orchestrator; they do not call each other directly.

Project-wide conventions:
- All character offsets are HALF-OPEN: [start, end), like Python slicing.
- All confidence values are floats in [0.0, 1.0].
- Every structured pipeline output carries a `type` discriminator.

Design notes:
- Pydantic v2 models for runtime validation. Schema violations fail loudly
  rather than silently producing weird downstream behavior.
- A single `Span` primitive is reused everywhere text is located. The earlier
  draft had three slightly different span shapes; that was a footgun.
- Tagged union (StructuredPipelineOutput) over the five in-pipeline agent
  outputs. The meta-agent (prompt rewriter) is intentionally NOT in this
  union — it runs after the eval, persists to its own table, and shouldn't
  share an envelope with per-job pipeline outputs.
- Compliance invariants (e.g. retrieval must do >= 2 hops) are enforced at
  the orchestrator level via helper methods like `RetrievalOutput.is_compliant()`,
  NOT as schema constraints. This way a non-compliant agent emits a logged
  PolicyViolation instead of a parse error — the system catches and audits
  the failure rather than rejecting it silently.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


class Span(BaseModel):
    """Half-open [start, end) char offsets into some target string.

    `text` carries the actual substring when emitted by an LLM, so traces are
    self-describing and don't require resolving offsets back to source strings
    to be readable. `text` is optional because some span shapes (e.g. computed
    spans in the eval scorer) may not need to materialize it.
    """

    start: int
    end: int
    text: str | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Span:
        if self.start < 0:
            raise ValueError(f"Span.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"Span.end ({self.end}) must be >= Span.start ({self.start}) "
                "(half-open intervals; equal start/end means empty span)"
            )
        return self


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Citations and provenance
# --------------------------------------------------------------------------- #


class CitationRef(BaseModel):
    """Pointer from an agent claim back to a retrieved chunk."""

    chunk_id: str
    source_doc: str
    claim_span: Span | None = None  # span within the SOURCE CHUNK text
    relevance_score: float | None = None


class ProvenanceEntry(BaseModel):
    """One sentence in the final answer mapped back to its source agent and chunks."""

    sentence_index: int  # 0-indexed sentence position in final_answer
    text_span: Span  # span within the FINAL ANSWER string
    source_agent: str
    source_output_id: str
    citations: list[CitationRef] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Sub-tasks (decomposition output)
# --------------------------------------------------------------------------- #


class SubTask(BaseModel):
    """A typed sub-task produced by the decomposition agent."""

    id: str = Field(default_factory=_new_id)
    description: str
    task_type: Literal["retrieval", "computation", "synthesis", "verification", "other"]
    task_type_detail: str | None = None  # required iff task_type == "other"
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    assigned_agent: str | None = None

    @model_validator(mode="after")
    def _other_requires_detail(self) -> SubTask:
        if self.task_type == "other" and not self.task_type_detail:
            raise ValueError("SubTask.task_type_detail is required when task_type == 'other'")
        return self


# --------------------------------------------------------------------------- #
# Critique
# --------------------------------------------------------------------------- #


class ClaimScore(BaseModel):
    """Structured confidence score for one claim within a target output."""

    claim_span: Span  # span within the target agent's output text
    confidence: float = Field(ge=0.0, le=1.0)
    note: str | None = None


class CritiqueDisagreement(BaseModel):
    """A specific span the critique agent flags. Span-level, not whole-output."""

    span: Span  # span within the target agent's output text
    target_agent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggested_correction: str | None = None


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


class RetrievalHop(BaseModel):
    """One hop in multi-hop retrieval. The brief requires >= 2 hops; that
    invariant is enforced at the orchestrator level (see is_compliant)."""

    hop_index: int
    query: str
    retrieved_chunk_ids: list[str]
    rationale: str | None = None


class AnswerChunkLink(BaseModel):
    """Which chunk supported which span of the retrieval agent's draft answer."""

    chunk_id: str
    answer_span: Span  # span within RetrievalOutput.answer_draft
    contribution_note: str | None = None


# --------------------------------------------------------------------------- #
# Five in-pipeline structured outputs (discriminated union)
# --------------------------------------------------------------------------- #


class DecompositionOutput(BaseModel):
    type: Literal["decomposition"] = "decomposition"
    sub_tasks: list[SubTask]
    rationale: str | None = None


class RetrievalOutput(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    answer_draft: str
    hops: list[RetrievalHop]
    chunk_links: list[AnswerChunkLink]

    def is_compliant(self) -> bool:
        """Brief-mandated invariant: at least 2 retrieval hops before forming
        an answer. Non-compliance is logged as a PolicyViolation at the
        orchestrator level rather than enforced here, so we have an audit
        trail of the agent attempting to short-cut multi-hop reasoning."""
        return len(self.hops) >= 2


class CritiqueOutput(BaseModel):
    type: Literal["critique"] = "critique"
    target_output_id: str  # the AgentOutput.id being critiqued (one pass = one target)
    claim_scores: list[ClaimScore]
    disagreements: list[CritiqueDisagreement]


class SynthesisOutput(BaseModel):
    type: Literal["synthesis"] = "synthesis"
    final_answer: str
    provenance: list[ProvenanceEntry]  # ordered by sentence_index
    resolution_notes: str | None = None  # how contradictions were resolved


class PreservedRef(BaseModel):
    """A reference preserved losslessly through compression."""

    ref_type: Literal["tool_invocation", "agent_output", "citation", "subtask"]
    ref_id: str


class CompressionOutput(BaseModel):
    """Output of the compression agent.

    Tracks both `original_token_count` and `compressed_token_count` so the
    trace can audit the compression ratio and the eval scorer can attribute
    context-budget compliance correctly.
    """

    type: Literal["compression"] = "compression"
    lossy_summary: str
    preserved_refs: list[PreservedRef]
    original_token_count: int = Field(ge=0)
    compressed_token_count: int = Field(ge=0)
    notes: str | None = None


StructuredPipelineOutput = Annotated[
    DecompositionOutput | RetrievalOutput | CritiqueOutput | SynthesisOutput | CompressionOutput,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# Meta-agent output (separate from in-pipeline union; persists to its own table)
# --------------------------------------------------------------------------- #


class MetaRewriteOutput(BaseModel):
    """One proposed prompt rewrite from the meta-agent. Stored in
    prompt_versions / rewrite_proposals — not in AgentOutput."""

    eval_run_id: str
    target_prompt_key: str  # e.g. "orchestrator.system", "retrieval.agent"
    old_prompt: str
    proposed_prompt: str
    structured_diff: str  # unified diff string
    rationale: str
    worst_dimension: str | None = None
    worst_case_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tool invocations
# --------------------------------------------------------------------------- #


class ToolInvocation(BaseModel):
    """One tool call. Logged with full input/output for reproducibility."""

    id: str = Field(default_factory=_new_id)
    tool_name: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float | None = None
    accepted_by_agent: bool | None = None  # set after agent reviews the result
    retry_of: str | None = None  # ID of the previous invocation if this is a retry
    timestamp: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Agent output envelope
# --------------------------------------------------------------------------- #


class AgentOutput(BaseModel):
    """One agent turn. The common envelope around every pipeline agent's output.

    `content` is the human-readable stream/log line (used for SSE).
    `structured_output` is the parsed and validated typed payload.
    """

    id: str = Field(default_factory=_new_id)
    agent_name: str
    content: str
    structured_output: StructuredPipelineOutput | None = None
    citations: list[CitationRef] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_hash: str | None = None  # SHA-256 of rendered prompt; reproducibility
    timestamp: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Policy violations
# --------------------------------------------------------------------------- #


_RAW_PREVIEW_MAX = 500


class PolicyViolation(BaseModel):
    """An agent did something that violated a hard rule. Logged, not silently fixed."""

    agent_name: str
    violation_type: Literal[
        "budget_overflow",
        "schema_violation",
        "direct_agent_call",
        "retrieval_undersourced",
    ]
    detail: str
    raw_preview: str | None = None  # truncated raw model output for diagnosis
    timestamp: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _cap_raw_preview(self) -> PolicyViolation:
        if self.raw_preview is not None and len(self.raw_preview) > _RAW_PREVIEW_MAX:
            self.raw_preview = self.raw_preview[:_RAW_PREVIEW_MAX] + "...[truncated]"
        return self


# --------------------------------------------------------------------------- #
# Routing decisions
# --------------------------------------------------------------------------- #


class RoutingDecision(BaseModel):
    """The orchestrator's reasoning for the next agent to invoke."""

    id: str = Field(default_factory=_new_id)
    next_agent: str | None  # None when the pipeline is finished
    justification: str
    candidates_considered: list[str]
    timestamp: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Shared context (the single object every agent reads/writes via the orchestrator)
# --------------------------------------------------------------------------- #


class SharedContext(BaseModel):
    """The single object every agent reads from and writes to via the orchestrator.

    Intentionally a single growing record per job because critique and
    synthesis need to see the full history. Budget management ensures it
    doesn't grow unboundedly. Provenance is sourced from the synthesis
    output, not duplicated here, to keep it diff-able for regression tests.
    """

    job_id: str = Field(default_factory=_new_id)
    user_query: str
    sub_tasks: list[SubTask] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    disagreements: list[CritiqueDisagreement] = Field(default_factory=list)
    routing_log: list[RoutingDecision] = Field(default_factory=list)
    policy_violations: list[PolicyViolation] = Field(default_factory=list)
    final_answer: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    # ---- convenience accessors ----

    def get_output(self, output_id: str) -> AgentOutput | None:
        return next((o for o in self.agent_outputs if o.id == output_id), None)

    def outputs_by_agent(self, agent_name: str) -> list[AgentOutput]:
        return [o for o in self.agent_outputs if o.agent_name == agent_name]

    def get_subtask(self, task_id: str) -> SubTask | None:
        return next((t for t in self.sub_tasks if t.id == task_id), None)

    def ready_subtasks(self) -> list[SubTask]:
        """Sub-tasks whose dependencies are all complete and that haven't run yet."""
        completed = {t.id for t in self.sub_tasks if t.status == TaskStatus.COMPLETE}
        return [
            t
            for t in self.sub_tasks
            if t.status == TaskStatus.PENDING and all(d in completed for d in t.depends_on)
        ]

    def latest_synthesis(self) -> SynthesisOutput | None:
        """The most recent synthesis output, if any. Source of truth for provenance."""
        for output in reversed(self.agent_outputs):
            if isinstance(output.structured_output, SynthesisOutput):
                return output.structured_output
        return None


def hash_prompt(prompt: str) -> str:
    """SHA-256 of a rendered prompt; used for reproducibility and prompt versioning."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
