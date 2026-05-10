"""Retrieval agent.

Wraps the MultiHopRetriever primitive with an LLM-driven loop that decides
what to query at each hop and when to stop. Produces a RetrievalOutput with
hops, an answer_draft, and chunk_links binding answer spans to chunk IDs.

Loop shape:
    hop 1: query = user's question (lightly normalized)
    hop 2..N: LLM picks the next query given prior chunks, or signals "done"
    final: LLM writes answer_draft + chunk_links over the union of retrieved chunks

Per the brief, retrieval must produce >= 2 hops. We enforce that as a logged
PolicyViolation (retrieval_undersourced), not a parse error — the agent's
output is preserved for audit.
"""

from __future__ import annotations

from typing import ClassVar

from app.agents.base import AgentExecutionError, BaseAgent
from app.agents.prompts import get_prompt, register_prompt
from app.context import (
    AgentOutput,
    PolicyViolation,
    RetrievalHop,
    RetrievalOutput,
    SharedContext,
    hash_prompt,
)
from app.llm import LLMMessage, MalformedJSONError
from app.rag import MultiHopRetriever, RetrievedChunk, VectorStore

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_HOP_PLANNER_SYSTEM = """You are the Retrieval Agent's hop planner.

You see the user's query and the chunks retrieved so far. Decide whether \
another retrieval hop would help, and if so, what to query next.

Rules:
- If the existing chunks already cover the query's information needs, return \
{"action": "stop"}.
- Otherwise, return {"action": "query", "query": "...", "rationale": "..."}.
- Each new query should target an UNDER-COVERED aspect of the original \
question, not rephrase what we already retrieved.
- We have a hard cap of 3 hops total. Aim for 2-3 unless the question is \
genuinely simple.

Return JSON ONLY."""

_HOP_PLANNER_USER = """Original user query: {user_query}

Hops completed so far: {hop_count}

Chunks retrieved (across all hops):
{chunks_summary}

Should we hop again? Output JSON with {{"action": "stop"}} or \
{{"action": "query", "query": "...", "rationale": "..."}}."""

_SYNTHESIS_SYSTEM = """You are the Retrieval Agent's draft writer.

Given the user's query and a set of retrieved chunks, write a draft answer \
and explicitly link each part of the answer back to the chunks that supported \
it.

Output schema (return JSON ONLY):
{
  "type": "retrieval",
  "answer_draft": "<full answer text>",
  "hops": [],
  "chunk_links": [
    {
      "chunk_id": "<chunk_id>",
      "answer_span": {"start": <int>, "end": <int>, "text": "<exact substring>"},
      "contribution_note": "<one-line explanation>"
    },
    ...
  ]
}

Rules:
- answer_span offsets are HALF-OPEN [start, end) into answer_draft, like \
Python slicing. The substring answer_draft[start:end] MUST equal the \"text\" \
field exactly.
- Every meaningful claim in the answer should have at least one chunk_link.
- Do NOT cite chunks that did not actually contribute. Penalty for fabrication.
- Leave \"hops\" as []. The orchestrator fills it in from the actual hop \
sequence.
- If retrieved chunks do not support an answer, say so honestly in \
answer_draft and produce zero chunk_links."""

_SYNTHESIS_USER = """User query: {user_query}

Retrieved chunks:
{chunks_full}

Write the draft answer with chunk_links. JSON ONLY."""


register_prompt("retrieval.hop_planner_system", _HOP_PLANNER_SYSTEM)
register_prompt("retrieval.hop_planner_user", _HOP_PLANNER_USER)
register_prompt("retrieval.synthesis_system", _SYNTHESIS_SYSTEM)
register_prompt("retrieval.synthesis_user", _SYNTHESIS_USER)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class RetrievalAgent(BaseAgent[RetrievalOutput]):
    name: ClassVar[str] = "retrieval"
    default_budget: ClassVar[int] = 6000  # bigger because chunks land in context
    output_type = RetrievalOutput

    MIN_HOPS: ClassVar[int] = 2
    MAX_HOPS: ClassVar[int] = 3

    def __init__(self, llm, budgets, vector_store: VectorStore, top_k_per_hop: int = 3):
        super().__init__(llm, budgets)
        self._store = vector_store
        self._top_k = top_k_per_hop

    def build_messages(self, ctx, **kwargs):
        # Not used directly — we override `run()` to drive the multi-hop loop.
        # build_messages is called for the synthesis step inside run().
        chunks: list[RetrievedChunk] = kwargs["chunks"]
        return [
            LLMMessage(role="system", content=get_prompt("retrieval.synthesis_system")),
            LLMMessage(
                role="user",
                content=get_prompt("retrieval.synthesis_user").format(
                    user_query=ctx.user_query,
                    chunks_full=_format_chunks_full(chunks),
                ),
            ),
        ]

    async def run(self, ctx: SharedContext, *, budget: int | None = None, **kwargs) -> AgentOutput:
        """Execute the multi-hop loop, then call the parent's _invoke_llm for
        the synthesis step that produces the final RetrievalOutput."""
        max_tokens = budget if budget is not None else self.default_budget
        self._budgets.declare(self.name, max_tokens)

        retriever = MultiHopRetriever(store=self._store, top_k_per_hop=self._top_k)

        # Hop 1: always run with the user's query verbatim.
        retriever.hop(ctx.user_query)

        # Subsequent hops: LLM-driven, up to MAX_HOPS total.
        while len(retriever.hops) < self.MAX_HOPS:
            decision = await self._plan_next_hop(ctx, retriever)
            if decision is None:
                break  # LLM said "stop" or returned undecidable response
            retriever.hop(decision)

        all_chunks = retriever.result().all_chunks

        # Synthesis: build a RetrievalOutput from the chunks via the parent
        # run path (validation + budget tracking + AgentOutput envelope).
        messages = self.build_messages(ctx, chunks=all_chunks)
        prompt_h = hash_prompt("\n\n".join(m.content for m in messages))
        agent_output = await self._invoke_llm(ctx, messages, prompt_h)

        # The LLM left `hops: []` — we overwrite with the actual hop sequence.
        retrieval_out: RetrievalOutput = agent_output.structured_output  # type: ignore[assignment]
        actual_hops = [
            RetrievalHop(
                hop_index=h.hop_index,
                query=h.query,
                retrieved_chunk_ids=[c.chunk_id for c in h.chunks],
            )
            for h in retriever.hops
        ]
        # Validate that chunk_links reference chunks we actually retrieved.
        valid_chunk_ids = {c.chunk_id for c in all_chunks}
        for link in retrieval_out.chunk_links:
            if link.chunk_id not in valid_chunk_ids:
                ctx.policy_violations.append(
                    PolicyViolation(
                        agent_name=self.name,
                        violation_type="schema_violation",
                        detail=f"chunk_link references unknown chunk_id {link.chunk_id!r}",
                        raw_preview=str(retrieval_out.model_dump())[:500],
                    )
                )
                raise AgentExecutionError(f"retrieval: fabricated chunk citation {link.chunk_id!r}")

        retrieval_out = retrieval_out.model_copy(update={"hops": actual_hops})

        # Compliance check: minimum hops. Logged, not raised.
        if not retrieval_out.is_compliant():
            ctx.policy_violations.append(
                PolicyViolation(
                    agent_name=self.name,
                    violation_type="retrieval_undersourced",
                    detail=(
                        f"retrieval produced only {len(retrieval_out.hops)} hop(s); "
                        f"brief requires >= {self.MIN_HOPS}"
                    ),
                )
            )

        return agent_output.model_copy(update={"structured_output": retrieval_out})

    async def _plan_next_hop(self, ctx: SharedContext, retriever: MultiHopRetriever) -> str | None:
        """Ask the LLM whether to continue and what to query next.

        Returns the next query string, or None if the LLM signals stop or
        the response is malformed (in which case we stop gracefully — no
        further hops).
        """
        messages = [
            LLMMessage(
                role="system",
                content=get_prompt("retrieval.hop_planner_system"),
            ),
            LLMMessage(
                role="user",
                content=get_prompt("retrieval.hop_planner_user").format(
                    user_query=ctx.user_query,
                    hop_count=len(retriever.hops),
                    chunks_summary=_format_chunks_summary(retriever.result().all_chunks),
                ),
            ),
        ]
        try:
            parsed, resp = await self._llm.complete_json(messages)
        except MalformedJSONError:
            return None  # Treat as "stop" — graceful degradation
        self._budgets.consume(self.name, resp.input_tokens + resp.output_tokens)

        action = parsed.get("action")
        if action == "stop":
            return None
        if action == "query":
            q = parsed.get("query")
            return q if isinstance(q, str) and q.strip() else None
        return None  # Unknown action -> stop

    def _render_content(self, structured) -> str:
        out: RetrievalOutput = structured
        return f"Retrieval: {len(out.hops)} hop(s), {len(out.chunk_links)} citation(s)"


# --------------------------------------------------------------------------- #
# Chunk formatting helpers
# --------------------------------------------------------------------------- #


def _format_chunks_summary(chunks: list[RetrievedChunk]) -> str:
    """Compact list for the hop planner — chunk_id + title + first ~30 words."""
    if not chunks:
        return "(none yet)"
    lines = []
    for c in chunks:
        title = c.metadata.get("title", "Untitled")
        snippet = " ".join(c.text.split()[:30])
        lines.append(f"- {c.chunk_id} | {title} | {snippet}...")
    return "\n".join(lines)


def _format_chunks_full(chunks: list[RetrievedChunk]) -> str:
    """Full chunk text for the synthesis step. Preserves chunk_id labeling so
    the LLM can cite by ID."""
    if not chunks:
        return "(no chunks retrieved)"
    sections = []
    for c in chunks:
        title = c.metadata.get("title", "Untitled")
        sections.append(f"--- {c.chunk_id} | {title} ---\n{c.text}")
    return "\n\n".join(sections)
