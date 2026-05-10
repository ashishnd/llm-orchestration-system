"""Synthesis agent.

Merges outputs from prior agents (typically retrieval + critique), resolves
contradictions the critique agent flagged, and produces a final answer with
a sentence-level provenance map.

The provenance map is the audit trail: each sentence in the final answer
maps back to the source agent and source chunks that contributed to it.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.agents.base import BaseAgent
from app.agents.prompts import get_prompt, register_prompt
from app.context import SharedContext, SynthesisOutput
from app.llm import LLMMessage

_SYSTEM = """You are the Synthesis Agent.

You receive: (1) a draft answer from the Retrieval Agent, (2) the chunks \
that were retrieved, (3) the Critique Agent's claim scores and \
disagreements. Your job is to produce a final answer that:

1. Resolves contradictions the critique flagged (don't ignore them — either \
incorporate the correction or explain in resolution_notes why you didn't).
2. Maps every sentence in the final answer back to (a) the source agent that \
contributed it and (b) the chunk citations that support it.

Provenance rules:
- Number sentences 0, 1, 2, ... in order. Provenance entries reference these \
indices.
- text_span uses HALF-OPEN [start, end) char offsets into final_answer.
- Each provenance entry has a source_agent (e.g. "retrieval", "critique") \
and source_output_id (an AgentOutput.id from the shared context).
- citations[] within each entry references the retrieved chunk_ids that \
support that specific sentence.

Return JSON ONLY:
{
  "type": "synthesis",
  "final_answer": "<full answer>",
  "provenance": [
    {
      "sentence_index": 0,
      "text_span": {"start": 0, "end": 42, "text": "<the sentence>"},
      "source_agent": "retrieval",
      "source_output_id": "<id>",
      "citations": [
        {"chunk_id": "<id>", "source_doc": "<doc>", "relevance_score": 0.8}
      ]
    }
  ],
  "resolution_notes": "<how contradictions were resolved, or null>"
}"""

_USER_TEMPLATE = """Original user query: {user_query}

Retrieval draft (output_id={retrieval_id}):
\"\"\"{retrieval_content}\"\"\"

Critique disagreements ({n_disagreements}):
{disagreements_text}

Produce the final synthesis with provenance. JSON ONLY."""

register_prompt("synthesis.system", _SYSTEM)
register_prompt("synthesis.user_template", _USER_TEMPLATE)


class SynthesisAgent(BaseAgent[SynthesisOutput]):
    name: ClassVar[str] = "synthesis"
    default_budget: ClassVar[int] = 4000
    output_type = SynthesisOutput

    def build_messages(self, ctx: SharedContext, **kwargs: Any) -> list[LLMMessage]:
        # Pull the latest retrieval and critique outputs from the shared context.
        retrieval_outputs = ctx.outputs_by_agent("retrieval")
        critique_outputs = ctx.outputs_by_agent("critique")
        if not retrieval_outputs:
            raise ValueError("synthesis requires at least one retrieval output in context")

        latest_retrieval = retrieval_outputs[-1]
        disagreements = []
        for c in critique_outputs:
            if c.structured_output is not None:
                disagreements.extend(c.structured_output.disagreements)  # type: ignore[union-attr]

        disagreements_text = (
            "\n".join(
                f"- [{d.span.text!r}] (target={d.target_agent}, conf={d.confidence}): {d.reason}"
                for d in disagreements
            )
            or "(none)"
        )

        return [
            LLMMessage(role="system", content=get_prompt("synthesis.system")),
            LLMMessage(
                role="user",
                content=get_prompt("synthesis.user_template").format(
                    user_query=ctx.user_query,
                    retrieval_id=latest_retrieval.id,
                    retrieval_content=latest_retrieval.content,
                    n_disagreements=len(disagreements),
                    disagreements_text=disagreements_text,
                ),
            ),
        ]

    def _render_content(self, structured) -> str:
        s: SynthesisOutput = structured
        return f"Synthesis: final answer with {len(s.provenance)} sentence-level citation(s)"
