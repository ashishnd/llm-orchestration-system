"""Critique agent.

Reviews ONE target agent output per pass. For each meaningful claim, emits
a structured confidence score with a span. For specific spans it disagrees
with, emits a CritiqueDisagreement with the disputed text and a reason.

The brief is explicit: critique is span-level, not whole-output. The
prompt enforces this.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.agents.base import BaseAgent
from app.agents.prompts import get_prompt, register_prompt
from app.context import AgentOutput, CritiqueOutput, SharedContext
from app.llm import LLMMessage

_SYSTEM = """You are the Critique Agent.

Your job is to review ONE specific output from another agent and produce \
structured per-claim confidence scores and per-span disagreements.

Rules:
- For each meaningful claim in the target output, emit a claim_score with \
the exact character span [start, end) of the claim and a confidence in [0, 1].
- For spans you actively disagree with, ALSO emit a disagreement entry \
with the span, your reason, and an optional suggested_correction.
- Spans use HALF-OPEN [start, end) char offsets into the target output text. \
The substring target[start:end] MUST equal the span's "text" field exactly.
- Do NOT critique the output as a whole — only specific spans.
- If the target output is correct and well-supported, emit claim_scores with \
high confidences and an empty disagreements list.

Return JSON ONLY, with this shape:
{
  "type": "critique",
  "target_output_id": "<the target's AgentOutput.id>",
  "claim_scores": [
    {
      "claim_span": {"start": <int>, "end": <int>, "text": "<exact substring>"},
      "confidence": <float 0..1>,
      "note": "<optional brief reason>"
    }
  ],
  "disagreements": [
    {
      "span": {"start": <int>, "end": <int>, "text": "<exact substring>"},
      "target_agent": "<target's agent_name>",
      "confidence": <float 0..1>,
      "reason": "<why you disagree>",
      "suggested_correction": "<optional>"
    }
  ]
}"""

_USER_TEMPLATE = """Original user query: {user_query}

Target output to critique:
- agent_name: {target_agent}
- output_id: {target_output_id}
- content:
\"\"\"{target_content}\"\"\"

Produce claim_scores and disagreements per the schema. JSON ONLY."""

register_prompt("critique.system", _SYSTEM)
register_prompt("critique.user_template", _USER_TEMPLATE)


class CritiqueAgent(BaseAgent[CritiqueOutput]):
    name: ClassVar[str] = "critique"
    default_budget: ClassVar[int] = 3000
    output_type = CritiqueOutput

    def build_messages(self, ctx: SharedContext, **kwargs: Any) -> list[LLMMessage]:
        target: AgentOutput = kwargs["target"]
        return [
            LLMMessage(role="system", content=get_prompt("critique.system")),
            LLMMessage(
                role="user",
                content=get_prompt("critique.user_template").format(
                    user_query=ctx.user_query,
                    target_agent=target.agent_name,
                    target_output_id=target.id,
                    target_content=target.content,
                ),
            ),
        ]

    def _render_content(self, structured) -> str:
        c: CritiqueOutput = structured
        return (
            f"Critique of {c.target_output_id}: "
            f"{len(c.claim_scores)} claim(s), {len(c.disagreements)} disagreement(s)"
        )
