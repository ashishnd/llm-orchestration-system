"""Compression agent.

Triggered when an agent's context utilization crosses the compression
threshold (BudgetManager.needs_compression). The brief: "lossless for
structured data (tool outputs, scores, citations), lossy only for
conversational filler."

We produce both before/after token counts so the trace can audit
compression ratio.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.agents.base import BaseAgent
from app.agents.prompts import get_prompt, register_prompt
from app.context import CompressionOutput, SharedContext
from app.llm import LLMMessage

_SYSTEM = """You are the Compression Agent.

Given older context that is taking up budget, produce a compressed version \
that preserves all structured/factual content while compressing conversational \
filler.

Lossless preservation REQUIRED for:
- Tool invocation results (input, output, error, latency)
- Critique claim scores and disagreements
- Citations / chunk references
- Sub-task IDs and their statuses
- Policy violations

Lossy compression ALLOWED for:
- Verbose narrative / explanation text
- Restated user query
- Agent-internal "thinking" prose
- Repetition

Return JSON ONLY:
{
  "type": "compression",
  "lossy_summary": "<the compressed narrative summary>",
  "preserved_refs": [
    {"ref_type": "tool_invocation", "ref_id": "<id>"},
    {"ref_type": "agent_output", "ref_id": "<id>"},
    {"ref_type": "citation", "ref_id": "<chunk_id>"},
    {"ref_type": "subtask", "ref_id": "<id>"}
  ],
  "original_token_count": <int>,
  "compressed_token_count": <int>,
  "notes": "<optional compression notes>"
}"""

_USER_TEMPLATE = """Original token count (caller-supplied): {original_tokens}

Context to compress:
\"\"\"{context_text}\"\"\"

Compress per the rules. JSON ONLY."""

register_prompt("compression.system", _SYSTEM)
register_prompt("compression.user_template", _USER_TEMPLATE)


class CompressionAgent(BaseAgent[CompressionOutput]):
    name: ClassVar[str] = "compression"
    default_budget: ClassVar[int] = 3000
    output_type = CompressionOutput

    def build_messages(self, ctx: SharedContext, **kwargs: Any) -> list[LLMMessage]:
        context_text: str = kwargs["context_text"]
        original_tokens: int = kwargs.get("original_tokens", self._llm.count_tokens(context_text))
        return [
            LLMMessage(role="system", content=get_prompt("compression.system")),
            LLMMessage(
                role="user",
                content=get_prompt("compression.user_template").format(
                    original_tokens=original_tokens,
                    context_text=context_text,
                ),
            ),
        ]

    def _render_content(self, structured) -> str:
        c: CompressionOutput = structured
        ratio = c.compressed_token_count / c.original_token_count if c.original_token_count else 0
        return (
            f"Compression: {c.original_token_count} -> {c.compressed_token_count} "
            f"tokens ({ratio:.0%}), {len(c.preserved_refs)} ref(s) preserved"
        )
