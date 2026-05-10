"""Self-reflection tool.

The brief: the agent can re-read its own previous outputs within the
session and identify contradictions.

Implementation: scans `SharedContext.agent_outputs` for the calling
agent (or a target agent name passed as input), uses an LLM call to
detect contradictions across the outputs, returns a structured list
of pairs of conflicting claims.

We bind the SharedContext via constructor injection rather than
passing it through `_execute` — the orchestrator builds one tool
instance per job and the context is the job's context.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.context import SharedContext
from app.llm import LLMClient, LLMMessage, MalformedJSONError
from app.tools.base import Tool, ToolResult, ToolStatus

_SYSTEM = """You scan a list of prior agent outputs from this session and \
identify pairs that contradict each other. Return JSON ONLY:

{
  "contradictions": [
    {
      "output_id_a": "...",
      "output_id_b": "...",
      "claim_a": "<exact text from a>",
      "claim_b": "<exact text from b>",
      "explanation": "why these conflict"
    }
  ]
}

Empty contradictions list is fine if you find none. Don't invent conflicts."""


class SelfReflectionTool(Tool):
    name: ClassVar[str] = "self_reflect"
    default_timeout_s: ClassVar[float] = 10.0

    def __init__(self, llm: LLMClient, ctx: SharedContext) -> None:
        self._llm = llm
        self._ctx = ctx

    async def _execute(
        self,
        target_agent: str | None = None,
        **_: Any,
    ) -> ToolResult:
        outputs = (
            self._ctx.outputs_by_agent(target_agent) if target_agent else self._ctx.agent_outputs
        )
        if len(outputs) < 2:
            return ToolResult(
                status=ToolStatus.EMPTY,
                payload={
                    "contradictions": [],
                    "reason": "fewer than 2 outputs to compare",
                },
            )

        formatted = "\n\n".join(f"--- {o.id} ({o.agent_name}) ---\n{o.content}" for o in outputs)
        try:
            parsed, _ = await self._llm.complete_json(
                [
                    LLMMessage(role="system", content=_SYSTEM),
                    LLMMessage(role="user", content=f"Outputs to scan:\n{formatted}"),
                ]
            )
        except MalformedJSONError as e:
            raise ValueError(
                f"self_reflect LLM call returned malformed JSON: {e.raw_first[:120]}"
            ) from e

        contradictions = parsed.get("contradictions", [])
        if not isinstance(contradictions, list):
            raise ValueError("self_reflect: LLM returned non-list contradictions field")

        return ToolResult(
            status=ToolStatus.OK,
            payload={
                "contradictions": contradictions,
                "outputs_scanned": len(outputs),
            },
        )
