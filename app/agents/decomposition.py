"""Decomposition agent.

Breaks an ambiguous user query into typed sub-tasks with an explicit
dependency graph. Per the brief: dependent sub-tasks must not execute
until their dependencies resolve — so this agent must produce IDs and
`depends_on` references the orchestrator can topologically order.

Design notes:
- The agent emits sub-task IDs of its own choosing (short slugs like
  "t1", "t2", ...) and references them in `depends_on`. We then re-stamp
  with our hash IDs at validation time so downstream code uses canonical
  IDs while the LLM works with readable ones. This keeps prompts simple
  and IDs stable.
- The prompt encourages caution: simple queries get 1-2 tasks, not 5.
  Over-decomposition wastes tool calls (penalized in eval scoring) and
  degrades synthesis quality (more pieces to reconcile).
- `task_type="other"` requires a `task_type_detail`. The prompt explicitly
  mentions this rule so the LLM doesn't emit "other" without explanation.
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

from app.agents.base import AgentExecutionError, BaseAgent
from app.agents.prompts import register_prompt
from app.context import (
    DecompositionOutput,
    PolicyViolation,
    SharedContext,
    SubTask,
)
from app.llm import LLMMessage

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_SYSTEM = """You are the Decomposition Agent in a multi-agent retrieval system.

Your job: break the user's query into a small number of typed sub-tasks with \
explicit dependencies between them. The system will execute sub-tasks in \
topological order — so dependencies matter.

CRITICAL RULES:
1. Be conservative. A simple, single-fact query needs ONE sub-task, not five.
   Only decompose when the query genuinely requires multiple distinct steps.
2. Each sub-task has an `id` (short slug like "t1", "t2", ...), a `description`,
   a `task_type`, and a `depends_on` list of prior task ids.
3. Allowed task_type values: "retrieval", "computation", "synthesis", \
"verification", or "other". If you use "other", you MUST include a \
`task_type_detail` field explaining what it is.
4. A "synthesis" task should depend on the "retrieval" tasks whose results \
it consumes. A "computation" task should depend on whatever produces the \
data it computes on.
5. If the query has a single, well-defined retrieval intent, emit ONE \
retrieval task with no dependencies. Do not invent verification or \
synthesis steps for the sake of it.

Return JSON ONLY, with this shape:
{
  "type": "decomposition",
  "sub_tasks": [
    {
      "id": "t1",
      "description": "...",
      "task_type": "retrieval",
      "depends_on": []
    },
    ...
  ],
  "rationale": "one or two sentences explaining your decomposition"
}
"""

_USER_TEMPLATE = """User query: {query}

Decompose this query into the minimum sub-tasks needed to answer it well. \
Remember: most queries are simpler than they look. Output JSON only."""

register_prompt("decomposition.system", _SYSTEM)
register_prompt("decomposition.user_template", _USER_TEMPLATE)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class DecompositionAgent(BaseAgent[DecompositionOutput]):
    name: ClassVar[str] = "decomposition"
    default_budget: ClassVar[int] = 2000
    output_type = DecompositionOutput

    def build_messages(self, ctx: SharedContext, **kwargs: Any) -> list[LLMMessage]:
        from app.agents.prompts import get_prompt

        return [
            LLMMessage(role="system", content=get_prompt(self.name + ".system")),
            LLMMessage(
                role="user",
                content=get_prompt(self.name + ".user_template").format(query=ctx.user_query),
            ),
        ]

    def _validate_output(self, ctx, parsed, raw):
        """Validate AND re-stamp slug IDs with canonical IDs.

        The LLM emits readable slug IDs ("t1", "t2") in `id` and `depends_on`.
        After validation, we map these to fresh canonical IDs (uuid4 hex
        prefixes) so the rest of the system uses stable IDs without LLM
        influence. Cycles and dangling refs in `depends_on` would be logical
        bugs — we detect and log them as schema_violations before re-stamping.
        """
        # Run standard Pydantic validation first. Catches missing fields,
        # bad task_type, conditional task_type_detail, etc.
        validated = super()._validate_output(ctx, parsed, raw)
        if validated is None:
            return None

        decomp: DecompositionOutput = validated  # type: ignore[assignment]

        # Detect dependency cycles BEFORE re-stamping. Cycles are a logical
        # error in the LLM's output, not a schema-shape error.
        if _has_cycle(decomp.sub_tasks):
            ctx.policy_violations.append(
                PolicyViolation(
                    agent_name=self.name,
                    violation_type="schema_violation",
                    detail="Decomposition produced a dependency cycle",
                    raw_preview=raw,
                )
            )
            raise AgentExecutionError("decomposition: dependency cycle detected")

        # Detect dangling depends_on (references a non-existent task).
        all_slugs = {t.id for t in decomp.sub_tasks}
        for t in decomp.sub_tasks:
            for dep in t.depends_on:
                if dep not in all_slugs:
                    ctx.policy_violations.append(
                        PolicyViolation(
                            agent_name=self.name,
                            violation_type="schema_violation",
                            detail=(f"Sub-task {t.id!r} depends on unknown task {dep!r}"),
                            raw_preview=raw,
                        )
                    )
                    raise AgentExecutionError(f"decomposition: dangling dependency {dep!r}")

        # Re-stamp: generate fresh canonical IDs and rewrite depends_on in
        # lockstep so the graph stays consistent.
        slug_to_canonical = {t.id: uuid4().hex[:12] for t in decomp.sub_tasks}
        rewritten: list[SubTask] = [
            t.model_copy(
                update={
                    "id": slug_to_canonical[t.id],
                    "depends_on": [slug_to_canonical[d] for d in t.depends_on],
                }
            )
            for t in decomp.sub_tasks
        ]
        return decomp.model_copy(update={"sub_tasks": rewritten})

    def _render_content(self, structured) -> str:
        decomp: DecompositionOutput = structured
        return f"Decomposed into {len(decomp.sub_tasks)} sub-task(s): " + ", ".join(
            f"{t.task_type}({t.description[:40]})" for t in decomp.sub_tasks
        )


# --------------------------------------------------------------------------- #
# Cycle detection
# --------------------------------------------------------------------------- #


def _has_cycle(tasks: list[SubTask]) -> bool:
    """Standard DFS cycle detection on the dependency graph."""
    # Map by id for O(1) lookup
    by_id = {t.id: t for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(by_id, WHITE)

    def dfs(node_id: str) -> bool:
        color[node_id] = GRAY
        for dep in by_id[node_id].depends_on:
            if dep not in by_id:
                # Dangling dep — handled separately. Don't treat as cycle.
                continue
            if color[dep] == GRAY:
                return True
            if color[dep] == WHITE and dfs(dep):
                return True
        color[node_id] = BLACK
        return False

    return any(color[tid] == WHITE and dfs(tid) for tid in by_id)
