"""Dynamic routing.

The orchestrator asks the LLM to pick the next agent from a list of valid
candidates given the current context state. This is the routing decision
the brief requires; we log every decision with its justification.

Routing is not pure free-form — pipeline phases give structure, but within
each phase the LLM picks the next agent and order. Different runs over the
same query produce different (but valid) agent sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.context import RoutingDecision, SharedContext
from app.llm import LLMClient, LLMMessage, MalformedJSONError

_ROUTER_SYSTEM = """You are the orchestrator's router. Given the current state of \
a multi-agent pipeline, pick the next agent to invoke from a provided list of \
candidates.

Return JSON ONLY: {"next_agent": "<name>", "justification": "<why>"}.

If no further agent should run (pipeline is complete), return \
{"next_agent": null, "justification": "<why complete>"}.

Use only agent names from the candidates list. Justify in 1-2 sentences."""


@dataclass
class RouterContext:
    user_query: str
    completed_agents: list[str]
    candidates: list[str]
    last_agent_summary: str | None  # short content from the last agent's output
    open_subtasks: int
    open_disagreements: int


async def pick_next_agent(
    llm: LLMClient,
    ctx: SharedContext,
    candidates: list[str],
) -> RoutingDecision:
    """Pick the next agent. Logs the decision to ctx.routing_log and returns it.

    On LLM malformed-JSON, falls back to the first candidate (graceful
    degradation — the routing log records the fallback as the justification).
    """
    rctx = _build_router_context(ctx, candidates)
    user = (
        f"User query: {rctx.user_query}\n"
        f"Completed agents (in order): {rctx.completed_agents}\n"
        f"Open sub-tasks: {rctx.open_subtasks}\n"
        f"Open disagreements: {rctx.open_disagreements}\n"
        f"Last agent output summary: {rctx.last_agent_summary or '(none)'}\n"
        f"Candidates for next agent: {rctx.candidates}\n"
        "Pick the next agent or signal completion. JSON ONLY."
    )

    try:
        parsed, _ = await llm.complete_json(
            [
                LLMMessage(role="system", content=_ROUTER_SYSTEM),
                LLMMessage(role="user", content=user),
            ]
        )
        next_agent = parsed.get("next_agent")
        if isinstance(next_agent, str) and next_agent not in candidates:
            # LLM hallucinated an agent name — fall back to first candidate
            decision = RoutingDecision(
                next_agent=candidates[0] if candidates else None,
                justification=(
                    f"router proposed unknown agent {next_agent!r}; falling back "
                    f"to first candidate"
                ),
                candidates_considered=candidates,
            )
        else:
            decision = RoutingDecision(
                next_agent=next_agent if isinstance(next_agent, str) else None,
                justification=str(parsed.get("justification", "")),
                candidates_considered=candidates,
            )
    except MalformedJSONError:
        decision = RoutingDecision(
            next_agent=candidates[0] if candidates else None,
            justification="router produced malformed JSON twice; fallback to first candidate",
            candidates_considered=candidates,
        )

    ctx.routing_log.append(decision)
    return decision


def _build_router_context(ctx: SharedContext, candidates: list[str]) -> RouterContext:
    completed = [o.agent_name for o in ctx.agent_outputs]
    last_summary = ctx.agent_outputs[-1].content[:200] if ctx.agent_outputs else None
    open_disagreements = sum(
        len(c.structured_output.disagreements)  # type: ignore[union-attr]
        for c in ctx.outputs_by_agent("critique")
        if c.structured_output is not None
    )
    return RouterContext(
        user_query=ctx.user_query,
        completed_agents=completed,
        candidates=candidates,
        last_agent_summary=last_summary,
        open_subtasks=len(ctx.ready_subtasks()),
        open_disagreements=open_disagreements,
    )
