"""Orchestrator tests — focused on the integration concerns:
routing decisions logged, tool retry with modified input, compression-on-
overflow, and end-to-end phase dispatch."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context import (
    AgentOutput,
    BudgetViolation,
    SharedContext,
    ToolInvocation,
)
from app.llm.client import LLMResponse
from app.orchestrator import (
    MAX_RETRIES,
    dispatch_tool,
    mark_invocation_accepted,
    pick_next_agent,
)
from app.tools import Tool, ToolResult, ToolStatus

# --------------------------------------------------------------------------- #
# Minimal stub LLM driving routing tests
# --------------------------------------------------------------------------- #


class StubLLM:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)

    def count_tokens(self, t):
        return max(1, len(t) // 4)

    def count_message_tokens(self, msgs):
        return sum(self.count_tokens(m.content) + 4 for m in msgs)

    async def complete_json(self, messages, **kwargs):
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        resp = LLMResponse(text=json.dumps(nxt), input_tokens=10, output_tokens=20, model="stub")
        return nxt, resp


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_routing_decision_logged_with_justification():
    """Every routing decision goes into ctx.routing_log with its justification."""
    ctx = SharedContext(user_query="anything")
    llm = StubLLM([{"next_agent": "decomposition", "justification": "phase 1 always starts here"}])
    decision = await pick_next_agent(llm, ctx, candidates=["decomposition"])
    assert decision.next_agent == "decomposition"
    assert "phase 1" in decision.justification.lower()
    assert ctx.routing_log == [decision]


@pytest.mark.asyncio
async def test_routing_falls_back_when_llm_picks_unknown_agent():
    """If the LLM hallucinates an agent name, fall back to first candidate
    and surface that in the justification."""
    ctx = SharedContext(user_query="x")
    llm = StubLLM([{"next_agent": "phantom_agent", "justification": "I'm making this up"}])
    decision = await pick_next_agent(llm, ctx, candidates=["decomposition", "retrieval"])
    assert decision.next_agent == "decomposition"  # first candidate
    assert "unknown agent" in decision.justification.lower()


# --------------------------------------------------------------------------- #
# Tool dispatch with retries
# --------------------------------------------------------------------------- #


class _FlakyTool(Tool):
    """Returns EMPTY for the first N attempts, then OK."""

    name = "flaky"

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self._attempts = 0

    async def _execute(self, **kwargs):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            return ToolResult(status=ToolStatus.EMPTY, payload={})
        return ToolResult(status=ToolStatus.OK, payload={"q": kwargs.get("query")})


@pytest.mark.asyncio
async def test_tool_dispatch_retries_with_modified_input():
    """EMPTY is retryable; the planner provides modified input each time."""
    ctx = SharedContext(user_query="x")
    tool = _FlakyTool(fail_count=2)

    async def planner(last_input, last_result):
        # broaden the query each retry
        new_q = (last_input.get("query") or "") + " more"
        return {**last_input, "query": new_q.strip()}

    result = await dispatch_tool(
        tool, ctx, initial_input={"query": "narrow"}, retry_planner=planner
    )
    assert result.is_ok
    # 1 initial + 2 retries = 3 invocations, all logged
    assert len(ctx.tool_invocations) == 3
    # Retry chain is linked
    assert ctx.tool_invocations[0].retry_of is None
    assert ctx.tool_invocations[1].retry_of == ctx.tool_invocations[0].id
    assert ctx.tool_invocations[2].retry_of == ctx.tool_invocations[1].id
    # Inputs were modified across retries
    assert ctx.tool_invocations[0].input_payload["query"] == "narrow"
    assert ctx.tool_invocations[1].input_payload["query"] == "narrow more"
    assert ctx.tool_invocations[2].input_payload["query"] == "narrow more more"


@pytest.mark.asyncio
async def test_tool_dispatch_caps_at_max_retries():
    """Per the brief: up to 2 retries. Third failure is final."""
    ctx = SharedContext(user_query="x")
    tool = _FlakyTool(fail_count=10)  # always fails

    async def planner(last_input, last_result):
        return last_input

    result = await dispatch_tool(tool, ctx, initial_input={"query": "x"}, retry_planner=planner)
    assert result.status == ToolStatus.EMPTY
    # 1 initial + MAX_RETRIES retries = MAX_RETRIES + 1 total
    assert len(ctx.tool_invocations) == MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_mark_invocation_accepted_records_decision():
    ctx = SharedContext(user_query="x")
    inv = ToolInvocation(tool_name="t", input_payload={})
    ctx.tool_invocations.append(inv)
    mark_invocation_accepted(ctx, inv.id, accepted=True)
    assert ctx.tool_invocations[0].accepted_by_agent is True


# --------------------------------------------------------------------------- #
# Compression-on-overflow
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_compression_triggers_on_budget_violation():
    """First agent.run() raises BudgetViolation, second succeeds. Verify the
    compression agent ran in between."""
    from app.orchestrator.orchestrator import Orchestrator

    # Build a real-ish orchestrator but replace agent stubs after construction
    llm = StubLLM([])
    budgets = MagicMock()
    store = MagicMock()
    orch = Orchestrator(llm=llm, budgets=budgets, vector_store=store)

    # Replace agents with mocks
    failing_agent = MagicMock()
    success_output = AgentOutput(agent_name="x", content="ok")
    failing_agent.name = "test_agent"
    failing_agent.run = AsyncMock(
        side_effect=[BudgetViolation("test_agent", 1000, 100), success_output]
    )

    # Compression agent stub
    comp_output = AgentOutput(agent_name="compression", content="compressed")
    orch._compression = MagicMock()
    orch._compression.run = AsyncMock(return_value=comp_output)
    orch._compression.name = "compression"

    ctx = SharedContext(user_query="x")
    # Need at least 2 outputs for compression to kick in (per implementation)
    ctx.agent_outputs.append(AgentOutput(agent_name="prior1", content="a"))
    ctx.agent_outputs.append(AgentOutput(agent_name="prior2", content="b"))

    out = await orch._run_with_compression(failing_agent, ctx)
    assert out is success_output
    # Compression was invoked exactly once between the two attempts
    assert orch._compression.run.await_count == 1
    # The failing agent was retried once (so 2 calls total)
    assert failing_agent.run.await_count == 2
