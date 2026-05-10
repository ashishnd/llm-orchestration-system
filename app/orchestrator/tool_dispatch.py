"""Tool dispatch with explicit retry logic.

Per the brief: an agent that receives a tool result and decides it is
insufficient must be able to re-call the tool with modified input, up to
two retries, with each retry logged separately.

We expose `dispatch_tool()` which:
1. Calls the tool, logs the ToolInvocation
2. If retryable (TIMEOUT/EMPTY) and a retry_planner is provided, asks
   it for modified input; up to MAX_RETRIES total attempts
3. Returns the final ToolResult and the chain of invocations
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.context import SharedContext, ToolInvocation
from app.tools import Tool, ToolResult

MAX_RETRIES = 2  # per the brief

# A retry planner takes (last_input, last_result) and returns modified
# input dict, or None to stop retrying.
RetryPlanner = Callable[[dict[str, Any], ToolResult], Awaitable[dict[str, Any] | None]]


async def dispatch_tool(
    tool: Tool,
    ctx: SharedContext,
    initial_input: dict[str, Any],
    *,
    retry_planner: RetryPlanner | None = None,
) -> ToolResult:
    """Dispatch a tool with up to 2 retries on retryable failures.

    Every invocation (initial + each retry) is logged to ctx.tool_invocations
    with full input/output/latency and a `retry_of` link to the prior
    invocation. The agent's accept/reject decision is recorded by the caller
    (we don't have visibility into the agent's logic here).
    """
    current_input = dict(initial_input)
    last_invocation_id: str | None = None
    result = await _call_and_log(tool, ctx, current_input, retry_of=None)
    last_invocation_id = ctx.tool_invocations[-1].id

    attempts = 1
    while result.is_retryable and attempts <= MAX_RETRIES and retry_planner is not None:
        modified = await retry_planner(current_input, result)
        if modified is None:
            break
        current_input = modified
        result = await _call_and_log(tool, ctx, current_input, retry_of=last_invocation_id)
        last_invocation_id = ctx.tool_invocations[-1].id
        attempts += 1

    return result


async def _call_and_log(
    tool: Tool,
    ctx: SharedContext,
    input_payload: dict[str, Any],
    *,
    retry_of: str | None,
) -> ToolResult:
    result = await tool.call(**input_payload)
    invocation = ToolInvocation(
        tool_name=tool.name,
        input_payload=input_payload,
        output_payload=result.payload if result.is_ok else None,
        error=result.error_message,
        latency_ms=result.latency_ms,
        retry_of=retry_of,
    )
    ctx.tool_invocations.append(invocation)
    return result


def mark_invocation_accepted(ctx: SharedContext, invocation_id: str, accepted: bool) -> None:
    """Record the agent's accept/reject decision on a tool result. Called by
    the agent or the orchestrator after the agent reviews the output."""
    for inv in ctx.tool_invocations:
        if inv.id == invocation_id:
            inv.accepted_by_agent = accepted
            return
    raise KeyError(f"unknown invocation {invocation_id!r}")
