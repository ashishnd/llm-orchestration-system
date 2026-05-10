"""SSE event bus.

The orchestrator writes events to per-job asyncio.Queues. The FastAPI
SSE endpoint consumes from the queue and emits sse-starlette events.

Design notes:
- One queue per job_id, held by EventBus. The endpoint registers as a
  consumer for a job_id; the orchestrator publishes by job_id.
- We don't persist events here — the JobTrace is the source of truth for
  replay. Events are ephemeral, for live observability.
- The "stream agent tokens" requirement is intentionally narrow: agents
  emit JSON, which isn't useful mid-stream. We surface higher-signal
  events instead: agent_start, agent_complete, tool_start, tool_complete,
  routing, budget_status, final_answer. (Documented in AI_COLLABORATION
  Section 8.)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    ROUTING = "routing"
    BUDGET_STATUS = "budget_status"
    FINAL_ANSWER = "final_answer"
    POLICY_VIOLATION = "policy_violation"
    DONE = "done"  # terminal sentinel
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_sse_data(self) -> str:
        return json.dumps({"type": self.type.value, "ts": self.ts, **self.payload})


class EventBus:
    """Per-job event queue registry.

    The orchestrator gets an asyncio.Queue for a job by calling
    `register(job_id)` at the start of execution and emits via `publish()`.
    The SSE endpoint calls `consume(job_id)` to get an async iterator.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Event]] = {}

    def register(self, job_id: str) -> asyncio.Queue[Event]:
        if job_id in self._queues:
            return self._queues[job_id]
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._queues[job_id] = q
        return q

    async def publish(self, event: Event) -> None:
        q = self._queues.get(event.job_id)
        if q is None:
            return  # no consumer; event dropped (acceptable — trace has the truth)
        await q.put(event)

    def has_subscriber(self, job_id: str) -> bool:
        return job_id in self._queues

    def cleanup(self, job_id: str) -> None:
        """Drop the queue after the consumer disconnects."""
        self._queues.pop(job_id, None)

    async def consume(self, job_id: str):
        """Async iterator over events for `job_id`. Stops on Event.DONE."""
        q = self.register(job_id)
        try:
            while True:
                event = await q.get()
                yield event
                if event.type in (EventType.DONE, EventType.ERROR):
                    return
        finally:
            self.cleanup(job_id)


# Module-level singleton; FastAPI dependency injection points at this.
_BUS = EventBus()


def get_event_bus() -> EventBus:
    return _BUS


# --------------------------------------------------------------------------- #
# Convenience publishers — used by the orchestrator
# --------------------------------------------------------------------------- #


async def emit_agent_start(bus: EventBus, job_id: str, agent_name: str) -> None:
    await bus.publish(Event(EventType.AGENT_START, job_id, {"agent": agent_name}))


async def emit_agent_complete(bus: EventBus, job_id: str, agent_name: str, content: str) -> None:
    await bus.publish(
        Event(
            EventType.AGENT_COMPLETE,
            job_id,
            {"agent": agent_name, "content": content[:500]},
        )
    )


async def emit_tool_start(
    bus: EventBus, job_id: str, tool_name: str, input_payload: dict[str, Any]
) -> None:
    await bus.publish(
        Event(
            EventType.TOOL_START,
            job_id,
            {"tool": tool_name, "input": input_payload},
        )
    )


async def emit_tool_complete(
    bus: EventBus,
    job_id: str,
    tool_name: str,
    status: str,
    latency_ms: float,
) -> None:
    await bus.publish(
        Event(
            EventType.TOOL_COMPLETE,
            job_id,
            {"tool": tool_name, "status": status, "latency_ms": latency_ms},
        )
    )


async def emit_routing(
    bus: EventBus, job_id: str, next_agent: str | None, justification: str
) -> None:
    await bus.publish(
        Event(
            EventType.ROUTING,
            job_id,
            {"next_agent": next_agent, "justification": justification},
        )
    )


async def emit_budget_status(
    bus: EventBus, job_id: str, agent_name: str, used: int, max_tokens: int
) -> None:
    await bus.publish(
        Event(
            EventType.BUDGET_STATUS,
            job_id,
            {
                "agent": agent_name,
                "used_tokens": used,
                "max_tokens": max_tokens,
                "remaining": max_tokens - used,
            },
        )
    )


async def emit_final_answer(bus: EventBus, job_id: str, answer: str | None) -> None:
    await bus.publish(Event(EventType.FINAL_ANSWER, job_id, {"answer": answer}))


async def emit_done(bus: EventBus, job_id: str) -> None:
    await bus.publish(Event(EventType.DONE, job_id))


async def emit_error(bus: EventBus, job_id: str, message: str) -> None:
    await bus.publish(Event(EventType.ERROR, job_id, {"message": message}))


async def emit_policy_violation(
    bus: EventBus, job_id: str, agent_name: str, violation_type: str, detail: str
) -> None:
    await bus.publish(
        Event(
            EventType.POLICY_VIOLATION,
            job_id,
            {"agent": agent_name, "type": violation_type, "detail": detail[:300]},
        )
    )
