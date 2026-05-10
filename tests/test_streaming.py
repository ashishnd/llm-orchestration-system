"""SSE event bus tests."""

from __future__ import annotations

import json

import pytest

from app.streaming import (
    Event,
    EventBus,
    EventType,
    emit_agent_start,
    emit_done,
    emit_routing,
)


@pytest.mark.asyncio
async def test_publish_consume_roundtrip():
    bus = EventBus()
    bus.register("job-1")
    await emit_agent_start(bus, "job-1", "decomposition")
    await emit_done(bus, "job-1")

    received = []
    async for ev in bus.consume("job-1"):
        received.append(ev)
    assert [e.type for e in received] == [EventType.AGENT_START, EventType.DONE]
    assert received[0].payload["agent"] == "decomposition"


@pytest.mark.asyncio
async def test_publish_without_subscriber_drops_silently():
    """No subscriber -> event dropped (acceptable; trace is source of truth)."""
    bus = EventBus()
    await emit_agent_start(bus, "job-ghost", "x")  # no register() called
    # Should not raise. Nothing to assert on success.


@pytest.mark.asyncio
async def test_consume_terminates_on_done_event():
    """consume() yields events until DONE/ERROR, then terminates."""
    bus = EventBus()
    bus.register("job-2")
    await emit_routing(bus, "job-2", "decomposition", "first phase")
    await emit_done(bus, "job-2")
    # Push another event AFTER done; should not be yielded
    await emit_agent_start(bus, "job-2", "should-be-dropped")

    received = []
    async for ev in bus.consume("job-2"):
        received.append(ev)
    assert len(received) == 2
    assert received[-1].type == EventType.DONE


@pytest.mark.asyncio
async def test_event_to_sse_data_is_valid_json():
    ev = Event(EventType.AGENT_START, "job-x", {"agent": "decomposition"})
    parsed = json.loads(ev.to_sse_data())
    assert parsed["type"] == "agent_start"
    assert parsed["agent"] == "decomposition"
    assert "ts" in parsed
