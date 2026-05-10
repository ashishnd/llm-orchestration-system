"""Tests for the context budget manager.

Focus: budget declaration, consumption tracking, overflow detection as a
policy violation (not silent truncation), and the compression threshold.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.context.budget import BudgetManager, BudgetViolation
from app.context.schema import SharedContext


@pytest.fixture
def llm():
    """A mock LLM client. Budget manager only needs token counting from it."""
    m = MagicMock()
    m.count_tokens.side_effect = lambda s: len(s.split())
    return m


@pytest.fixture
def manager(llm):
    return BudgetManager(llm=llm, compression_threshold=0.85)


@pytest.fixture
def ctx():
    return SharedContext(user_query="test")


def test_declare_creates_budget(manager):
    b = manager.declare("decomposition", 1000)
    assert b.max_tokens == 1000
    assert b.used_tokens == 0
    assert b.remaining == 1000


def test_check_passes_within_budget(manager):
    manager.declare("retrieval", 1000)
    remaining = manager.check("retrieval", 200)
    assert remaining == 800


def test_check_raises_on_overflow(manager):
    manager.declare("retrieval", 1000)
    with pytest.raises(BudgetViolation) as exc_info:
        manager.check("retrieval", 1500)
    assert exc_info.value.agent_name == "retrieval"
    assert exc_info.value.requested == 1500
    assert exc_info.value.remaining == 1000


def test_consume_decrements_remaining(manager):
    manager.declare("synthesis", 2000)
    manager.consume("synthesis", 500)
    assert manager.remaining("synthesis") == 1500


def test_compression_threshold_triggers_at_85_percent(manager):
    manager.declare("retrieval", 1000)
    manager.consume("retrieval", 800)
    assert not manager.needs_compression("retrieval")
    manager.consume("retrieval", 50)  # now at 850/1000 = 0.85
    assert manager.needs_compression("retrieval")


def test_redeclare_resets_budget(manager):
    """Same agent invoked twice gets a fresh budget on each turn."""
    manager.declare("critique", 1000)
    manager.consume("critique", 800)
    manager.declare("critique", 1000)  # second turn
    assert manager.remaining("critique") == 1000


def test_check_unknown_agent_raises(manager):
    with pytest.raises(KeyError, match="did not declare a budget"):
        manager.check("phantom", 100)


def test_record_violation_appends_to_context(manager, ctx):
    """Overflow must be logged as a policy violation, not silently fixed."""
    manager.declare("synthesis", 100)
    try:
        manager.check("synthesis", 500)
    except BudgetViolation as e:
        manager.record_violation(ctx, "synthesis", str(e))

    assert len(ctx.policy_violations) == 1
    v = ctx.policy_violations[0]
    assert v.agent_name == "synthesis"
    assert v.violation_type == "budget_overflow"
    assert "500" in v.detail


def test_utilization_clamps_to_one_when_full(manager):
    manager.declare("decomposition", 100)
    manager.consume("decomposition", 100)
    assert manager.utilization("decomposition") == 1.0
    assert manager.remaining("decomposition") == 0
