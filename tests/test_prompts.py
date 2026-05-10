"""Tests for the prompt registry.

The registry is process-global by design (so the meta-agent and the
orchestrator share state), so each test snapshots and restores around
itself rather than depending on isolated fixtures.
"""

from __future__ import annotations

import pytest

from app.agents import prompts as P


@pytest.fixture(autouse=True)
def restore_registry():
    """Snapshot active state, run the test, restore."""
    baseline_snap = dict(P._BASELINE)
    active_snap = dict(P._ACTIVE)
    yield
    P._BASELINE.clear()
    P._BASELINE.update(baseline_snap)
    P._ACTIVE.clear()
    P._ACTIVE.update(active_snap)


def test_register_and_get_roundtrip():
    P.register_prompt("test.system", "you are helpful")
    assert P.get_prompt("test.system") == "you are helpful"


def test_register_idempotent_same_text():
    """Re-registering with the same text is a no-op (allows module reimports)."""
    P.register_prompt("test.idem", "hello")
    P.register_prompt("test.idem", "hello")  # should not raise
    assert P.get_prompt("test.idem") == "hello"


def test_register_conflicting_text_raises():
    """Re-registering with different text indicates a bug — registry must
    be append-only."""
    P.register_prompt("test.conflict", "v1")
    with pytest.raises(ValueError, match="re-registered with different text"):
        P.register_prompt("test.conflict", "v2")


def test_get_unknown_key_raises_with_known_keys():
    P.register_prompt("known.one", "x")
    with pytest.raises(KeyError, match="known.one"):
        P.get_prompt("phantom.key")


def test_set_active_prompt_overrides_baseline():
    P.register_prompt("test.set", "baseline text")
    previous = P.set_active_prompt("test.set", "new text")
    assert previous == "baseline text"
    assert P.get_prompt("test.set") == "new text"
    assert P.get_baseline("test.set") == "baseline text"  # baseline preserved


def test_set_active_unknown_key_raises():
    with pytest.raises(KeyError, match="phantom"):
        P.set_active_prompt("phantom", "x")


def test_reset_to_baseline_rolls_back():
    P.register_prompt("test.reset", "original")
    P.set_active_prompt("test.reset", "modified")
    P.reset_to_baseline("test.reset")
    assert P.get_prompt("test.reset") == "original"


def test_all_keys_sorted():
    P.register_prompt("zeta.system", "x")
    P.register_prompt("alpha.system", "y")
    keys = P.all_keys()
    # Sorted
    assert keys == sorted(keys)
    assert "alpha.system" in keys and "zeta.system" in keys


def test_snapshot_returns_copy():
    P.register_prompt("test.snap", "x")
    snap = P.snapshot()
    assert snap["test.snap"] == "x"
    snap["test.snap"] = "mutated"
    # Original unchanged
    assert P.get_prompt("test.snap") == "x"


def test_decomposition_prompts_registered_on_import():
    """Importing the agent module must register its prompts."""
    import app.agents.decomposition  # noqa: F401  side effect: register_prompt

    assert "decomposition.system" in P.all_keys()
    assert "decomposition.user_template" in P.all_keys()
