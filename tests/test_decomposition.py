"""Tests for the decomposition agent.

We don't call OpenAI. We stub the LLM client to return canned JSON
responses for each scenario, and assert on:
- AgentOutput shape and content
- Sub-task ID re-stamping (slug -> canonical ID)
- Dependency graph integrity (cycles / dangling refs detected)
- Schema violations recorded as PolicyViolation
- Budget consumption tracked correctly
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agents import AgentExecutionError, DecompositionAgent
from app.context import (
    BudgetManager,
    DecompositionOutput,
    SharedContext,
)
from app.llm import MalformedJSONError
from app.llm.client import LLMResponse

# --------------------------------------------------------------------------- #
# Stub LLM client
# --------------------------------------------------------------------------- #


class StubLLM:
    """Minimal LLM stub.

    `complete_json` returns the next canned response from `responses` queue.
    `count_tokens` and `count_message_tokens` return small fixed values so
    budget arithmetic is predictable.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages) -> int:
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    async def complete_json(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("StubLLM ran out of canned responses")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        # next_resp is a dict; package it like complete_json's return shape
        resp = LLMResponse(
            text=json.dumps(next_resp),
            input_tokens=10,
            output_tokens=20,
            model="stub",
            finish_reason="stop",
        )
        return next_resp, resp


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def ctx():
    return SharedContext(user_query="Has anyone improved on GLUE since 2023?")


@pytest.fixture
def budgets(stub_llm):
    return BudgetManager(llm=stub_llm)


@pytest.fixture
def stub_llm():
    return StubLLM([])


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestDecompositionHappyPath:
    @pytest.mark.asyncio
    async def test_simple_query_emits_single_retrieval(self, ctx, budgets):
        """Brief: simple queries should emit ONE sub-task, not five."""
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "Look up papers describing BERT",
                            "task_type": "retrieval",
                            "depends_on": [],
                        }
                    ],
                    "rationale": "Single fact lookup; no further decomposition needed.",
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]
        ctx_simple = SharedContext(user_query="What is BERT?")

        out = await agent.run(ctx_simple)

        assert out.agent_name == "decomposition"
        assert isinstance(out.structured_output, DecompositionOutput)
        assert len(out.structured_output.sub_tasks) == 1
        assert out.structured_output.sub_tasks[0].task_type == "retrieval"
        assert out.input_tokens == 10 and out.output_tokens == 20
        assert out.prompt_hash is not None and len(out.prompt_hash) == 64

    @pytest.mark.asyncio
    async def test_complex_query_with_dependency_graph(self, ctx, budgets):
        """Brief: dependent sub-tasks must reference upstream task IDs."""
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "Find papers on parameter-efficient FT",
                            "task_type": "retrieval",
                            "depends_on": [],
                        },
                        {
                            "id": "t2",
                            "description": "Find papers reporting GLUE benchmark",
                            "task_type": "retrieval",
                            "depends_on": [],
                        },
                        {
                            "id": "t3",
                            "description": "Cross-reference the two retrieval sets",
                            "task_type": "synthesis",
                            "depends_on": ["t1", "t2"],
                        },
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        out = await agent.run(ctx)

        decomp: DecompositionOutput = out.structured_output  # type: ignore[assignment]
        assert len(decomp.sub_tasks) == 3
        # Slug IDs should have been re-stamped to canonical IDs (12 hex chars)
        ids = [t.id for t in decomp.sub_tasks]
        assert all(len(i) == 12 for i in ids), f"got: {ids}"
        # And depends_on should reference those canonical IDs, not the slugs
        t3 = decomp.sub_tasks[2]
        assert set(t3.depends_on) == {ids[0], ids[1]}
        # Slug strings should not survive anywhere
        assert "t1" not in [d for t in decomp.sub_tasks for d in t.depends_on]

    @pytest.mark.asyncio
    async def test_other_task_type_with_detail(self, ctx, budgets):
        """task_type='other' is allowed if task_type_detail is set."""
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "Cluster topics across retrieved abstracts",
                            "task_type": "other",
                            "task_type_detail": "topic_clustering",
                            "depends_on": [],
                        }
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        out = await agent.run(ctx)
        decomp: DecompositionOutput = out.structured_output  # type: ignore[assignment]
        assert decomp.sub_tasks[0].task_type == "other"
        assert decomp.sub_tasks[0].task_type_detail == "topic_clustering"


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestDecompositionFailureModes:
    @pytest.mark.asyncio
    async def test_dangling_dependency_logs_violation(self, ctx, budgets):
        """A depends_on pointing to a non-existent task must be caught."""
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "real task",
                            "task_type": "retrieval",
                            "depends_on": ["t99"],  # phantom
                        }
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        with pytest.raises(AgentExecutionError, match="dangling dependency"):
            await agent.run(ctx)

        violations = [v for v in ctx.policy_violations if v.agent_name == "decomposition"]
        assert len(violations) == 1
        assert violations[0].violation_type == "schema_violation"
        assert "t99" in violations[0].detail

    @pytest.mark.asyncio
    async def test_dependency_cycle_logs_violation(self, ctx, budgets):
        """A cycle in the dependency graph is a logical bug; we refuse it."""
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "a",
                            "task_type": "synthesis",
                            "depends_on": ["t2"],
                        },
                        {
                            "id": "t2",
                            "description": "b",
                            "task_type": "synthesis",
                            "depends_on": ["t1"],
                        },
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        with pytest.raises(AgentExecutionError, match="cycle"):
            await agent.run(ctx)

        violations = [v for v in ctx.policy_violations if v.agent_name == "decomposition"]
        assert len(violations) == 1
        assert "cycle" in violations[0].detail.lower()

    @pytest.mark.asyncio
    async def test_invalid_task_type_logs_schema_violation(self, ctx, budgets):
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "x",
                            "task_type": "not_a_real_type",
                            "depends_on": [],
                        }
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        with pytest.raises(AgentExecutionError):
            await agent.run(ctx)

        violations = [v for v in ctx.policy_violations if v.agent_name == "decomposition"]
        assert len(violations) == 1
        assert violations[0].violation_type == "schema_violation"
        # raw_preview should carry the bad output
        assert "not_a_real_type" in (violations[0].raw_preview or "")

    @pytest.mark.asyncio
    async def test_other_without_detail_logs_violation(self, ctx, budgets):
        """task_type='other' without task_type_detail violates SubTask validator."""
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "x",
                            "task_type": "other",
                            # missing task_type_detail
                            "depends_on": [],
                        }
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        with pytest.raises(AgentExecutionError):
            await agent.run(ctx)

        assert any(v.violation_type == "schema_violation" for v in ctx.policy_violations)

    @pytest.mark.asyncio
    async def test_malformed_json_logs_violation_and_raises(self, ctx, budgets):
        """If complete_json fails after repair, agent logs schema_violation."""
        llm = StubLLM(
            [MalformedJSONError("twice failed", raw_first="garbage 1", raw_retry="garbage 2")]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        with pytest.raises(AgentExecutionError, match="malformed JSON"):
            await agent.run(ctx)

        violations = ctx.policy_violations
        assert len(violations) == 1
        assert violations[0].violation_type == "schema_violation"
        assert violations[0].raw_preview is not None
        assert "garbage 1" in violations[0].raw_preview
        assert "garbage 2" in violations[0].raw_preview


# --------------------------------------------------------------------------- #
# Budget integration
# --------------------------------------------------------------------------- #


class TestDecompositionBudget:
    @pytest.mark.asyncio
    async def test_declares_budget_on_run(self, ctx, budgets):
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "x",
                            "task_type": "retrieval",
                            "depends_on": [],
                        }
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        await agent.run(ctx)

        # Budget should be declared and consumed
        assert "decomposition" in budgets.budgets
        assert budgets.budgets["decomposition"].used_tokens == 30  # 10 in + 20 out

    @pytest.mark.asyncio
    async def test_custom_budget_overrides_default(self, ctx, budgets):
        llm = StubLLM(
            [
                {
                    "type": "decomposition",
                    "sub_tasks": [
                        {
                            "id": "t1",
                            "description": "x",
                            "task_type": "retrieval",
                            "depends_on": [],
                        }
                    ],
                }
            ]
        )
        budgets._llm = llm  # type: ignore[attr-defined]
        agent = DecompositionAgent(llm=llm, budgets=budgets)  # type: ignore[arg-type]

        await agent.run(ctx, budget=500)

        assert budgets.budgets["decomposition"].max_tokens == 500
