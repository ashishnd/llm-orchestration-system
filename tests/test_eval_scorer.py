"""Tests for the eval scorer.

Mechanical dimensions (citation, tool efficiency, budget compliance,
critique agreement) are pure functions of SharedContext — we test them
directly without LLM mocks. LLM-judge dimensions (correctness,
contradiction resolution) are smoke-tested with a stub judge.
"""

from __future__ import annotations

import json

import pytest

from app.context import (
    AgentOutput,
    CitationRef,
    ClaimScore,
    CritiqueOutput,
    PolicyViolation,
    ProvenanceEntry,
    SharedContext,
    Span,
    SubTask,
    SynthesisOutput,
    ToolInvocation,
)
from app.eval.cases import TestCase
from app.eval.scorer import (
    score_case,
    score_citation_accuracy,
    score_context_budget_compliance,
    score_critique_agreement,
    score_tool_selection_efficiency,
)
from app.llm.client import LLMResponse


def _ctx_with_synthesis(cited_chunks: list[str], answer: str = "the answer") -> SharedContext:
    ctx = SharedContext(user_query="x")
    ctx.final_answer = answer
    synthesis_out = SynthesisOutput(
        final_answer=answer,
        provenance=[
            ProvenanceEntry(
                sentence_index=0,
                text_span=Span(start=0, end=len(answer), text=answer),
                source_agent="retrieval",
                source_output_id="abc",
                citations=[CitationRef(chunk_id=cid, source_doc=cid) for cid in cited_chunks],
            )
        ],
    )
    ctx.agent_outputs.append(
        AgentOutput(agent_name="synthesis", content="x", structured_output=synthesis_out)
    )
    return ctx


# --------------------------------------------------------------------------- #
# Citation accuracy
# --------------------------------------------------------------------------- #


def test_citation_accuracy_full_match():
    case = TestCase(
        id="t",
        category="baseline",
        query="x",
        expected_chunk_ids=["arxiv:A", "arxiv:B"],
    )
    ctx = _ctx_with_synthesis(["arxiv:A", "arxiv:B"])
    s = score_citation_accuracy(case, ctx)
    assert s.score == 1.0


def test_citation_accuracy_partial_match():
    case = TestCase(
        id="t", category="baseline", query="x", expected_chunk_ids=["arxiv:A", "arxiv:B"]
    )
    ctx = _ctx_with_synthesis(["arxiv:A"])  # half
    s = score_citation_accuracy(case, ctx)
    assert s.score == 0.5


def test_citation_accuracy_no_synthesis_zero():
    case = TestCase(id="t", category="baseline", query="x", expected_chunk_ids=["arxiv:A"])
    ctx = SharedContext(user_query="x")
    s = score_citation_accuracy(case, ctx)
    assert s.score == 0.0


# --------------------------------------------------------------------------- #
# Tool selection efficiency
# --------------------------------------------------------------------------- #


def test_tool_efficiency_no_calls_is_perfect():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    s = score_tool_selection_efficiency(case, ctx)
    assert s.score == 1.0


def test_tool_efficiency_penalizes_rejected_results():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    ctx.sub_tasks.append(SubTask(description="x", task_type="retrieval"))
    ctx.tool_invocations.append(
        ToolInvocation(tool_name="t", input_payload={}, accepted_by_agent=False)
    )
    s = score_tool_selection_efficiency(case, ctx)
    assert s.score < 1.0


def test_tool_efficiency_penalizes_excess_calls():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    ctx.sub_tasks.append(SubTask(description="x", task_type="retrieval"))
    # Budget = 1 sub-task * 2 = 2; submit 5 calls
    for _ in range(5):
        ctx.tool_invocations.append(ToolInvocation(tool_name="t", input_payload={}))
    s = score_tool_selection_efficiency(case, ctx)
    assert s.score < 1.0


# --------------------------------------------------------------------------- #
# Budget compliance
# --------------------------------------------------------------------------- #


def test_budget_compliance_no_violations_is_perfect():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    s = score_context_budget_compliance(case, ctx)
    assert s.score == 1.0


def test_budget_compliance_penalizes_overflows():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    ctx.policy_violations.append(
        PolicyViolation(agent_name="x", violation_type="budget_overflow", detail="...")
    )
    ctx.policy_violations.append(
        PolicyViolation(agent_name="x", violation_type="schema_violation", detail="...")
    )  # not a budget overflow
    s = score_context_budget_compliance(case, ctx)
    # 1 budget violation -> 1.0 - 0.2 = 0.8
    assert s.score == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Critique agreement
# --------------------------------------------------------------------------- #


def test_critique_agreement_no_passes_is_neutral():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    s = score_critique_agreement(case, ctx)
    assert s.score == 0.5


def test_critique_agreement_high_confidence_no_disagreements():
    case = TestCase(id="t", category="baseline", query="x")
    ctx = SharedContext(user_query="x")
    crit = CritiqueOutput(
        target_output_id="abc",
        claim_scores=[
            ClaimScore(claim_span=Span(start=0, end=5), confidence=0.9),
            ClaimScore(claim_span=Span(start=5, end=10), confidence=0.85),
        ],
        disagreements=[],
    )
    ctx.agent_outputs.append(
        AgentOutput(agent_name="critique", content="x", structured_output=crit)
    )
    s = score_critique_agreement(case, ctx)
    assert s.score == pytest.approx(0.875)


# --------------------------------------------------------------------------- #
# Full case scoring (smoke test with stub judge)
# --------------------------------------------------------------------------- #


class StubJudge:
    """LLM stub for the judge dimensions. Returns a fixed high score."""

    def count_tokens(self, text):
        return len(text) // 4

    def count_message_tokens(self, msgs):
        return sum(self.count_tokens(m.content) + 4 for m in msgs)

    async def complete_json(self, messages, **kwargs):
        canned = {"score": 0.85, "justification": "stub judge says ok"}
        resp = LLMResponse(text=json.dumps(canned), input_tokens=10, output_tokens=20, model="stub")
        return canned, resp


@pytest.mark.asyncio
async def test_score_case_returns_six_dimensions():
    case = TestCase(
        id="t",
        category="baseline",
        query="x",
        expected_facts=["fact one"],
        expected_chunk_ids=["arxiv:A"],
    )
    ctx = _ctx_with_synthesis(["arxiv:A"], answer="fact one is here")
    scores = await score_case(case, ctx, judge_llm=StubJudge())

    dimensions = {s.dimension for s in scores}
    assert dimensions == {
        "answer_correctness",
        "citation_accuracy",
        "contradiction_resolution",
        "tool_selection_efficiency",
        "context_budget_compliance",
        "critique_agreement_rate",
    }
    # All scores should be valid floats in [0, 1]
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    # All scores should carry a justification string
    assert all(s.justification for s in scores)
