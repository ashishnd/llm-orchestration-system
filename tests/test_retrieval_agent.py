"""Retrieval agent tests — focused on the three things only this agent can break:
multi-hop loop sequencing, citation fabrication, and undersourced compliance.
Schema-level concerns are covered by tests/test_schema.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.agents import AgentExecutionError, RetrievalAgent
from app.context import (
    BudgetManager,
    RetrievalOutput,
    SharedContext,
)
from app.llm.client import LLMResponse
from app.rag import VectorStore
from app.rag.vector_store import RetrievedChunk


def _stub_embedder(texts):
    # 16-dim deterministic; quality irrelevant for these tests
    return [[0.1] * 16 for _ in texts]


def _chunk(cid: str, text: str = "stub text"):
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata={"arxiv_id": cid.split(":", 1)[-1], "title": f"Title for {cid}"},
        distance=0.5,
    )


class StubLLM:
    """Returns canned responses in order. count_tokens returns small fixed values."""

    def __init__(self, responses):
        self._responses = list(responses)

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages):
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    async def complete_json(self, messages, **kwargs):
        if not self._responses:
            raise RuntimeError("StubLLM ran out of canned responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        resp = LLMResponse(text=json.dumps(nxt), input_tokens=10, output_tokens=20, model="stub")
        return nxt, resp


@pytest.fixture
def store():
    """Vector store with 3 chunks pre-loaded. Mock the query method directly so
    we control which chunks come back per hop without depending on similarity."""
    s = MagicMock(spec=VectorStore)
    return s


@pytest.fixture
def ctx():
    return SharedContext(user_query="What are recent improvements to BERT-style models?")


@pytest.mark.asyncio
async def test_happy_path_two_hops_with_citations(store, ctx):
    """Hop 1 retrieves 2 chunks. LLM plans hop 2, retrieves 1 more chunk.
    LLM signals stop. Synthesis produces answer_draft + valid chunk_links."""
    store.query.side_effect = [
        [_chunk("arxiv:2401.001"), _chunk("arxiv:2401.002")],  # hop 1
        [_chunk("arxiv:2401.003")],  # hop 2
    ]

    llm = StubLLM(
        [
            # hop planner: continue with new query
            {"action": "query", "query": "BERT variants 2024", "rationale": "fill gap"},
            # hop planner: stop
            {"action": "stop"},
            # synthesis
            {
                "type": "retrieval",
                "answer_draft": "Recent BERT-style improvements include X and Y.",
                "hops": [],
                "chunk_links": [
                    {
                        "chunk_id": "arxiv:2401.001",
                        "answer_span": {
                            "start": 0,
                            "end": 36,
                            "text": "Recent BERT-style improvements include",
                        },
                        "contribution_note": "names X",
                    }
                ],
            },
        ]
    )
    budgets = BudgetManager(llm=llm)
    agent = RetrievalAgent(llm=llm, budgets=budgets, vector_store=store)

    out = await agent.run(ctx)

    ro: RetrievalOutput = out.structured_output
    assert isinstance(ro, RetrievalOutput)
    assert len(ro.hops) == 2  # compliant
    assert ro.hops[0].query == ctx.user_query
    assert ro.hops[1].query == "BERT variants 2024"
    assert ro.is_compliant()
    assert len(ro.chunk_links) == 1
    assert ro.chunk_links[0].chunk_id == "arxiv:2401.001"
    # No policy violations on a clean run
    assert not any(v.agent_name == "retrieval" for v in ctx.policy_violations)


@pytest.mark.asyncio
async def test_fabricated_citation_raises_and_logs(store, ctx):
    """If synthesis cites a chunk that wasn't retrieved, that's hallucination —
    raise and log schema_violation."""
    store.query.side_effect = [
        [_chunk("arxiv:real:1")],
        [_chunk("arxiv:real:2")],
    ]
    llm = StubLLM(
        [
            {"action": "query", "query": "second hop", "rationale": "x"},
            {"action": "stop"},
            {
                "type": "retrieval",
                "answer_draft": "Some answer.",
                "hops": [],
                "chunk_links": [
                    {
                        "chunk_id": "arxiv:HALLUCINATED",  # not retrieved
                        "answer_span": {"start": 0, "end": 5, "text": "Some "},
                        "contribution_note": "fake",
                    }
                ],
            },
        ]
    )
    budgets = BudgetManager(llm=llm)
    agent = RetrievalAgent(llm=llm, budgets=budgets, vector_store=store)

    with pytest.raises(AgentExecutionError, match="fabricated"):
        await agent.run(ctx)

    violations = [v for v in ctx.policy_violations if v.agent_name == "retrieval"]
    assert len(violations) == 1
    assert violations[0].violation_type == "schema_violation"
    assert "HALLUCINATED" in violations[0].detail


@pytest.mark.asyncio
async def test_undersourced_logged_as_policy_violation(store, ctx):
    """If the LLM signals stop after hop 1, we still produce output but log
    retrieval_undersourced. The brief requires >=2 hops; we surface that as
    an audit signal, not a refusal."""
    store.query.side_effect = [[_chunk("arxiv:only:1")]]  # only hop 1 runs
    llm = StubLLM(
        [
            # hop planner says stop after hop 1 — undersourced
            {"action": "stop"},
            # synthesis still runs
            {
                "type": "retrieval",
                "answer_draft": "Limited answer based on one chunk.",
                "hops": [],
                "chunk_links": [
                    {
                        "chunk_id": "arxiv:only:1",
                        "answer_span": {"start": 0, "end": 7, "text": "Limited"},
                    }
                ],
            },
        ]
    )
    budgets = BudgetManager(llm=llm)
    agent = RetrievalAgent(llm=llm, budgets=budgets, vector_store=store)

    out = await agent.run(ctx)

    ro: RetrievalOutput = out.structured_output
    assert len(ro.hops) == 1  # only one hop ran
    assert not ro.is_compliant()
    # Undersourcing is logged but doesn't fail the run
    violations = [v for v in ctx.policy_violations if v.agent_name == "retrieval"]
    assert len(violations) == 1
    assert violations[0].violation_type == "retrieval_undersourced"
