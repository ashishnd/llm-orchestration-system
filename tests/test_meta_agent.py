"""Meta-agent tests.

We don't call OpenAI; the LLM is stubbed. The interesting paths are:
- find_worst_dimension picks the right (agent, dimension) pair
- propose_rewrite persists a PromptRewrite with status=pending
- propose_rewrite returns None when no dimension is below threshold
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.eval.meta_agent import propose_rewrite
from app.llm.client import LLMResponse
from app.persistence import repository
from app.persistence.models import Base


class StubLLM:
    def __init__(self, response):
        self._response = response

    def count_tokens(self, t):
        return max(1, len(t) // 4)

    def count_message_tokens(self, msgs):
        return sum(self.count_tokens(m.content) + 4 for m in msgs)

    async def complete_json(self, messages, temperature=0.2, **kwargs):
        resp = LLMResponse(
            text=json.dumps(self._response),
            input_tokens=10,
            output_tokens=20,
            model="stub",
        )
        return self._response, resp


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        # Make sure all default prompts are registered
        import app.agents  # noqa: F401  side-effect: registers all prompts

        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_propose_rewrite_persists_proposal_for_worst_dimension(session):
    run_id = await repository.create_eval_run(session, prompt_snapshot={})
    # Two dimensions: one fine (0.9), one bad (0.4 mean)
    for case_id, dim, score in [
        ("c1", "answer_correctness", 0.9),
        ("c2", "answer_correctness", 0.85),
        ("c1", "citation_accuracy", 0.4),  # worst
        ("c2", "citation_accuracy", 0.4),
    ]:
        await repository.add_eval_score(
            session,
            eval_run_id=run_id,
            test_case_id=case_id,
            test_case_category="baseline",
            dimension=dim,
            score=score,
            justification=f"sample for {dim}",
        )

    llm = StubLLM(
        {
            "proposed_prompt": "REWRITTEN: be more careful with citations",
            "rationale": "the original prompt didn't emphasize citing sources",
        }
    )
    out = await propose_rewrite(session=session, eval_run_id=run_id, judge_llm=llm)

    assert out is not None
    # citation_accuracy maps to the retrieval agent in our mapping
    assert out.target_prompt_key == "retrieval.synthesis_system"
    assert out.worst_dimension == "citation_accuracy"
    assert "REWRITTEN" in out.proposed_prompt
    assert out.structured_diff  # non-empty unified diff

    # Persisted as pending
    pending = await repository.get_pending_rewrites(session)
    assert len(pending) == 1
    assert pending[0].target_prompt_key == "retrieval.synthesis_system"


@pytest.mark.asyncio
async def test_propose_rewrite_returns_none_when_all_dimensions_above_threshold(session):
    run_id = await repository.create_eval_run(session, prompt_snapshot={})
    for case_id in ["c1", "c2"]:
        await repository.add_eval_score(
            session,
            eval_run_id=run_id,
            test_case_id=case_id,
            test_case_category="baseline",
            dimension="answer_correctness",
            score=0.9,
            justification="ok",
        )

    llm = StubLLM({"proposed_prompt": "should not be called", "rationale": ""})
    out = await propose_rewrite(session=session, eval_run_id=run_id, judge_llm=llm)
    assert out is None
    pending = await repository.get_pending_rewrites(session)
    assert pending == []
