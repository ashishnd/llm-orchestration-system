"""Persistence tests using in-memory aiosqlite.

We don't run against real Postgres here; SQLAlchemy's portable SQL covers
the surface we use. Postgres-specific JSONB falls back to JSON via the
.with_variant() column.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.context import SharedContext
from app.persistence import repository
from app.persistence.models import Base


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_job_and_save_trace_roundtrip(session):
    job_id = await repository.create_job(session, user_query="What is BERT?")
    assert len(job_id) == 12

    ctx = SharedContext(job_id=job_id, user_query="What is BERT?")
    ctx.final_answer = "It's a transformer encoder."
    await repository.save_trace(session, ctx)
    await repository.mark_job_completed(session, job_id, final_answer=ctx.final_answer)

    loaded = await repository.get_trace(session, job_id)
    assert loaded is not None
    assert loaded["user_query"] == "What is BERT?"
    assert loaded["final_answer"] == "It's a transformer encoder."


@pytest.mark.asyncio
async def test_eval_run_lifecycle_and_score_persistence(session):
    run_id = await repository.create_eval_run(
        session, prompt_snapshot={"decomposition.system": "..."}
    )
    await repository.add_eval_score(
        session,
        eval_run_id=run_id,
        test_case_id="baseline_01",
        test_case_category="baseline",
        dimension="answer_correctness",
        score=0.85,
        justification="Answer matched expected key facts.",
    )
    await repository.complete_eval_run(session, run_id)

    latest = await repository.get_latest_eval_run(session)
    assert latest is not None
    assert latest.id == run_id
    assert latest.completed_at is not None

    scores = await repository.get_eval_scores(session, run_id)
    assert len(scores) == 1
    assert scores[0].score == 0.85


@pytest.mark.asyncio
async def test_prompt_rewrite_decision_flow(session):
    eval_run_id = await repository.create_eval_run(session, prompt_snapshot={})
    rewrite_id = await repository.create_prompt_rewrite(
        session,
        eval_run_id=eval_run_id,
        target_prompt_key="decomposition.system",
        old_prompt="OLD",
        proposed_prompt="NEW",
        structured_diff="--- old\n+++ new",
        rationale="The old prompt over-decomposed simple queries.",
    )
    pending = await repository.get_pending_rewrites(session)
    assert len(pending) == 1

    rw = await repository.decide_prompt_rewrite(
        session, rewrite_id, approved=True, decided_by="reviewer"
    )
    assert rw.status == "approved"
    assert rw.decided_at is not None

    # Cannot re-decide
    with pytest.raises(ValueError, match="already decided"):
        await repository.decide_prompt_rewrite(session, rewrite_id, approved=False)
