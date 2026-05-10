"""Repository layer.

Thin functions over the SQLAlchemy models. The orchestrator and API
endpoints call these; we don't expose ORM models directly to the caller
to keep the dependency direction clean.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.context import SharedContext
from app.persistence.models import EvalRun, EvalScore, Job, JobTrace, PromptRewrite


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Jobs and traces
# --------------------------------------------------------------------------- #


async def create_job(session: AsyncSession, *, user_query: str) -> str:
    """Create a job row. Returns the new job_id."""
    job_id = uuid4().hex[:12]
    session.add(Job(id=job_id, user_query=user_query, status="pending"))
    await session.commit()
    return job_id


async def mark_job_completed(
    session: AsyncSession, job_id: str, *, final_answer: str | None
) -> None:
    job = await session.get(Job, job_id)
    if job is None:
        raise KeyError(f"unknown job {job_id!r}")
    job.status = "completed" if final_answer else "failed"
    job.final_answer = final_answer
    job.completed_at = _utcnow()
    await session.commit()


async def save_trace(session: AsyncSession, ctx: SharedContext) -> None:
    """Persist the full SharedContext as JSON for replay."""
    payload = ctx.model_dump(mode="json")
    existing = await session.get(JobTrace, ctx.job_id)
    if existing:
        existing.context_json = payload
        existing.saved_at = _utcnow()
    else:
        session.add(JobTrace(job_id=ctx.job_id, context_json=payload))
    await session.commit()


async def get_trace(session: AsyncSession, job_id: str) -> dict[str, Any] | None:
    trace = await session.get(JobTrace, job_id)
    return trace.context_json if trace else None


# --------------------------------------------------------------------------- #
# Eval runs
# --------------------------------------------------------------------------- #


async def create_eval_run(
    session: AsyncSession, *, prompt_snapshot: dict[str, str], notes: str | None = None
) -> str:
    run_id = uuid4().hex[:12]
    session.add(EvalRun(id=run_id, prompt_snapshot=prompt_snapshot, notes=notes))
    await session.commit()
    return run_id


async def add_eval_score(
    session: AsyncSession,
    *,
    eval_run_id: str,
    test_case_id: str,
    test_case_category: str,
    dimension: str,
    score: float,
    justification: str,
    job_id: str | None = None,
) -> None:
    session.add(
        EvalScore(
            eval_run_id=eval_run_id,
            test_case_id=test_case_id,
            test_case_category=test_case_category,
            dimension=dimension,
            score=score,
            justification=justification,
            job_id=job_id,
        )
    )
    await session.commit()


async def complete_eval_run(session: AsyncSession, eval_run_id: str) -> None:
    run = await session.get(EvalRun, eval_run_id)
    if run is None:
        raise KeyError(f"unknown eval run {eval_run_id!r}")
    run.completed_at = _utcnow()
    await session.commit()


async def get_latest_eval_run(session: AsyncSession) -> EvalRun | None:
    stmt = select(EvalRun).order_by(EvalRun.started_at.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_eval_scores(session: AsyncSession, eval_run_id: str) -> list[EvalScore]:
    stmt = select(EvalScore).where(EvalScore.eval_run_id == eval_run_id)
    result = await session.execute(stmt)
    return list(result.scalars())


# --------------------------------------------------------------------------- #
# Prompt rewrites (meta-agent proposals)
# --------------------------------------------------------------------------- #


async def create_prompt_rewrite(
    session: AsyncSession,
    *,
    eval_run_id: str,
    target_prompt_key: str,
    old_prompt: str,
    proposed_prompt: str,
    structured_diff: str,
    rationale: str,
    worst_dimension: str | None = None,
    worst_case_ids: list[str] | None = None,
) -> str:
    rid = uuid4().hex[:12]
    session.add(
        PromptRewrite(
            id=rid,
            eval_run_id=eval_run_id,
            target_prompt_key=target_prompt_key,
            old_prompt=old_prompt,
            proposed_prompt=proposed_prompt,
            structured_diff=structured_diff,
            rationale=rationale,
            worst_dimension=worst_dimension,
            worst_case_ids=worst_case_ids or [],
        )
    )
    await session.commit()
    return rid


async def decide_prompt_rewrite(
    session: AsyncSession,
    rewrite_id: str,
    *,
    approved: bool,
    decided_by: str | None = None,
) -> PromptRewrite:
    rw = await session.get(PromptRewrite, rewrite_id)
    if rw is None:
        raise KeyError(f"unknown rewrite {rewrite_id!r}")
    if rw.status != "pending":
        raise ValueError(f"rewrite {rewrite_id!r} already decided ({rw.status})")
    rw.status = "approved" if approved else "rejected"
    rw.decided_at = _utcnow()
    rw.decided_by = decided_by
    await session.commit()
    return rw


async def get_pending_rewrites(session: AsyncSession) -> list[PromptRewrite]:
    stmt = select(PromptRewrite).where(PromptRewrite.status == "pending")
    result = await session.execute(stmt)
    return list(result.scalars())
