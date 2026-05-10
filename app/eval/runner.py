"""Eval runner.

Executes the orchestrator on each test case, scores the result across all
six dimensions, and persists the scores. Supports targeted re-eval on a
subset of case IDs (used by /eval/rerun-failed).

Reproducibility:
- Each EvalRun row stores `prompt_snapshot` — the active prompts at run
  time. Re-running with the same prompts produces the same scores
  (modulo small LLM nondeterminism at temp=0).
- Each EvalScore row carries the originating job_id so any score can be
  traced back to the specific run that produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompts import snapshot as prompt_snapshot
from app.context import BudgetManager, SharedContext
from app.eval.cases import CASES, TestCase, get_case
from app.eval.scorer import DimensionScore, score_case
from app.llm import LLMClient
from app.orchestrator import Orchestrator
from app.persistence import repository
from app.rag import VectorStore

log = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case_id: str
    scores: list[DimensionScore]
    job_id: str


async def run_one_case(
    case: TestCase,
    *,
    session: AsyncSession,
    llm: LLMClient,
    vector_store: VectorStore,
) -> CaseResult:
    """Run the pipeline on a single case and score the result. Persists a Job
    row + JobTrace; the caller persists the scores via add_eval_score."""
    job_id = await repository.create_job(session, user_query=case.query)
    ctx = SharedContext(job_id=job_id, user_query=case.query)
    budgets = BudgetManager(llm=llm)
    orchestrator = Orchestrator(llm=llm, budgets=budgets, vector_store=vector_store)

    try:
        await orchestrator.run(ctx)
    except Exception:
        log.exception("Pipeline crashed during eval for %s", case.id)

    await repository.save_trace(session, ctx)
    await repository.mark_job_completed(session, job_id, final_answer=ctx.final_answer)

    scores = await score_case(case, ctx, judge_llm=llm)
    return CaseResult(case_id=case.id, scores=scores, job_id=job_id)


async def run_eval(
    *,
    session: AsyncSession,
    llm: LLMClient,
    vector_store: VectorStore,
    case_ids: list[str] | None = None,
    notes: str | None = None,
) -> str:
    """Run the full eval (or a subset) and return the eval_run_id.

    Persists EvalRun + EvalScore rows. Each score links to the
    underlying job_id so traces are queryable.
    """
    cases = [get_case(cid) for cid in case_ids] if case_ids is not None else CASES

    eval_run_id = await repository.create_eval_run(
        session, prompt_snapshot=prompt_snapshot(), notes=notes
    )

    for case in cases:
        result = await run_one_case(case, session=session, llm=llm, vector_store=vector_store)
        for ds in result.scores:
            await repository.add_eval_score(
                session,
                eval_run_id=eval_run_id,
                test_case_id=case.id,
                test_case_category=case.category,
                dimension=ds.dimension,
                score=ds.score,
                justification=ds.justification,
                job_id=result.job_id,
            )

    await repository.complete_eval_run(session, eval_run_id)
    return eval_run_id


async def rerun_failed_cases(
    *,
    session: AsyncSession,
    llm: LLMClient,
    vector_store: VectorStore,
    base_eval_run_id: str,
    score_threshold: float = 0.6,
) -> tuple[str, list[str], dict[str, float]]:
    """Re-run only cases that scored below threshold on any dimension in the
    base run. Returns (new_run_id, rerun_case_ids, delta_summary).

    delta_summary maps dimension -> mean score delta (new - base) on the
    re-run cases. Positive = improvement.
    """
    base_scores = await repository.get_eval_scores(session, base_eval_run_id)
    if not base_scores:
        raise KeyError(f"unknown base eval run {base_eval_run_id!r}")

    # Failed case IDs: any dimension below threshold qualifies
    failed_ids = sorted({s.test_case_id for s in base_scores if s.score < score_threshold})
    if not failed_ids:
        new_run = await repository.create_eval_run(
            session,
            prompt_snapshot=prompt_snapshot(),
            notes=f"rerun of {base_eval_run_id}; no failures to re-run",
        )
        await repository.complete_eval_run(session, new_run)
        return new_run, [], {}

    new_run_id = await run_eval(
        session=session,
        llm=llm,
        vector_store=vector_store,
        case_ids=failed_ids,
        notes=f"rerun of {base_eval_run_id} for failed cases",
    )

    # Compute delta per dimension on the failed cases
    new_scores = await repository.get_eval_scores(session, new_run_id)
    base_failed_only = [s for s in base_scores if s.test_case_id in failed_ids]
    delta = _compute_delta(base_failed_only, new_scores)
    return new_run_id, failed_ids, delta


def _compute_delta(
    base_scores: list,
    new_scores: list,
) -> dict[str, float]:
    """Mean score delta per dimension across the re-run cases."""
    by_dim_base: dict[str, list[float]] = {}
    by_dim_new: dict[str, list[float]] = {}
    for s in base_scores:
        by_dim_base.setdefault(s.dimension, []).append(s.score)
    for s in new_scores:
        by_dim_new.setdefault(s.dimension, []).append(s.score)

    out: dict[str, float] = {}
    for dim, new_vals in by_dim_new.items():
        base_vals = by_dim_base.get(dim, [])
        if not base_vals:
            continue
        out[dim] = (sum(new_vals) / len(new_vals)) - (sum(base_vals) / len(base_vals))
    return out
