"""FastAPI application with the five required endpoints.

1. POST /jobs                         submit query, return SSE stream
2. GET  /jobs/{job_id}/trace          retrieve full execution trace
3. GET  /eval/latest                  latest eval run summary by category/dimension
4. POST /prompt-rewrites/{id}/decide  human approval/rejection
5. POST /eval/rerun-failed            targeted re-eval on previously failed cases

Error responses follow the brief: machine-readable code, human-readable
message, optional job_id.

The orchestrator runs in the *foreground* of the SSE request so the stream
yields events as the pipeline executes. A real production setup would push
the run to RQ / Celery / etc. and stream from a Redis pub/sub; for assessment
scope, in-process is honest and simpler.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.deps import AppContainer, get_container, set_container
from app.api.schemas import (
    CategorySummary,
    DecideRewriteRequest,
    DecideRewriteResponse,
    DimensionSummary,
    ErrorResponse,
    EvalSummaryResponse,
    RerunFailedRequest,
    RerunFailedResponse,
    SubmitJobRequest,
    TraceResponse,
)
from app.context import BudgetManager, SharedContext
from app.llm import get_llm_client
from app.orchestrator import Orchestrator
from app.persistence import get_sessionmaker, init_db, repository
from app.persistence.models import EvalScore
from app.rag import make_persistent_store
from app.settings import get_settings
from app.streaming import get_event_bus

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan: build the container at startup
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    container = AppContainer(
        llm=get_llm_client(),
        vector_store=make_persistent_store(settings.chroma_persist_dir, settings.openai_api_key),
        event_bus=get_event_bus(),
        sessionmaker=get_sessionmaker(),
    )
    set_container(container)
    yield


app = FastAPI(
    title="Multi-Agent LLM Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Session dependency
# --------------------------------------------------------------------------- #


async def get_session() -> AsyncSession:
    container = get_container()
    async with container.sessionmaker() as session:
        yield session


# --------------------------------------------------------------------------- #
# Error helper
# --------------------------------------------------------------------------- #


def _error(status_code: int, code: str, message: str, job_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error_code=code, message=message, job_id=job_id).model_dump(),
    )


# --------------------------------------------------------------------------- #
# 1. POST /jobs — submit query, return SSE stream
# --------------------------------------------------------------------------- #


async def _run_pipeline_in_background(
    container: AppContainer,
    job_id: str,
    user_query: str,
):
    """Run the full pipeline. Always emit DONE so the consumer terminates."""
    ctx = SharedContext(job_id=job_id, user_query=user_query)
    budgets = BudgetManager(llm=container.llm)
    orchestrator = Orchestrator(
        llm=container.llm,
        budgets=budgets,
        vector_store=container.vector_store,
        event_bus=container.event_bus,
    )
    try:
        result = await orchestrator.run(ctx)
        # Persist trace + final answer
        async with container.sessionmaker() as session:
            await repository.save_trace(session, ctx)
            await repository.mark_job_completed(session, job_id, final_answer=result.final_answer)
    except Exception as e:
        log.exception("Pipeline failed for job %s", job_id)
        from app.streaming import emit_done, emit_error

        await emit_error(container.event_bus, job_id, str(e))
        await emit_done(container.event_bus, job_id)


@app.post("/jobs", responses={400: {"model": ErrorResponse}})
async def submit_job(
    req: SubmitJobRequest,
    session: AsyncSession = Depends(get_session),
):
    """Submit a query. Returns an SSE stream of pipeline events.

    The job_id is also embedded in the first event so clients can later
    fetch the persisted trace. We run the orchestrator concurrently with
    the SSE consumer so the stream sees events as they happen.
    """
    container = get_container()
    job_id = await repository.create_job(session, user_query=req.query)
    container.event_bus.register(job_id)

    asyncio.create_task(_run_pipeline_in_background(container, job_id, req.query))

    async def event_generator():
        # First event: job_id so clients can correlate with later trace fetches
        yield {
            "event": "job_started",
            "data": f'{{"job_id": "{job_id}"}}',
        }
        async for event in container.event_bus.consume(job_id):
            yield {
                "event": event.type.value,
                "data": event.to_sse_data(),
            }

    return EventSourceResponse(event_generator())


# --------------------------------------------------------------------------- #
# 2. GET /jobs/{job_id}/trace
# --------------------------------------------------------------------------- #


@app.get(
    "/jobs/{job_id}/trace",
    response_model=TraceResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_trace(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    trace = await repository.get_trace(session, job_id)
    if trace is None:
        return _error(
            status.HTTP_404_NOT_FOUND,
            "trace_not_found",
            f"No trace found for job_id {job_id!r}",
            job_id=job_id,
        )
    return TraceResponse(job_id=job_id, context=trace)


# --------------------------------------------------------------------------- #
# 3. GET /eval/latest
# --------------------------------------------------------------------------- #


@app.get(
    "/eval/latest",
    response_model=EvalSummaryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_latest_eval(session: AsyncSession = Depends(get_session)):
    run = await repository.get_latest_eval_run(session)
    if run is None:
        return _error(
            status.HTTP_404_NOT_FOUND,
            "no_eval_run",
            "No eval run has been completed yet.",
        )

    scores = await repository.get_eval_scores(session, run.id)
    return EvalSummaryResponse(
        eval_run_id=run.id,
        started_at=run.started_at,
        completed_at=run.completed_at,
        by_category=_summarize_by_category(scores),
        n_test_cases=len({s.test_case_id for s in scores}),
    )


def _summarize_by_category(scores: list[EvalScore]) -> list[CategorySummary]:
    """Group scores by (category, dimension), compute mean."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        grouped[s.test_case_category][s.dimension].append(s.score)

    out: list[CategorySummary] = []
    for category, dims in grouped.items():
        out.append(
            CategorySummary(
                category=category,
                dimensions=[
                    DimensionSummary(
                        dimension=dim,
                        mean_score=sum(values) / len(values),
                        n=len(values),
                    )
                    for dim, values in dims.items()
                ],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 4. POST /prompt-rewrites/{rewrite_id}/decide
# --------------------------------------------------------------------------- #


@app.post(
    "/prompt-rewrites/{rewrite_id}/decide",
    response_model=DecideRewriteResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def decide_prompt_rewrite(
    rewrite_id: str,
    req: DecideRewriteRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        rw = await repository.decide_prompt_rewrite(
            session,
            rewrite_id,
            approved=req.approved,
            decided_by=req.decided_by,
        )
    except KeyError:
        return _error(
            status.HTTP_404_NOT_FOUND,
            "rewrite_not_found",
            f"No rewrite found with id {rewrite_id!r}",
        )
    except ValueError as e:
        return _error(
            status.HTTP_400_BAD_REQUEST,
            "rewrite_already_decided",
            str(e),
        )

    # If approved, swap the active prompt at runtime.
    if req.approved:
        from app.agents.prompts import set_active_prompt

        try:
            set_active_prompt(rw.target_prompt_key, rw.proposed_prompt)
        except KeyError:
            log.warning("Approved rewrite targets unknown prompt key %s", rw.target_prompt_key)

    return DecideRewriteResponse(
        rewrite_id=rw.id,
        status=rw.status,
        target_prompt_key=rw.target_prompt_key,
        decided_at=rw.decided_at,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# 5. POST /eval/rerun-failed
# --------------------------------------------------------------------------- #


@app.post(
    "/eval/rerun-failed",
    response_model=RerunFailedResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def rerun_failed(
    req: RerunFailedRequest,
    session: AsyncSession = Depends(get_session),
):
    """Re-run only the test cases that scored below `score_threshold` on any
    dimension in `base_eval_run_id`, using whatever prompts are currently
    active. Returns a delta summary against the base run.
    """
    from app.eval import rerun_failed_cases

    container = get_container()
    try:
        new_run_id, failed_ids, delta = await rerun_failed_cases(
            session=session,
            llm=container.llm,
            vector_store=container.vector_store,
            base_eval_run_id=req.base_eval_run_id,
            score_threshold=req.score_threshold,
        )
    except KeyError:
        return _error(
            status.HTTP_404_NOT_FOUND,
            "base_run_not_found",
            f"No scores found for eval_run_id {req.base_eval_run_id!r}",
        )

    return RerunFailedResponse(
        new_eval_run_id=new_run_id,
        rerun_case_ids=failed_ids,
        delta_summary=delta,
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@app.get("/healthz")
async def health():
    return {"status": "ok"}
