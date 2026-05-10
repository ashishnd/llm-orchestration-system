"""API endpoint tests.

We substitute the AppContainer with fakes (in-memory sqlite, fake LLM,
empty event bus) and exercise the four non-streaming endpoints directly.
SSE behavior is covered by the orchestrator + streaming tests; here we
just verify the route wiring, schemas, and error responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.deps import AppContainer, set_container
from app.api.main import app
from app.persistence import repository
from app.persistence.models import Base
from app.streaming import EventBus


@pytest.fixture
async def app_with_fakes():
    """Build the FastAPI app with an in-memory DB and fake non-DB deps."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    container = AppContainer(
        llm=MagicMock(),  # not used by the endpoints we test
        vector_store=MagicMock(),
        event_bus=EventBus(),
        sessionmaker=Session,
    )
    set_container(container)

    yield app, Session
    await engine.dispose()


@pytest.fixture
async def client(app_with_fakes):
    application, _ = app_with_fakes
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_healthz_returns_ok(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# GET /jobs/{job_id}/trace
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_trace_returns_persisted_context(client, app_with_fakes):
    """A trace persisted by the repo should round-trip through the API."""
    from app.context import SharedContext

    _, Session = app_with_fakes
    async with Session() as session:
        job_id = await repository.create_job(session, user_query="What is BERT?")
        ctx = SharedContext(job_id=job_id, user_query="What is BERT?")
        ctx.final_answer = "It's a transformer encoder."
        await repository.save_trace(session, ctx)

    r = await client.get(f"/jobs/{job_id}/trace")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["context"]["final_answer"] == "It's a transformer encoder."


@pytest.mark.asyncio
async def test_get_trace_404_for_unknown_job(client):
    r = await client.get("/jobs/does_not_exist/trace")
    assert r.status_code == 404
    body = r.json()
    assert body["error_code"] == "trace_not_found"
    assert body["job_id"] == "does_not_exist"


# --------------------------------------------------------------------------- #
# GET /eval/latest
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_eval_latest_returns_summary_grouped_by_category(client, app_with_fakes):
    _, Session = app_with_fakes
    async with Session() as session:
        run_id = await repository.create_eval_run(session, prompt_snapshot={})
        # Two baseline cases, one adversarial; two dimensions
        for case_id, category, dim, score in [
            ("b1", "baseline", "answer_correctness", 0.9),
            ("b2", "baseline", "answer_correctness", 0.7),
            ("b1", "baseline", "citation_accuracy", 0.8),
            ("a1", "adversarial", "answer_correctness", 0.4),
        ]:
            await repository.add_eval_score(
                session,
                eval_run_id=run_id,
                test_case_id=case_id,
                test_case_category=category,
                dimension=dim,
                score=score,
                justification="...",
            )
        await repository.complete_eval_run(session, run_id)

    r = await client.get("/eval/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["eval_run_id"] == run_id
    assert body["n_test_cases"] == 3  # b1, b2, a1

    by_cat = {c["category"]: c for c in body["by_category"]}
    assert "baseline" in by_cat and "adversarial" in by_cat
    baseline_dims = {d["dimension"]: d for d in by_cat["baseline"]["dimensions"]}
    assert baseline_dims["answer_correctness"]["mean_score"] == pytest.approx(0.8)
    assert baseline_dims["answer_correctness"]["n"] == 2


@pytest.mark.asyncio
async def test_eval_latest_404_when_no_runs(client):
    r = await client.get("/eval/latest")
    assert r.status_code == 404
    assert r.json()["error_code"] == "no_eval_run"


# --------------------------------------------------------------------------- #
# POST /prompt-rewrites/{id}/decide
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_decide_rewrite_approve_swaps_active_prompt(client, app_with_fakes):
    """Approving a rewrite must update the active prompt registry."""
    from app.agents.prompts import _ACTIVE, _BASELINE, get_prompt, register_prompt

    # Snapshot + restore around the test
    baseline_snap = dict(_BASELINE)
    active_snap = dict(_ACTIVE)
    try:
        register_prompt("test.target_prompt", "ORIGINAL TEXT")

        _, Session = app_with_fakes
        async with Session() as session:
            run_id = await repository.create_eval_run(session, prompt_snapshot={})
            rewrite_id = await repository.create_prompt_rewrite(
                session,
                eval_run_id=run_id,
                target_prompt_key="test.target_prompt",
                old_prompt="ORIGINAL TEXT",
                proposed_prompt="REWRITTEN TEXT",
                structured_diff="--- old\n+++ new",
                rationale="for testing",
            )

        r = await client.post(
            f"/prompt-rewrites/{rewrite_id}/decide",
            json={"approved": True, "decided_by": "tester"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "approved"
        assert get_prompt("test.target_prompt") == "REWRITTEN TEXT"
    finally:
        _BASELINE.clear()
        _BASELINE.update(baseline_snap)
        _ACTIVE.clear()
        _ACTIVE.update(active_snap)


@pytest.mark.asyncio
async def test_decide_rewrite_404_for_unknown_id(client):
    r = await client.post(
        "/prompt-rewrites/does_not_exist/decide",
        json={"approved": True},
    )
    assert r.status_code == 404
    assert r.json()["error_code"] == "rewrite_not_found"


@pytest.mark.asyncio
async def test_decide_rewrite_400_when_already_decided(client, app_with_fakes):
    _, Session = app_with_fakes
    async with Session() as session:
        run_id = await repository.create_eval_run(session, prompt_snapshot={})
        rid = await repository.create_prompt_rewrite(
            session,
            eval_run_id=run_id,
            target_prompt_key="x",
            old_prompt="o",
            proposed_prompt="n",
            structured_diff="diff",
            rationale="r",
        )
        await repository.decide_prompt_rewrite(session, rid, approved=False)

    r = await client.post(f"/prompt-rewrites/{rid}/decide", json={"approved": True})
    assert r.status_code == 400
    assert r.json()["error_code"] == "rewrite_already_decided"


# --------------------------------------------------------------------------- #
# POST /eval/rerun-failed (stub for now)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rerun_failed_identifies_below_threshold_cases(client, app_with_fakes, monkeypatch):
    """Endpoint identifies failed cases and delegates to the runner. We mock
    the runner since real execution requires a live LLM."""
    _, Session = app_with_fakes
    async with Session() as session:
        run_id = await repository.create_eval_run(session, prompt_snapshot={})
        for case_id, score in [
            ("c_pass", 0.9),
            ("c_fail_one", 0.4),
            ("c_fail_two", 0.5),
        ]:
            await repository.add_eval_score(
                session,
                eval_run_id=run_id,
                test_case_id=case_id,
                test_case_category="baseline",
                dimension="answer_correctness",
                score=score,
                justification="...",
            )

    # Patch the runner's rerun_failed_cases so we don't try to call the LLM
    async def fake_runner(*, session, llm, vector_store, base_eval_run_id, score_threshold):
        base = await repository.get_eval_scores(session, base_eval_run_id)
        failed = sorted({s.test_case_id for s in base if s.score < score_threshold})
        return ("new-run-id", failed, {"answer_correctness": 0.2})

    from app import eval as eval_pkg

    monkeypatch.setattr(eval_pkg, "rerun_failed_cases", fake_runner)

    r = await client.post(
        "/eval/rerun-failed",
        json={"base_eval_run_id": run_id, "score_threshold": 0.6},
    )
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["rerun_case_ids"]) == ["c_fail_one", "c_fail_two"]
    assert body["new_eval_run_id"] == "new-run-id"
    assert body["delta_summary"]["answer_correctness"] == 0.2


@pytest.mark.asyncio
async def test_rerun_failed_404_for_unknown_base_run(client):
    r = await client.post(
        "/eval/rerun-failed",
        json={"base_eval_run_id": "does_not_exist", "score_threshold": 0.6},
    )
    assert r.status_code == 404
    assert r.json()["error_code"] == "base_run_not_found"
