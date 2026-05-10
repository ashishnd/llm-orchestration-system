"""SQLAlchemy models for persistence.

Schema:
- jobs: one row per submitted query (id, user_query, status, final_answer,
  timestamps)
- job_traces: full SharedContext serialized as JSONB so any job can be
  replayed exactly. One row per job.
- eval_runs: one row per `python -m scripts.run_eval` invocation
- eval_scores: per-test-case-per-dimension scores (15 cases × N dimensions)
- prompt_rewrites: meta-agent proposals + human approval/rejection state

We use Alembic-free schema management (Base.metadata.create_all) for
assessment scope. Documented in Known Limitations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


# JSONB for postgres, JSON otherwise (sqlite for tests)
def _json_column():
    """Use JSONB on Postgres, JSON elsewhere. Decided at table-create time
    via SQLAlchemy's dialect-specific column types."""
    from sqlalchemy.dialects.postgresql import JSONB

    return JSON().with_variant(JSONB, "postgresql")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobTrace(Base):
    """Full SharedContext as JSON. Source of truth for trace replay."""

    __tablename__ = "job_traces"

    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("jobs.id"), primary_key=True)
    context_json: Mapped[dict[str, Any]] = mapped_column(_json_column(), nullable=False)
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Snapshot of active prompts at run time, so re-runs can detect prompt
    # changes between runs.
    prompt_snapshot: Mapped[dict[str, Any]] = mapped_column(_json_column(), default=dict)

    scores: Mapped[list[EvalScore]] = relationship(
        "EvalScore", back_populates="eval_run", cascade="all, delete-orphan"
    )


class EvalScore(Base):
    """One score for one (eval_run, test_case, dimension) tuple."""

    __tablename__ = "eval_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[str] = mapped_column(String(32), ForeignKey("eval_runs.id"), nullable=False)
    test_case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    test_case_category: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # baseline | ambiguous | adversarial
    dimension: Mapped[str] = mapped_column(String(48), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    job_id: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # link back to the run that produced this score

    eval_run: Mapped[EvalRun] = relationship("EvalRun", back_populates="scores")


class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    eval_run_id: Mapped[str] = mapped_column(String(32), ForeignKey("eval_runs.id"), nullable=False)
    target_prompt_key: Mapped[str] = mapped_column(String(96), nullable=False)
    old_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    structured_diff: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    worst_dimension: Mapped[str | None] = mapped_column(String(48), nullable=True)
    worst_case_ids: Mapped[list[str]] = mapped_column(_json_column(), default=list)
    status: Mapped[str] = mapped_column(
        String(16), default="pending"
    )  # pending | approved | rejected
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
