"""API request and response schemas.

Kept separate from app.context schemas because the API surface is
external-contract — we don't want a refactor of an internal Pydantic
model to silently change the API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ErrorResponse(BaseModel):
    """Per the brief: machine-readable code, human-readable message, optional job_id."""

    error_code: str
    message: str
    job_id: str | None = None


# --------------------------------------------------------------------------- #
# POST /jobs
# --------------------------------------------------------------------------- #


class SubmitJobRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


# --------------------------------------------------------------------------- #
# GET /jobs/{job_id}/trace
# --------------------------------------------------------------------------- #


class TraceResponse(BaseModel):
    job_id: str
    context: dict[str, Any]


# --------------------------------------------------------------------------- #
# GET /eval/latest
# --------------------------------------------------------------------------- #


class DimensionSummary(BaseModel):
    dimension: str
    mean_score: float
    n: int


class CategorySummary(BaseModel):
    category: str  # baseline | ambiguous | adversarial
    dimensions: list[DimensionSummary]


class EvalSummaryResponse(BaseModel):
    eval_run_id: str
    started_at: datetime
    completed_at: datetime | None
    by_category: list[CategorySummary]
    n_test_cases: int


# --------------------------------------------------------------------------- #
# POST /prompt-rewrites/{rewrite_id}/decide
# --------------------------------------------------------------------------- #


class DecideRewriteRequest(BaseModel):
    approved: bool
    decided_by: str | None = None


class DecideRewriteResponse(BaseModel):
    rewrite_id: str
    status: str  # approved | rejected
    target_prompt_key: str
    decided_at: datetime


# --------------------------------------------------------------------------- #
# POST /eval/rerun-failed
# --------------------------------------------------------------------------- #


class RerunFailedRequest(BaseModel):
    base_eval_run_id: str  # the eval whose failed cases we re-run
    score_threshold: float = 0.6  # cases below this on any dimension count as "failed"


class RerunFailedResponse(BaseModel):
    new_eval_run_id: str
    rerun_case_ids: list[str]
    delta_summary: dict[str, float]  # dimension -> mean score delta vs base run
