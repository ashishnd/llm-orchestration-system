"""Eval harness: 15 test cases, multi-dimensional scorer, runner, meta-agent."""

from app.eval.cases import CASES, TestCase, cases_by_category, get_case
from app.eval.meta_agent import propose_rewrite
from app.eval.runner import CaseResult, rerun_failed_cases, run_eval, run_one_case
from app.eval.scorer import DimensionScore, score_case

__all__ = [
    "CASES",
    "CaseResult",
    "DimensionScore",
    "TestCase",
    "cases_by_category",
    "get_case",
    "propose_rewrite",
    "rerun_failed_cases",
    "run_eval",
    "run_one_case",
    "score_case",
]
