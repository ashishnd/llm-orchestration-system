"""Multi-dimensional scorer.

Six scoring dimensions per the brief:
1. answer_correctness        — LLM judge against expected_facts
2. citation_accuracy         — chunk_id overlap with expected_chunk_ids
3. contradiction_resolution  — were critique disagreements addressed in synthesis?
4. tool_selection_efficiency — penalize unnecessary tool calls
5. context_budget_compliance — count of policy_violations
6. critique_agreement_rate   — mean confidence of critique claim_scores

Each dimension produces (score: float, justification: str). No black-box
framework; the scoring logic is in this file and reviewable.

Reproducibility:
- Mechanical dimensions (citation_accuracy, tool_selection_efficiency,
  context_budget_compliance, critique_agreement_rate) are pure functions
  of the SharedContext.
- LLM-judge dimensions (answer_correctness, contradiction_resolution) use
  temperature=0 and a fixed judge prompt. Same inputs -> same outputs
  (modulo upstream LLM nondeterminism, which is small at temp=0).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.context import (
    CritiqueOutput,
    SharedContext,
)
from app.eval.cases import TestCase
from app.llm import LLMClient, LLMMessage, MalformedJSONError


@dataclass
class DimensionScore:
    dimension: str
    score: float  # in [0, 1]
    justification: str


# --------------------------------------------------------------------------- #
# Mechanical dimensions
# --------------------------------------------------------------------------- #


def score_citation_accuracy(case: TestCase, ctx: SharedContext) -> DimensionScore:
    """Fraction of expected chunks that appear in the synthesis citations.

    If no expected chunks are specified (open-ended cases), score is 1.0
    when there's at least one citation, else 0.5 (citations exist but we
    can't verify them — neither pass nor fail).
    """
    synthesis = ctx.latest_synthesis()
    if synthesis is None:
        return DimensionScore("citation_accuracy", 0.0, "No synthesis output produced.")

    cited_chunks: set[str] = set()
    for entry in synthesis.provenance:
        for c in entry.citations:
            cited_chunks.add(c.chunk_id)

    if not case.expected_chunk_ids:
        if cited_chunks:
            return DimensionScore(
                "citation_accuracy",
                1.0,
                f"No expected chunks specified; {len(cited_chunks)} citations present (acceptable).",
            )
        return DimensionScore(
            "citation_accuracy",
            0.5,
            "No expected chunks specified and no citations produced.",
        )

    expected = set(case.expected_chunk_ids)
    matched = expected & cited_chunks
    score = len(matched) / len(expected)
    return DimensionScore(
        "citation_accuracy",
        score,
        f"Matched {len(matched)}/{len(expected)} expected chunks. "
        f"Cited: {sorted(cited_chunks)[:5]}.",
    )


def score_tool_selection_efficiency(case: TestCase, ctx: SharedContext) -> DimensionScore:
    """Penalize unnecessary tool calls and rejected results.

    1.0 if zero tool calls or all accepted; decreases with rejected results
    and excess invocations relative to a heuristic budget (1 tool call
    per sub-task is plenty).
    """
    invocations = ctx.tool_invocations
    if not invocations:
        return DimensionScore(
            "tool_selection_efficiency", 1.0, "No tool calls — nothing to penalize."
        )

    rejected = sum(1 for inv in invocations if inv.accepted_by_agent is False)
    n_sub_tasks = max(1, len(ctx.sub_tasks))
    budget = n_sub_tasks * 2  # generous
    excess = max(0, len(invocations) - budget)

    # Linear penalty, clipped at 0
    score = max(0.0, 1.0 - 0.15 * rejected - 0.1 * excess)
    return DimensionScore(
        "tool_selection_efficiency",
        score,
        f"{len(invocations)} call(s), {rejected} rejected, "
        f"{excess} excess over budget {budget}.",
    )


def score_context_budget_compliance(case: TestCase, ctx: SharedContext) -> DimensionScore:
    """1.0 minus 0.2 per budget_overflow violation, floored at 0."""
    overflows = [v for v in ctx.policy_violations if v.violation_type == "budget_overflow"]
    score = max(0.0, 1.0 - 0.2 * len(overflows))
    if not overflows:
        return DimensionScore("context_budget_compliance", 1.0, "No budget overflows.")
    return DimensionScore(
        "context_budget_compliance",
        score,
        f"{len(overflows)} budget overflow violation(s) recorded.",
    )


def score_critique_agreement(case: TestCase, ctx: SharedContext) -> DimensionScore:
    """Mean of all critique claim_scores' confidence values across all critique
    passes. High agreement = critique generally trusts the targets it reviewed.

    If there were disagreements, agreement is reduced by the disagreements'
    weight: each strong (>0.5 confidence) disagreement reduces score by 0.1.
    """
    critique_outputs = [
        c.structured_output
        for c in ctx.outputs_by_agent("critique")
        if isinstance(c.structured_output, CritiqueOutput)
    ]
    if not critique_outputs:
        return DimensionScore(
            "critique_agreement_rate",
            0.5,
            "No critique passes ran — neutral score.",
        )

    all_confidences = [cs.confidence for c in critique_outputs for cs in c.claim_scores]
    if not all_confidences:
        mean_conf = 0.5
    else:
        mean_conf = sum(all_confidences) / len(all_confidences)

    strong_disagreements = sum(
        1 for c in critique_outputs for d in c.disagreements if d.confidence > 0.5
    )
    score = max(0.0, mean_conf - 0.1 * strong_disagreements)
    return DimensionScore(
        "critique_agreement_rate",
        score,
        f"Mean claim confidence {mean_conf:.2f} across "
        f"{len(all_confidences)} claims; {strong_disagreements} strong disagreements.",
    )


# --------------------------------------------------------------------------- #
# LLM-judge dimensions
# --------------------------------------------------------------------------- #


_FACT_JUDGE_SYSTEM = """You are an impartial evaluator. Given a final answer and \
a list of expected facts that the answer should contain, score the answer's \
factual coverage on a 0.0-1.0 scale.

Return JSON ONLY:
{
  "score": <float 0..1>,
  "justification": "<one short sentence on which expected facts were/weren't covered>"
}

Scoring guide:
- 1.0: all expected facts clearly addressed
- 0.7: most facts addressed, minor gaps
- 0.4: some facts addressed, significant gaps
- 0.0: facts missing or contradicted

Be conservative; partial coverage is partial credit. A confident wrong answer \
should score lower than a hedged correct one."""


_RESOLUTION_JUDGE_SYSTEM = """You are an impartial evaluator. Given a synthesis \
output and the disagreements the critique agent flagged, score 0.0-1.0 how well \
the synthesis resolved those disagreements.

Return JSON ONLY:
{
  "score": <float 0..1>,
  "justification": "<one short sentence>"
}

Scoring guide:
- 1.0: every disagreement either incorporated or explicitly addressed in resolution_notes
- 0.5: some addressed, some ignored
- 0.0: disagreements completely ignored, or no disagreements existed (so nothing to resolve)

If there were zero disagreements, return 1.0 with justification "no contradictions to resolve"."""


async def score_answer_correctness(
    case: TestCase, ctx: SharedContext, judge_llm: LLMClient
) -> DimensionScore:
    """LLM judge against expected_facts. Temperature=0 for reproducibility."""
    if ctx.final_answer is None:
        return DimensionScore("answer_correctness", 0.0, "No final answer produced.")
    if not case.expected_facts:
        return DimensionScore("answer_correctness", 0.5, "No expected facts to judge against.")

    prompt = (
        f'Final answer:\n"""{ctx.final_answer}"""\n\n'
        f"Expected facts (the answer should cover these):\n"
        + "\n".join(f"- {f}" for f in case.expected_facts)
    )
    return await _llm_judge(judge_llm, "answer_correctness", _FACT_JUDGE_SYSTEM, prompt)


async def score_contradiction_resolution(
    case: TestCase, ctx: SharedContext, judge_llm: LLMClient
) -> DimensionScore:
    synthesis = ctx.latest_synthesis()
    if synthesis is None:
        return DimensionScore(
            "contradiction_resolution",
            0.0,
            "No synthesis output to evaluate resolution against.",
        )

    critique_outputs = [
        c.structured_output
        for c in ctx.outputs_by_agent("critique")
        if isinstance(c.structured_output, CritiqueOutput)
    ]
    all_disagreements = [d for c in critique_outputs for d in c.disagreements]
    if not all_disagreements:
        return DimensionScore(
            "contradiction_resolution",
            1.0,
            "No critique disagreements to resolve.",
        )

    disagreements_text = "\n".join(
        f"- [{d.span.text!r}] (target={d.target_agent}): {d.reason}" for d in all_disagreements
    )
    prompt = (
        f"Synthesis final_answer:\n\"\"\"{synthesis.final_answer}\"\"\"\n\n"
        f"Resolution notes:\n{synthesis.resolution_notes or '(none)'}\n\n"
        f"Critique disagreements that needed resolution:\n{disagreements_text}"
    )
    return await _llm_judge(judge_llm, "contradiction_resolution", _RESOLUTION_JUDGE_SYSTEM, prompt)


async def _llm_judge(
    llm: LLMClient,
    dimension: str,
    system: str,
    user: str,
) -> DimensionScore:
    """Run an LLM judge call. On malformed JSON, return a 0.5 with a note —
    we don't penalize the run for our judge's failure."""
    try:
        parsed, _ = await llm.complete_json(
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            temperature=0.0,
        )
    except MalformedJSONError:
        return DimensionScore(
            dimension,
            0.5,
            "Judge returned malformed JSON; neutral score assigned.",
        )

    score = parsed.get("score")
    justification = parsed.get("justification", "")
    if not isinstance(score, int | float) or not 0.0 <= score <= 1.0:
        return DimensionScore(
            dimension,
            0.5,
            f"Judge returned out-of-range score {score!r}; neutral score assigned.",
        )
    return DimensionScore(dimension, float(score), str(justification))


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #


async def score_case(
    case: TestCase, ctx: SharedContext, judge_llm: LLMClient
) -> list[DimensionScore]:
    """Score one test case across all six dimensions."""
    return [
        await score_answer_correctness(case, ctx, judge_llm),
        score_citation_accuracy(case, ctx),
        await score_contradiction_resolution(case, ctx, judge_llm),
        score_tool_selection_efficiency(case, ctx),
        score_context_budget_compliance(case, ctx),
        score_critique_agreement(case, ctx),
    ]
