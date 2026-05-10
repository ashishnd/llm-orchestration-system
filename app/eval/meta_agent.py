"""Meta-agent: proposes prompt rewrites for the worst-performing prompt.

Reads scores from a completed EvalRun, identifies the prompt-key whose
agent scored worst on its weakest dimension, and emits a structured
rewrite proposal. The proposal is persisted to prompt_rewrites with
status='pending' — application of the rewrite waits for human approval
via /prompt-rewrites/{id}/decide.

Mapping: each prompt_key maps to an agent (the substring before the
first dot, e.g. "decomposition.system" -> "decomposition"). We pick
the agent with the worst score on its weakest dimension, then pick
the .system prompt for that agent (the heaviest-weight prompt, the
one most likely to influence behavior).
"""

from __future__ import annotations

import difflib
import logging
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompts import all_keys, get_prompt
from app.context import MetaRewriteOutput
from app.llm import LLMClient, LLMMessage, MalformedJSONError
from app.persistence import repository
from app.persistence.models import EvalScore

log = logging.getLogger(__name__)


_AGENT_TO_DEFAULT_PROMPT_KEY = {
    "decomposition": "decomposition.system",
    "retrieval": "retrieval.synthesis_system",
    "critique": "critique.system",
    "synthesis": "synthesis.system",
    "compression": "compression.system",
}


# Map scoring dimensions to the agent that primarily influences them, so
# a low score on a dimension points us to the right prompt to rewrite.
_DIMENSION_TO_AGENT = {
    "answer_correctness": "synthesis",
    "citation_accuracy": "retrieval",
    "contradiction_resolution": "synthesis",
    "tool_selection_efficiency": "decomposition",
    "context_budget_compliance": "compression",
    "critique_agreement_rate": "critique",
}


_REWRITE_SYSTEM = """You are the Meta-Agent. You read a prompt that has been \
producing poor results on a specific scoring dimension and propose a \
rewritten version.

Constraints on your rewrite:
- Preserve the prompt's structural contract (any required JSON shape, \
schema rules, output format) unchanged. Reviewers reject rewrites that \
break the contract.
- Improve guidance on the failing dimension. Be specific about what to do \
differently, not vague exhortations.
- Match the original tone and voice. Don't add markdown if the original is \
plain text, don't add headers if the original is paragraphs.
- Don't add references to specific test cases or eval scores. The rewrite \
must be general.

Return JSON ONLY:
{
  "proposed_prompt": "<full rewritten prompt text>",
  "rationale": "<1-3 sentences on what you changed and why>"
}"""


_REWRITE_USER = """Prompt key: {key}

Failing dimension: {dimension}
Mean score on this dimension: {score}

Sample failure justifications:
{justifications}

Current prompt:
\"\"\"{old_prompt}\"\"\"

Propose a rewritten version. JSON ONLY."""


@dataclass
class _DimAgentScore:
    agent: str
    dimension: str
    mean_score: float
    n_samples: int
    sample_justifications: list[str]


async def propose_rewrite(
    *,
    session: AsyncSession,
    eval_run_id: str,
    judge_llm: LLMClient,
) -> MetaRewriteOutput | None:
    """Identify the worst (agent, dimension) pair in the run and propose a
    rewrite for that agent's primary prompt. Persists the proposal as
    PromptRewrite(status='pending'). Returns the proposal or None if the
    run had no problematic dimensions (every score >= 0.7) or the
    rewrite call failed."""
    scores = await repository.get_eval_scores(session, eval_run_id)
    if not scores:
        log.info("No scores for eval run %s", eval_run_id)
        return None

    worst = _find_worst_dimension(scores)
    if worst is None:
        log.info("No dimensions below threshold for eval run %s", eval_run_id)
        return None

    target_key = _AGENT_TO_DEFAULT_PROMPT_KEY.get(worst.agent)
    if target_key is None or target_key not in all_keys():
        log.warning("No registered prompt key for agent %s; skipping rewrite", worst.agent)
        return None

    old_prompt = get_prompt(target_key)
    proposed_text, rationale = await _llm_propose(judge_llm, target_key, worst, old_prompt)
    if proposed_text is None:
        return None

    diff = "\n".join(
        difflib.unified_diff(
            old_prompt.splitlines(),
            proposed_text.splitlines(),
            fromfile="baseline",
            tofile="proposed",
            lineterm="",
        )
    )

    failed_case_ids = sorted({s.test_case_id for s in scores if s.score < 0.6})
    rewrite_id = await repository.create_prompt_rewrite(
        session,
        eval_run_id=eval_run_id,
        target_prompt_key=target_key,
        old_prompt=old_prompt,
        proposed_prompt=proposed_text,
        structured_diff=diff,
        rationale=rationale,
        worst_dimension=worst.dimension,
        worst_case_ids=failed_case_ids,
    )
    log.info(
        "Created prompt rewrite %s targeting %s (worst dim %s, mean=%.2f)",
        rewrite_id,
        target_key,
        worst.dimension,
        worst.mean_score,
    )

    return MetaRewriteOutput(
        eval_run_id=eval_run_id,
        target_prompt_key=target_key,
        old_prompt=old_prompt,
        proposed_prompt=proposed_text,
        structured_diff=diff,
        rationale=rationale,
        worst_dimension=worst.dimension,
        worst_case_ids=failed_case_ids,
    )


def _find_worst_dimension(scores: list[EvalScore]) -> _DimAgentScore | None:
    """Group by dimension, find the lowest mean. Returns None if all means
    are above the 'concerning' threshold (0.7) — no rewrite needed."""
    by_dim: dict[str, list[EvalScore]] = defaultdict(list)
    for s in scores:
        by_dim[s.dimension].append(s)

    candidates: list[_DimAgentScore] = []
    for dim, dim_scores in by_dim.items():
        agent = _DIMENSION_TO_AGENT.get(dim)
        if agent is None:
            continue
        mean = sum(s.score for s in dim_scores) / len(dim_scores)
        if mean >= 0.7:
            continue  # not concerning enough to propose a rewrite
        # Pull a few sample justifications from the lowest-scoring rows
        sorted_low = sorted(dim_scores, key=lambda s: s.score)[:3]
        candidates.append(
            _DimAgentScore(
                agent=agent,
                dimension=dim,
                mean_score=mean,
                n_samples=len(dim_scores),
                sample_justifications=[s.justification for s in sorted_low],
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda c: c.mean_score)
    return candidates[0]


async def _llm_propose(
    llm: LLMClient,
    key: str,
    worst: _DimAgentScore,
    old_prompt: str,
) -> tuple[str | None, str]:
    """Call the meta-agent LLM. Returns (proposed_prompt, rationale) or
    (None, error_message) on failure."""
    user = _REWRITE_USER.format(
        key=key,
        dimension=worst.dimension,
        score=f"{worst.mean_score:.3f}",
        justifications="\n".join(f"- {j}" for j in worst.sample_justifications),
        old_prompt=old_prompt,
    )
    try:
        parsed, _ = await llm.complete_json(
            [
                LLMMessage(role="system", content=_REWRITE_SYSTEM),
                LLMMessage(role="user", content=user),
            ],
            temperature=0.2,
        )
    except MalformedJSONError:
        log.warning("Meta-agent returned malformed JSON; skipping rewrite proposal")
        return None, "malformed JSON"

    proposed = parsed.get("proposed_prompt")
    rationale = parsed.get("rationale", "")
    if not isinstance(proposed, str) or not proposed.strip():
        log.warning("Meta-agent returned empty proposed_prompt; skipping")
        return None, "empty proposal"
    if proposed.strip() == old_prompt.strip():
        log.info("Meta-agent produced identical rewrite; skipping")
        return None, "identical proposal"

    return proposed, str(rationale)
