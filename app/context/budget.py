"""Per-agent context budget tracking.

The brief requires:
- Each agent declares its max budget before execution.
- Agents can query remaining budget before adding to their context.
- Overflow is a *logged policy violation*, not silent truncation.
- A compression agent rewrites older context when the threshold is hit, and
  compression is lossless for structured data (tool outputs, citations) and
  lossy only for conversational filler.

This module owns budget bookkeeping. The compression agent itself lives in
app/agents/compression.py — this module decides *when* to compress, not *how*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.context.schema import PolicyViolation, SharedContext


class TokenCounter(Protocol):
    """Anything with a count_tokens method. Decouples budgeting from the LLM client."""

    def count_tokens(self, text: str) -> int: ...


@dataclass
class AgentBudget:
    agent_name: str
    max_tokens: int
    used_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def utilization(self) -> float:
        if self.max_tokens == 0:
            return 1.0
        return self.used_tokens / self.max_tokens


class BudgetViolation(Exception):
    """Raised when an agent would exceed its declared budget.

    The orchestrator catches this, logs a PolicyViolation, and either invokes
    the compression agent or refuses to proceed. We do NOT silently truncate.
    """

    def __init__(self, agent_name: str, requested: int, remaining: int) -> None:
        self.agent_name = agent_name
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Agent {agent_name!r} requested {requested} tokens but only {remaining} "
            f"remain in budget"
        )


@dataclass
class BudgetManager:
    """Tracks token consumption per agent per job."""

    llm: TokenCounter
    compression_threshold: float = 0.85
    budgets: dict[str, AgentBudget] = field(default_factory=dict)

    def declare(self, agent_name: str, max_tokens: int) -> AgentBudget:
        """Called at the start of an agent's turn to declare its budget.

        Re-declaring resets used_tokens for that agent, since each turn is
        independent. (A single agent invoked twice in the pipeline gets a
        fresh budget each time.)
        """
        budget = AgentBudget(agent_name=agent_name, max_tokens=max_tokens)
        self.budgets[agent_name] = budget
        return budget

    def check(self, agent_name: str, additional_tokens: int) -> int:
        """Returns remaining budget if the addition fits; raises BudgetViolation otherwise.

        Agents call this *before* assembling context to decide whether to
        request compression or proceed.
        """
        budget = self._require(agent_name)
        if additional_tokens > budget.remaining:
            raise BudgetViolation(agent_name, additional_tokens, budget.remaining)
        return budget.remaining - additional_tokens

    def consume(self, agent_name: str, tokens: int) -> None:
        """Record actual usage. Called after a context assembly."""
        budget = self._require(agent_name)
        budget.used_tokens += tokens

    def needs_compression(self, agent_name: str) -> bool:
        """True if utilization has crossed the compression threshold."""
        budget = self._require(agent_name)
        return budget.utilization >= self.compression_threshold

    def remaining(self, agent_name: str) -> int:
        return self._require(agent_name).remaining

    def utilization(self, agent_name: str) -> float:
        return self._require(agent_name).utilization

    def record_violation(
        self,
        ctx: SharedContext,
        agent_name: str,
        detail: str,
    ) -> None:
        """Append a policy violation to the shared context.

        The brief specifies that overflow must be 'caught and logged as a
        policy violation, not silently truncated'. This is the logging side.
        """
        ctx.policy_violations.append(
            PolicyViolation(
                agent_name=agent_name,
                violation_type="budget_overflow",
                detail=detail,
            )
        )

    def _require(self, agent_name: str) -> AgentBudget:
        if agent_name not in self.budgets:
            raise KeyError(
                f"Agent {agent_name!r} did not declare a budget before checking it. "
                "Call declare() at the start of the agent's turn."
            )
        return self.budgets[agent_name]
