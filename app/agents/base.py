"""Base agent.

Every pipeline agent inherits from `BaseAgent`. The base provides:
- Budget declaration on entry (each agent says how many tokens it needs).
- Prompt assembly from the registry + agent-specific kwargs.
- LLM call (JSON mode by default; non-JSON for narrative agents).
- Structured-output validation against the agent's declared output type.
- Policy violation logging on schema/budget failures.
- A consistent AgentOutput envelope on success.

Agents implement three things:
- `name`: stable identifier matching the prompt registry namespace.
- `output_type`: the Pydantic model the structured_output should match.
- `build_messages(ctx, **kwargs)`: render prompts into LLMMessage list.

The base handles the rest. Concrete agents add agent-specific behavior in
overridable hooks (e.g. RetrievalAgent owns the multi-hop loop and only
calls `super().run()` for the final summarization).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.context import (
    AgentOutput,
    BudgetManager,
    BudgetViolation,
    PolicyViolation,
    SharedContext,
    StructuredPipelineOutput,
    hash_prompt,
)
from app.llm import LLMClient, LLMMessage, MalformedJSONError

OutputT = TypeVar("OutputT", bound=BaseModel)


class AgentExecutionError(Exception):
    """Raised when an agent run fails non-recoverably.

    The base agent records a PolicyViolation on the shared context BEFORE
    raising, so the orchestrator (which catches this) has the audit trail
    even if it decides to abort the whole pipeline.
    """


class BaseAgent(ABC, Generic[OutputT]):
    """Abstract base for all pipeline agents.

    Subclasses must set:
        name: ClassVar[str]            — registry namespace, e.g. "decomposition"
        output_type: type[OutputT]     — the Pydantic model for structured_output
        default_budget: ClassVar[int]  — declared max-token budget per turn

    And implement:
        build_messages(ctx, **kwargs) -> list[LLMMessage]
    """

    name: ClassVar[str]
    default_budget: ClassVar[int] = 3000
    json_mode: ClassVar[bool] = True
    output_type: type[OutputT]

    def __init__(self, llm: LLMClient, budgets: BudgetManager) -> None:
        self._llm = llm
        self._budgets = budgets

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_messages(self, ctx: SharedContext, **kwargs: Any) -> list[LLMMessage]:
        """Render the agent's prompts into a message list.

        Subclasses load prompts via `get_prompt(key)` and substitute via
        `.format(**vars)`. The base agent doesn't prescribe a specific
        prompt-key layout because different agents have different needs
        (some have only a system prompt, some have multi-turn templates).
        """

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        ctx: SharedContext,
        *,
        budget: int | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Execute one turn.

        On success: return an AgentOutput; the caller (orchestrator) appends
        it to `ctx.agent_outputs`.
        On schema violation: log a PolicyViolation and raise
        AgentExecutionError. The orchestrator decides whether to retry,
        skip, or abort.
        On budget violation: log a PolicyViolation and raise BudgetViolation.
        """
        max_tokens = budget if budget is not None else self.default_budget
        self._budgets.declare(self.name, max_tokens)

        messages = self.build_messages(ctx, **kwargs)
        rendered = "\n\n".join(m.content for m in messages)
        prompt_h = hash_prompt(rendered)

        # Pre-flight budget check on the assembled prompt. The orchestrator
        # may choose to invoke compression first if this raises.
        try:
            self._budgets.check(self.name, self._llm.count_message_tokens(messages))
        except BudgetViolation as e:
            self._budgets.record_violation(ctx, self.name, str(e))
            raise

        return await self._invoke_llm(ctx, messages, prompt_h)

    async def _invoke_llm(
        self,
        ctx: SharedContext,
        messages: list[LLMMessage],
        prompt_hash: str,
    ) -> AgentOutput:
        """Issue the LLM call and validate the response.

        Separated so subclasses can override the call shape (e.g. retrieval
        wraps this in a multi-hop loop) without re-implementing budget
        accounting and validation.
        """
        try:
            if self.json_mode:
                parsed, resp = await self._llm.complete_json(messages)
                structured = self._validate_output(ctx, parsed, raw=resp.text)
                content = self._render_content(structured)
            else:
                resp = await self._llm.complete(messages)
                structured = None
                content = resp.text
        except MalformedJSONError as e:
            ctx.policy_violations.append(
                PolicyViolation(
                    agent_name=self.name,
                    violation_type="schema_violation",
                    detail="LLM returned malformed JSON twice; aborting agent turn",
                    raw_preview=e.raw_preview,
                )
            )
            raise AgentExecutionError(f"{self.name}: malformed JSON after repair retry") from e

        self._budgets.consume(self.name, resp.input_tokens + resp.output_tokens)

        return AgentOutput(
            agent_name=self.name,
            content=content,
            structured_output=structured,  # type: ignore[arg-type]
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            prompt_hash=prompt_hash,
        )

    def _validate_output(
        self, ctx: SharedContext, parsed: dict, raw: str
    ) -> StructuredPipelineOutput | None:
        """Validate the parsed JSON against the agent's declared output_type.

        On validation failure, log a schema_violation PolicyViolation with a
        raw_preview and raise AgentExecutionError. We do NOT silently coerce
        or drop bad fields — the system's posture is "log and fail loudly."
        """
        try:
            return self.output_type.model_validate(parsed)  # type: ignore[return-value]
        except ValidationError as e:
            ctx.policy_violations.append(
                PolicyViolation(
                    agent_name=self.name,
                    violation_type="schema_violation",
                    detail=f"Output failed {self.output_type.__name__} validation: {e}",
                    raw_preview=raw,
                )
            )
            raise AgentExecutionError(f"{self.name}: output failed schema validation") from e

    def _render_content(self, structured: BaseModel) -> str:
        """Produce a short human-readable summary for `AgentOutput.content`.

        Default is the type name + a one-line summary delegated to the model
        if it implements `__str__`. Subclasses override for more useful
        content (e.g. RetrievalAgent.content includes the answer_draft).
        """
        return f"{self.output_type.__name__} produced"
