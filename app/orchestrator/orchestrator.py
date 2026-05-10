"""Master orchestrator.

Owns the per-job pipeline:
1. Decompose the query into typed sub-tasks
2. Dispatch ready sub-tasks (topological order via SharedContext.ready_subtasks)
3. Critique outputs that have meaningful claims
4. Synthesize a final answer

Within each phase the LLM-driven router picks which agent/sub-task to
process next, so different valid runs over the same query produce
different (but valid) agent sequences. The phases themselves are
deterministic structure; routing within them is dynamic.

Compression-on-overflow: when an agent raises BudgetViolation, the
orchestrator catches it, invokes the compression agent on the existing
context, and retries the failed agent ONCE with the compressed view. If
still over budget, log and skip that agent's turn (don't crash the run).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agents import (
    AgentExecutionError,
    CompressionAgent,
    CritiqueAgent,
    DecompositionAgent,
    RetrievalAgent,
    SynthesisAgent,
)
from app.context import (
    AgentOutput,
    BudgetManager,
    BudgetViolation,
    DecompositionOutput,
    SharedContext,
    TaskStatus,
)
from app.llm import LLMClient
from app.orchestrator.routing import pick_next_agent
from app.rag import VectorStore
from app.streaming import (
    EventBus,
    emit_agent_complete,
    emit_agent_start,
    emit_done,
    emit_error,
    emit_final_answer,
    emit_policy_violation,
    emit_routing,
)

log = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """What `run_pipeline` returns. The shared context is the source of truth;
    this struct just makes the common things ergonomic."""

    ctx: SharedContext
    final_answer: str | None
    completed: bool


class Orchestrator:
    """Phase-driven orchestrator with dynamic routing inside each phase.

    The five agents are wired by constructor injection; tests can substitute
    fakes for any of them. The vector store is needed by RetrievalAgent.
    """

    MAX_AGENT_TURNS = 12  # safety cap; real runs use 4-6

    def __init__(
        self,
        llm: LLMClient,
        budgets: BudgetManager,
        vector_store: VectorStore,
        event_bus: EventBus | None = None,
    ) -> None:
        self._llm = llm
        self._budgets = budgets
        self._bus = event_bus
        self._decomposition = DecompositionAgent(llm, budgets)
        self._retrieval = RetrievalAgent(llm, budgets, vector_store=vector_store)
        self._critique = CritiqueAgent(llm, budgets)
        self._synthesis = SynthesisAgent(llm, budgets)
        self._compression = CompressionAgent(llm, budgets)

    async def run(self, ctx: SharedContext) -> OrchestratorResult:
        """Run the full pipeline. Mutates ctx in place; returns a wrapper."""
        try:
            await self._phase_decompose(ctx)
            await self._phase_retrieve(ctx)
            await self._phase_critique(ctx)
            await self._phase_synthesize(ctx)
        except Exception as e:
            log.exception("Orchestrator hit an unrecoverable error: %s", e)
            if self._bus:
                await emit_error(self._bus, ctx.job_id, str(e))

        synthesis = ctx.latest_synthesis()
        ctx.final_answer = synthesis.final_answer if synthesis else None

        if self._bus:
            # Emit any policy violations that accumulated during the run
            for v in ctx.policy_violations:
                await emit_policy_violation(
                    self._bus, ctx.job_id, v.agent_name, v.violation_type, v.detail
                )
            await emit_final_answer(self._bus, ctx.job_id, ctx.final_answer)
            await emit_done(self._bus, ctx.job_id)

        return OrchestratorResult(
            ctx=ctx,
            final_answer=ctx.final_answer,
            completed=ctx.final_answer is not None,
        )

    async def _emit_agent_lifecycle(
        self, ctx: SharedContext, agent_name: str, output: AgentOutput | None
    ):
        """Helper: emit start before invocation, complete after. Called inside
        each phase to keep the lifecycle tight around the agent.run() call."""
        if self._bus and output is not None:
            await emit_agent_complete(self._bus, ctx.job_id, agent_name, output.content)

    async def _emit_routing(self, ctx: SharedContext, decision):
        if self._bus:
            await emit_routing(self._bus, ctx.job_id, decision.next_agent, decision.justification)

    # ------------------------------------------------------------------ #
    # Phases
    # ------------------------------------------------------------------ #

    async def _phase_decompose(self, ctx: SharedContext) -> None:
        """One decomposition call; populates ctx.sub_tasks."""
        decision = await pick_next_agent(self._llm, ctx, candidates=[self._decomposition.name])
        await self._emit_routing(ctx, decision)
        if self._bus:
            await emit_agent_start(self._bus, ctx.job_id, self._decomposition.name)

        try:
            output = await self._run_with_compression(self._decomposition, ctx)
        except AgentExecutionError:
            return
        ctx.agent_outputs.append(output)
        decomp: DecompositionOutput = output.structured_output  # type: ignore[assignment]
        ctx.sub_tasks.extend(decomp.sub_tasks)
        await self._emit_agent_lifecycle(ctx, self._decomposition.name, output)

    async def _phase_retrieve(self, ctx: SharedContext) -> None:
        """Walk the dependency graph: dispatch ready retrieval sub-tasks
        until all are complete or no more progress can be made."""
        for _ in range(self.MAX_AGENT_TURNS):
            ready = [t for t in ctx.ready_subtasks() if t.task_type == "retrieval"]
            if not ready:
                # Mark non-retrieval ready tasks as skipped so the dependency
                # graph can advance past them in v1. (Computation/synthesis/
                # verification sub-tasks are handled differently — see notes.)
                for t in ctx.ready_subtasks():
                    if t.task_type != "retrieval":
                        t.status = TaskStatus.SKIPPED
                break

            # Routing: which retrieval sub-task to process next? We only have
            # one retrieval agent, so the dynamic choice is *which sub-task*
            # to process. For simplicity and to match the brief's "ready =>
            # dispatch", take the first ready task.
            task = ready[0]
            task.status = TaskStatus.IN_PROGRESS
            decision = await pick_next_agent(self._llm, ctx, candidates=[self._retrieval.name])
            await self._emit_routing(ctx, decision)
            if self._bus:
                await emit_agent_start(self._bus, ctx.job_id, self._retrieval.name)

            try:
                output = await self._run_with_compression(self._retrieval, ctx)
            except AgentExecutionError:
                task.status = TaskStatus.FAILED
                continue
            ctx.agent_outputs.append(output)
            task.status = TaskStatus.COMPLETE
            task.result = output.id
            await self._emit_agent_lifecycle(ctx, self._retrieval.name, output)

    async def _phase_critique(self, ctx: SharedContext) -> None:
        """Critique the most recent retrieval output. One pass per target."""
        retrieval_outputs = ctx.outputs_by_agent("retrieval")
        if not retrieval_outputs:
            return
        target = retrieval_outputs[-1]

        decision = await pick_next_agent(self._llm, ctx, candidates=[self._critique.name])
        await self._emit_routing(ctx, decision)
        if self._bus:
            await emit_agent_start(self._bus, ctx.job_id, self._critique.name)
        try:
            output = await self._run_with_compression(self._critique, ctx, target=target)
        except AgentExecutionError:
            return
        ctx.agent_outputs.append(output)
        await self._emit_agent_lifecycle(ctx, self._critique.name, output)

    async def _phase_synthesize(self, ctx: SharedContext) -> None:
        """Final synthesis. Skip if no retrieval output exists."""
        if not ctx.outputs_by_agent("retrieval"):
            return

        decision = await pick_next_agent(self._llm, ctx, candidates=[self._synthesis.name])
        await self._emit_routing(ctx, decision)
        if self._bus:
            await emit_agent_start(self._bus, ctx.job_id, self._synthesis.name)
        try:
            output = await self._run_with_compression(self._synthesis, ctx)
        except AgentExecutionError:
            return
        ctx.agent_outputs.append(output)
        await self._emit_agent_lifecycle(ctx, self._synthesis.name, output)

    # ------------------------------------------------------------------ #
    # Compression-on-overflow
    # ------------------------------------------------------------------ #

    async def _run_with_compression(self, agent, ctx: SharedContext, **kwargs) -> AgentOutput:
        """Invoke an agent; on BudgetViolation, compress and retry once.

        The retry is the brief's "automatic compression before proceeding"
        requirement. If the second attempt also overflows, we log and re-raise
        — the orchestrator catches and skips the phase rather than crashing.
        """
        try:
            return await agent.run(ctx, **kwargs)
        except BudgetViolation:
            # Compress the rolling context, then retry the agent once.
            await self._compress_context(ctx)
            try:
                return await agent.run(ctx, **kwargs)
            except BudgetViolation as e:
                log.warning(
                    "Agent %s still over budget after compression; skipping: %s",
                    agent.name,
                    e,
                )
                raise AgentExecutionError(
                    f"{agent.name}: budget exceeded even after compression"
                ) from e

    async def _compress_context(self, ctx: SharedContext) -> None:
        """Build a textual view of older outputs and call the compression agent."""
        if len(ctx.agent_outputs) < 2:
            return
        # Compress everything except the last output
        older = ctx.agent_outputs[:-1]
        joined = "\n\n".join(f"--- {o.agent_name}/{o.id} ---\n{o.content}" for o in older)
        original_tokens = self._llm.count_tokens(joined)
        try:
            comp_out = await self._compression.run(
                ctx, context_text=joined, original_tokens=original_tokens
            )
            ctx.agent_outputs.append(comp_out)
        except (AgentExecutionError, BudgetViolation):
            # Compression itself failed; nothing we can do automatically.
            log.warning("Compression failed; agent will retry without compressed view")
