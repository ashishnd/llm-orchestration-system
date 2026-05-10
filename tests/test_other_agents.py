"""Happy-path tests for critique, synthesis, compression agents.

These three agents use the standard BaseAgent.run() path. Schema
validation, budget tracking, and policy violation logging are covered
by base + decomposition tests. Here we just verify each agent wires
its prompts and structured output correctly.
"""

from __future__ import annotations

import json

import pytest

from app.agents import CompressionAgent, CritiqueAgent, SynthesisAgent
from app.context import (
    AgentOutput,
    BudgetManager,
    CompressionOutput,
    CritiqueOutput,
    SharedContext,
)
from app.llm.client import LLMResponse


class StubLLM:
    def __init__(self, response):
        self._response = response

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages):
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    async def complete_json(self, messages, **kwargs):
        resp = LLMResponse(
            text=json.dumps(self._response),
            input_tokens=10,
            output_tokens=20,
            model="stub",
        )
        return self._response, resp


@pytest.fixture
def ctx():
    return SharedContext(user_query="What is BERT?")


@pytest.mark.asyncio
async def test_critique_produces_claim_scores_and_disagreements(ctx):
    target = AgentOutput(
        agent_name="retrieval",
        content="BERT was introduced in 2018 and uses self-attention.",
    )
    llm = StubLLM(
        {
            "type": "critique",
            "target_output_id": target.id,
            "claim_scores": [
                {
                    "claim_span": {"start": 0, "end": 30, "text": "BERT was introduced in 2018 an"},
                    "confidence": 0.95,
                }
            ],
            "disagreements": [
                {
                    "span": {"start": 38, "end": 52, "text": "self-attention"},
                    "target_agent": "retrieval",
                    "confidence": 0.7,
                    "reason": "BERT uses bidirectional attention specifically; not just self-attention.",
                }
            ],
        }
    )
    budgets = BudgetManager(llm=llm)
    agent = CritiqueAgent(llm=llm, budgets=budgets)

    out = await agent.run(ctx, target=target)

    co: CritiqueOutput = out.structured_output  # type: ignore[assignment]
    assert isinstance(co, CritiqueOutput)
    assert co.target_output_id == target.id
    assert len(co.claim_scores) == 1
    assert len(co.disagreements) == 1


@pytest.mark.asyncio
async def test_synthesis_produces_final_answer_with_provenance(ctx):
    # Seed context with a retrieval output so synthesis has something to merge.
    retrieval_out = AgentOutput(
        agent_name="retrieval",
        content="BERT is a transformer-based bidirectional encoder model.",
    )
    ctx.agent_outputs.append(retrieval_out)

    llm = StubLLM(
        {
            "type": "synthesis",
            "final_answer": "BERT is a bidirectional transformer encoder.",
            "provenance": [
                {
                    "sentence_index": 0,
                    "text_span": {
                        "start": 0,
                        "end": 44,
                        "text": "BERT is a bidirectional transformer encoder.",
                    },
                    "source_agent": "retrieval",
                    "source_output_id": retrieval_out.id,
                    "citations": [
                        {
                            "chunk_id": "arxiv:1810.04805",
                            "source_doc": "arxiv:1810.04805",
                            "relevance_score": 0.9,
                        }
                    ],
                }
            ],
            "resolution_notes": None,
        }
    )
    budgets = BudgetManager(llm=llm)
    agent = SynthesisAgent(llm=llm, budgets=budgets)

    out = await agent.run(ctx)

    so = out.structured_output
    assert so.type == "synthesis"
    assert "BERT" in so.final_answer
    assert len(so.provenance) == 1
    assert so.provenance[0].source_output_id == retrieval_out.id


@pytest.mark.asyncio
async def test_compression_reports_before_after_token_counts(ctx):
    llm = StubLLM(
        {
            "type": "compression",
            "lossy_summary": "Brief summary preserving facts.",
            "preserved_refs": [
                {"ref_type": "tool_invocation", "ref_id": "tool-1"},
                {"ref_type": "citation", "ref_id": "arxiv:2401.001"},
            ],
            "original_token_count": 1000,
            "compressed_token_count": 200,
        }
    )
    budgets = BudgetManager(llm=llm)
    agent = CompressionAgent(llm=llm, budgets=budgets)

    long_context = "verbose narrative " * 200
    out = await agent.run(ctx, context_text=long_context, original_tokens=1000)

    co: CompressionOutput = out.structured_output  # type: ignore[assignment]
    assert isinstance(co, CompressionOutput)
    assert co.original_token_count == 1000
    assert co.compressed_token_count == 200
    assert len(co.preserved_refs) == 2
    # Compression ratio renderable via content
    assert "20%" in out.content
