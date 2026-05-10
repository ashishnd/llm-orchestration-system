"""Tool tests — happy path + key failure contract per tool.

We focus on the failure-contract behavior since that's the brief's
distinctive requirement: tools must return structured failure modes,
not raise.
"""

from __future__ import annotations

import json

import pytest

from app.context import AgentOutput, SharedContext
from app.llm.client import LLMResponse
from app.tools import (
    CodeExecutionTool,
    SelfReflectionTool,
    SQLLookupTool,
    ToolStatus,
    WebSearchTool,
)

# --------------------------------------------------------------------------- #
# Web search
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_web_search_happy_path():
    tool = WebSearchTool()
    result = await tool.call(query="BERT", top_k=2)
    assert result.is_ok
    assert "results" in result.payload
    assert len(result.payload["results"]) == 2
    # Each result has the structured shape the brief requires
    r0 = result.payload["results"][0]
    assert "url" in r0 and "relevance_score" in r0
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_web_search_empty_query_is_malformed_input():
    """Per the failure contract, an empty query is malformed input, not error."""
    tool = WebSearchTool()
    result = await tool.call(query="")
    assert result.status == ToolStatus.MALFORMED_INPUT
    assert not result.is_retryable  # bad input — agent must fix, not retry


# --------------------------------------------------------------------------- #
# Code execution
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_code_exec_happy_path_returns_stdout():
    tool = CodeExecutionTool()
    result = await tool.call(code="print(2 + 2)")
    assert result.is_ok
    assert result.payload["stdout"].strip() == "4"
    assert result.payload["exit_code"] == 0


@pytest.mark.asyncio
async def test_code_exec_timeout_returns_TIMEOUT_status():
    """Timeout is retryable per the contract."""
    tool = CodeExecutionTool()
    result = await tool.call(code="import time; time.sleep(5)", timeout_s=0.5)
    assert result.status == ToolStatus.TIMEOUT
    assert result.is_retryable


# --------------------------------------------------------------------------- #
# SQL lookup
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sql_lookup_select_returns_rows():
    tool = SQLLookupTool()
    result = await tool.call(sql="SELECT arxiv_id, title FROM papers WHERE year >= 2020")
    assert result.is_ok
    assert result.payload["row_count"] >= 1
    assert "arxiv_id" in result.payload["rows"][0]


@pytest.mark.asyncio
async def test_sql_lookup_rejects_non_select():
    """Read-only enforcement — DELETE/UPDATE/etc are malformed input."""
    tool = SQLLookupTool()
    result = await tool.call(sql="DELETE FROM papers")
    assert result.status == ToolStatus.MALFORMED_INPUT
    assert "read-only" in (result.error_message or "").lower()


@pytest.mark.asyncio
async def test_sql_lookup_empty_results_returns_EMPTY_status():
    """Empty results are retryable (agent might want to broaden the query)."""
    tool = SQLLookupTool()
    result = await tool.call(sql="SELECT * FROM papers WHERE year = 1900")
    assert result.status == ToolStatus.EMPTY
    assert result.is_retryable


# --------------------------------------------------------------------------- #
# Self-reflection
# --------------------------------------------------------------------------- #


class StubLLM:
    def __init__(self, response):
        self._response = response

    def count_tokens(self, text):
        return len(text) // 4

    def count_message_tokens(self, messages):
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    async def complete_json(self, messages, **kwargs):
        resp = LLMResponse(
            text=json.dumps(self._response), input_tokens=10, output_tokens=20, model="stub"
        )
        return self._response, resp


@pytest.mark.asyncio
async def test_self_reflect_returns_empty_when_fewer_than_two_outputs():
    """The contract: empty when there's nothing to compare. NOT an error."""
    ctx = SharedContext(user_query="x")
    ctx.agent_outputs.append(AgentOutput(agent_name="retrieval", content="only one"))
    tool = SelfReflectionTool(llm=StubLLM({"contradictions": []}), ctx=ctx)
    result = await tool.call()
    assert result.status == ToolStatus.EMPTY
    assert "fewer than 2" in result.payload["reason"]


@pytest.mark.asyncio
async def test_self_reflect_finds_contradictions_via_llm():
    ctx = SharedContext(user_query="x")
    ctx.agent_outputs.append(
        AgentOutput(agent_name="retrieval", content="BERT was published in 2018.")
    )
    ctx.agent_outputs.append(
        AgentOutput(agent_name="retrieval", content="BERT was published in 2020.")
    )
    canned = {
        "contradictions": [
            {
                "output_id_a": ctx.agent_outputs[0].id,
                "output_id_b": ctx.agent_outputs[1].id,
                "claim_a": "2018",
                "claim_b": "2020",
                "explanation": "different years",
            }
        ]
    }
    tool = SelfReflectionTool(llm=StubLLM(canned), ctx=ctx)
    result = await tool.call()
    assert result.is_ok
    assert len(result.payload["contradictions"]) == 1
    assert result.payload["outputs_scanned"] == 2
