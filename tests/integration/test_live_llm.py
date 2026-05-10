"""Live OpenAI smoke tests. Not run by default.

Requires:
  RUN_LIVE_LLM=1
  OPENAI_API_KEY

Run:  RUN_LIVE_LLM=1 pytest -m live_llm tests/integration -v
"""

from __future__ import annotations

import os

import pytest

from app.llm import LLMClient, LLMMessage

pytestmark = pytest.mark.live_llm


def _live_llm_enabled() -> bool:
    return os.environ.get("RUN_LIVE_LLM") == "1"


def _client() -> LLMClient:
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return LLMClient(api_key=key, model=model, base_url=base)


@pytest.mark.asyncio
async def test_openai_complete_smoke():
    if not _live_llm_enabled():
        pytest.skip("Set RUN_LIVE_LLM=1 to enable live OpenAI tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    client = _client()
    resp = await client.complete(
        [LLMMessage(role="user", content='Reply with exactly the word OK and nothing else.')],
        temperature=0,
        max_tokens=16,
        json_mode=False,
    )
    parts = resp.text.strip().split()
    assert parts
    first = parts[0].upper().strip("\"'.")
    assert first == "OK"


@pytest.mark.asyncio
async def test_openai_complete_json_smoke():
    if not _live_llm_enabled():
        pytest.skip("Set RUN_LIVE_LLM=1 to enable live OpenAI tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    client = _client()
    data, resp = await client.complete_json(
        [
            LLMMessage(
                role="user",
                content='Return a JSON object with a single key "greeting" whose value is "hi".',
            )
        ],
        temperature=0,
        max_tokens=64,
    )
    assert data.get("greeting") == "hi"
    assert resp.output_tokens > 0
