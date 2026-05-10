"""Tests for the LLM client.

Focus: retry semantics, JSON-mode parsing with the 1-retry repair loop, and
deterministic token counting. Uses respx to mock the OpenAI HTTP layer rather
than monkey-patching the SDK — this exercises the real client code path
including retry decorators.

These tests do NOT call any real API. They are hermetic.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.llm import LLMClient, LLMMessage, MalformedJSONError


@pytest.fixture
def client() -> LLMClient:
    return LLMClient(
        api_key="test-key-not-real",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )


def _chat_response(content: str, *, model: str = "gpt-4o-mini") -> dict:
    """Minimal OpenAI chat-completion response shape."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


class TestTokenCounting:
    """Token counting uses tiktoken which downloads BPE tables on first use.
    To keep these tests hermetic we stub the encoder. The behavioral contract
    being tested is: the LLMClient delegates correctly to its encoder,
    overhead is added on top of bare content counts, and counts are
    deterministic — none of which depend on the BPE table choice."""

    def _stub_encoder(self, client: LLMClient) -> None:
        """Inject a deterministic word-counting stub in place of tiktoken."""

        class _Stub:
            def encode(self, s: str) -> list[int]:
                # A token per whitespace-separated word; deterministic and
                # monotonic in length, which is all these tests assert.
                return list(range(len(s.split())))

        client._encoder = _Stub()  # type: ignore[assignment]

    def test_count_tokens_is_deterministic(self, client: LLMClient):
        self._stub_encoder(client)
        n1 = client.count_tokens("hello world")
        n2 = client.count_tokens("hello world")
        assert n1 == n2 and n1 > 0

    def test_longer_text_has_more_tokens(self, client: LLMClient):
        self._stub_encoder(client)
        short = client.count_tokens("hi")
        long = client.count_tokens("hi " * 100)
        assert long > short

    def test_count_message_tokens_includes_overhead(self, client: LLMClient):
        self._stub_encoder(client)
        msgs = [
            LLMMessage(role="user", content="hello"),
            LLMMessage(role="assistant", content="hi"),
        ]
        # Message-level count should be strictly greater than the bare content count
        # because of per-message and reply-priming overhead.
        bare = client.count_tokens("hello") + client.count_tokens("hi")
        msg_total = client.count_message_tokens(msgs)
        assert msg_total > bare


# --------------------------------------------------------------------------- #
# Retry behavior
# --------------------------------------------------------------------------- #


class TestRetryBehavior:
    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_on_429_rate_limit(self, client: LLMClient):
        """RateLimitError is retryable. After two 429s and a success, we should
        get the success without an exception."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": {"message": "rate limited"}}),
                httpx.Response(429, json={"error": {"message": "rate limited"}}),
                httpx.Response(200, json=_chat_response("ok")),
            ]
        )
        resp = await client.complete([LLMMessage(role="user", content="hi")])
        assert resp.text == "ok"
        assert route.call_count == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_does_not_retry_on_400_bad_request(self, client: LLMClient):
        """4xx user errors indicate bugs, not transient failures. No retry."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                400, json={"error": {"message": "bad request", "type": "invalid_request_error"}}
            )
        )
        with pytest.raises(Exception):  # noqa: B017 — openai wraps it; we just don't retry
            await client.complete([LLMMessage(role="user", content="hi")])
        assert route.call_count == 1  # exactly one attempt

    @pytest.mark.asyncio
    @respx.mock
    async def test_does_not_retry_on_401_auth_error(self, client: LLMClient):
        """Authentication errors are not transient. No retry."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "bad key", "type": "invalid_request_error"}}
            )
        )
        with pytest.raises(Exception):  # noqa: B017
            await client.complete([LLMMessage(role="user", content="hi")])
        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_gives_up_after_max_attempts(self, client: LLMClient):
        """3 attempts max. After 3 retryable failures we surface the error."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
        )
        with pytest.raises(Exception):  # noqa: B017
            await client.complete([LLMMessage(role="user", content="hi")])
        assert route.call_count == 3


# --------------------------------------------------------------------------- #
# JSON mode + repair retry
# --------------------------------------------------------------------------- #


class TestCompleteJson:
    @pytest.mark.asyncio
    @respx.mock
    async def test_valid_json_first_try(self, client: LLMClient):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_chat_response(json.dumps({"key": "value", "n": 1})),
            )
        )
        parsed, resp = await client.complete_json([LLMMessage(role="user", content="hi")])
        assert parsed == {"key": "value", "n": 1}

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_first_then_repair_succeeds(self, client: LLMClient):
        """First attempt returns garbage; repair retry returns valid JSON.
        We expect exactly 2 HTTP calls and a successful parse."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_chat_response("not json at all {")),
                httpx.Response(200, json=_chat_response('{"recovered": true}')),
            ]
        )
        parsed, _ = await client.complete_json([LLMMessage(role="user", content="hi")])
        assert parsed == {"recovered": True}
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_twice_raises_with_raw_preview(self, client: LLMClient):
        """Both attempts fail. Should raise MalformedJSONError carrying both
        raw outputs so the orchestrator can log a schema_violation."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_chat_response("attempt one garbage")),
                httpx.Response(200, json=_chat_response("attempt two also garbage")),
            ]
        )
        with pytest.raises(MalformedJSONError) as exc_info:
            await client.complete_json([LLMMessage(role="user", content="hi")])
        err = exc_info.value
        assert "attempt one garbage" in err.raw_first
        assert "attempt two also garbage" in err.raw_retry
        # The combined preview is what gets logged on PolicyViolation.raw_preview
        preview = err.raw_preview
        assert "attempt 1" in preview and "attempt 2" in preview

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_object_json_triggers_repair(self, client: LLMClient):
        """A valid JSON array (not an object) should also trigger the repair
        retry — agents expect dict-shaped structured outputs."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_chat_response('["not", "an", "object"]')),
                httpx.Response(200, json=_chat_response('{"ok": 1}')),
            ]
        )
        parsed, _ = await client.complete_json([LLMMessage(role="user", content="hi")])
        assert parsed == {"ok": 1}
        assert route.call_count == 2
