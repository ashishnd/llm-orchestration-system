"""Provider-agnostic LLM client.

Currently wraps OpenAI's chat completions API but is structured so that swapping
to Anthropic, a local Ollama instance, or any OpenAI-compatible endpoint is a
single-class change.

Design choices:
- Streaming and non-streaming calls share a single Pydantic-typed response shape
  so callers don't branch on transport.
- Retries use tenacity with exponential backoff. We retry on transient API
  errors only (rate limits, timeouts), never on 4xx user errors.
- Structured output is enforced via response_format=json_object and a
  post-validation step. Malformed JSON raises rather than returning silently
  bad data — agents downstream rely on schema integrity.
- Token counting uses tiktoken. We count input tokens locally so the context
  budget manager can run a pre-flight check without a network round trip.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.settings import get_settings


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str | None = None


# Errors we retry on. 4xx user errors (bad request, auth) are NOT retried —
# they indicate a bug, not a transient failure.
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


class MalformedJSONError(Exception):
    """Raised when the LLM returns invalid JSON on BOTH the initial attempt
    and the constrained repair retry.

    Carries the raw outputs of both attempts so the orchestrator can log a
    schema_violation PolicyViolation with `raw_preview` populated for
    diagnostic auditing.
    """

    def __init__(self, message: str, raw_first: str, raw_retry: str) -> None:
        super().__init__(message)
        self.raw_first = raw_first
        self.raw_retry = raw_retry

    @property
    def raw_preview(self) -> str:
        """Combined preview of both attempts, suitable for raw_preview field
        on PolicyViolation. The PolicyViolation validator caps to 500 chars."""
        return f"[attempt 1] {self.raw_first[:240]} | [attempt 2] {self.raw_retry[:240]}"


class LLMClient:
    """Async OpenAI-compatible chat client with retry and structured-output support."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        # max_retries=0 disables the OpenAI SDK's internal retry layer. We own
        # retry policy via tenacity (see RETRYABLE_ERRORS); composing two
        # retry layers would multiply attempt counts and hide the real
        # behavior from the budget manager and the trace.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._model = model
        self._encoder: tiktoken.Encoding | None = None  # lazy-loaded on first use

    @property
    def model(self) -> str:
        return self._model

    def _get_encoder(self) -> tiktoken.Encoding:
        """Lazily load the tiktoken encoder.

        We don't materialize the encoder at construction time because tiktoken
        downloads the BPE table from a remote URL on first use, and we want
        construction to be hermetic (no network) for tests and for fast app
        startup. The encoder is cached after the first call.
        """
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self._model)
            except KeyError:
                # Unknown model — fall back to a known-good encoding.
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens locally; used by the context budget manager."""
        return len(self._get_encoder().encode(text))

    def count_message_tokens(self, messages: list[LLMMessage]) -> int:
        # OpenAI's per-message overhead is ~4 tokens for role/separators.
        # This is an estimate — exact counts are not critical for budgeting.
        return sum(self.count_tokens(m.content) + 4 for m in messages) + 2

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Single-shot completion. Use this when you don't need streaming."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model,
            finish_reason=choice.finish_reason,
        )

    async def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Completion with strict JSON parsing and one constrained repair retry.

        Behavior:
        1. First attempt at temperature=`temperature` with json_mode=True.
        2. On malformed JSON or non-object, ONE retry at temperature=0.0 with
           a hard repair instruction appended. The retry sees the original
           messages plus a clarifying assistant/user pair pointing out the
           failure.
        3. If the retry also fails, raise MalformedJSONError carrying the
           raw output so the caller can log it on a PolicyViolation.

        Network-layer transient errors are still retried by the parent
        @retry decorator on `complete`; this method only handles the
        JSON-parse failure mode.
        """
        resp = await self.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=True
        )
        parsed = self._try_parse_json_object(resp.text)
        if parsed is not None:
            return parsed, resp

        # Repair attempt: one retry, lower temperature, with explicit instruction
        repair_messages = list(messages) + [
            LLMMessage(role="assistant", content=resp.text),
            LLMMessage(
                role="user",
                content=(
                    "Your previous response was not valid JSON. "
                    "Return ONLY a single valid JSON object, with no preamble, "
                    "no markdown fences, and no trailing commentary."
                ),
            ),
        ]
        retry_resp = await self.complete(
            repair_messages,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        retry_parsed = self._try_parse_json_object(retry_resp.text)
        if retry_parsed is not None:
            return retry_parsed, retry_resp

        # Both attempts failed. Raise with the raw output so the caller can
        # log a schema_violation PolicyViolation with raw_preview populated.
        raise MalformedJSONError(
            "LLM returned malformed JSON on both initial and repair attempts",
            raw_first=resp.text,
            raw_retry=retry_resp.text,
        )

    @staticmethod
    def _try_parse_json_object(text: str) -> dict[str, Any] | None:
        """Parse text as a JSON object. Returns None if it's not a valid object."""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Token-by-token stream. Used by SSE endpoint to surface real-time output."""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


@lru_cache
def get_llm_client() -> LLMClient:
    s = get_settings()
    return LLMClient(
        api_key=s.openai_api_key,
        model=s.openai_model,
        base_url=s.openai_base_url,
    )
