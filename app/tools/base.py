"""Tool base class with explicit failure contract.

Every tool returns a `ToolResult`, never raises. Failure modes are
encoded as `status` values: ok / timeout / empty / malformed_input /
error. The orchestrator pattern-matches on status to decide what to do
(accept, retry with modified input, fall back, etc.).

This module defines the contract. Concrete tools live in sibling files.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class ToolStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    EMPTY = "empty"
    MALFORMED_INPUT = "malformed_input"
    ERROR = "error"


@dataclass
class ToolResult:
    status: ToolStatus
    payload: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    latency_ms: float = 0.0

    @property
    def is_ok(self) -> bool:
        return self.status == ToolStatus.OK

    @property
    def is_retryable(self) -> bool:
        """Whether the orchestrator should consider retrying with modified input.
        Empty results and timeouts are retryable; malformed input usually isn't
        (the agent needs to fix its call), nor are unexpected errors."""
        return self.status in (ToolStatus.TIMEOUT, ToolStatus.EMPTY)


class Tool(ABC):
    """Base for all tools. Subclasses implement `_execute()`; the base wraps
    it with timing and error containment so a misbehaving tool can never
    crash the orchestrator."""

    name: ClassVar[str]
    default_timeout_s: ClassVar[float] = 10.0

    async def call(self, **kwargs: Any) -> ToolResult:
        """Public entrypoint. Times the call, contains exceptions, returns
        a ToolResult regardless of what the underlying tool does."""
        start = time.monotonic()
        try:
            result = await self._execute(**kwargs)
        except TimeoutError as e:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error_message=str(e) or f"{self.name} timed out",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except (TypeError, ValueError) as e:
            # Likely malformed input from the agent
            return ToolResult(
                status=ToolStatus.MALFORMED_INPUT,
                error_message=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error_message=f"{type(e).__name__}: {e}",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        result.latency_ms = (time.monotonic() - start) * 1000
        return result

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Tool-specific implementation. May raise; the base wraps."""
