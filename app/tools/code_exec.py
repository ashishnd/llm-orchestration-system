"""Code execution sandbox.

Runs a Python snippet in a subprocess with a hard timeout. Returns
stdout, stderr, and exit code in the payload.

Sandboxing posture: this is NOT a security boundary. We use a
subprocess to isolate state and enforce a timeout, but a malicious
snippet could still do damage. In production this would run in a
gVisor / Firecracker / Docker-in-Docker container with no network
and a temp working dir. Documented in Known Limitations.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from app.tools.base import Tool, ToolResult, ToolStatus


class CodeExecutionTool(Tool):
    name: ClassVar[str] = "code_exec"
    default_timeout_s: ClassVar[float] = 5.0

    async def _execute(
        self,
        code: str | None = None,
        timeout_s: float | None = None,
        **_: Any,
    ) -> ToolResult:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code_exec requires a non-empty `code` string")

        timeout = timeout_s if timeout_s is not None else self.default_timeout_s

        proc = await asyncio.create_subprocess_exec(
            "python",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"code_exec exceeded {timeout}s") from e

        return ToolResult(
            status=ToolStatus.OK,
            payload={
                "stdout": stdout_b.decode("utf-8", errors="replace"),
                "stderr": stderr_b.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode,
            },
        )
