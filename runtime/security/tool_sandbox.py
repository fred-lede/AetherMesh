from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("security.sandbox")

ALLOWED_SANDBOX_PATHS: list[str] = []


def configure_sandbox_paths(paths: list[str]) -> None:
    global ALLOWED_SANDBOX_PATHS
    ALLOWED_SANDBOX_PATHS = list(paths)


class ToolSandbox:
    def run_shell(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        command = str(call.arguments.get("command", ""))
        if not command:
            return ToolResult(call=call, output="No command provided", is_error=True)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = result.stdout + result.stderr
            return ToolResult(call=call, output=output[:100_000], is_error=result.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult(call=call, output=f"Command timed out after {timeout_s}s", is_error=True)
        except Exception as e:
            return ToolResult(call=call, output=str(e), is_error=True)

    def run_python(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        code = str(call.arguments.get("code", ""))
        if not code:
            return ToolResult(call=call, output="No code provided", is_error=True)
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = result.stdout + result.stderr
            return ToolResult(call=call, output=output[:100_000], is_error=result.returncode != 0)
        except subprocess.TimeoutExpired:
            return ToolResult(call=call, output=f"Python execution timed out after {timeout_s}s", is_error=True)
        except Exception as e:
            return ToolResult(call=call, output=str(e), is_error=True)

    def read_file(self, call: ToolCall) -> ToolResult:
        path = str(call.arguments.get("path", ""))
        if not path:
            return ToolResult(call=call, output="No path provided", is_error=True)
        p = Path(path).resolve()
        if ALLOWED_SANDBOX_PATHS:
            allowed = any(str(p).startswith(allowed) for allowed in ALLOWED_SANDBOX_PATHS)
            if not allowed:
                return ToolResult(call=call, output=f"Path not allowed: {path}", is_error=True)
        try:
            content = p.read_text(encoding="utf-8")
            return ToolResult(call=call, output=content[:100_000])
        except Exception as e:
            return ToolResult(call=call, output=str(e), is_error=True)

    def write_file(self, call: ToolCall) -> ToolResult:
        path = str(call.arguments.get("path", ""))
        content = str(call.arguments.get("content", ""))
        if not path:
            return ToolResult(call=call, output="No path provided", is_error=True)
        p = Path(path).resolve()
        if ALLOWED_SANDBOX_PATHS:
            allowed = any(str(p).startswith(allowed) for allowed in ALLOWED_SANDBOX_PATHS)
            if not allowed:
                return ToolResult(call=call, output=f"Path not allowed: {path}", is_error=True)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(call=call, output=f"Written to {path}")
        except Exception as e:
            return ToolResult(call=call, output=str(e), is_error=True)


tool_sandbox = ToolSandbox()
