from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from runtime.tools.tool_result import ToolResult

logger = logging.getLogger("mcp.sandbox")

ALLOWED_MCP_PATHS: list[str] = []
MAX_OUTPUT_SIZE = 100_000
DEFAULT_TIMEOUT_S = 60


def configure_allowed_paths(paths: list[str]) -> None:
    global ALLOWED_MCP_PATHS
    ALLOWED_MCP_PATHS = list(paths)


def _path_allowed(path: str) -> bool:
    if not ALLOWED_MCP_PATHS:
        return True
    resolved = Path(path).resolve()
    return any(str(resolved).startswith(allowed) for allowed in ALLOWED_MCP_PATHS)


class MCPSandbox:
    def __init__(self) -> None:
        self._temp_dirs: list[Path] = []

    def create_temp_dir(self, prefix: str = "mcp_sandbox_") -> Path:
        tmp = Path(tempfile.mkdtemp(prefix=prefix))
        self._temp_dirs.append(tmp)
        return tmp

    def run_stdio(
        self,
        command: str,
        args: list[str],
        input_data: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        try:
            result = subprocess.run(
                [command] + args,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
            duration_ms = (time.monotonic() - start) * 1000
            return {
                "stdout": result.stdout[:MAX_OUTPUT_SIZE],
                "stderr": result.stderr[:MAX_OUTPUT_SIZE],
                "returncode": result.returncode,
                "duration_ms": duration_ms,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            duration_ms = (time.monotonic() - start) * 1000
            return {
                "stdout": "",
                "stderr": f"Process timed out after {timeout_s}s",
                "returncode": -1,
                "duration_ms": duration_ms,
                "timed_out": True,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Command not found: {command}",
                "returncode": -1,
                "duration_ms": 0,
                "timed_out": False,
            }
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "duration_ms": duration_ms,
                "timed_out": False,
            }

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str]:
        if tool_name in {"read_file", "write_file"}:
            path = str(arguments.get("path", ""))
            if path and not _path_allowed(path):
                return False, f"Path not allowed by sandbox policy: {path}"
        if tool_name in {"shell", "python", "bash"}:
            return True, ""
        if tool_name == "http_request":
            url = str(arguments.get("url", ""))
            if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
                reason = f"URL not allowed by sandbox policy: {url}"
                return False, reason
        return True, ""

    def cleanup(self) -> None:
        for tmp in self._temp_dirs:
            try:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()


mcp_sandbox = MCPSandbox()
