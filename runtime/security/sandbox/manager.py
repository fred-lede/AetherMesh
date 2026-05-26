from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles, default_profile
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("sandbox.manager")


class _NullSandbox(PlatformSandbox):
    def execute(
        self,
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        logger.warning("No sandbox implementation for platform %s — executing without isolation", sys.platform)
        import subprocess
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=profile.timeout_sec,
                cwd=cwd,
                env=env,
                input=stdin,
            )
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(
                output=(result.stdout + result.stderr)[:100_000],
                return_code=result.returncode,
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(output="", return_code=-1, duration_ms=duration, timed_out=True)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(output=str(e), return_code=-1, duration_ms=duration)


class SandboxManager:
    def __init__(
        self,
        profiles: dict[str, SandboxProfile] | None = None,
    ) -> None:
        if sys.platform == "linux":
            from runtime.security.sandbox.linux_sandbox import LinuxSandbox
            self._platform: PlatformSandbox = LinuxSandbox()
        elif sys.platform == "darwin":
            from runtime.security.sandbox.mac_sandbox import MacSandbox
            self._platform = MacSandbox()
        else:
            self._platform = _NullSandbox()
        self._profiles = profiles or builtin_profiles()

    def execute(
        self,
        tool_name: str,
        handler_fn: Callable[[ToolCall], ToolResult],
        call: ToolCall,
    ) -> ToolResult:
        profile = self._profiles.get(tool_name)
        if profile is None:
            profile = default_profile(tool_name)
        if not profile.enabled:
            return handler_fn(call)

        if profile.sandbox_type == "policy":
            return self._execute_policy(profile, handler_fn, call)
        return self._execute_process(profile, call)

    def _execute_policy(
        self,
        profile: SandboxProfile,
        handler_fn: Callable[[ToolCall], ToolResult],
        call: ToolCall,
    ) -> ToolResult:
        if call.name in ("filesystem", "read_file", "write_file"):
            path = str(call.arguments.get("path", ""))
            if profile.allowed_paths is not None:
                resolved = Path(path).resolve()
                allowed = any(
                    str(resolved).startswith(str(Path(p).resolve()))
                    for p in profile.allowed_paths
                )
                if not allowed:
                    return ToolResult(
                        call=call,
                        output=f"Path not allowed: {path}",
                        is_error=True,
                    )
        elif call.name in ("web_fetch", "http_request"):
            url = str(call.arguments.get("url", ""))
            if not profile.allow_network:
                return ToolResult(
                    call=call,
                    output="Network access disabled for this tool",
                    is_error=True,
                )
            if profile.allowed_domains and profile.allowed_domains != ["*"]:
                parsed = urllib.parse.urlparse(url)
                allowed = any(parsed.hostname.endswith(d) for d in profile.allowed_domains)
                if not allowed:
                    return ToolResult(
                        call=call,
                        output=f"Domain not allowed: {parsed.hostname}",
                        is_error=True,
                    )

        return handler_fn(call)

    def _execute_process(
        self,
        profile: SandboxProfile,
        call: ToolCall,
    ) -> ToolResult:
        scratch_dir = tempfile.mkdtemp(prefix="aether_sandbox_")
        try:
            if call.name == "python":
                code = str(call.arguments.get("code", ""))
                result = self._platform.execute(
                    ["python3", "-c", code],
                    profile=profile,
                    cwd=scratch_dir,
                )
            elif call.name == "shell":
                command = str(call.arguments.get("command", ""))
                result = self._platform.execute(
                    ["sh", "-c", command],
                    profile=profile,
                    cwd=scratch_dir,
                )
            else:
                return ToolResult(
                    call=call,
                    output=f"No process handler for tool: {call.name}",
                    is_error=True,
                )

            return ToolResult(
                call=call,
                output=result.output,
                is_error=result.return_code != 0,
                duration_ms=result.duration_ms,
            )
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
