# Execution Environment Abstraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add process-level isolation (resource limits, filesystem isolation, network control) for all builtin tools via a `SandboxManager` + platform-specific `PlatformSandbox` layer.

**Architecture:** Three-layer design — `SandboxProfile` (per-tool config), `PlatformSandbox` (abstract base with Mac/Linux implementations), `SandboxManager` (orchestration + cleanup). Injected as optional dependency into `ToolExecutor`.

**Tech Stack:** Python `subprocess`, `resource`, `tempfile`, `shutil`, `unshare` (Linux-only), `unittest.mock`.

---

### Task 1: SandboxProfile dataclass + builtin defaults

**Files:**
- Create: `runtime/security/sandbox/__init__.py`
- Create: `runtime/security/sandbox/profile.py`
- Create: `tests/test_execution_environment.py`

- [ ] **Step 1: Create module init with exports**

```python
# runtime/security/sandbox/__init__.py
from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles, default_profile
from runtime.security.sandbox.platform import SandboxResult, PlatformSandbox
from runtime.security.sandbox.manager import SandboxManager

__all__ = [
    "SandboxProfile",
    "SandboxResult",
    "PlatformSandbox",
    "SandboxManager",
    "builtin_profiles",
    "default_profile",
]
```

- [ ] **Step 2: Write failing test for SandboxProfile**

```python
# tests/test_execution_environment.py (append to this file throughout)
from __future__ import annotations

from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles, default_profile


def test_sandbox_profile_defaults():
    profile = SandboxProfile()
    assert profile.enabled is True
    assert profile.sandbox_type == "process"
    assert profile.allow_network is False
    assert profile.max_cpu_time_sec == 30.0
    assert profile.max_memory_bytes == 524288000
    assert profile.timeout_sec == 30


def test_sandbox_profile_custom():
    profile = SandboxProfile(
        enabled=False,
        sandbox_type="policy",
        allow_network=True,
        max_memory_bytes=268435456,
    )
    assert profile.enabled is False
    assert profile.sandbox_type == "policy"
    assert profile.allow_network is True


def test_builtin_profiles_exist():
    profiles = builtin_profiles()
    assert "python" in profiles
    assert "shell" in profiles
    assert "filesystem" in profiles
    assert "web_fetch" in profiles
    assert "http_request" in profiles


def test_builtin_profile_values():
    profiles = builtin_profiles()
    py = profiles["python"]
    assert py.sandbox_type == "process"
    assert py.allow_network is False

    fs = profiles["filesystem"]
    assert fs.sandbox_type == "policy"


def test_default_profile_fallback():
    profile = default_profile("unknown_tool")
    assert profile.enabled is True
    assert profile.sandbox_type == "process"
    assert profile.allow_network is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_execution_environment.py -v`
Expected: 5 FAILED (ImportError: cannot import name 'SandboxProfile')

- [ ] **Step 4: Implement SandboxProfile + builtin defaults**

```python
# runtime/security/sandbox/profile.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxProfile:
    enabled: bool = True
    sandbox_type: str = "process"  # "process" | "policy"
    allow_network: bool = False
    max_cpu_time_sec: float = 30.0
    max_memory_bytes: int = 524288000  # 500 MB
    allowed_paths: list[str] | None = None
    allowed_domains: list[str] | None = None
    timeout_sec: int = 30


def builtin_profiles() -> dict[str, SandboxProfile]:
    return {
        "python": SandboxProfile(sandbox_type="process", allow_network=False),
        "shell": SandboxProfile(sandbox_type="process", allow_network=False),
        "filesystem": SandboxProfile(sandbox_type="policy", allowed_paths=[]),
        "web_fetch": SandboxProfile(sandbox_type="policy", allow_network=True, allowed_domains=["*"]),
        "http_request": SandboxProfile(sandbox_type="policy", allow_network=True, allowed_domains=["*"]),
    }


def default_profile(tool_name: str) -> SandboxProfile:
    return SandboxProfile(
        enabled=True,
        sandbox_type="process",
        allow_network=False,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_execution_environment.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add runtime/security/sandbox/__init__.py runtime/security/sandbox/profile.py tests/test_execution_environment.py
git commit -m "feat: add SandboxProfile dataclass with builtin defaults"
```

---

### Task 2: PlatformSandbox ABC + SandboxResult

**Files:**
- Create: `runtime/security/sandbox/platform.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_execution_environment.py
import pytest

from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile


def test_sandbox_result_defaults():
    result = SandboxResult(output="hello", return_code=0, duration_ms=10.0)
    assert result.output == "hello"
    assert result.return_code == 0
    assert result.timed_out is False


def test_platform_sandbox_is_abstract():
    with pytest.raises(TypeError):
        PlatformSandbox()  # type: ignore
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_execution_environment.py::test_sandbox_result_defaults tests/test_execution_environment.py::test_platform_sandbox_is_abstract -v`
Expected: ImportError for PlatformSandbox

- [ ] **Step 3: Implement PlatformSandbox ABC**

```python
# runtime/security/sandbox/platform.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from runtime.security.sandbox.profile import SandboxProfile


@dataclass
class SandboxResult:
    output: str
    return_code: int
    duration_ms: float
    timed_out: bool = False


class PlatformSandbox(ABC):
    @abstractmethod
    def execute(
        self,
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult: ...
```

- [ ] **Step 4: Pass tests**

Run: `pytest tests/test_execution_environment.py::test_sandbox_result_defaults tests/test_execution_environment.py::test_platform_sandbox_is_abstract -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add runtime/security/sandbox/platform.py tests/test_execution_environment.py
git commit -m "feat: add PlatformSandbox ABC and SandboxResult"
```

---

### Task 3: MacSandbox

**Files:**
- Create: `runtime/security/sandbox/mac_sandbox.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_execution_environment.py
import platform
import subprocess

import pytest

from runtime.security.sandbox.mac_sandbox import MacSandbox
from runtime.security.sandbox.profile import SandboxProfile


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific")
def test_mac_sandbox_execute_simple():
    sandbox = MacSandbox()
    profile = SandboxProfile()
    result = sandbox.execute(["echo", "hello"], profile)
    assert result.return_code == 0
    assert "hello" in result.output


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific")
def test_mac_sandbox_timeout():
    sandbox = MacSandbox()
    profile = SandboxProfile(timeout_sec=1)
    result = sandbox.execute(["sleep", "5"], profile)
    assert result.timed_out is True


def test_mac_sandbox_resource_limits():
    import resource
    sandbox = MacSandbox()
    profile = SandboxProfile(max_cpu_time_sec=60, max_memory_bytes=104857600)
    sandbox._apply_resource_limits(profile)
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    assert soft == 60
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    assert soft == 104857600


def test_mac_sandbox_scratch_cleanup():
    import os
    import tempfile
    sandbox = MacSandbox()
    scratch = sandbox._create_scratch_dir()
    assert os.path.isdir(scratch)
    sandbox._cleanup_scratch(scratch)
    assert not os.path.exists(scratch)
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_execution_environment.py::test_mac_sandbox_execute_simple tests/test_execution_environment.py::test_mac_sandbox_timeout tests/test_execution_environment.py::test_mac_sandbox_resource_limits tests/test_execution_environment.py::test_mac_sandbox_scratch_cleanup -v`
Expected: 4 FAILED (ImportError for MacSandbox)

- [ ] **Step 3: Implement MacSandbox**

```python
# runtime/security/sandbox/mac_sandbox.py
from __future__ import annotations

import logging
import os
import resource
import shutil
import subprocess
import tempfile
import time

from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile

logger = logging.getLogger("sandbox.mac")


class MacSandbox(PlatformSandbox):
    def execute(
        self,
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        exec_env = os.environ.copy()
        exec_env["NO_PROXY"] = "*"
        exec_env["HTTP_PROXY"] = ""
        exec_env["HTTPS_PROXY"] = ""
        exec_env["http_proxy"] = ""
        exec_env["https_proxy"] = ""

        if env:
            exec_env.update(env)

        scratch = self._create_scratch_dir()
        workdir = cwd or scratch

        start = time.monotonic()
        try:
            preexec = lambda: self._apply_resource_limits(profile)  # noqa: E731
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=profile.timeout_sec,
                cwd=workdir,
                env=exec_env,
                preexec_fn=preexec,
                stdin=subprocess.PIPE if stdin else None,
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
            return SandboxResult(
                output="",
                return_code=-1,
                duration_ms=duration,
                timed_out=True,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return SandboxResult(
                output=str(e),
                return_code=-1,
                duration_ms=duration,
            )
        finally:
            self._cleanup_scratch(scratch)

    def _apply_resource_limits(self, profile: SandboxProfile) -> None:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(profile.max_cpu_time_sec), int(profile.max_cpu_time_sec)),
        )
        resource.setrlimit(
            resource.RLIMIT_DATA,
            (profile.max_memory_bytes, profile.max_memory_bytes),
        )

    def _create_scratch_dir(self) -> str:
        return tempfile.mkdtemp(prefix="aether_sandbox_")

    def _cleanup_scratch(self, path: str) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logger.warning("Failed to clean up scratch dir %s", path)
```

- [ ] **Step 4: Pass tests**

Run: `pytest tests/test_execution_environment.py -v`
Expected: all previous + 4 new passed (or skipped on Linux)

- [ ] **Step 5: Commit**

```bash
git add runtime/security/sandbox/mac_sandbox.py tests/test_execution_environment.py
git commit -m "feat: add MacSandbox with resource limits and scratch directory"
```

---

### Task 4: LinuxSandbox

**Files:**
- Create: `runtime/security/sandbox/linux_sandbox.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_execution_environment.py
@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific")
def test_linux_sandbox_execute_simple():
    from runtime.security.sandbox.linux_sandbox import LinuxSandbox
    sandbox = LinuxSandbox()
    profile = SandboxProfile()
    result = sandbox.execute(["echo", "hello"], profile)
    assert result.return_code == 0
    assert "hello" in result.output


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific")
def test_linux_sandbox_network_isolation():
    from runtime.security.sandbox.linux_sandbox import LinuxSandbox
    sandbox = LinuxSandbox()
    profile = SandboxProfile(allow_network=False)
    result = sandbox.execute(["ping", "-c", "1", "8.8.8.8"], profile)
    # should fail because network is isolated
    assert result.return_code != 0


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific")
def test_linux_sandbox_timeout():
    from runtime.security.sandbox.linux_sandbox import LinuxSandbox
    sandbox = LinuxSandbox()
    profile = SandboxProfile(timeout_sec=1)
    result = sandbox.execute(["sleep", "5"], profile)
    assert result.timed_out is True
```

- [ ] **Step 2: Implement LinuxSandbox**

```python
# runtime/security/sandbox/linux_sandbox.py
from __future__ import annotations

import logging
import os
import resource
import shutil
import subprocess
import tempfile
import time

from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile

logger = logging.getLogger("sandbox.linux")

_CLONE_NEWNET = 0x40000000
_CLONE_NEWNS = 0x00020000


class LinuxSandbox(PlatformSandbox):
    def execute(
        self,
        command: list[str],
        profile: SandboxProfile,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        scratch = self._create_scratch_dir()
        workdir = cwd or scratch

        start = time.monotonic()
        try:
            preexec = lambda: self._apply_linux_limits(profile)  # noqa: E731
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=profile.timeout_sec,
                cwd=workdir,
                env=exec_env,
                preexec_fn=preexec,
                stdin=subprocess.PIPE if stdin else None,
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
        finally:
            self._cleanup_scratch(scratch)

    def _apply_linux_limits(self, profile: SandboxProfile) -> None:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(profile.max_cpu_time_sec), int(profile.max_cpu_time_sec)),
        )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (profile.max_memory_bytes, profile.max_memory_bytes),
        )
        if not profile.allow_network:
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.unshare(_CLONE_NEWNET)
            except Exception:
                logger.warning("unshare(CLONE_NEWNET) failed — network isolation not enforced")

    def _create_scratch_dir(self) -> str:
        return tempfile.mkdtemp(prefix="aether_sandbox_")

    def _cleanup_scratch(self, path: str) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logger.warning("Failed to clean up scratch dir %s", path)
```

- [ ] **Step 3: Pass tests (on Linux — skipped on macOS)**

Run: `pytest tests/test_execution_environment.py -v`
Expected: all tests pass or skip appropriately

- [ ] **Step 4: Commit**

```bash
git add runtime/security/sandbox/linux_sandbox.py tests/test_execution_environment.py
git commit -m "feat: add LinuxSandbox with network namespace isolation"
```

---

### Task 5: SandboxManager

**Files:**
- Create: `runtime/security/sandbox/manager.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_execution_environment.py
from unittest.mock import MagicMock, patch

from runtime.security.sandbox.manager import SandboxManager
from runtime.tools.tool_result import ToolCall, ToolResult


def test_sandbox_manager_init_default():
    manager = SandboxManager()
    assert manager._profiles is not None
    assert "python" in manager._profiles
    assert "shell" in manager._profiles


def test_sandbox_manager_passthrough_when_disabled():
    manager = SandboxManager(profiles={
        "python": SandboxProfile(enabled=False),
    })
    handler = MagicMock(return_value=ToolResult(
        call=ToolCall(id="1", name="python", arguments={}),
        output="direct output",
    ))
    call = ToolCall(id="1", name="python", arguments={"code": "print(1)"})
    result = manager.execute("python", handler, call)
    assert result.output == "direct output"
    handler.assert_called_once_with(call)


def test_sandbox_manager_policy_isolation():
    manager = SandboxManager(profiles={
        "filesystem": SandboxProfile(sandbox_type="policy", allowed_paths=["/tmp/workspace"]),
    })
    call = ToolCall(id="1", name="filesystem", arguments={"path": "/etc/passwd"})
    handler = MagicMock()
    result = manager.execute("filesystem", handler, call)
    assert result.is_error is True
    assert "Path not allowed" in result.output


def test_sandbox_manager_policy_allows_valid_path():
    import tempfile
    manager = SandboxManager(profiles={
        "filesystem": SandboxProfile(
            sandbox_type="policy",
            allowed_paths=[tempfile.gettempdir()],
        ),
    })
    handler = MagicMock(return_value=ToolResult(
        call=ToolCall(id="1", name="filesystem", arguments={}),
        output="valid content",
    ))
    call = ToolCall(id="1", name="filesystem", arguments={"path": tempfile.gettempdir() + "/test.txt"})
    result = manager.execute("filesystem", handler, call)
    assert result.is_error is False
    assert result.output == "valid content"


def test_sandbox_manager_process_isolation():
    mock_platform = MagicMock()
    mock_platform.execute.return_value = SandboxResult(output="42", return_code=0, duration_ms=5.0)

    manager = SandboxManager(profiles={
        "python": SandboxProfile(sandbox_type="process", allow_network=False),
    })
    manager._platform = mock_platform

    call = ToolCall(id="1", name="python", arguments={"code": "print(42)"})
    handler = MagicMock()
    result = manager.execute("python", handler, call)
    assert result.output == "42"
    assert result.is_error is False
    mock_platform.execute.assert_called_once()
    handler.assert_not_called()


def test_sandbox_manager_fallback_to_default_profile():
    manager = SandboxManager(profiles={})
    call = ToolCall(id="1", name="unknown_tool", arguments={"code": "x"})
    handler = MagicMock(return_value=ToolResult(
        call=call, output="fallback", duration_ms=1.0,
    ))
    result = manager.execute("unknown_tool", handler, call)
    handler.assert_called_once()
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_execution_environment.py -v`
Expected: 6 new FAILED (ImportError for SandboxManager)

- [ ] **Step 3: Implement SandboxManager**

```python
# runtime/security/sandbox/manager.py
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles, default_profile
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("sandbox.manager")


class SandboxManager:
    def __init__(
        self,
        profiles: dict[str, SandboxProfile] | None = None,
    ) -> None:
        if sys.platform == "linux":
            from runtime.security.sandbox.linux_sandbox import LinuxSandbox
            self._platform: PlatformSandbox = LinuxSandbox()
        else:
            from runtime.security.sandbox.mac_sandbox import MacSandbox
            self._platform = MacSandbox()
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
                    str(resolved).startswith(p)
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
```

- [ ] **Step 4: Pass tests**

Run: `pytest tests/test_execution_environment.py -v`
Expected: all 11+ tests passing

- [ ] **Step 5: Commit**

```bash
git add runtime/security/sandbox/manager.py tests/test_execution_environment.py
git commit -m "feat: add SandboxManager with policy and process isolation"
```

---

### Task 6: ToolExecutor integration

**Files:**
- Modify: `runtime/tools/tool_executor.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_execution_environment.py
from runtime.tools.tool_executor import ToolExecutor
from runtime.tools.tool_registry import ToolRegistry, ToolDescriptor


def test_tool_executor_with_sandbox():
    sandbox = MagicMock()
    sandbox.execute.return_value = ToolResult(
        call=ToolCall(id="1", name="python", arguments={}),
        output="sandboxed output",
    )

    registry = ToolRegistry()
    registry.register(
        "python",
        ToolDescriptor(
            name="python",
            handler=lambda c: ToolResult(call=c, output="unsandboxed"),
        ),
    )

    executor = ToolExecutor(registry=registry, sandbox_manager=sandbox)
    call = ToolCall(id="1", name="python", arguments={"code": "print(1)"})
    result = executor.execute(call)

    assert result.output == "sandboxed output"
    sandbox.execute.assert_called_once()


def test_tool_executor_without_sandbox():
    registry = ToolRegistry()
    registry.register(
        "python",
        ToolDescriptor(
            name="python",
            handler=lambda c: ToolResult(call=c, output="direct output"),
        ),
    )
    executor = ToolExecutor(registry=registry, sandbox_manager=None)
    call = ToolCall(id="1", name="python", arguments={"code": "print(1)"})
    result = executor.execute(call)
    assert result.output == "direct output"


def test_tool_executor_unknown_tool():
    executor = ToolExecutor(sandbox_manager=MagicMock())
    call = ToolCall(id="1", name="nonexistent", arguments={})
    result = executor.execute(call)
    assert result.is_error is True
    assert "not found" in result.output
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_execution_environment.py -v`
Expected: 3 new FAILED (ToolExecutor.__init__() got unexpected keyword 'sandbox_manager')

- [ ] **Step 3: Modify ToolExecutor**

```python
# modify runtime/tools/tool_executor.py
# In __init__ signature, add sandbox_manager parameter

class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        sandbox_manager: Any | None = None,
    ) -> None:
        self._registry = registry or default_registry
        self._sandbox_manager = sandbox_manager

    def execute(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        descriptor = self._registry.resolve(call.name)
        if not descriptor:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' not found in registry",
                is_error=True,
            )
        if not descriptor.handler:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' has no handler registered",
                is_error=True,
            )

        if self._sandbox_manager:
            return self._sandbox_manager.execute(call.name, descriptor.handler, call)

        start = time.monotonic()
        try:
            result = descriptor.handler(call)
            # ... rest unchanged
```

Full file after edit:

```python
# runtime/tools/tool_executor.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from runtime.tools.tool_registry import ToolRegistry, tool_registry as default_registry
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("tool_executor")


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        sandbox_manager: Any | None = None,
    ) -> None:
        self._registry = registry or default_registry
        self._sandbox_manager = sandbox_manager

    def execute(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        descriptor = self._registry.resolve(call.name)
        if not descriptor:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' not found in registry",
                is_error=True,
            )
        if not descriptor.handler:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' has no handler registered",
                is_error=True,
            )

        if self._sandbox_manager:
            return self._sandbox_manager.execute(call.name, descriptor.handler, call)

        start = time.monotonic()
        try:
            result = descriptor.handler(call)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import asyncio.tasks

                    result = asyncio.run_coroutine_threadsafe(result, loop).result(timeout=timeout_s)
                else:
                    result = asyncio.run(result)
            duration = (time.monotonic() - start) * 1000
            if isinstance(result, ToolResult):
                result.duration_ms = duration
                return result
            return ToolResult(call=call, output=result, duration_ms=duration)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.exception("Tool %s execution failed", call.name)
            return ToolResult(call=call, output=str(e), is_error=True, duration_ms=duration)

    async def execute_async(self, call: ToolCall, timeout_s: int = 30) -> ToolResult:
        descriptor = self._registry.resolve(call.name)
        if not descriptor:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' not found in registry",
                is_error=True,
            )
        if not descriptor.handler:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' has no handler registered",
                is_error=True,
            )

        if self._sandbox_manager:
            return self._sandbox_manager.execute(call.name, descriptor.handler, call)

        start = time.monotonic()
        try:
            result = descriptor.handler(call)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout_s)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(descriptor.handler, call), timeout=timeout_s
                )
            duration = (time.monotonic() - start) * 1000
            if isinstance(result, ToolResult):
                result.duration_ms = duration
                return result
            return ToolResult(call=call, output=result, duration_ms=duration)
        except asyncio.TimeoutError:
            return ToolResult(
                call=call,
                output=f"Tool '{call.name}' timed out after {timeout_s}s",
                is_error=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.exception("Tool %s async execution failed", call.name)
            return ToolResult(call=call, output=str(e), is_error=True, duration_ms=duration)


tool_executor = ToolExecutor()
```

- [ ] **Step 4: Pass tests**

Run: `pytest tests/test_execution_environment.py -v`
Expected: all 14+ tests passing

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -x -v`
Expected: 239+ tests passing (baseline + new sandbox tests). The 22 `test_dashboard_auth.py` failures + 1 `test_capabilities.py` may pre-exist.

- [ ] **Step 6: Commit**

```bash
git add runtime/tools/tool_executor.py tests/test_execution_environment.py
git commit -m "feat: integrate SandboxManager into ToolExecutor"
```

---

### Task 7: Settings integration

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_execution_environment.py
from config.settings import settings


def test_settings_has_sandbox_profiles():
    assert hasattr(settings, "sandbox_profiles")
    assert isinstance(settings.sandbox_profiles, dict)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_execution_environment.py::test_settings_has_sandbox_profiles -v`
Expected: FAILED (Settings has no sandbox_profiles)

- [ ] **Step 3: Add sandbox_profiles to Settings**

```python
# modify config/settings.py — add field to Settings dataclass

@dataclass
class Settings:
    ...
    sandbox_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def sandbox_manager(self) -> Any:
        from runtime.security.sandbox.manager import SandboxManager
        from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles

        profiles = builtin_profiles()
        for tool_name, overrides in self.sandbox_profiles.items():
            if tool_name in profiles:
                existing = profiles[tool_name]
                profiles[tool_name] = SandboxProfile(
                    enabled=overrides.get("enabled", existing.enabled),
                    sandbox_type=overrides.get("sandbox_type", existing.sandbox_type),
                    allow_network=overrides.get("allow_network", existing.allow_network),
                    max_cpu_time_sec=overrides.get("max_cpu_time_sec", existing.max_cpu_time_sec),
                    max_memory_bytes=overrides.get("max_memory_bytes", existing.max_memory_bytes),
                    allowed_paths=overrides.get("allowed_paths", existing.allowed_paths),
                    allowed_domains=overrides.get("allowed_domains", existing.allowed_domains),
                    timeout_sec=overrides.get("timeout_sec", existing.timeout_sec),
                )
            else:
                profiles[tool_name] = SandboxProfile(
                    enabled=overrides.get("enabled", True),
                    sandbox_type=overrides.get("sandbox_type", "process"),
                    allow_network=overrides.get("allow_network", False),
                    max_cpu_time_sec=overrides.get("max_cpu_time_sec", 30.0),
                    max_memory_bytes=overrides.get("max_memory_bytes", 524288000),
                    allowed_paths=overrides.get("allowed_paths", None),
                    allowed_domains=overrides.get("allowed_domains", None),
                    timeout_sec=overrides.get("timeout_sec", 30),
                )

        return SandboxManager(profiles=profiles)
```

- [ ] **Step 4: Pass tests**

Run: `pytest tests/test_execution_environment.py::test_settings_has_sandbox_profiles -v`
Expected: PASSED

- [ ] **Step 5: Add integration test for sandbox_manager()**

```python
# append to tests/test_execution_environment.py
def test_settings_sandbox_manager_creation():
    manager = settings.sandbox_manager()
    from runtime.security.sandbox.manager import SandboxManager
    assert isinstance(manager, SandboxManager)


def test_settings_sandbox_manager_custom_profile():
    overrides = settings.sandbox_profiles.copy()
    overrides["python"] = {"max_memory_bytes": 268435456}
    result = settings.sandbox_manager()
    assert result._profiles["python"].max_memory_bytes == 268435456
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x -v`
Expected: all tests pass (baseline + new sandbox tests)

- [ ] **Step 7: Commit**

```bash
git add config/settings.py tests/test_execution_environment.py
git commit -m "feat: add sandbox_profiles config and sandbox_manager() factory"
```

---

### Self-Review Checklist

- [ ] **Spec coverage:** Every section from the spec is covered by a task:
  - SandboxProfile dataclass + builtin defaults → Task 1 ✓
  - PlatformSandbox ABC + SandboxResult → Task 2 ✓
  - MacSandbox → Task 3 ✓
  - LinuxSandbox → Task 4 ✓
  - SandboxManager (policy + process isolation) → Task 5 ✓
  - ToolExecutor integration → Task 6 ✓
  - Settings integration → Task 7 ✓
  - Testing → all tasks ✓
- [ ] **Placeholder scan:** No TODOs, TBDs, or "implement later" patterns.
- [ ] **Type consistency:** `SandboxProfile`, `SandboxResult`, `PlatformSandbox.execute()` return types, `ToolResult` all consistent across tasks.
- [ ] **Import consistency:** All cross-module references match actual module paths.
