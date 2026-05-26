from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from runtime.security.sandbox.mac_sandbox import MacSandbox
from runtime.security.sandbox.manager import SandboxManager
from runtime.security.sandbox.platform import PlatformSandbox, SandboxResult
from runtime.security.sandbox.profile import SandboxProfile, builtin_profiles, default_profile
from runtime.tools.tool_result import ToolCall, ToolResult


# ── Task 1: SandboxProfile ──────────────────────────────────────────────

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


# ── Task 2: PlatformSandbox ABC + SandboxResult ─────────────────────────

def test_sandbox_result_defaults():
    result = SandboxResult(output="hello", return_code=0, duration_ms=10.0)
    assert result.output == "hello"
    assert result.return_code == 0
    assert result.timed_out is False


def test_platform_sandbox_is_abstract():
    with pytest.raises(TypeError):
        PlatformSandbox()  # type: ignore


# ── Task 3: MacSandbox ──────────────────────────────────────────────────

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


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific")
def test_mac_sandbox_resource_limits():
    sandbox = MacSandbox()
    profile = SandboxProfile(max_cpu_time_sec=60, max_memory_bytes=104857600)
    result = sandbox.execute(
        ["python3", "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_CPU)[0])"],
        profile,
    )
    assert result.return_code == 0
    assert "60" in result.output


def test_mac_sandbox_scratch_cleanup():
    import os
    sandbox = MacSandbox()
    scratch = sandbox._create_scratch_dir()
    assert os.path.isdir(scratch)
    sandbox._cleanup_scratch(scratch)
    assert not os.path.exists(scratch)


# ── Task 4: LinuxSandbox ────────────────────────────────────────────────

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
    assert result.return_code != 0


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific")
def test_linux_sandbox_timeout():
    from runtime.security.sandbox.linux_sandbox import LinuxSandbox
    sandbox = LinuxSandbox()
    profile = SandboxProfile(timeout_sec=1)
    result = sandbox.execute(["sleep", "5"], profile)
    assert result.timed_out is True


# ── Task 5: SandboxManager ──────────────────────────────────────────────

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
    handler = MagicMock()
    result = manager.execute("unknown_tool", handler, call)
    # falls through to default profile with process isolation, but unknown_tool
    # has no process handler, so returns error
    assert result.is_error is True
    assert "No process handler" in result.output
    handler.assert_not_called()


# ── Task 6: ToolExecutor integration ────────────────────────────────────

def test_tool_executor_with_sandbox():
    from runtime.tools.tool_executor import ToolExecutor
    from runtime.tools.tool_registry import ToolRegistry, ToolDescriptor

    sandbox = MagicMock()
    sandbox.execute.return_value = ToolResult(
        call=ToolCall(id="1", name="python", arguments={}),
        output="sandboxed output",
    )

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(
            name="python",
            description="Run Python code",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            handler=lambda c: ToolResult(call=c, output="unsandboxed"),
        ),
    )

    executor = ToolExecutor(registry=registry, sandbox_manager=sandbox)
    call = ToolCall(id="1", name="python", arguments={"code": "print(1)"})
    result = executor.execute(call)

    assert result.output == "sandboxed output"
    sandbox.execute.assert_called_once()


def test_tool_executor_without_sandbox():
    from runtime.tools.tool_executor import ToolExecutor
    from runtime.tools.tool_registry import ToolRegistry, ToolDescriptor

    registry = ToolRegistry()
    registry.register(
        ToolDescriptor(
            name="python",
            description="Run Python code",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            handler=lambda c: ToolResult(call=c, output="direct output"),
        ),
    )
    executor = ToolExecutor(registry=registry, sandbox_manager=None)
    call = ToolCall(id="1", name="python", arguments={"code": "print(1)"})
    result = executor.execute(call)
    assert result.output == "direct output"


def test_tool_executor_unknown_tool():
    from runtime.tools.tool_executor import ToolExecutor

    executor = ToolExecutor(sandbox_manager=MagicMock())
    call = ToolCall(id="1", name="nonexistent", arguments={})
    result = executor.execute(call)
    assert result.is_error is True
    assert "not found" in result.output


# ── Task 7: Settings integration ────────────────────────────────────────

def test_settings_has_sandbox_profiles():
    from config.settings import settings
    assert hasattr(settings, "sandbox_profiles")
    assert isinstance(settings.sandbox_profiles, dict)
