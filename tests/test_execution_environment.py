from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from runtime.security.sandbox.mac_sandbox import MacSandbox
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
