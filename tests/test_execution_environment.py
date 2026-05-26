from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

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
