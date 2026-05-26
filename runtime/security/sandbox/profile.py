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
