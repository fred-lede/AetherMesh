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
