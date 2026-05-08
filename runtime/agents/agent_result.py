from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from runtime.agents.agent_step import AgentStep


@dataclass
class AgentResult:
    task: str = ""
    output: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    tool_call_count: int = 0
    tool_error_count: int = 0
    step_count: int = 0
    error: str | None = None
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.time()

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def has_tool_calls(self) -> bool:
        return self.tool_call_count > 0

    def finalize(self) -> None:
        self.total_duration_ms = (time.time() - self.started_at) * 1000
        self.step_count = len(self.steps)
        self.tool_call_count = sum(len(s.tool_calls) for s in self.steps)
        self.tool_error_count = sum(1 for s in self.steps for r in s.tool_results if r.is_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task[:200],
            "output_length": len(self.output),
            "steps": self.step_count,
            "total_duration_ms": self.total_duration_ms,
            "tool_calls": self.tool_call_count,
            "tool_errors": self.tool_error_count,
            "succeeded": self.succeeded,
            "error": self.error,
        }

    @classmethod
    def from_error(cls, task: str, error: str) -> AgentResult:
        return cls(task=task, error=error, started_at=time.time())
