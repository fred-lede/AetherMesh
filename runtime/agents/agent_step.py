from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from runtime.tools.tool_result import ToolCall, ToolResult


@dataclass
class AgentStep:
    step_number: int
    action: str = "think"
    model_call: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    observation: str = ""
    reasoning: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.time()

    @property
    def is_tool_step(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_complete(self) -> bool:
        return self.action == "complete"

    @property
    def has_error(self) -> bool:
        return any(r.is_error for r in self.tool_results)

    def complete(self, observation: str = "") -> None:
        self.completed_at = time.time()
        self.duration_ms = (self.completed_at - self.started_at) * 1000
        if observation:
            self.observation = observation

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_number,
            "action": self.action,
            "tool_calls": len(self.tool_calls),
            "tool_errors": sum(1 for r in self.tool_results if r.is_error),
            "duration_ms": self.duration_ms,
            "has_reasoning": bool(self.reasoning),
        }
