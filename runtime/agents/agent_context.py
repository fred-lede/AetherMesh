from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from runtime.agents.agent_step import AgentStep


@dataclass
class AgentContext:
    session_id: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)
    max_steps: int = 25
    system_prompt: str = ""
    task: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"agent_{uuid.uuid4().hex[:12]}"

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)

    def last_step(self) -> AgentStep | None:
        return self.steps[-1] if self.steps else None

    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    def total_tool_errors(self) -> int:
        return sum(1 for s in self.steps for r in s.tool_results if r.is_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "steps": len(self.steps),
            "max_steps": self.max_steps,
            "tool_calls": self.total_tool_calls(),
            "tool_errors": self.total_tool_errors(),
            "memory_keys": list(self.memory.keys()),
        }
