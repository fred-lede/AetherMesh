from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from runtime.tools.tool_executor import tool_executor
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("agents.loop")


@dataclass
class AgentStep:
    step_number: int
    action: str = "think"
    model_call: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    observation: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0

    @property
    def is_tool_step(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class AgentResult:
    task: str = ""
    output: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    tool_call_count: int = 0
    error: str | None = None


class AgentContext:
    def __init__(self, session_id: str = "", tools: list[dict[str, Any]] | None = None) -> None:
        self.session_id = session_id
        self.tools = tools or []
        self.memory: dict[str, Any] = {}
        self.steps: list[AgentStep] = []
        self.max_steps: int = 25
        self.system_prompt: str = ""

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)


class AgentLoop:
    def run(self, context: AgentContext, task: str) -> AgentResult:
        started = time.time()
        result = AgentResult(task=task)
        context.steps = []

        for step_num in range(context.max_steps):
            step_started = time.time()
            step = AgentStep(step_number=step_num + 1, started_at=step_started)

            if step_num >= context.max_steps - 1:
                step.action = "complete"
                step.observation = "Max steps reached"
                step.completed_at = time.time()
                step.duration_ms = (step.completed_at - step_started) * 1000
                context.add_step(step)
                break

            step.action = "think"
            step.completed_at = time.time()
            step.duration_ms = (step.completed_at - step_started) * 1000
            context.add_step(step)

            if step_num == 0 and not step.tool_calls:
                step.action = "complete"
                break

        result.steps = context.steps
        result.total_duration_ms = (time.time() - started) * 1000
        result.tool_call_count = sum(len(s.tool_calls) for s in context.steps)
        return result


agent_loop = AgentLoop()
