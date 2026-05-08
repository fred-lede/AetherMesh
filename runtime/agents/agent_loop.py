from __future__ import annotations

import logging
import time
from typing import Any

from runtime.agents.agent_context import AgentContext
from runtime.agents.agent_result import AgentResult
from runtime.agents.agent_step import AgentStep
from runtime.tools.tool_executor import tool_executor
from runtime.tools.tool_result import ToolCall, ToolResult

logger = logging.getLogger("agents.loop")


class AgentLoop:
    def run(self, context: AgentContext, task: str) -> AgentResult:
        started = time.time()
        result = AgentResult(task=task, started_at=started)
        context.steps = []
        context.task = task

        for step_num in range(context.max_steps):
            step_started = time.time()
            step = AgentStep(step_number=step_num + 1, started_at=step_started)

            if step_num >= context.max_steps - 1:
                step.action = "complete"
                step.complete("Max steps reached")
                context.add_step(step)
                break

            step.action = "think"
            step.complete()
            context.add_step(step)

            if step_num == 0 and not step.tool_calls:
                step.action = "complete"
                break

        result.steps = context.steps
        result.finalize()
        return result


agent_loop = AgentLoop()
