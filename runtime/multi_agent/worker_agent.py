from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("multi_agent.worker")


@dataclass
class SubtaskResult:
    agent_id: str
    task: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    success: bool = True
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task": self.task[:100],
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "tool_calls": self.tool_calls,
        }


class WorkerAgent:
    def __init__(
        self,
        agent_id: str,
        handler: Callable[[str, dict[str, Any]], Any] | None = None,
        tools: list[str] | None = None,
        system_prompt: str = "",
    ) -> None:
        self.agent_id = agent_id
        self._handler = handler
        self.tools = tools or []
        self.system_prompt = system_prompt

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> SubtaskResult:
        started = time.time()
        ctx = context or {}
        logger.info("Worker %s executing: %s", self.agent_id, task[:80])

        try:
            if self._handler:
                output = await self._handler(task, ctx)
            else:
                output = f"[{self.agent_id}] processed: {task[:50]}"
            duration = (time.time() - started) * 1000
            logger.debug("Worker %s completed in %.0fms", self.agent_id, duration)
            return SubtaskResult(
                agent_id=self.agent_id,
                task=task,
                output=output,
                duration_ms=duration,
                success=True,
            )
        except Exception as exc:
            duration = (time.time() - started) * 1000
            logger.error("Worker %s failed after %.0fms: %s", self.agent_id, duration, exc)
            return SubtaskResult(
                agent_id=self.agent_id,
                task=task,
                error=str(exc),
                duration_ms=duration,
                success=False,
            )
