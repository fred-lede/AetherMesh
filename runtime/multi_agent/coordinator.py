from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.multi_agent.worker_agent import WorkerAgent, SubtaskResult

logger = logging.getLogger("multi_agent.coordinator")


@dataclass
class OrchestrationPlan:
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    results: dict[str, SubtaskResult] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "steps": {k: v.to_dict() for k, v in self.results.items()},
        }


class Coordinator:
    def __init__(self) -> None:
        self._agents: dict[str, WorkerAgent] = {}

    def register_agent(self, agent_id: str, agent: WorkerAgent) -> None:
        self._agents[agent_id] = agent
        logger.info("Agent registered: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        logger.info("Agent unregistered: %s", agent_id)

    def get_agent(self, agent_id: str) -> WorkerAgent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    async def delegate(
        self,
        task: str,
        agent_id: str,
        context: dict[str, Any] | None = None,
    ) -> SubtaskResult:
        agent = self._agents.get(agent_id)
        if not agent:
            return SubtaskResult(
                agent_id=agent_id,
                task=task,
                error=f"Unknown agent: {agent_id}",
                success=False,
            )
        return await agent.execute(task, context)

    async def fan_out(
        self,
        task: str,
        agent_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[SubtaskResult]:
        ctx = context or {}
        agents = [aid for aid in agent_ids if aid in self._agents]
        if not agents:
            return []
        tasks = [self._agents[aid].execute(task, ctx) for aid in agents]
        return await asyncio.gather(*tasks)

    async def orchestrate(
        self,
        plan: OrchestrationPlan,
        context: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        started = time.time()
        result = OrchestrationResult()
        step_outputs: dict[str, Any] = {}

        for i, step in enumerate(plan.steps):
            agent_id = step.get("agent_id", "")
            task = step.get("task", "")
            deps = step.get("dependencies", [])
            step_id = step.get("id", f"step_{i}")

            step_context = dict(context or {})
            for dep_id in deps:
                if dep_id in step_outputs:
                    step_context[f"input_from_{dep_id}"] = step_outputs[dep_id]

            sub_result = await self.delegate(task, agent_id, step_context)
            result.results[step_id] = sub_result
            step_outputs[step_id] = sub_result.output

            if not sub_result.success:
                result.success = False
                logger.error("Orchestration failed at step %s: %s", step_id, sub_result.error)
                break

        result.elapsed_ms = (time.time() - started) * 1000
        return result
