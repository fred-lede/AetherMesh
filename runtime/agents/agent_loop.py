from __future__ import annotations

import logging
import time
from typing import Any, Callable

from runtime.agents.agent_context import AgentContext
from runtime.agents.agent_result import AgentResult
from runtime.agents.agent_step import AgentStep
from runtime.orchestration.execution_plan import ExecutionPlan
from runtime.orchestration.graph import ExecutionNode, NodeType
from runtime.orchestration.graph_executor import GraphExecutor
from runtime.orchestration.planner import Planner
from runtime.orchestration.retry_policy import RetryPolicy

logger = logging.getLogger("agents.loop")


class AgentLoop:
    def __init__(self) -> None:
        self.executor = GraphExecutor(retry_policy=RetryPolicy(max_retries=2))
        self.planner = Planner()

    def register_handler(self, node_type: str, handler: Callable[[ExecutionNode], Any]) -> None:
        self.executor.register_handler(node_type, handler)

    async def run(self, context: AgentContext, task: str) -> AgentResult:
        started = time.time()
        result = AgentResult(task=task, started_at=started)
        context.steps = []
        context.task = task

        plan = ExecutionPlan.from_task(task, self.planner)
        context.metadata["graph_nodes"] = list(plan.graph.nodes.keys())
        context.metadata["graph_plan"] = "planned"

        exec_result = await self.executor.execute(plan.graph, context={"task": task})

        step = AgentStep(step_number=1, started_at=started)
        step.complete(str(exec_result.output or exec_result.node_results))
        context.add_step(step)

        result.steps = context.steps
        result.output = exec_result.output or exec_result.node_results
        result.finalize()
        return result


agent_loop = AgentLoop()
