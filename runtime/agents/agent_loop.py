from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from runtime.agents.agent_context import AgentContext
from runtime.agents.agent_result import AgentResult
from runtime.agents.agent_step import AgentStep
from runtime.memory.memory_manager import memory_manager
from runtime.multi_agent.coordinator import coordinator
from runtime.orchestration.execution_plan import ExecutionPlan
from runtime.orchestration.graph import ExecutionNode
from runtime.orchestration.graph_executor import GraphExecutor
from runtime.orchestration.planner import Planner
from runtime.orchestration.retry_policy import RetryPolicy
from runtime.tools.tool_executor import ToolExecutor
from runtime.tools.tool_result import ToolCall

logger = logging.getLogger("agents.loop")


class AgentLoop:
    def __init__(self) -> None:
        self.executor = GraphExecutor(retry_policy=RetryPolicy(max_retries=2))
        self.planner = Planner()

    async def run(self, context: AgentContext, task: str) -> AgentResult:
        started = time.time()
        result = AgentResult(task=task, started_at=started)
        context.steps = []
        context.task = task

        self.executor.register_handler("llm_call", _make_llm_handler())
        self.executor.register_handler("tool_call", _make_tool_handler())
        self.executor.register_handler("conditional", _make_conditional_handler())
        self.executor.register_handler("agent_call", _make_agent_handler())

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

        memory_manager.episodic.record(
            session_id=context.session_id or task,
            model="agent",
            provider="agent_loop",
            task_summary=task[:200],
            duration_ms=exec_result.elapsed_ms,
            success=exec_result.success,
            error="; ".join(exec_result.node_errors.values()) if exec_result.node_errors else None,
        )

        return result


def _make_llm_handler():
    async def handler(node: ExecutionNode) -> Any:
        return {"text": "LLM handler not yet wired", "provider": "", "model": ""}
    return handler


def _make_tool_handler():
    executor = ToolExecutor()
    async def handler(node: ExecutionNode) -> Any:
        tool_name = node.config.get("tool", "")
        arguments = node.config.get("arguments", {})
        tool_call = ToolCall(
            id=node.id,
            name=tool_name,
            arguments=arguments,
            source_provider="agent_loop",
            source_model="",
        )
        tool_result = await asyncio.to_thread(executor.execute, tool_call)
        return {"tool": tool_name, "result": tool_result.output, "error": tool_result.is_error}
    return handler


def _make_conditional_handler():
    def _eval(condition: str, ctx: dict) -> bool:
        return bool(eval(condition, {"__builtins__": {}}, ctx))

    async def handler(node: ExecutionNode) -> Any:
        condition = node.config.get("condition", "True")
        ctx = node.config.get("context", {})
        try:
            result = await asyncio.to_thread(_eval, condition, ctx)
        except Exception as exc:
            logger.warning("Conditional eval failed for node %s: %s", node.id, exc)
            result = False
        return result
    return handler


def _make_agent_handler():
    async def handler(node: ExecutionNode) -> Any:
        agent_id = node.config.get("agent_id", "")
        task_str = node.config.get("task", "")
        sub_result = await coordinator.delegate(task_str, agent_id)
        return {"agent": agent_id, "success": sub_result.success, "output": str(sub_result.output)}
    return handler


agent_loop = AgentLoop()
