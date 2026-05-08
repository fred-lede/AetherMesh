from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.abi.lifecycle_manager import runtime_lifecycle
from runtime.context.execution_context import ExecutionContext
from runtime.multi_agent import coordinator

logger = logging.getLogger("agents.lifecycle")


class AgentRuntimeLifecycle(RuntimeComponent):
    async def initialize(self, ctx: ExecutionContext) -> None:
        logger.debug("AgentRuntime initialized for execution %s", ctx.execution_id)

    async def start(self, ctx: ExecutionContext) -> None:
        logger.debug("AgentRuntime started for execution %s", ctx.execution_id)

    async def pause(self, ctx: ExecutionContext) -> None:
        logger.debug("AgentRuntime paused for execution %s", ctx.execution_id)

    async def resume(self, ctx: ExecutionContext) -> None:
        logger.debug("AgentRuntime resumed for execution %s", ctx.execution_id)

    async def cancel(self, ctx: ExecutionContext) -> None:
        for agent_id in coordinator.list_agents():
            coordinator.unregister_agent(agent_id)
        logger.debug("AgentRuntime cancelled for execution %s", ctx.execution_id)

    async def shutdown(self, ctx: ExecutionContext) -> None:
        coordinator._agents.clear()
        logger.debug("AgentRuntime shut down")


agent_lifecycle = AgentRuntimeLifecycle()
runtime_lifecycle.register("agent", agent_lifecycle)
