from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.abi.lifecycle_manager import runtime_lifecycle
from runtime.context.execution_context import ExecutionContext
from runtime.tools.tool_runtime import tool_runtime

logger = logging.getLogger("tools.lifecycle")


class ToolRuntimeLifecycle(RuntimeComponent):
    async def initialize(self, ctx: ExecutionContext) -> None:
        logger.debug("ToolRuntime initialized for execution %s", ctx.execution_id)

    async def start(self, ctx: ExecutionContext) -> None:
        logger.debug("ToolRuntime started for execution %s", ctx.execution_id)

    async def pause(self, ctx: ExecutionContext) -> None:
        logger.debug("ToolRuntime paused for execution %s", ctx.execution_id)

    async def resume(self, ctx: ExecutionContext) -> None:
        logger.debug("ToolRuntime resumed for execution %s", ctx.execution_id)

    async def cancel(self, ctx: ExecutionContext) -> None:
        logger.debug("ToolRuntime cancelled for execution %s", ctx.execution_id)

    async def shutdown(self, ctx: ExecutionContext) -> None:
        logger.debug("ToolRuntime shut down for execution %s", ctx.execution_id)


tool_lifecycle = ToolRuntimeLifecycle()
runtime_lifecycle.register("tool", tool_lifecycle)
