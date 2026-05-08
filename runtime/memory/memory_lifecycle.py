from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.abi.lifecycle_manager import runtime_lifecycle
from runtime.context.execution_context import ExecutionContext

logger = logging.getLogger("memory.lifecycle")


class MemoryRuntimeLifecycle(RuntimeComponent):
    async def initialize(self, ctx: ExecutionContext) -> None:
        logger.debug("MemoryRuntime initialized for execution %s", ctx.execution_id)

    async def start(self, ctx: ExecutionContext) -> None:
        logger.debug("MemoryRuntime started for execution %s", ctx.execution_id)

    async def pause(self, ctx: ExecutionContext) -> None:
        logger.debug("MemoryRuntime paused for execution %s", ctx.execution_id)

    async def resume(self, ctx: ExecutionContext) -> None:
        logger.debug("MemoryRuntime resumed for execution %s", ctx.execution_id)

    async def cancel(self, ctx: ExecutionContext) -> None:
        logger.debug("MemoryRuntime cancelled for execution %s", ctx.execution_id)

    async def shutdown(self, ctx: ExecutionContext) -> None:
        logger.debug("MemoryRuntime shut down")


memory_lifecycle = MemoryRuntimeLifecycle()
runtime_lifecycle.register("memory", memory_lifecycle)
