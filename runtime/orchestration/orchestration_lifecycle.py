from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.abi.lifecycle_manager import runtime_lifecycle
from runtime.context.execution_context import ExecutionContext

logger = logging.getLogger("orchestration.lifecycle")


class OrchestrationRuntimeLifecycle(RuntimeComponent):
    async def initialize(self, ctx: ExecutionContext) -> None:
        logger.debug("OrchestrationRuntime initialized for execution %s", ctx.execution_id)

    async def start(self, ctx: ExecutionContext) -> None:
        logger.debug("OrchestrationRuntime started for execution %s", ctx.execution_id)

    async def pause(self, ctx: ExecutionContext) -> None:
        logger.debug("OrchestrationRuntime paused for execution %s", ctx.execution_id)

    async def resume(self, ctx: ExecutionContext) -> None:
        logger.debug("OrchestrationRuntime resumed for execution %s", ctx.execution_id)

    async def cancel(self, ctx: ExecutionContext) -> None:
        logger.debug("OrchestrationRuntime cancelled for execution %s", ctx.execution_id)

    async def shutdown(self, ctx: ExecutionContext) -> None:
        logger.debug("OrchestrationRuntime shut down")


orchestration_lifecycle = OrchestrationRuntimeLifecycle()
runtime_lifecycle.register("orchestration", orchestration_lifecycle)
