from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.abi.lifecycle_manager import runtime_lifecycle
from runtime.context.execution_context import ExecutionContext
from runtime.skills.builtin_skills import register_builtin_skills

logger = logging.getLogger("skills.lifecycle")


class SkillRuntimeLifecycle(RuntimeComponent):
    async def initialize(self, ctx: ExecutionContext) -> None:
        register_builtin_skills()
        logger.debug("Skills initialized for execution %s", ctx.execution_id)

    async def start(self, ctx: ExecutionContext) -> None:
        logger.debug("Skills started for execution %s", ctx.execution_id)

    async def pause(self, ctx: ExecutionContext) -> None:
        logger.debug("Skills paused for execution %s", ctx.execution_id)

    async def resume(self, ctx: ExecutionContext) -> None:
        logger.debug("Skills resumed for execution %s", ctx.execution_id)

    async def cancel(self, ctx: ExecutionContext) -> None:
        logger.debug("Skills cancelled for execution %s", ctx.execution_id)

    async def shutdown(self, ctx: ExecutionContext) -> None:
        logger.debug("Skills shut down for execution %s", ctx.execution_id)


skill_lifecycle = SkillRuntimeLifecycle()
runtime_lifecycle.register("skill", skill_lifecycle)
