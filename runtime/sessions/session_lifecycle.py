from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.abi.lifecycle_manager import runtime_lifecycle
from runtime.context.execution_context import ExecutionContext
from runtime.sessions.session_manager import session_manager
from runtime.sessions.session_store import session_store

logger = logging.getLogger("sessions.lifecycle")


class SessionRuntimeLifecycle(RuntimeComponent):
    async def initialize(self, ctx: ExecutionContext) -> None:
        logger.debug("SessionRuntime initialized for execution %s", ctx.execution_id)

    async def start(self, ctx: ExecutionContext) -> None:
        logger.debug("SessionRuntime started for execution %s", ctx.execution_id)

    async def pause(self, ctx: ExecutionContext) -> None:
        logger.debug("SessionRuntime paused for execution %s", ctx.execution_id)

    async def resume(self, ctx: ExecutionContext) -> None:
        logger.debug("SessionRuntime resumed for execution %s", ctx.execution_id)

    async def cancel(self, ctx: ExecutionContext) -> None:
        logger.debug("SessionRuntime cancelled for execution %s", ctx.execution_id)

    async def shutdown(self, ctx: ExecutionContext) -> None:
        expired = session_store.evict_expired() if hasattr(session_store, 'evict_expired') else 0
        logger.debug("SessionRuntime shut down (evicted %d expired)", expired)


session_lifecycle = SessionRuntimeLifecycle()
runtime_lifecycle.register("session", session_lifecycle)
