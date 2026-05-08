from __future__ import annotations

import logging

from runtime import __version__, __runtime_name__
from runtime.context.execution_context import ExecutionContext
from runtime.events.bus import runtime_event_bus
from runtime.events.event_trace import event_trace
from runtime.events.event_types import EventType
from runtime.event_bridge import connect_event_buses
from runtime.replay.recorder import execution_recorder
from runtime.state.runtime_state_machine import RuntimeStateMachine
from runtime.state.execution_state import RuntimeStatus
from runtime.abi.lifecycle_manager import runtime_lifecycle

logger = logging.getLogger("runtime.kernel")


class AetherKernel:
    def __init__(self) -> None:
        self._initialized = False
        self._version = __version__
        self._runtime_name = __runtime_name__

    @property
    def version(self) -> str:
        return self._version

    @property
    def name(self) -> str:
        return self._runtime_name

    async def initialize(self) -> None:
        if self._initialized:
            return
        connect_event_buses()
        ctx = ExecutionContext(execution_id="kernel_init", session_id="kernel")
        await runtime_lifecycle.initialize_all(ctx)
        self._initialized = True
        logger.info(
            "%s v%s initialized with %d runtime components",
            self._runtime_name, self._version,
            runtime_lifecycle.component_count,
        )

    async def create_execution(
        self,
        session_id: str = "",
    ) -> tuple[ExecutionContext, RuntimeStateMachine]:
        ctx = ExecutionContext(session_id=session_id)
        state_machine = RuntimeStateMachine(ctx.execution_id)
        state_machine.transition(RuntimeStatus.CREATED)
        runtime_event_bus.publish_sync(
            __import__("runtime.events.event", fromlist=["event_from_type"]).event_from_type(
                EventType.EXECUTION_CREATED,
                execution_id=ctx.execution_id,
                source="kernel",
                session_id=session_id,
            )
        )
        return ctx, state_machine

    async def start_execution(
        self,
        ctx: ExecutionContext,
        state_machine: RuntimeStateMachine,
    ) -> None:
        if state_machine.transition(RuntimeStatus.PLANNING):
            ctx.start()
            await runtime_lifecycle.start_all(ctx)
            state_machine.transition(RuntimeStatus.EXECUTING)

    async def pause_execution(
        self,
        ctx: ExecutionContext,
        state_machine: RuntimeStateMachine,
    ) -> None:
        if state_machine.transition(RuntimeStatus.PAUSED):
            await runtime_lifecycle.pause_all(ctx)

    async def resume_execution(
        self,
        ctx: ExecutionContext,
        state_machine: RuntimeStateMachine,
    ) -> None:
        if state_machine.transition(RuntimeStatus.EXECUTING):
            await runtime_lifecycle.resume_all(ctx)

    async def cancel_execution(
        self,
        ctx: ExecutionContext,
        state_machine: RuntimeStateMachine,
    ) -> None:
        if state_machine.transition(RuntimeStatus.CANCELLED):
            ctx.complete()
            await runtime_lifecycle.cancel_all(ctx)

    async def fail_execution(
        self,
        ctx: ExecutionContext,
        state_machine: RuntimeStateMachine,
        error: str,
    ) -> None:
        if state_machine.transition(RuntimeStatus.FAILED):
            ctx.fail(error)
            runtime_event_bus.publish_sync(
                __import__("runtime.events.event", fromlist=["event_from_type"]).event_from_type(
                    EventType.EXECUTION_FAILED,
                    execution_id=ctx.execution_id,
                    source="kernel",
                    error=error,
                )
            )

    async def complete_execution(
        self,
        ctx: ExecutionContext,
        state_machine: RuntimeStateMachine,
    ) -> None:
        if state_machine.transition(RuntimeStatus.COMPLETED):
            ctx.complete()
            runtime_event_bus.publish_sync(
                __import__("runtime.events.event", fromlist=["event_from_type"]).event_from_type(
                    EventType.EXECUTION_COMPLETED,
                    execution_id=ctx.execution_id,
                    source="kernel",
                    elapsed_ms=ctx.elapsed_ms(),
                )
            )
            await runtime_lifecycle.shutdown_all(ctx)

    async def shutdown(self) -> None:
        ctx = ExecutionContext(execution_id="kernel_shutdown", session_id="kernel")
        await runtime_lifecycle.shutdown_all(ctx)
        runtime_event_bus.clear()
        event_trace.clear()
        execution_recorder.clear()
        self._initialized = False
        logger.info("%s shut down", self._runtime_name)

    def status(self) -> dict[str, object]:
        return {
            "name": self._runtime_name,
            "version": self._version,
            "initialized": self._initialized,
            "components": runtime_lifecycle.status(),
            "event_bus_subscribers": runtime_event_bus.subscriber_count,
            "recorded_executions": len(
                __import__("runtime.replay.recorder", fromlist=["execution_recorder"]).execution_recorder._recordings
            ),
        }


kernel = AetherKernel()
