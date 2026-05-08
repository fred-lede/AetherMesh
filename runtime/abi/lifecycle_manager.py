from __future__ import annotations

import logging
from typing import Any

from runtime.abi.runtime_contract import RuntimeComponent
from runtime.context.execution_context import ExecutionContext

logger = logging.getLogger("abi.lifecycle")


class RuntimeLifecycleManager:
    def __init__(self) -> None:
        self._components: dict[str, RuntimeComponent] = {}
        self._states: dict[str, str] = {}

    def register(self, name: str, component: RuntimeComponent) -> None:
        self._components[name] = component
        self._states[name] = "created"
        logger.info("Registered runtime component: %s", name)

    def unregister(self, name: str) -> None:
        self._components.pop(name, None)
        self._states.pop(name, None)

    async def initialize_all(self, ctx: ExecutionContext) -> None:
        for name, component in self._components.items():
            try:
                await component.initialize(ctx)
                self._states[name] = "initialized"
                logger.debug("Initialized component: %s", name)
            except Exception as e:
                logger.error("Failed to initialize %s: %s", name, e)

    async def start_all(self, ctx: ExecutionContext) -> None:
        for name, component in self._components.items():
            try:
                await component.start(ctx)
                self._states[name] = "running"
            except Exception as e:
                logger.error("Failed to start %s: %s", name, e)

    async def pause_all(self, ctx: ExecutionContext) -> None:
        for name, component in self._components.items():
            try:
                await component.pause(ctx)
                self._states[name] = "paused"
            except Exception as e:
                logger.error("Failed to pause %s: %s", name, e)

    async def resume_all(self, ctx: ExecutionContext) -> None:
        for name, component in self._components.items():
            try:
                await component.resume(ctx)
                self._states[name] = "running"
            except Exception as e:
                logger.error("Failed to resume %s: %s", name, e)

    async def cancel_all(self, ctx: ExecutionContext) -> None:
        for name, component in self._components.items():
            try:
                await component.cancel(ctx)
                self._states[name] = "cancelled"
            except Exception as e:
                logger.error("Failed to cancel %s: %s", name, e)

    async def shutdown_all(self, ctx: ExecutionContext) -> None:
        for name, component in self._components.items():
            try:
                await component.shutdown(ctx)
                self._states[name] = "shutdown"
            except Exception as e:
                logger.error("Failed to shutdown %s: %s", name, e)

    def get_state(self, name: str) -> str:
        return self._states.get(name, "unknown")

    def status(self) -> dict[str, str]:
        return dict(self._states)

    @property
    def component_count(self) -> int:
        return len(self._components)


runtime_lifecycle = RuntimeLifecycleManager()
