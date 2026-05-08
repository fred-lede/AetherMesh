from __future__ import annotations

import logging
from typing import Any

from runtime.events.bus import runtime_event_bus
from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType
from runtime.state.execution_state import (
    RuntimeStatus,
    StreamStatus,
    SessionStatus,
    AgentStatus,
    ProviderStatus,
    validate_transition,
    transition_event,
)

logger = logging.getLogger("state.machine")


class RuntimeStateMachine:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self._current: RuntimeStatus = RuntimeStatus.CREATED
        self._history: list[dict[str, Any]] = []

    @property
    def current(self) -> RuntimeStatus:
        return self._current

    def transition(self, target: RuntimeStatus) -> bool:
        if not validate_transition(self._current, target):
            logger.warning(
                "Invalid transition %s -> %s for execution %s",
                self._current.value, target.value, self.execution_id,
            )
            return False
        previous = self._current
        self._current = target
        self._history.append({
            "from": previous.value,
            "to": target.value,
            "execution_id": self.execution_id,
        })
        event = transition_event(
            self.execution_id,
            "execution",
            previous.value,
            target.value,
        )
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.ensure_future(runtime_event_bus.publish(event))
            else:
                runtime_event_bus.publish_sync(event)
        except RuntimeError:
            runtime_event_bus.publish_sync(event)
        logger.debug(
            "State transition: %s -> %s (%s)",
            previous.value, target.value, self.execution_id,
        )
        return True

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def can_transition_to(self, target: RuntimeStatus) -> bool:
        return validate_transition(self._current, target)

    def reset(self) -> None:
        self._current = RuntimeStatus.CREATED
        self._history.clear()
