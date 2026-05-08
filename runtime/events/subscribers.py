from __future__ import annotations

import logging
from typing import Any, Callable

from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType

logger = logging.getLogger("events.subscribers")

EventHandler = Callable[[RuntimeEvent], Any]


class Subscriber:
    def __init__(
        self,
        handler: EventHandler,
        event_type: EventType | None = None,
        name: str = "",
    ) -> None:
        self.handler = handler
        self.event_type = event_type
        self.name = name or getattr(handler, "__name__", "anonymous")

    def matches(self, event: RuntimeEvent) -> bool:
        return self.event_type is None or event.event_type == self.event_type

    async def handle(self, event: RuntimeEvent) -> None:
        try:
            result = self.handler(event)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.exception("Subscriber %s failed handling %s", self.name, event.type_name)
