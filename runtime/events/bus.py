from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType
from runtime.events.subscribers import EventHandler, Subscriber

logger = logging.getLogger("events.bus")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: list[RuntimeEvent] = []
        self._max_history: int = 10000

    def subscribe(
        self,
        handler: EventHandler,
        event_type: EventType | None = None,
        name: str = "",
    ) -> None:
        sub = Subscriber(handler=handler, event_type=event_type, name=name)
        self._subscribers.append(sub)

    def unsubscribe(self, handler: EventHandler) -> None:
        self._subscribers = [
            s for s in self._subscribers if s.handler is not handler
        ]

    async def publish(self, event: RuntimeEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for subscriber in self._subscribers:
            if subscriber.matches(event):
                await subscriber.handle(event)

    def publish_sync(self, event: RuntimeEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for subscriber in self._subscribers:
            if subscriber.matches(event):
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.ensure_future(subscriber.handle(event))
                        continue
                except RuntimeError:
                    pass
                try:
                    result = subscriber.handler(event)
                    if hasattr(result, "__await__"):
                        import asyncio
                        asyncio.run(result)
                except Exception:
                    logger.exception("Sync subscriber %s failed", subscriber.name)

    def clear(self) -> None:
        self._subscribers.clear()
        self._history.clear()

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[RuntimeEvent]:
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def __aenter__(self) -> EventBus:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.clear()


runtime_event_bus = EventBus()
