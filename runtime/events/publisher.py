from __future__ import annotations

from abc import ABC, abstractmethod

from runtime.events.event import RuntimeEvent


class Publisher(ABC):
    @abstractmethod
    async def publish(self, event: RuntimeEvent) -> None:
        ...
