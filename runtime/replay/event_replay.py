from __future__ import annotations

import asyncio
import logging
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType
from runtime.events.bus import runtime_event_bus

logger = logging.getLogger("replay.event_replay")


class EventReplay:
    async def replay_events(
        self,
        events: list[RuntimeEvent],
        execution_id: str = "",
        speed: float = 1.0,
    ) -> int:
        count = 0
        for event in events:
            if execution_id and event.execution_id != execution_id:
                continue
            if speed > 0:
                await asyncio.sleep(event.duration_ms / 1000.0 / speed)
            await runtime_event_bus.publish(event)
            count += 1
        logger.info("Replayed %d events", count)
        return count

    def filter_events(
        self,
        events: list[RuntimeEvent],
        event_type: EventType | None = None,
        source: str = "",
    ) -> list[RuntimeEvent]:
        result = events
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if source:
            result = [e for e in result if e.source == source]
        return result

    def summarize(
        self,
        events: list[RuntimeEvent],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for event in events:
            type_name = event.type_name
            summary[type_name] = summary.get(type_name, 0) + 1
        return {
            "total_events": len(events),
            "unique_types": len(summary),
            "breakdown": summary,
        }


event_replay = EventReplay()
