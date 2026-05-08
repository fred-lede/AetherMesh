from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from runtime.events.event import RuntimeEvent, event_from_type
from runtime.events.event_types import EventType
from runtime.events.bus import runtime_event_bus
from runtime.replay.execution_snapshot import ExecutionSnapshot
from runtime.replay.recorder import RecordedExecution

logger = logging.getLogger("replay.engine")


class ReplayEngine:
    def __init__(self) -> None:
        self._replays: dict[str, RecordedExecution] = {}

    def load(self, path: str | Path) -> RecordedExecution | None:
        path = Path(path)
        if not path.exists():
            logger.warning("Replay file not found: %s", path)
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load replay file: %s", e)
            return None

        execution = RecordedExecution(
            execution_id=data.get("execution_id", ""),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            success=data.get("success", True),
        )
        for edata in data.get("events", []):
            try:
                event_type = EventType(edata["type"])
            except ValueError:
                continue
            event = RuntimeEvent(
                event_type=event_type,
                timestamp=edata.get("timestamp", 0.0),
                execution_id=execution.execution_id,
                source=edata.get("source", ""),
                payload=edata.get("payload", {}),
                error=edata.get("error", ""),
                duration_ms=edata.get("duration_ms", 0.0),
            )
            execution.events.append(event)
        for sdata in data.get("snapshots", []):
            snapshot = ExecutionSnapshot(**sdata)
            execution.snapshots.append(snapshot)

        self._replays[execution.execution_id] = execution
        logger.info(
            "Loaded replay %s (%d events, %d snapshots)",
            execution.execution_id, len(execution.events), len(execution.snapshots),
        )
        return execution

    async def replay(
        self,
        execution_id: str,
        speed: float = 1.0,
    ) -> RecordedExecution | None:
        execution = self._replays.get(execution_id)
        if execution is None:
            logger.warning("No replay found for %s", execution_id)
            return None

        logger.info(
            "Replaying execution %s (%d events at %.1fx speed)",
            execution_id, len(execution.events), speed,
        )
        for event in execution.events:
            if speed > 0:
                await asyncio.sleep(event.duration_ms / 1000.0 / speed)
            # Re-publish event to the bus
            await runtime_event_bus.publish(event)

        logger.info("Replay of %s completed", execution_id)
        return execution

    def get_replay(self, execution_id: str) -> RecordedExecution | None:
        return self._replays.get(execution_id)

    def list_replays(self) -> list[str]:
        return list(self._replays.keys())

    def clear(self) -> None:
        self._replays.clear()


import asyncio

replay_engine = ReplayEngine()
