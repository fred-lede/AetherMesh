from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.events.event import RuntimeEvent
from runtime.events.event_types import EventType
from runtime.events.bus import runtime_event_bus
from runtime.replay.execution_snapshot import ExecutionSnapshot

logger = logging.getLogger("replay.recorder")


@dataclass
class RecordedExecution:
    execution_id: str = ""
    events: list[RuntimeEvent] = field(default_factory=list)
    snapshots: list[ExecutionSnapshot] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = True


class ExecutionRecorder:
    def __init__(self) -> None:
        self._recordings: dict[str, RecordedExecution] = {}
        self._active: bool = False

    @property
    def is_recording(self) -> bool:
        return self._active

    def start_recording(self, execution_id: str) -> None:
        recording = RecordedExecution(
            execution_id=execution_id,
            start_time=time.time(),
        )
        self._recordings[execution_id] = recording
        self._active = True
        logger.info("Started recording execution %s", execution_id)

    def record_event(self, event: RuntimeEvent) -> None:
        recording = self._recordings.get(event.execution_id)
        if recording is None:
            return
        recording.events.append(event)

    def snapshot(self, execution_id: str, **kwargs: Any) -> ExecutionSnapshot:
        recording = self._recordings.get(execution_id)
        if recording is None:
            snapshot = ExecutionSnapshot(execution_id=execution_id, timestamp=time.time())
            return snapshot
        snapshot = ExecutionSnapshot(
            execution_id=execution_id,
            timestamp=time.time(),
            event_count=len(recording.events),
            **kwargs,
        )
        recording.snapshots.append(snapshot)
        return snapshot

    def stop_recording(self, execution_id: str, success: bool = True) -> RecordedExecution | None:
        recording = self._recordings.get(execution_id)
        if recording is None:
            return None
        recording.end_time = time.time()
        recording.success = success
        self._active = any(r.end_time == 0 for eid, r in self._recordings.items() if eid != execution_id)
        logger.info(
            "Stopped recording execution %s (%d events, %d snapshots)",
            execution_id, len(recording.events), len(recording.snapshots),
        )
        return recording

    def get_recording(self, execution_id: str) -> RecordedExecution | None:
        return self._recordings.get(execution_id)

    def clear(self) -> None:
        self._recordings.clear()
        self._active = False

    def save_to_file(self, execution_id: str, path: str | Path) -> None:
        recording = self._recordings.get(execution_id)
        if recording is None:
            logger.warning("No recording found for %s", execution_id)
            return
        data = {
            "execution_id": recording.execution_id,
            "start_time": recording.start_time,
            "end_time": recording.end_time,
            "success": recording.success,
            "events": [
                {
                    "type": e.type_name,
                    "timestamp": e.timestamp,
                    "source": e.source,
                    "payload": e.payload,
                    "error": e.error,
                    "duration_ms": e.duration_ms,
                }
                for e in recording.events
            ],
            "snapshots": [s.to_dict() for s in recording.snapshots],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved recording to %s (%d events)", path, len(recording.events))


execution_recorder = ExecutionRecorder()
