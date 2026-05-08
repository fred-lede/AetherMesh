from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from runtime.events.event_types import EventType


@dataclass
class RuntimeEvent:
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    execution_id: str = ""
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0

    @property
    def type_name(self) -> str:
        return self.event_type.value


def event_from_type(
    event_type: EventType,
    execution_id: str = "",
    source: str = "",
    **payload: Any,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        execution_id=execution_id,
        source=source,
        payload=payload,
    )
