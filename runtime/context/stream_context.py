from __future__ import annotations

from dataclasses import dataclass, field

from runtime.state.execution_state import StreamStatus


@dataclass
class StreamState:
    active: bool = False
    status: StreamStatus = StreamStatus.ACTIVE
    bytes_sent: int = 0
    chunk_count: int = 0
    duration_ms: float = 0.0
    error: str = ""
