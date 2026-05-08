from __future__ import annotations

from dataclasses import dataclass, field

from runtime.state.execution_state import StreamStatus, validate_transition


@dataclass
class StreamMachineState:
    status: StreamStatus = StreamStatus.ACTIVE
    bytes_sent: int = 0
    chunk_count: int = 0
    started_at: float = 0.0
    duration_ms: float = 0.0

    def transition(self, target: StreamStatus) -> bool:
        if validate_transition(self.status, target):
            self.status = target
            return True
        return False
