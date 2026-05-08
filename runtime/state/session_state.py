from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.state.execution_state import SessionStatus, validate_transition


@dataclass
class SessionMachineState:
    status: SessionStatus = SessionStatus.ACTIVE
    idle_since: float = 0.0
    expires_at: float = 0.0

    def transition(self, target: SessionStatus) -> bool:
        if validate_transition(self.status, target):
            self.status = target
            return True
        return False
