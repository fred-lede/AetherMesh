from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.state.execution_state import SessionStatus


@dataclass
class SessionState:
    session_id: str = ""
    client_id: str = ""
    client_type: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    message_count: int = 0
    created_at: float = 0.0
    last_active_at: float = 0.0
    ttl_s: int = 3600
    metadata: dict[str, Any] = field(default_factory=dict)
