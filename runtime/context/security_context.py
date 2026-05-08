from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecurityScope:
    authenticated: bool = False
    client_id: str = ""
    roles: list[str] = field(default_factory=list)
    rate_limit_remaining: int = 0
    rate_limit_reset_s: float = 0.0
    validation_errors: list[str] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
