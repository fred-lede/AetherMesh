from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolState:
    pending_calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    current_tool: str = ""
    total_calls: int = 0
    total_errors: int = 0
    timeout_s: int = 30
