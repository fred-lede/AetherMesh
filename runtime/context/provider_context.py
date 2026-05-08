from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderState:
    selected_provider: str = ""
    selected_model: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    selection_score: float = 0.0
    selection_reason: str = ""
    rules_applied: list[str] = field(default_factory=list)
    provider_latency_ms: float = 0.0
    retry_count: int = 0
