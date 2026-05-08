from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphState:
    graph_id: str = ""
    node_states: dict[str, str] = field(default_factory=dict)
    node_results: dict[str, Any] = field(default_factory=dict)
    node_errors: dict[str, str] = field(default_factory=dict)
    node_durations: dict[str, float] = field(default_factory=dict)
    current_group: int = 0
    total_groups: int = 0
    cancelled: bool = False
