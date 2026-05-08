from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUState:
    allocations: dict[str, dict[str, Any]] = field(default_factory=dict)
    requested_vram_mb: int = 0
    allocated_vram_mb: int = 0
    devices_used: list[str] = field(default_factory=list)
    warm_pool_hit: bool = False
    model_loaded: str = ""
