from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryState:
    short_term_keys: list[str] = field(default_factory=list)
    semantic_queries: int = 0
    episodic_records: int = 0
    last_retrieval_ms: float = 0.0
    last_storage_ms: float = 0.0
    retrieval_count: int = 0
    storage_count: int = 0
