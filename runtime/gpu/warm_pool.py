from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("gpu.warm_pool")


@dataclass
class WarmPoolEntry:
    model_name: str
    node_id: str
    worker_port: int
    kept_warm_until: float = 0.0
    keepalive_interval_s: int = 300


class WarmPool:
    def __init__(self) -> None:
        self._entries: list[WarmPoolEntry] = []

    def keep_warm(self, model: str, node_id: str, port: int, ttl_s: int = 300) -> None:
        for entry in self._entries:
            if entry.model_name == model and entry.node_id == node_id and entry.worker_port == port:
                entry.kept_warm_until = time.time() + ttl_s
                return
        self._entries.append(
            WarmPoolEntry(model_name=model, node_id=node_id, worker_port=port,
                          kept_warm_until=time.time() + ttl_s)
        )

    def is_warm(self, model: str, node_id: str, port: int) -> bool:
        for entry in self._entries:
            if entry.model_name == model and entry.node_id == node_id and entry.worker_port == port:
                return time.time() < entry.kept_warm_until
        return False

    def evict_stale(self) -> None:
        now = time.time()
        self._entries = [e for e in self._entries if e.kept_warm_until > now]

    def warm_models(self) -> list[WarmPoolEntry]:
        self.evict_stale()
        return list(self._entries)


warm_pool = WarmPool()
