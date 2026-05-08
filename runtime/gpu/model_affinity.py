from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("gpu.affinity")


@dataclass
class ModelAffinityRecord:
    model_name: str
    node_id: str
    worker_port: int
    loaded: bool = False
    load_count: int = 0
    last_used: float = 0.0
    avg_latency_ms: float = 0.0


class ModelAffinityTracker:
    def __init__(self) -> None:
        self._records: dict[str, list[ModelAffinityRecord]] = {}

    def record_load(self, model: str, node_id: str, port: int) -> None:
        self._records.setdefault(model, []).append(
            ModelAffinityRecord(model_name=model, node_id=node_id, worker_port=port, loaded=True, load_count=1)
        )

    def record_unload(self, model: str, node_id: str, port: int) -> None:
        records = self._records.get(model, [])
        for r in records:
            if r.node_id == node_id and r.worker_port == port:
                r.loaded = False

    def best_worker(self, model: str) -> tuple[str, int] | None:
        records = self._records.get(model, [])
        loaded = [r for r in records if r.loaded]
        if loaded:
            best = min(loaded, key=lambda r: r.avg_latency_ms)
            return (best.node_id, best.worker_port)
        if records:
            best = min(records, key=lambda r: r.load_count)
            return (best.node_id, best.worker_port)
        return None

    def all_models(self) -> list[str]:
        return list(self._records.keys())


model_affinity = ModelAffinityTracker()
