from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NodeRecord:
    node_id: str
    ip: str
    gpus: list[dict[str, Any]] = field(default_factory=list)
    workers: list[int] = field(default_factory=list)
    status: str = "healthy"
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ip": self.ip,
            "gpus": list(self.gpus),
            "workers": list(self.workers),
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "metadata": dict(self.metadata),
        }


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeRecord] = {}
        self._lock = threading.RLock()

    def upsert_node(
        self,
        *,
        node_id: str,
        ip: str,
        gpus: list[dict[str, Any]],
        workers: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> NodeRecord:
        with self._lock:
            record = self._nodes.get(node_id)
            if record is None:
                record = NodeRecord(node_id=node_id, ip=ip)
                self._nodes[node_id] = record
            record.ip = ip
            record.gpus = list(gpus)
            record.workers = list(workers)
            record.status = "healthy"
            record.last_heartbeat = time.time()
            record.metadata = dict(metadata or {})
            return record

    def heartbeat(
        self,
        *,
        node_id: str,
        ip: str,
        gpus: list[dict[str, Any]],
        workers: list[int],
    ) -> NodeRecord:
        return self.upsert_node(node_id=node_id, ip=ip, gpus=gpus, workers=workers)

    def list_nodes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.to_dict() for record in self._nodes.values()]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._nodes.get(node_id)
            return record.to_dict() if record else None

    def remove_stale(self, stale_after_s: int) -> list[str]:
        cutoff = time.time() - stale_after_s
        stale_nodes: list[str] = []
        with self._lock:
            for node_id, record in list(self._nodes.items()):
                if record.last_heartbeat < cutoff:
                    record.status = "stale"
                    stale_nodes.append(node_id)
        return stale_nodes
