from __future__ import annotations

import time

from control_plane.worker_registry import WorkerRegistry


def _registry_with_worker() -> WorkerRegistry:
    registry = WorkerRegistry()
    registry.register_node_workers(
        node_id="node-01",
        host="127.0.0.1",
        gpus=[{"id": 0, "name": "RTX 5090", "memory": 32768}],
        workers=[11434],
        model_assignments={"node-01:11434": ["gemma4:26b"]},
    )
    return registry


def test_release_removes_matching_assignment() -> None:
    registry = _registry_with_worker()

    registry.acquire("node-01:11434", assignment_id="a1")
    registry.acquire("node-01:11434", assignment_id="a2")
    registry.release("node-01:11434", assignment_id="a1")

    worker = registry.get_worker("node-01:11434")
    assert worker is not None
    assert worker["queue_size"] == 1
    assert worker["metadata"]["active_assignments"] == {"a2": worker["metadata"]["active_assignments"]["a2"]}


def test_expired_assignments_are_reclaimed(monkeypatch) -> None:
    registry = _registry_with_worker()
    registry.acquire("node-01:11434", assignment_id="stale")
    monkeypatch.setattr("control_plane.worker_registry.settings.worker_assignment_ttl_s", 1)

    worker = registry._workers["node-01:11434"]
    worker.metadata["active_assignments"]["stale"] = time.time() - 10

    refreshed = registry.get_worker("node-01:11434")

    assert refreshed is not None
    assert refreshed["queue_size"] == 0
    assert "active_assignments" not in refreshed["metadata"]
