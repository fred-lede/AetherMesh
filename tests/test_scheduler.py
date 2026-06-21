from __future__ import annotations

import sys
import types

import pytest

redis_stub = types.ModuleType("redis")
redis_stub.RedisError = Exception
redis_stub.from_url = lambda *args, **kwargs: None
sys.modules.setdefault("redis", redis_stub)

from control_plane.scheduler import SchedulerOverloadedError, Scheduler
from control_plane.worker_registry import WorkerRegistry


class DummyQueue:
    def enqueue(self, payload):
        return {"payload": payload}

    def length(self):
        return 0


def _registry_with_worker(*, queue_size: int = 0, gpu_utilization: float = 0.0) -> WorkerRegistry:
    registry = WorkerRegistry()
    registry.register_node_workers(
        node_id="node-01",
        host="127.0.0.1",
        gpus=[{"id": 0, "name": "RTX 5090", "memory": 32768, "utilization": gpu_utilization}],
        workers=[11434],
        model_assignments={"node-01:11434": ["gemma4:26b"]},
    )
    if queue_size:
        registry.update_worker_runtime("node-01:11434", queue_size=queue_size)
    return registry


def test_dispatch_reports_queue_capacity_only_when_queue_is_full() -> None:
    scheduler = Scheduler(
        _registry_with_worker(queue_size=8),
        DummyQueue(),
        max_worker_queue_size=8,
    )

    with pytest.raises(SchedulerOverloadedError, match="queue capacity"):
        scheduler.dispatch(model="gemma4:26b", provider="ollama")


def test_dispatch_reports_gpu_saturation_as_unavailable() -> None:
    scheduler = Scheduler(
        _registry_with_worker(gpu_utilization=90.0),
        DummyQueue(),
        max_worker_queue_size=8,
    )

    with pytest.raises(RuntimeError, match="GPU saturated"):
        scheduler.dispatch(model="gemma4:26b", provider="ollama")


def test_dispatch_blocks_model_switch_while_worker_is_active() -> None:
    registry = WorkerRegistry()
    registry.register_node_workers(
        node_id="node-01",
        host="127.0.0.1",
        gpus=[{"id": 0, "name": "RTX 5090", "memory": 32768}],
        workers=[11434],
        model_assignments={"node-01:11434": ["gemma4:31b-it-qat", "qwen3-coder:30b"]},
        node_metadata={
            "worker_runtime": {
                "11434": {"ps_model_vram_mb": {"gemma4:31b-it-qat": 19333}},
            },
        },
    )
    registry.acquire("node-01:11434", assignment_id="active")
    scheduler = Scheduler(registry, DummyQueue(), max_worker_queue_size=8)

    with pytest.raises(RuntimeError, match="Insufficient VRAM"):
        scheduler.dispatch(model="qwen3-coder:30b", provider="ollama")


def test_dispatch_allows_idle_worker_model_switch() -> None:
    registry = WorkerRegistry()
    registry.register_node_workers(
        node_id="node-01",
        host="127.0.0.1",
        gpus=[{"id": 0, "name": "RTX 5090", "memory": 32768}],
        workers=[11434],
        model_assignments={"node-01:11434": ["gemma4:31b-it-qat", "qwen3-coder:30b"]},
        node_metadata={
            "worker_runtime": {
                "11434": {"ps_model_vram_mb": {"gemma4:31b-it-qat": 19333}},
            },
        },
    )
    scheduler = Scheduler(registry, DummyQueue(), max_worker_queue_size=8)

    dispatch = scheduler.dispatch(model="qwen3-coder:30b", provider="ollama")

    assert dispatch["status"] == "assigned"
