from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings


@dataclass(slots=True)
class WorkerRecord:
    worker_id: str
    node_id: str
    host: str
    port: int
    provider: str
    gpu_id: int
    gpu_name: str
    gpu_memory: int
    gpu_utilization: float = 0.0
    temperature: float = 0.0
    queue_size: int = 0
    status: str = "healthy"
    last_heartbeat: float = field(default_factory=time.time)
    models: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    consecutive_errors: int = 0
    degraded_until: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        output = {
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "provider": self.provider,
            "gpu_id": self.gpu_id,
            "gpu_name": self.gpu_name,
            "gpu_memory": self.gpu_memory,
            "gpu_utilization": self.gpu_utilization,
            "temperature": self.temperature,
            "queue_size": self.queue_size,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "models": list(self.models),
            "metadata": dict(self.metadata),
        }
        output["metadata"]["error_streak"] = self.consecutive_errors
        if self.degraded_until > 0:
            output["metadata"]["degraded_until"] = self.degraded_until
        return output


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = threading.RLock()

    def _refresh_state(self, record: WorkerRecord, now: float) -> None:
        if record.status == "degraded" and now >= record.degraded_until and record.queue_size == 0:
            record.status = "healthy"
            record.consecutive_errors = 0
            record.metadata.pop("last_error", None)
            record.metadata.pop("last_error_at", None)

    def register_node_workers(
        self,
        *,
        node_id: str,
        host: str,
        gpus: list[dict[str, Any]],
        workers: list[int],
        model_assignments: dict[str, list[str]] | None = None,
        node_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        registered: list[dict[str, Any]] = []
        with self._lock:
            now = time.time()
            gpu_lookup = {int(gpu.get("id", idx)): gpu for idx, gpu in enumerate(gpus)}
            for index, port in enumerate(workers):
                gpu = gpu_lookup.get(index) or gpu_lookup.get(index % max(len(gpu_lookup), 1), {})
                worker_id = f"{node_id}:{port}"
                record = self._workers.get(worker_id)
                if record is None:
                    record = WorkerRecord(
                        worker_id=worker_id,
                        node_id=node_id,
                        host=host,
                        port=int(port),
                        provider="ollama",
                        gpu_id=int(gpu.get("id", index)),
                        gpu_name=str(gpu.get("name", "unknown")),
                        gpu_memory=int(gpu.get("memory", 0)),
                    )
                    self._workers[worker_id] = record

                self._refresh_state(record, now)
                record.host = host
                record.port = int(port)
                record.gpu_id = int(gpu.get("id", index))
                record.gpu_name = str(gpu.get("name", "unknown"))
                record.gpu_memory = int(gpu.get("memory", 0))
                record.gpu_utilization = float(gpu.get("utilization", 0.0))
                record.temperature = float(gpu.get("temperature", 0.0))
                if record.status == "dead":
                    record.status = "healthy"
                    record.consecutive_errors = 0
                    record.degraded_until = 0.0
                record.last_heartbeat = now

                specific_key = f"{node_id}:{int(port)}"
                fallback_key = str(int(port))
                assignments = model_assignments or {}
                record.models = list(assignments.get(specific_key, assignments.get(fallback_key, [])))

                runtime_map = node_metadata.get("worker_runtime", {}) if isinstance(node_metadata, dict) else {}
                runtime = runtime_map.get(str(int(port)), {}) if isinstance(runtime_map, dict) else {}
                if isinstance(runtime, dict):
                    record.metadata.update(runtime)

                registered.append(record.to_dict())
        return registered

    def update_worker_runtime(
        self,
        worker_id: str,
        *,
        gpu_utilization: float | None = None,
        temperature: float | None = None,
        queue_size: int | None = None,
        status: str | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            now = time.time()
            self._refresh_state(record, now)
            if gpu_utilization is not None:
                record.gpu_utilization = float(gpu_utilization)
            if temperature is not None:
                record.temperature = float(temperature)
            if queue_size is not None:
                record.queue_size = max(0, int(queue_size))
            if status is not None:
                record.status = status
            if last_error:
                record.metadata["last_error"] = last_error
                record.metadata["last_error_at"] = now
            record.last_heartbeat = now
            return record.to_dict()

    def release(self, worker_id: str, success: bool = True) -> dict[str, Any] | None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None

            now = time.time()
            record.queue_size = max(0, record.queue_size - 1)
            if success:
                record.status = "healthy"
                record.consecutive_errors = 0
                record.degraded_until = 0.0
                record.metadata.pop("last_error", None)
                record.metadata.pop("last_error_at", None)
            else:
                record.consecutive_errors += 1
                record.metadata["last_error_at"] = now
                threshold = max(1, int(settings.worker_degrade_after_errors))
                if record.consecutive_errors >= threshold:
                    record.status = "degraded"
                    cooldown = max(1, int(settings.worker_degrade_cooldown_s))
                    record.degraded_until = now + cooldown

            record.last_heartbeat = now
            return record.to_dict()

    def acquire(self, worker_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            now = time.time()
            self._refresh_state(record, now)
            record.queue_size += 1
            record.last_heartbeat = now
            return record.to_dict()

    def list_workers(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.time()
            for record in self._workers.values():
                self._refresh_state(record, now)
            return [record.to_dict() for record in self._workers.values()]

    def get_worker(self, worker_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            self._refresh_state(record, time.time())
            return record.to_dict()

    def available_workers(
        self,
        *,
        model: str | None = None,
        provider: str = "ollama",
    ) -> list[dict[str, Any]]:
        with self._lock:
            now = time.time()
            workers = []
            for record in self._workers.values():
                self._refresh_state(record, now)
                if record.provider != provider:
                    continue
                if record.status == "dead":
                    continue
                if model and record.models and model not in record.models:
                    continue
                workers.append(record.to_dict())
            return workers

    def remove_stale_workers(self, stale_after_s: int, max_worker_queue_size: int = 8) -> list[dict[str, Any]]:
        """
        Remove stale workers and return tasks that need to be requeued.
        Returns list of task dicts for requeue.
        """
        cutoff = time.time() - stale_after_s
        dead_workers: list[dict[str, Any]] = []
        with self._lock:
            for worker_id, record in self._workers.items():
                if record.last_heartbeat < cutoff and record.status != "dead":
                    record.status = "dead"
                    record.metadata["death_reason"] = "heartbeat_timeout"
                    record.metadata["death_at"] = time.time()
                    # Return worker info for task requeue
                    dead_workers.append({
                        "worker_id": worker_id,
                        "node_id": record.node_id,
                        "port": record.port,
                        "queue_size": record.queue_size,
                        "death_reason": "heartbeat_timeout",
                    })
                    # Note: Tasks in queue are managed by RedisTaskQueue
                    # Control plane should query Redis for pending tasks from this worker
        return dead_workers

    def mark_worker_dead(self, worker_id: str, reason: str = "manual") -> bool:
        """Mark a worker as dead and return True if found."""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = "dead"
                self._workers[worker_id].metadata["death_reason"] = reason
                self._workers[worker_id].metadata["death_at"] = time.time()
                return True
        return False

    def get_dead_workers(self) -> list[dict[str, Any]]:
        """Get list of dead workers."""
        dead = []
        with self._lock:
            for worker_id, record in self._workers.items():
                if record.status == "dead":
                    dead.append({
                        "worker_id": worker_id,
                        "node_id": record.node_id,
                        "death_reason": record.metadata.get("death_reason"),
                        "death_at": record.metadata.get("death_at"),
                    })
        return dead
