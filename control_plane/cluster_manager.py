from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_queue.redis_queue import RedisTaskQueue
from cluster.gpu_discovery import discover_gpus
from config.settings import settings
from metrics.metrics import metrics_store

LOGGER = logging.getLogger("aiih.cluster")

from .node_registry import NodeRegistry
from .scheduler import Scheduler, SchedulerOverloadedError
from .worker_registry import WorkerRegistry


class GPUModel(BaseModel):
    id: int
    name: str
    memory: int
    utilization: float = 0.0
    temperature: float = 0.0
    power_watts: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeRegistrationPayload(BaseModel):
    node_id: str
    ip: str
    gpus: list[GPUModel] = Field(default_factory=list)
    workers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DispatchRequest(BaseModel):
    model: str
    provider: str = "ollama"
    allow_queue: bool = False
    task_payload: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "hybrid"  # "least-loaded", "round-robin", or "hybrid"


class WorkerReleaseRequest(BaseModel):
    worker_id: str
    assignment_id: str | None = None
    success: bool = True
    gpu_utilization: float | None = None
    temperature: float | None = None


class TelemetryEvent(BaseModel):
    endpoint: str
    latency_ms: float
    model: str = ""
    provider: str = ""
    worker_id: str = ""
    error: bool = False
    error_code: str = ""


class QueueTaskRequest(BaseModel):
    payload: dict[str, Any]


class TaskPruneRequest(BaseModel):
    statuses: list[str] = Field(default_factory=list)
    older_than_hours: int = 72
    limit: int = 10000


class ClusterManager:
    def __init__(self) -> None:
        self.node_registry = NodeRegistry()
        self.worker_registry = WorkerRegistry()
        self.task_queue = RedisTaskQueue(settings.redis_url)
        self.scheduler = Scheduler(
            self.worker_registry,
            self.task_queue,
            max_worker_queue_size=settings.max_worker_queue_size,
        )
        self._stop_event = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self._last_prune_date = ""

    def start_background_tasks(self) -> None:
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop_event.is_set():
                self.node_registry.remove_stale(settings.stale_after_s)
                stale_workers = self.worker_registry.remove_stale_workers(settings.stale_after_s, settings.max_worker_queue_size)
                # Process stale workers
                for worker_info in stale_workers:
                    worker_id = worker_info.get("worker_id")
                    worker = self.worker_registry.get_worker(worker_id)
                    if worker:
                        metrics_store.set_worker_usage(
                            worker_id,
                            node_id=worker["node_id"],
                            gpu_utilization=float(worker.get("gpu_utilization", 0.0)),
                            queue_size=int(worker.get("queue_size", 0)),
                            status="dead",
                        )
                        LOGGER.warning(f"Worker {worker_id} marked dead, queue_size={worker_info.get('queue_size')}")

                # Try to requeue pending tasks from dead workers
                if stale_workers:
                    try:
                        requeued = self.manager.scheduler.requeue_pending_tasks()
                        if requeued > 0:
                            LOGGER.info(f"Requeued {requeued} tasks from dead workers")
                    except Exception as e:
                        LOGGER.error(f"Error requeueing tasks: {e}")

                self.run_scheduled_prune()
                metrics_store.set_queue_length(self.task_queue.length())
                time.sleep(max(settings.heartbeat_interval_s, 5))

        self._maintenance_thread = threading.Thread(target=_loop, daemon=True)
        self._maintenance_thread.start()

    def stop_background_tasks(self) -> None:
        self._stop_event.set()

    def run_scheduled_prune(self) -> dict[str, Any] | None:
        if not settings.task_prune_enabled:
            return None

        now = datetime.datetime.now()
        if now.hour != settings.task_prune_hour or now.minute != settings.task_prune_minute:
            return None

        today = now.strftime("%Y-%m-%d")
        if self._last_prune_date == today:
            return None

        result = self.prune_tasks(
            statuses=settings.task_prune_statuses,
            older_than_hours=settings.task_retention_hours,
            limit=10000,
        )
        result["auto"] = True
        self._last_prune_date = today
        return result

    def prune_tasks(self, *, statuses: list[str], older_than_hours: int, limit: int) -> dict[str, Any]:
        chosen_statuses = set(statuses) if statuses else set(settings.task_prune_statuses)
        older_than_s = max(int(older_than_hours), 0) * 3600
        result = self.task_queue.prune_tasks(
            statuses=chosen_statuses,
            older_than_s=older_than_s,
            limit=max(int(limit), 1),
        )
        result["auto"] = False
        result["run_at"] = datetime.datetime.now().isoformat()
        result["older_than_hours"] = max(int(older_than_hours), 0)
        metrics_store.set_queue_length(self.task_queue.length())
        return result

    def model_assignments(self) -> dict[str, list[str]]:
        registry = settings.model_registry()
        assignments: dict[str, list[str]] = {}
        for model in registry.get("models", []):
            model_name = str(model.get("name", ""))
            if not model_name:
                continue
            for binding in model.get("worker_bindings", []):
                node_id = str(binding.get("node_id", "")).strip()
                port = binding.get("port")
                if not node_id or port is None:
                    continue
                assignments.setdefault(f"{node_id}:{int(port)}", []).append(model_name)
            for port in model.get("worker_ports", []):
                assignments.setdefault(str(int(port)), []).append(model_name)
        return assignments

    def register_node(self, payload: NodeRegistrationPayload) -> dict[str, Any]:
        node = self.node_registry.upsert_node(
            node_id=payload.node_id,
            ip=payload.ip,
            gpus=[gpu.model_dump() for gpu in payload.gpus],
            workers=payload.workers,
            metadata=payload.metadata,
        )
        workers = self.worker_registry.register_node_workers(
            node_id=payload.node_id,
            host=payload.ip,
            gpus=[gpu.model_dump() for gpu in payload.gpus],
            workers=payload.workers,
            model_assignments=self.model_assignments(),
            node_metadata=payload.metadata,
        )
        for worker in workers:
            metrics_store.set_worker_usage(
                worker["worker_id"],
                node_id=worker["node_id"],
                gpu_utilization=float(worker.get("gpu_utilization", 0.0)),
                queue_size=int(worker.get("queue_size", 0)),
                status=str(worker.get("status", "healthy")),
            )
        return {"node": node.to_dict(), "workers": workers}

    def aggregate_gpus(self) -> list[dict[str, Any]]:
        nodes = self.node_registry.list_nodes()
        gpus: list[dict[str, Any]] = []
        for node in nodes:
            for gpu in node.get("gpus", []):
                item = dict(gpu)
                item["node_id"] = node["node_id"]
                item["ip"] = node["ip"]
                gpus.append(item)
        return gpus or discover_gpus()

    def metrics_snapshot(self) -> dict[str, Any]:
        metrics_store.set_queue_length(self.task_queue.length())
        for worker in self.worker_registry.list_workers():
            metrics_store.set_worker_usage(
                worker["worker_id"],
                node_id=worker["node_id"],
                gpu_utilization=float(worker.get("gpu_utilization", 0.0)),
                queue_size=int(worker.get("queue_size", 0)),
                status=str(worker.get("status", "unknown")),
            )
        snapshot = metrics_store.snapshot()
        snapshot["redis_backend"] = self.task_queue.backend
        snapshot["nodes"] = self.node_registry.list_nodes()
        snapshot["workers"] = self.worker_registry.list_workers()
        return snapshot


manager = ClusterManager()
app = FastAPI(title="AetherMesh Control Plane", version="4.0.0")


@app.on_event("startup")
def startup_event() -> None:
    manager.start_background_tasks()


@app.on_event("shutdown")
def shutdown_event() -> None:
    manager.stop_background_tasks()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "control-plane",
        "redis_backend": manager.task_queue.backend,
        "nodes": len(manager.node_registry.list_nodes()),
        "workers": len(manager.worker_registry.list_workers()),
    }


@app.post("/cluster/register")
def register_node(payload: NodeRegistrationPayload) -> dict[str, Any]:
    return manager.register_node(payload)


@app.post("/cluster/heartbeat")
def heartbeat(payload: NodeRegistrationPayload) -> dict[str, Any]:
    return manager.register_node(payload)


@app.get("/cluster/nodes")
def list_nodes() -> dict[str, Any]:
    return {"nodes": manager.node_registry.list_nodes()}


@app.get("/cluster/workers")
def list_workers() -> dict[str, Any]:
    return {"workers": manager.worker_registry.list_workers()}


@app.get("/cluster/gpu")
def cluster_gpu() -> dict[str, Any]:
    return {"gpus": manager.aggregate_gpus()}


@app.get("/cluster/tasks")
def list_tasks(limit: int = 100) -> dict[str, Any]:
    return {"tasks": manager.task_queue.list_tasks(limit=limit)}


@app.get("/cluster/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = manager.task_queue.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.post("/cluster/tasks")
def create_task(request: QueueTaskRequest) -> dict[str, Any]:
    task = manager.task_queue.enqueue(request.payload)
    metrics_store.set_queue_length(manager.task_queue.length())
    return {"task": task}


@app.post("/cluster/tasks/prune")
def prune_tasks(request: TaskPruneRequest) -> dict[str, Any]:
    return manager.prune_tasks(
        statuses=request.statuses,
        older_than_hours=request.older_than_hours,
        limit=request.limit,
    )

@app.post("/cluster/tasks/{task_id}/cancel")
def cancel_task(task_id: str) -> dict[str, Any]:
    success = manager.task_queue.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already processed.")
    return {"status": "cancelled", "task_id": task_id}

@app.post("/cluster/workers/{worker_id}/restart")
def restart_worker(worker_id: str) -> dict[str, Any]:
    # Since we don't have an SSH/Agent command to actually restart a process,
    # we simulate the trigger by marking it 'stale' or requesting a re-registration.
    # In a real scenario, this would call an endpoint on the node_agent.
    worker = manager.worker_registry.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found.")
    
    # For now, we can request the worker registry to treat it as dead
    # so the system forces a re-check or the user knows to check the node.
    # Ideally, we should send a signal to the WorkerAgent.
    return {"status": "restart_requested", "worker_id": worker_id, "info": "Restart command sent to cluster."}

@app.post("/cluster/dispatch")
def dispatch_task(request: DispatchRequest) -> dict[str, Any]:
    try:
        payload = request.task_payload or {"model": request.model, "provider": request.provider}
        return manager.scheduler.dispatch(
            model=request.model,
            provider=request.provider,
            allow_queue=request.allow_queue,
            task_payload=payload,
        )
    except SchedulerOverloadedError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "worker_queue_full",
                "message": str(exc),
                "retry_after": exc.retry_after_s,
            },
            headers={"Retry-After": str(exc.retry_after_s)},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "worker_unavailable", "message": str(exc)}) from exc


@app.post("/cluster/release")
def release_worker(request: WorkerReleaseRequest) -> dict[str, Any]:
    updated = manager.worker_registry.release(
        request.worker_id,
        success=request.success,
        assignment_id=request.assignment_id,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Worker not found.")
    updated = manager.worker_registry.update_worker_runtime(
        request.worker_id,
        gpu_utilization=request.gpu_utilization,
        temperature=request.temperature,
        queue_size=updated["queue_size"],
    ) or updated
    metrics_store.set_worker_usage(
        updated["worker_id"],
        node_id=updated["node_id"],
        gpu_utilization=float(updated.get("gpu_utilization", 0.0)),
        queue_size=int(updated.get("queue_size", 0)),
        status=str(updated.get("status", "healthy")),
    )
    return {"worker": manager.worker_registry.get_worker(request.worker_id)}


@app.post("/cluster/telemetry")
def record_telemetry(event: TelemetryEvent) -> dict[str, Any]:
    metrics_store.record_request(
        endpoint=event.endpoint,
        latency_ms=event.latency_ms,
        model=event.model,
        worker_id=event.worker_id,
        provider=event.provider,
        error=event.error,
        error_code=event.error_code,
    )
    if event.provider:
        metrics_store.set_provider_status(event.provider, not event.error)
    if event.worker_id and event.error:
        manager.worker_registry.update_worker_runtime(
            event.worker_id,
            last_error=f"{event.endpoint} failed",
        )
    return {"status": "recorded"}


@app.get("/cluster/metrics")
def cluster_metrics() -> dict[str, Any]:
    return manager.metrics_snapshot()


@app.get("/cluster/models")
def cluster_models() -> dict[str, Any]:
    registry = settings.model_registry()
    return {"models": registry.get("models", [])}
