from __future__ import annotations

import logging
import uuid
from typing import Any

from cluster.load_balancer import choose_best_worker
from metrics.metrics import metrics_store
from ai_queue.redis_queue import RedisTaskQueue

from .worker_registry import WorkerRegistry

LOGGER = logging.getLogger("aiih.scheduler")


class SchedulerOverloadedError(RuntimeError):
    def __init__(self, message: str, *, retry_after_s: int = 3) -> None:
        super().__init__(message)
        self.retry_after_s = max(1, int(retry_after_s))


class Scheduler:
    def __init__(
        self,
        worker_registry: WorkerRegistry,
        task_queue: RedisTaskQueue,
        *,
        max_worker_queue_size: int,
    ) -> None:
        self.worker_registry = worker_registry
        self.task_queue = task_queue
        self.max_worker_queue_size = max_worker_queue_size

    def dispatch(
        self,
        *,
        model: str,
        provider: str,
        allow_queue: bool = False,
        task_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidates = self.worker_registry.available_workers(model=model, provider=provider)
        worker = choose_best_worker(
            candidates,
            max_queue_size=self.max_worker_queue_size,
            model=model,
        )
        if worker is not None:
            assignment_id = str(uuid.uuid4())
            self.worker_registry.acquire(worker["worker_id"])
            updated = self.worker_registry.get_worker(worker["worker_id"]) or worker
            metrics_store.set_worker_usage(
                updated["worker_id"],
                node_id=updated["node_id"],
                gpu_utilization=float(updated.get("gpu_utilization", 0.0)),
                queue_size=int(updated.get("queue_size", 0)),
                status=str(updated.get("status", "unknown")),
            )
            return {
                "status": "assigned",
                "assignment_id": assignment_id,
                "worker": updated,
            }

        if allow_queue and task_payload is not None:
            task = self.task_queue.enqueue(task_payload)
            metrics_store.set_queue_length(self.task_queue.length())
            return {"status": "queued", "task": task}

        if candidates and self.max_worker_queue_size > 0:
            raise SchedulerOverloadedError(
                "All matching workers are at queue capacity.",
                retry_after_s=3,
            )
        raise RuntimeError("No healthy worker available for the requested model.")

    def requeue_pending_tasks(self, max_retries: int = 3) -> int:
        """
        Requeue pending tasks from dead workers.
        Returns number of tasks requeued.
        """
        dead_workers = self.worker_registry.get_dead_workers()
        if not dead_workers:
            return 0

        requeued = 0
        for dead in dead_workers:
            try:
                # Get pending task from Redis for this worker
                # The pending task would be identified by worker_id in metadata
                # For now, we just log - actual requeue happens via Redis
                LOGGER.info(f"Worker {dead['worker_id']} is dead, checking for pending tasks to requeue")
            except Exception as e:
                LOGGER.error(f"Error requeueing tasks from {dead.get('worker_id')}: {e}")

        return requeued

    def multi_layer_fallback(
        self,
        *,
        model: str,
        provider: str,
        allow_queue: bool = False,
        task_payload: dict[str, Any] | None = None,
        fallback_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Multi-layer fallback: try model -> fallback model -> queue -> error.
        """
        # Layer 1: Original model
        try:
            return self.dispatch(
                model=model,
                provider=provider,
                allow_queue=allow_queue,
                task_payload=task_payload,
            )
        except (SchedulerOverloadedError, RuntimeError) as exc:
            # Layer 2+3: Try fallback models
            if fallback_models:
                for fallback_model in fallback_models:
                    try:
                        LOGGER.info(f"Falling back from {model} to {fallback_model}")
                        return self.dispatch(
                            model=fallback_model,
                            provider=provider,
                            allow_queue=allow_queue,
                            task_payload=task_payload,
                        )
                    except (SchedulerOverloadedError, RuntimeError):
                        continue

            # Layer 4: Queue if allowed
            if allow_queue and task_payload is not None:
                task = self.task_queue.enqueue(task_payload)
                return {"status": "queued", "task": task, "fallback_attempted": fallback_models}

            # Layer 5: Error
            raise exc
