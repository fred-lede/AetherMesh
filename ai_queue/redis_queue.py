from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from typing import Any

import redis


class RedisTaskQueue:
    def __init__(self, redis_url: str, queue_name: str = "aiih:tasks") -> None:
        self.redis_url = redis_url
        self.queue_name = queue_name
        self._memory_queue: deque[str] = deque()
        self._memory_tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._redis = None
        self._connect()

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _connect(self) -> None:
        try:
            client = redis.from_url(self.redis_url, decode_responses=True)
            client.ping()
            self._redis = client
        except redis.RedisError:
            self._redis = None

    def enqueue(self, payload: dict[str, Any], retries: int = 0) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "status": "queued",
            "retries": retries,
            "created_at": time.time(),
            "updated_at": time.time(),
            "payload": payload,
            "result": None,
            "error": None,
        }
        serialized = json.dumps(task)
        if self._redis is not None:
            self._redis.set(self._task_key(task_id), serialized)
            self._redis.rpush(self.queue_name, task_id)
        else:
            with self._lock:
                self._memory_tasks[task_id] = task
                self._memory_queue.append(task_id)
        return task

    def reserve(self, block_timeout_s: int = 1) -> dict[str, Any] | None:
        if self._redis is not None:
            item = self._redis.blpop(self.queue_name, timeout=block_timeout_s)
            if not item:
                return None
            _, task_id = item
            task = self.get_task(task_id)
            if not task:
                return None
            self.update_status(task_id, "running")
            return self.get_task(task_id)
        with self._lock:
            if not self._memory_queue:
                return None
            task_id = self._memory_queue.popleft()
            self._memory_tasks[task_id]["status"] = "running"
            self._memory_tasks[task_id]["updated_at"] = time.time()
            return dict(self._memory_tasks[task_id])

    def update_status(self, task_id: str, status: str, **fields: Any) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if not task:
            return None
        task.update(fields)
        task["status"] = status
        task["updated_at"] = time.time()
        if self._redis is not None:
            self._redis.set(self._task_key(task_id), json.dumps(task))
        else:
            with self._lock:
                self._memory_tasks[task_id] = task
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Removes a task from the queue and storage."""
        if self._redis is not None:
            # Remove from queue list
            # Note: LREM requires the exact value. 
            # We only remove if it's still in the queue (status usually 'queued')
            self._redis.lrem(self.queue_name, 0, task_id)
            # Remove from storage
            self._redis.delete(self._task_key(task_id))
            return True
        else:
            with self._lock:
                if task_id in self._memory_tasks:
                    self._memory_tasks.pop(task_id)
                    try:
                        self._memory_queue.remove(task_id)
                    except ValueError:
                        pass
                    return True
                return False

    def ack(self, task_id: str, result: Any) -> dict[str, Any] | None:
        return self.update_status(task_id, "completed", result=result, error=None)

    def fail(self, task_id: str, error: str, max_retries: int) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if not task:
            return None
        retries = int(task.get("retries", 0))
        if retries < max_retries:
            task["retries"] = retries + 1
            task["error"] = error
            task["status"] = "queued"
            task["updated_at"] = time.time()
            if self._redis is not None:
                self._redis.set(self._task_key(task_id), json.dumps(task))
                self._redis.rpush(self.queue_name, task_id)
            else:
                with self._lock:
                    self._memory_tasks[task_id] = task
                    self._memory_queue.append(task_id)
            return task
        return self.update_status(task_id, "failed", error=error)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        if self._redis is not None:
            payload = self._redis.get(self._task_key(task_id))
            return json.loads(payload) if payload else None
        with self._lock:
            task = self._memory_tasks.get(task_id)
            return dict(task) if task else None

    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        if self._redis is not None:
            keys = sorted(self._redis.scan_iter(f"{self.queue_name}:task:*"))[:limit]
            items = []
            for key in keys:
                payload = self._redis.get(key)
                if payload:
                    items.append(json.loads(payload))
            return sorted(items, key=lambda item: item.get("created_at", 0), reverse=True)
        with self._lock:
            items = list(self._memory_tasks.values())
            return sorted(items, key=lambda item: item.get("created_at", 0), reverse=True)[:limit]

    def length(self) -> int:
        if self._redis is not None:
            return int(self._redis.llen(self.queue_name))
        with self._lock:
            return len(self._memory_queue)

    def prune_tasks(
        self,
        *,
        statuses: set[str] | None = None,
        older_than_s: float = 0,
        limit: int = 10000,
    ) -> dict[str, Any]:
        now = time.time()
        statuses_normalized = {str(s).strip().lower() for s in (statuses or set()) if str(s).strip()}
        deleted = 0
        scanned = 0

        def should_prune(task: dict[str, Any]) -> bool:
            status = str(task.get("status", "")).lower()
            if statuses_normalized and status not in statuses_normalized:
                return False
            ref_ts = float(task.get("updated_at") or task.get("created_at") or 0)
            if older_than_s > 0 and (now - ref_ts) < older_than_s:
                return False
            return True

        if self._redis is not None:
            for key in self._redis.scan_iter(f"{self.queue_name}:task:*"):
                if scanned >= limit:
                    break
                payload = self._redis.get(key)
                scanned += 1
                if not payload:
                    continue
                try:
                    task = json.loads(payload)
                except ValueError:
                    self._redis.delete(key)
                    deleted += 1
                    continue
                if should_prune(task):
                    self._redis.delete(key)
                    deleted += 1
        else:
            with self._lock:
                for task_id, task in list(self._memory_tasks.items()):
                    if scanned >= limit:
                        break
                    scanned += 1
                    if should_prune(task):
                        self._memory_tasks.pop(task_id, None)
                        deleted += 1

        return {"deleted": deleted, "scanned": scanned, "statuses": sorted(statuses_normalized)}

    def _task_key(self, task_id: str) -> str:
        return f"{self.queue_name}:task:{task_id}"
