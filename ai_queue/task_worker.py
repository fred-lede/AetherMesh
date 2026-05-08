from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import requests

from config.settings import settings
from providers.base import ProviderError
from providers.gemini_adapter import GeminiAdapter
from providers.ollama_adapter import OllamaAdapter
from providers.openai_adapter import OpenAIAdapter
from providers.http_client import get_session
from ai_queue.redis_queue import RedisTaskQueue

LOGGER = logging.getLogger("aiih.task_worker")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def provider_for_model(model: str, requested: str | None = None) -> str:
    if requested:
        return requested.lower()
    registry = settings.model_registry()
    for item in registry.get("models", []):
        if item.get("name") == model:
            return str(item.get("provider", "ollama"))
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("gpt") or model.startswith("o"):
        return "openai"
    return "ollama"


class TaskWorker:
    def __init__(self, worker_id: str | None = None) -> None:
        self.worker_id = worker_id or _env_str("AIIH_TASK_WORKER_ID", "worker-1")
        self.queue = RedisTaskQueue(settings.redis_url)
        LOGGER.info(f"Task worker starting with ID: {self.worker_id}")

    def run_forever(self) -> None:
        while True:
            task = self.queue.reserve(block_timeout_s=5)
            if task is None:
                continue
            self._process_task(task)

    def _process_task(self, task: dict[str, Any]) -> None:
        payload = task.get("payload", {})
        model = payload.get("model")
        provider = provider_for_model(model or "", payload.get("provider"))
        worker = None
        started = time.perf_counter()
        error = None
        try:
            if provider == "ollama":
                dispatch = get_session().post(
                    f"{settings.control_plane_url}/cluster/dispatch",
                    json={"model": model, "provider": provider, "allow_queue": False, "task_payload": payload, "task_worker_id": self.worker_id},
                    timeout=10,
                )
                if not dispatch.ok:
                    raise ProviderError(dispatch.text)
                worker = dispatch.json()["worker"]
                adapter = OllamaAdapter(worker["base_url"])
            elif provider == "openai":
                adapter = OpenAIAdapter()
            else:
                adapter = GeminiAdapter()

            endpoint = payload.get("endpoint", "/v1/chat/completions")
            if endpoint == "/v1/embeddings":
                result = adapter.embeddings(payload)
            elif endpoint == "/v1/responses":
                result = adapter.responses(payload)
            elif endpoint == "/v1/rerank":
                result = adapter.rerank(payload)
            else:
                result = adapter.chat(payload)
            self.queue.ack(task["task_id"], result)
        except (ProviderError, requests.RequestException) as exc:
            error = str(exc)
            self.queue.fail(task["task_id"], error, max_retries=settings.max_task_retries)
        finally:
            if worker is not None:
                try:
                    get_session().post(
                        f"{settings.control_plane_url}/cluster/release",
                        json={"worker_id": worker["worker_id"], "success": error is None, "task_worker_id": self.worker_id},
                        timeout=5,
                    )
                except requests.RequestException:
                    pass
            try:
                get_session().post(
                    f"{settings.control_plane_url}/cluster/telemetry",
                    json={
                        "endpoint": payload.get("endpoint", "/cluster/tasks"),
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "model": model or "",
                        "provider": provider,
                        "error": error is not None,
                        "task_worker_id": self.worker_id,
                    },
                    timeout=5,
                )
            except requests.RequestException:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AetherMesh Task Worker")
    parser.add_argument("--worker-id", "-w", help="Worker ID (default: worker-1)")
    args = parser.parse_args()

    worker = TaskWorker(worker_id=args.worker_id)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        LOGGER.info(f"Task worker {worker.worker_id} shutting down")


if __name__ == "__main__":
    main()

