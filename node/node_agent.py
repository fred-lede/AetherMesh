from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any

import requests
from fastapi import FastAPI

from cluster.gpu_discovery import discover_gpus
from config.settings import settings


app = FastAPI(title="AetherMesh Node Agent", version="4.0.0")


class NodeAgent:
    def __init__(self) -> None:
        self.node_id = settings.local_node_id()
        self._status: dict[str, Any] = {"last_register": None, "last_heartbeat": None, "last_error": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cache_lock = threading.RLock()
        self._detail_refresh_s = max(5, int(os.getenv("AIIH_DETAIL_REFRESH", "30")))
        self._cached_workers: list[int] = []
        self._cached_worker_runtime: dict[str, Any] = {}
        self._cached_gpus: list[dict[str, Any]] = []
        self._last_detail_refresh = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def local_ip(self) -> str:
        configured = settings.local_node_ip()
        if configured:
            return configured

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                ip = sock.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except OSError:
            pass

        try:
            hostname = socket.gethostname()
            for _, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = sockaddr[0]
                if ip and not ip.startswith("127."):
                    return ip
        except OSError:
            pass

        return "127.0.0.1"

    def discover_workers(self) -> list[int]:
        ports = []
        for port in settings.local_worker_ports:
            try:
                response = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=2)
                if response.ok:
                    ports.append(port)
            except requests.RequestException:
                continue
        return ports

    def discover_worker_runtime(self, ports: list[int]) -> dict[str, Any]:
        runtime: dict[str, Any] = {}
        for port in ports:
            entry: dict[str, Any] = {
                "port": port,
                "ps_count": 0,
                "ps_models": [],
                "processors": [],
                "error": None,
            }
            try:
                response = requests.get(f"http://127.0.0.1:{port}/api/ps", timeout=2)
                if response.ok:
                    payload = response.json()
                    models = payload.get("models", []) if isinstance(payload, dict) else []
                    ps_models: list[str] = []
                    processors: list[str] = []
                    for item in models:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name") or item.get("model") or "").strip()
                        if name:
                            ps_models.append(name)
                        processor = str(item.get("processor") or "").strip()
                        if not processor:
                            details = item.get("details")
                            if isinstance(details, dict):
                                processor = str(details.get("processor") or "").strip()

                        if not processor:
                            size_vram = item.get("size_vram")
                            try:
                                if size_vram is not None and float(size_vram) > 0:
                                    processor = "GPU"
                                else:
                                    processor = "CPU"
                            except (TypeError, ValueError):
                                processor = ""

                        if processor:
                            processors.append(processor)
                    entry["ps_count"] = len(models)
                    entry["ps_models"] = sorted(set(ps_models))
                    entry["processors"] = sorted(set(processors))
                else:
                    entry["error"] = f"/api/ps status={response.status_code}"
            except requests.RequestException as exc:
                entry["error"] = str(exc)
            runtime[str(port)] = entry
        return runtime

    def _get_heavy_telemetry(self, workers: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        now = time.time()
        with self._cache_lock:
            workers_changed = workers != self._cached_workers
            cache_expired = (now - self._last_detail_refresh) >= self._detail_refresh_s
            should_refresh = workers_changed or cache_expired or not self._cached_worker_runtime

            if should_refresh:
                try:
                    gpus = discover_gpus()
                except OSError:
                    gpus = self._cached_gpus
                runtime = self.discover_worker_runtime(workers)
                self._cached_workers = list(workers)
                self._cached_gpus = list(gpus)
                self._cached_worker_runtime = dict(runtime)
                self._last_detail_refresh = now

            return list(self._cached_gpus), dict(self._cached_worker_runtime)

    def payload(self) -> dict[str, Any]:
        workers = self.discover_workers()
        gpus, worker_runtime = self._get_heavy_telemetry(workers)

        return {
            "node_id": self.node_id,
            "ip": self.local_ip(),
            "gpus": gpus,
            "workers": workers,
            "metadata": {
                "rpc_port": settings.worker_rpc_port,
                "node_port": settings.node_agent_port,
                "worker_runtime": worker_runtime,
                "detail_refresh_s": self._detail_refresh_s,
                "detail_cached_at": self._last_detail_refresh,
            },
        }

    def register(self) -> dict[str, Any]:
        response = requests.post(f"{settings.control_plane_url}/cluster/register", json=self.payload(), timeout=10)
        response.raise_for_status()
        self._status["last_register"] = time.time()
        self._status["last_error"] = None
        return response.json()

    def heartbeat(self) -> dict[str, Any]:
        response = requests.post(f"{settings.control_plane_url}/cluster/heartbeat", json=self.payload(), timeout=10)
        response.raise_for_status()
        self._status["last_heartbeat"] = time.time()
        self._status["last_error"] = None
        return response.json()

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                if self._status["last_register"] is None:
                    self.register()
                else:
                    self.heartbeat()
            except Exception as exc:
                self._status["last_error"] = str(exc)
            self._stop.wait(settings.heartbeat_interval_s)

    @property
    def status(self) -> dict[str, Any]:
        snapshot = dict(self._status)
        snapshot["node_id"] = self.node_id
        snapshot["payload"] = self.payload()
        return snapshot


agent = NodeAgent()


@app.on_event("startup")
def startup_event() -> None:
    agent.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    agent.stop()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "node-agent", "node": agent.status}


@app.get("/status")
def status() -> dict[str, Any]:
    return agent.status


@app.post("/register-now")
def register_now() -> dict[str, Any]:
    return agent.register()
