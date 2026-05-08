from __future__ import annotations

from typing import Any

import requests
from fastapi import FastAPI

from config.settings import settings


app = FastAPI(title="AetherMesh Worker Agent", version="4.0.0")


def discover_local_workers() -> list[dict[str, Any]]:
    workers = []
    for port in settings.local_worker_ports:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=2)
            if response.ok:
                data = response.json()
                workers.append(
                    {
                        "port": port,
                        "status": "healthy",
                        "models": [item.get("name") for item in data.get("models", [])],
                    }
                )
                continue
        except requests.RequestException:
            pass
        workers.append({"port": port, "status": "down", "models": []})
    return workers


@app.get("/health")
def health() -> dict[str, Any]:
    workers = discover_local_workers()
    healthy = sum(1 for worker in workers if worker["status"] == "healthy")
    return {"status": "ok", "service": "worker-agent", "healthy_workers": healthy, "workers": workers}


@app.get("/workers")
def list_workers() -> dict[str, Any]:
    return {"workers": discover_local_workers()}


@app.get("/rpc/ping")
def ping() -> dict[str, str]:
    return {"status": "pong"}
