from __future__ import annotations

from typing import Any

import pytest
from pytest import MonkeyPatch

from control_plane.worker_registry import WorkerRegistry
from node import node_agent
from providers import rerank_adapter
from providers.rerank_adapter import RerankAdapter


class _FakeResponse:
    def __init__(self, ok: bool, payload: dict[str, Any] | None = None, status_code: int = 200) -> None:
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _fake_get(ollama_ports: set[int], rerank_ports: set[int]):
    def fake_get(url: str, timeout: float | None = None) -> _FakeResponse:
        port = int(url.split("://")[1].split("/")[0].split(":")[1])
        if url.endswith("/api/tags"):
            return _FakeResponse(port in ollama_ports, {"models": []}, 200 if port in ollama_ports else 404)
        if url.endswith("/health"):
            return _FakeResponse(port in rerank_ports, {"status": "ok"}, 200 if port in rerank_ports else 404)
        return _FakeResponse(False, {}, 404)

    return fake_get


def _patch_rerank_workers(monkeypatch: MonkeyPatch, entries: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(
        type(node_agent.settings),
        "cluster_config",
        lambda self: {"local_workers": entries},
    )


def test_node_agent_reports_llama_rerank_workers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(node_agent.requests, "get", _fake_get({11434}, {11436}))
    _patch_rerank_workers(monkeypatch, [{"port": 11436, "gpu_id": 1, "role": "rerank"}])

    agent = node_agent.NodeAgent()
    workers = agent.discover_workers()
    rerank_workers = agent.discover_rerank_workers(workers)

    assert workers == [11434]
    assert rerank_workers == [{"port": 11436, "healthy": True, "engine": "llama-server", "gpu_id": 1}]


def test_node_agent_reports_no_rerank_workers_when_none_run(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(node_agent.requests, "get", _fake_get({11434}, set()))
    _patch_rerank_workers(monkeypatch, [{"port": 11436, "gpu_id": 1, "role": "rerank"}])

    agent = node_agent.NodeAgent()
    assert agent.discover_rerank_workers(agent.discover_workers()) == []


def test_rerank_workers_without_cluster_config_use_the_default_port(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(node_agent.requests, "get", _fake_get({11434}, {11436}))
    _patch_rerank_workers(monkeypatch, [])

    agent = node_agent.NodeAgent()
    assert agent.discover_rerank_workers(agent.discover_workers()) == [{"port": 11436, "healthy": True, "engine": "llama-server", "gpu_id": 0}]


def test_rerank_workers_are_not_reported_twice_as_ollama_workers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(node_agent.requests, "get", _fake_get({11436}, {11436}))
    _patch_rerank_workers(monkeypatch, [{"port": 11436, "gpu_id": 0, "role": "rerank"}])

    agent = node_agent.NodeAgent()
    assert agent.discover_rerank_workers(agent.discover_workers()) == []


def test_payload_metadata_carries_rerank_workers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(node_agent.requests, "get", _fake_get({11434}, {11436}))
    monkeypatch.setattr(node_agent, "discover_gpus", lambda: [])
    _patch_rerank_workers(monkeypatch, [{"port": 11436, "gpu_id": 1, "role": "rerank"}])

    payload = node_agent.NodeAgent().payload()

    assert payload["workers"] == [11434]
    assert payload["metadata"]["rerank_workers"] == [{"port": 11436, "healthy": True, "engine": "llama-server", "gpu_id": 1}]


def test_registry_registers_rerank_workers_as_llama_server() -> None:
    registry = WorkerRegistry()
    registry.register_node_workers(
        node_id="node-01",
        host="192.168.1.210",
        gpus=[{"id": 0, "name": "Tesla P40", "memory": 23040}, {"id": 1, "name": "Tesla P40", "memory": 23040}],
        workers=[11434],
        model_assignments={},
        node_metadata={"rerank_workers": [{"port": 11436, "healthy": True, "engine": "llama-server", "gpu_id": 1}]},
    )

    worker = registry.get_worker("node-01:11436")

    assert worker is not None
    assert worker["provider"] == "llama-server"
    assert worker["base_url"] == "http://192.168.1.210:11436"
    assert worker["gpu_id"] == 1
    assert worker["metadata"]["kind"] == "rerank"


def test_registry_skips_unhealthy_rerank_workers() -> None:
    registry = WorkerRegistry()
    registry.register_node_workers(
        node_id="node-01",
        host="192.168.1.210",
        gpus=[{"id": 0, "name": "Tesla P40", "memory": 23040}],
        workers=[11434],
        model_assignments={},
        node_metadata={"rerank_workers": [{"port": 11436, "healthy": False, "engine": "llama-server"}]},
    )

    assert registry.get_worker("node-01:11436") is None


def _reset_registry_cache(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_adapter, "_REGISTRY_CACHE", {"endpoints": [], "fetched_at": 0.0})


def test_registry_endpoints_prefer_local_loopback(monkeypatch: MonkeyPatch) -> None:
    workers = [
        {
            "worker_id": "node-mac-01:11436",
            "node_id": "node-mac-01",
            "port": 11436,
            "base_url": "http://192.168.1.100:11436",
            "provider": "llama-server",
            "status": "healthy",
            "gpu_memory": 64000,
            "metadata": {"kind": "rerank"},
        },
        {
            "worker_id": "node-01:11436",
            "node_id": "node-01",
            "port": 11436,
            "base_url": "http://192.168.1.210:11436",
            "provider": "llama-server",
            "status": "healthy",
            "gpu_memory": 23040,
            "metadata": {"kind": "rerank"},
        },
    ]
    monkeypatch.setattr(rerank_adapter.requests, "get", lambda url, timeout=None: _FakeResponse(True, {"workers": workers}))
    monkeypatch.setattr(type(rerank_adapter.settings), "local_node_id", lambda self: "node-01")
    _reset_registry_cache(monkeypatch)

    assert rerank_adapter._registry_endpoints() == ["http://127.0.0.1:11436", "http://192.168.1.100:11436"]


def test_registry_endpoints_skip_dead_workers(monkeypatch: MonkeyPatch) -> None:
    workers = [
        {
            "worker_id": "node-01:11436",
            "node_id": "node-01",
            "port": 11436,
            "base_url": "http://127.0.0.1:11436",
            "provider": "llama-server",
            "status": "dead",
            "gpu_memory": 23040,
            "metadata": {"kind": "rerank"},
        },
        {
            "worker_id": "node-mac-01:11436",
            "node_id": "node-mac-01",
            "port": 11436,
            "base_url": "http://192.168.1.100:11436",
            "provider": "llama-server",
            "status": "healthy",
            "gpu_memory": 64000,
            "metadata": {"kind": "rerank"},
        },
    ]
    monkeypatch.setattr(rerank_adapter.requests, "get", lambda url, timeout=None: _FakeResponse(True, {"workers": workers}))
    _reset_registry_cache(monkeypatch)

    assert rerank_adapter._registry_endpoints() == ["http://192.168.1.100:11436"]


def test_rerank_adapter_prefers_registry_endpoints(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_adapter, "_registry_endpoints", lambda: ["http://192.168.1.100:11436", "http://127.0.0.1:11436"])
    monkeypatch.setattr(type(rerank_adapter.settings), "rerank_pool", lambda self: ["http://127.0.0.1:11436"])

    adapter = RerankAdapter()

    assert adapter.endpoints == ["http://192.168.1.100:11436", "http://127.0.0.1:11436"]


def test_rerank_adapter_falls_back_to_config_pool(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_adapter, "_registry_endpoints", lambda: [])
    monkeypatch.setattr(type(rerank_adapter.settings), "rerank_pool", lambda self: ["http://127.0.0.1:11436"])

    adapter = RerankAdapter()

    assert adapter.endpoints == ["http://127.0.0.1:11436"]


def test_explicit_env_pool_beats_registry(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AIIH_RERANK_POOL", "http://manual:11436")
    monkeypatch.setattr(rerank_adapter, "_registry_endpoints", lambda: ["http://registry:11436"])

    adapter = RerankAdapter()

    assert adapter.endpoints == ["http://manual:11436"]
