from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import requests
from pytest import MonkeyPatch

from providers import rerank_adapter
from providers.rerank_adapter import RerankAdapter

import dashboard.dashboard_server as dashboard_server


def _fake_response(ok: bool = True, status_code: int = 200, data: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock()
    response.ok = ok
    response.status_code = status_code
    response.text = ""
    response.encoding = "utf-8"
    response.json.return_value = data or {}
    return response


class _StubSession:
    def __init__(self, health: dict[str, Any]) -> None:
        self._health = health
        self.gets: list[str] = []

    def get(self, url: str, timeout: Any = None):
        self.gets.append(url)
        behaviour = self._health.get(url.rsplit("/health", 1)[0])
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour

    def post(self, url: str, json: dict | None = None, timeout: Any = None):
        behaviour = self._health.get(url.rsplit("/rerank", 1)[0])
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


def _reset_pool(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_adapter, "_POOL_STATUS", {})
    monkeypatch.setattr(rerank_adapter, "_LAST_USED", {})


def test_pool_status_reports_health_and_cooldown(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    dead = "http://down:11436"
    live = "http://live:11436"
    session = _StubSession(
        {
            dead: requests.ConnectionError("connection refused"),
            live: _fake_response(ok=True, data={"status": "ok"}),
        }
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    adapter = RerankAdapter(endpoints=[dead, live])
    adapter._mark_dead(dead)

    entries = adapter.pool_status()

    assert [entry["base_url"] for entry in entries] == [dead, live]
    assert entries[0]["ok"] is False
    assert entries[0]["failures"] == 1
    assert entries[0]["in_cooldown"] is True
    assert entries[0]["cooldown_s"] > 0
    assert entries[1]["ok"] is True
    assert entries[1]["failures"] == 0
    assert entries[1]["in_cooldown"] is False
    assert all("last_used" in entry for entry in entries)


def test_pool_status_records_last_successful_endpoint(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    live = "http://live:11436"
    session = _StubSession({live: _fake_response(ok=True, data={"results": [{"index": 0, "relevance_score": 1.2}]})})
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    adapter = RerankAdapter(endpoints=[live])
    result = adapter.rerank({"query": "q", "documents": ["a", "b"]})

    assert result["rerank_via"] == live
    entries = adapter.pool_status()
    assert entries[0]["last_used"] is not None


def test_health_check_contract_is_preserved(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(
        {
            "http://down:11436": requests.ConnectionError("refused"),
            "http://live:11436": _fake_response(ok=True),
        }
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    health = RerankAdapter(endpoints=["http://down:11436", "http://live:11436"]).health_check()

    assert health["ok"] is True
    assert health["status_code"] == 200
    assert health["base_url"] == "http://live:11436"
    assert [entry["base_url"] for entry in health["endpoints"]] == ["http://down:11436", "http://live:11436"]


def _workers() -> list[dict[str, Any]]:
    return [
        {
            "worker_id": "node-01:11436",
            "node_id": "node-01",
            "base_url": "http://192.168.1.210:11436",
            "port": 11436,
            "provider": "llama-server",
            "gpu_name": "Tesla P40",
            "gpu_memory": 23040,
            "status": "healthy",
            "metadata": {"kind": "rerank", "engine": "llama-server"},
        },
        {
            "worker_id": "node-mac-01:11436",
            "node_id": "node-mac-01",
            "base_url": "http://192.168.1.100:11436",
            "port": 11436,
            "provider": "llama-server",
            "gpu_name": "Apple Silicon GPU (MPS)",
            "gpu_memory": 65536,
            "status": "healthy",
            "metadata": {"kind": "rerank", "engine": "llama-server"},
        },
    ]


def test_rerank_pool_status_attributes_endpoints_to_nodes(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    monkeypatch.setenv("AIIH_RERANK_POOL", "http://127.0.0.1:11436,http://192.168.1.100:11436")
    session = _StubSession(
        {
            "http://127.0.0.1:11436": _fake_response(ok=True),
            "http://192.168.1.100:11436": requests.ConnectionError("timeout"),
        }
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    pool = dashboard_server._rerank_pool_status(_workers())

    assert len(pool) == 2
    local = pool[0]
    assert local["node_id"] == "node-01"
    assert local["gpu_name"] == "Tesla P40"
    assert local["ok"] is True

    mac = pool[1]
    assert mac["node_id"] == "node-mac-01"
    assert mac["ok"] is False
    assert pool[0]["priority"] == 0
    assert pool[1]["priority"] == 1


def test_cluster_nodes_get_rerank_and_asr_columns() -> None:
    nodes = [
        {
            "node_id": "node-01",
            "ip": "192.168.1.210",
            "status": "healthy",
            "gpus": [{"name": "Tesla P40"}],
            "workers": [11434, 11436],
            "metadata": {
                "rerank_workers": [{"port": 11436, "healthy": True, "engine": "llama-server", "gpu_id": 1}],
                "asr_worker": {"port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"},
            },
        },
        {
            "node_id": "node-legacy",
            "ip": "192.168.1.50",
            "status": "healthy",
            "gpus": [],
            "workers": [11434],
            "metadata": {},
        },
    ]
    pool = [
        {"base_url": "http://127.0.0.1:11436", "ok": True, "node_id": "node-01", "priority": 0},
        {"base_url": "http://192.168.1.50:11436", "ok": False, "node_id": "node-legacy", "priority": 1},
    ]

    enriched = dashboard_server._enrich_cluster_nodes(nodes, pool)

    assert enriched[0]["rerank"] == {"healthy": True, "endpoints": ["11436"], "engines": ["llama-server"]}
    assert enriched[0]["asr"] == {"healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}
    assert enriched[1]["rerank"] == {"healthy": False, "endpoints": ["11436"], "engines": []}
    assert enriched[1]["asr"] == {"healthy": False, "engine": None, "model": None}


def test_alerts_report_rerank_pool_failures() -> None:
    alerts = dashboard_server._build_alerts(
        nodes=[{"node_id": "node-01", "status": "healthy"}],
        workers=[{"worker_id": "node-01:11436", "status": "healthy", "metadata": {"kind": "rerank"}}],
        metrics={},
        rerank_pool=[
            {"base_url": "http://127.0.0.1:11436", "ok": True, "node_id": "node-01"},
            {"base_url": "http://192.168.1.100:11436", "ok": False, "node_id": "node-mac-01", "cooldown_s": 30.0},
        ],
    )
    messages = [alert["message"] for alert in alerts]
    assert any("1 rerank endpoint" in message for message in messages)

    all_down = dashboard_server._build_alerts(
        nodes=[],
        workers=[],
        metrics={},
        rerank_pool=[
            {"base_url": "http://127.0.0.1:11436", "ok": False, "node_id": "node-01"},
            {"base_url": "http://192.168.1.100:11436", "ok": False, "node_id": "node-mac-01"},
        ],
    )
    levels = {alert["level"] for alert in all_down}
    assert "critical" in levels
    assert any("rerank" in alert["message"].lower() for alert in all_down)


def test_overview_exposes_rerank_pool_and_enriched_nodes(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    # Settings instances are read-only, so patch the class to keep the configured
    # pool deterministic in this test.
    monkeypatch.setattr(type(dashboard_server.settings), "rerank_pool", lambda self: [])
    workers = _workers()
    nodes = [
        {
            "node_id": "node-01",
            "ip": "192.168.1.210",
            "status": "healthy",
            "metadata": {"rerank_workers": [{"port": 11436, "healthy": True, "engine": "llama-server"}]},
        }
    ]

    def fake_fetch_json(path: str) -> dict[str, Any]:
        if path == "/cluster/nodes":
            return {"nodes": nodes}
        if path == "/cluster/workers":
            return {"workers": workers}
        if path == "/cluster/gpu":
            return {"gpus": []}
        if path == "/cluster/tasks":
            return {"tasks": []}
        if path == "/cluster/models":
            return {"models": []}
        return {}

    monkeypatch.setattr(dashboard_server, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(dashboard_server, "_fetch_router_metrics", lambda endpoint: {})
    session = _StubSession(
        {
            "http://127.0.0.1:11436": _fake_response(ok=True),
            "http://192.168.1.100:11436": requests.ConnectionError("timeout"),
        }
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    overview = dashboard_server._build_overview()

    assert "rerank_pool" in overview
    assert [entry["base_url"] for entry in overview["rerank_pool"]] == ["http://127.0.0.1:11436", "http://192.168.1.100:11436"]
    assert overview["rerank_summary"] == {"endpoints": 2, "healthy": 1, "active": "http://127.0.0.1:11436"}
    assert overview["nodes"][0]["rerank"]["healthy"] is True
    assert overview["nodes"][0]["asr"]["engine"] is None


def test_dashboard_templates_and_js_render_rerank_pool() -> None:
    repo = Path(dashboard_server.BASE_DIR).parent
    template = (repo / "dashboard/templates/index.html").read_text(encoding="utf-8")
    script = (repo / "dashboard/static/dashboard.js").read_text(encoding="utf-8")

    assert 'id="rerank-table"' in template
    assert "Rerank Pool" in template
    assert "<th>Rerank</th><th>ASR</th>" in template

    assert "rerankTable" in script
    assert "data.rerank_pool" in script
    assert "cooldown_s" in script
