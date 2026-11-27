from __future__ import annotations

import time
from typing import Any, Callable

from unittest.mock import MagicMock

import requests
from pytest import MonkeyPatch

from config.settings import Settings
from providers import rerank_adapter
from providers.base import ProviderError
from providers.rerank_adapter import RerankAdapter


def _fake_response(data: dict, ok: bool = True, status_code: int = 200, text: str = ""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = data
    return resp


class _StubSession:
    """Records every POST url so tests can assert which endpoints were tried."""

    def __init__(self, handler: Callable[[str], Any]) -> None:
        self._handler = handler
        self.posts: list[str] = []

    def post(self, url: str, json: dict | None = None, timeout: Any = None):
        self.posts.append(url)
        return self._handler(url)

    def get(self, url: str, timeout: Any = None):
        self.posts.append(url)
        return self._handler(url)


def _reset_pool(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rerank_adapter, "_POOL_STATUS", {})


def _fail(message: str) -> Any:
    raise requests.ConnectionError(message)


def _ok_result(index: int = 0, score: float = 1.0):
    return _fake_response({"results": [{"index": index, "relevance_score": score}]})


def test_rerank_pool_uses_env_order(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AIIH_RERANK_POOL", "http://mac:11436, http://um890:11436")
    assert Settings().rerank_pool() == ["http://mac:11436", "http://um890:11436"]


def test_rerank_pool_resolves_bindings_and_keeps_local_on_loopback(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("AIIH_RERANK_POOL", raising=False)
    models = {
        "models": [
            {
                "name": "rerank/bge-reranker-v2-m3",
                "provider": "llama-server",
                "capabilities": ["chat", "rerank"],
                "worker_bindings": [
                    {"node_id": "node-01", "port": 11436},
                    {"node_id": "node-mac-01", "port": 11436},
                ],
            }
        ]
    }
    monkeypatch.setattr(Settings, "model_registry", lambda self: models)
    monkeypatch.setattr(Settings, "node_hosts", lambda self: {"node-01": "192.168.1.210", "node-mac-01": "192.168.1.100"})
    monkeypatch.setattr(Settings, "local_node_id", lambda self: "node-01")

    assert Settings().rerank_pool() == ["http://127.0.0.1:11436", "http://192.168.1.100:11436"]


def test_rerank_pool_falls_back_to_single_default(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("AIIH_RERANK_POOL", raising=False)
    monkeypatch.setattr(Settings, "model_registry", lambda self: {})
    assert Settings().rerank_pool() == ["http://127.0.0.1:11436"]


def test_rerank_failover_uses_next_endpoint_when_first_is_down(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(
        lambda url: _ok_result() if "live" in url else _fail("connection refused")
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    adapter = RerankAdapter(endpoints=["http://dead:11436", "http://live:11436"])
    result = adapter.rerank({"query": "q", "documents": ["a", "b"]})

    assert session.posts == ["http://dead:11436/rerank", "http://live:11436/rerank"]
    assert result["data"][0]["index"] == 0


def test_rerank_failover_on_server_error_response(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(
        lambda url: _fake_response({}, ok=False, status_code=500, text="boom") if "dead" in url else _ok_result()
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    adapter = RerankAdapter(endpoints=["http://dead:11436", "http://live:11436"])
    result = adapter.rerank({"query": "q", "documents": ["a"]})

    assert session.posts == ["http://dead:11436/rerank", "http://live:11436/rerank"]
    assert result["data"][0]["relevance_score"] == 1.0


def test_rerank_skips_endpoint_in_cooldown_on_next_request(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(
        lambda url: _ok_result() if "live" in url else _fail("connection refused")
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    endpoints = ["http://dead:11436", "http://live:11436"]
    RerankAdapter(endpoints=endpoints).rerank({"query": "q", "documents": ["a"]})
    session.posts.clear()

    RerankAdapter(endpoints=endpoints).rerank({"query": "q", "documents": ["a"]})

    assert session.posts == ["http://live:11436/rerank"]


def test_rerank_retries_endpoint_after_cooldown(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    monkeypatch.setattr(rerank_adapter, "COOLDOWN_S", 0.0)
    session = _StubSession(
        lambda url: _ok_result() if "live" in url else _fail("connection refused")
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    endpoints = ["http://dead:11436", "http://live:11436"]
    RerankAdapter(endpoints=endpoints).rerank({"query": "q", "documents": ["a"]})
    session.posts.clear()

    RerankAdapter(endpoints=endpoints).rerank({"query": "q", "documents": ["a"]})

    assert session.posts == ["http://dead:11436/rerank", "http://live:11436/rerank"]


def test_rerank_raises_last_error_when_all_endpoints_fail(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(lambda url: _fail("all nodes down"))
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    adapter = RerankAdapter(endpoints=["http://a:11436", "http://b:11436"])
    try:
        adapter.rerank({"query": "q", "documents": ["a"]})
    except ProviderError as exc:
        assert "all nodes down" in str(exc)
        assert exc.code == "provider_unreachable"
    else:
        raise AssertionError("expected ProviderError")


def test_rerank_health_check_reports_every_endpoint(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(
        lambda url: _fake_response({}, ok=False, status_code=503) if "dead" in url else _fake_response({}, ok=True)
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    result = RerankAdapter(endpoints=["http://dead:11436", "http://live:11436"]).health_check()

    assert result["ok"] is True
    assert [entry["ok"] for entry in result["endpoints"]] == [False, True]
    assert result["base_url"] == "http://live:11436"


def test_pool_summary_reports_totals_and_active_endpoint(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(
        lambda url: _fake_response({}, ok=False, status_code=503) if "dead" in url else _fake_response({}, ok=True)
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    summary = RerankAdapter(endpoints=["http://dead:11436", "http://live:11436"]).pool_summary()

    assert summary["total"] == 2
    assert summary["healthy"] == 1
    assert summary["active"] == "http://live:11436"
    assert [entry["priority"] if "priority" in entry else None for entry in summary["endpoints"]] == [None, None]


def test_pool_summary_excludes_endpoint_in_cooldown_from_active(monkeypatch: MonkeyPatch) -> None:
    _reset_pool(monkeypatch)
    session = _StubSession(lambda url: _fake_response({}, ok=True))
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    adapter = RerankAdapter(endpoints=["http://a:11436", "http://b:11436"])
    rerank_adapter._POOL_STATUS["http://a:11436"] = {"failures": 1, "dead_until": time.monotonic() + 20}

    summary = adapter.pool_summary()

    assert summary["healthy"] == 2
    assert summary["active"] == "http://b:11436"
    first = summary["endpoints"][0]
    assert first["in_cooldown"] is True
    assert first["failures"] == 1
    assert first["cooldown_s"] > 0


def test_rerank_pool_route_exposes_summary(monkeypatch: MonkeyPatch) -> None:
    from router.openai.rerank_adapter import create_rerank_pool_route

    _reset_pool(monkeypatch)
    monkeypatch.setenv("AIIH_RERANK_POOL", "http://a:11436,http://b:11436")
    session = _StubSession(
        lambda url: _fake_response({}, ok=False, status_code=503) if "a:11436" in url else _fake_response({}, ok=True)
    )
    monkeypatch.setattr(rerank_adapter, "get_session", lambda: session)

    summary = create_rerank_pool_route()()

    assert summary["total"] == 2
    assert summary["healthy"] == 1
    assert summary["active"] == "http://b:11436"
