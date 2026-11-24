from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.providers_admin_router as providers_admin_module


@pytest.fixture
def client(monkeypatch):
    reload_mock = MagicMock(return_value={
        "UM890-Strata": {"api_type": "openai", "base_url": "http://192.168.1.201:47832/v1", "api_key": "sk-secret"},
        "M4-Strata": {"api_type": "openai", "base_url": "http://192.168.1.100:47832/v1", "api_key": "sk-other"},
    })
    monkeypatch.setattr(providers_admin_module, "reload_custom_providers", reload_mock)
    app = FastAPI()
    app.include_router(providers_admin_module.router)
    return TestClient(app), reload_mock


def test_reload_returns_ok_and_provider_names(client):
    c, reload_mock = client
    resp = c.post("/api/providers/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["providers"] == ["UM890-Strata", "M4-Strata"]
    reload_mock.assert_called_once_with()


def test_reload_response_does_not_leak_api_keys(client):
    c, _ = client
    resp = c.post("/api/providers/reload")
    assert "sk-secret" not in resp.text
    assert "sk-other" not in resp.text


def test_reload_invalidates_cached_credential_pools(monkeypatch):
    import runtime.orchestration.provider_router as provider_router
    import runtime.orchestration.routing_engine as routing_engine

    class FakePool:
        pass

    monkeypatch.setattr(provider_router, "_load_custom_providers", lambda: {
        "P": {"api_type": "openai", "base_url": "http://p:1/v1", "api_key": "sk-p"},
    })
    monkeypatch.setattr(routing_engine, "register_custom_providers", lambda *a, **k: None)
    monkeypatch.setattr(routing_engine, "unregister_custom_providers", lambda *a, **k: None)
    pools = {"P": FakePool(), "other": FakePool()}
    monkeypatch.setattr(provider_router, "_credential_pools", pools)
    data = provider_router.reload_custom_providers()
    assert "P" not in pools
    assert "other" in pools
    assert data["P"]["api_key"] == "sk-p"
