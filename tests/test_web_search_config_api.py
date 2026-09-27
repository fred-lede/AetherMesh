from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard import dashboard_server as dash


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(dash, "_auth_credentials_configured", lambda: True)
    monkeypatch.setattr(dash, "_session_auth_valid", lambda request: True)
    monkeypatch.setattr(dash, "_require_admin", lambda request: None)
    return TestClient(dash.app)


def test_get_web_search_config(client: TestClient):
    resp = client.get("/api/web-search/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["order"][0] == "exa"
    assert {p["name"] for p in body["providers"]} == {"exa", "tavily", "serper", "duckduckgo"}


def test_put_web_search_config_saves(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    saved: dict = {}
    monkeypatch.setattr(type(dash.settings), "save_web_search_config", lambda self, data: saved.update(data))
    resp = client.put("/api/web-search/config", json={"order": ["tavily", "exa", "serper", "duckduckgo"]})
    assert resp.status_code == 200
    assert saved["provider_order"] == ["tavily", "exa", "serper", "duckduckgo"]


def test_put_web_search_config_rejects_unknown(client: TestClient):
    resp = client.put("/api/web-search/config", json={"order": ["bogus"]})
    assert resp.status_code == 400


def test_put_web_search_config_requires_admin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dash, "_auth_credentials_configured", lambda: True)
    monkeypatch.setattr(dash, "_session_auth_valid", lambda request: True)
    client = TestClient(dash.app)
    resp = client.put("/api/web-search/config", json={"order": ["exa"]})
    assert resp.status_code == 403
