from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def cfg_file(tmp_path, monkeypatch):
    path = tmp_path / "services.json"
    monkeypatch.setattr(
        "dashboard.dashboard_server._services_config_path", lambda: path
    )
    return path


@pytest.fixture()
def client(monkeypatch):
    from dashboard.dashboard_server import app

    monkeypatch.setattr(
        "dashboard.dashboard_server.settings.dashboard_auth_enabled", False
    )
    with patch("dashboard.dashboard_server._require_admin", lambda r: None):
        yield TestClient(app)


class TestServicesAPI:
    def test_get_defaults_all_enabled(self, client, cfg_file) -> None:
        resp = client.get("/api/services")
        assert resp.status_code == 200
        services = resp.json()["services"]
        names = {s["name"] for s in services}
        assert "openai_router" in names
        assert all(s["enabled"] for s in services)

    def test_put_disable_then_get(self, client, cfg_file) -> None:
        resp = client.put("/api/services/openai_router", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "name": "openai_router", "enabled": False}
        saved = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert saved["openai_router"]["enabled"] is False
        listing = client.get("/api/services").json()["services"]
        entry = next(s for s in listing if s["name"] == "openai_router")
        assert entry["enabled"] is False

    def test_put_preserves_other_entries(self, client, cfg_file) -> None:
        client.put("/api/services/dashboard", json={"enabled": False})
        client.put("/api/services/metrics", json={"enabled": False})
        saved = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert saved["dashboard"]["enabled"] is False
        assert saved["metrics"]["enabled"] is False

    def test_put_reenable_overwrites(self, client, cfg_file) -> None:
        client.put("/api/services/openai_router", json={"enabled": False})
        client.put("/api/services/openai_router", json={"enabled": True})
        saved = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert saved["openai_router"]["enabled"] is True

    def test_unknown_service_404(self, client, cfg_file) -> None:
        resp = client.put("/api/services/nope", json={"enabled": False})
        assert resp.status_code == 404

    def test_malformed_file_treated_as_empty(self, client, cfg_file) -> None:
        cfg_file.write_text("{broken", encoding="utf-8")
        resp = client.get("/api/services")
        assert resp.status_code == 200
        assert all(s["enabled"] for s in resp.json()["services"])
