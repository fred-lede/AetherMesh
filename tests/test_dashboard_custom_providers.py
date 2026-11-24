from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

import dashboard.dashboard_server as dashboard_server


@pytest.fixture
def notify_mock(monkeypatch):
    mock = MagicMock(return_value=["http://127.0.0.1:8001", "http://127.0.0.1:8002"])
    monkeypatch.setattr(dashboard_server, "_notify_routers_custom_providers_reload", mock)
    return mock


class TestNotifyRoutersReload:
    def test_posts_reload_to_both_routers_with_api_key(self, monkeypatch):
        calls = []

        class FakeSession:
            def post(self, url, headers=None, timeout=None):
                calls.append({"url": url, "headers": headers, "timeout": timeout})

                class Resp:
                    ok = True

                    def json(self):
                        return {"ok": True, "providers": []}

                return Resp()

        monkeypatch.setattr(dashboard_server, "get_session", lambda: FakeSession())
        notified = dashboard_server._notify_routers_custom_providers_reload()
        urls = [c["url"] for c in calls]
        assert urls == [
            f"{dashboard_server.settings.router_url}/api/providers/reload",
            f"{dashboard_server.settings.anthropic_router_url}/api/providers/reload",
        ]
        assert calls[0]["headers"] == {"x-api-key": dashboard_server._router_api_key()}
        assert len(notified) == 2

    def test_router_down_does_not_raise(self, monkeypatch):
        fake = MagicMock()
        fake.post.side_effect = requests.ConnectionError("connection refused")
        monkeypatch.setattr(dashboard_server, "get_session", lambda: fake)
        assert dashboard_server._notify_routers_custom_providers_reload() == []

    def test_non_ok_response_not_counted(self, monkeypatch):
        class Resp:
            ok = False
            status_code = 503

        fake = MagicMock()
        fake.post.return_value = Resp()
        monkeypatch.setattr(dashboard_server, "get_session", lambda: fake)
        assert dashboard_server._notify_routers_custom_providers_reload() == []


class TestSaveEndpointsNotifyRouters:
    def _patch_storage(self, monkeypatch, data):
        saved = {}

        def save(self, d):
            saved.clear()
            saved.update(d)

        def load(self):
            return {k: dict(v) for k, v in data.items()}

        monkeypatch.setattr(type(dashboard_server.settings), "load_custom_providers", load)
        monkeypatch.setattr(type(dashboard_server.settings), "save_custom_providers", save)
        return saved

    def test_create_notifies_routers(self, monkeypatch, notify_mock):
        saved = self._patch_storage(monkeypatch, {})
        result = dashboard_server.custom_providers_create({
            "name": "NewProv",
            "base_url": "http://192.168.1.201:47832",
            "api_key": "sk-new",
        })
        assert result["ok"] is True
        assert saved["NewProv"]["api_key"] == "sk-new"
        assert saved["NewProv"]["base_url"].endswith("/v1")
        notify_mock.assert_called_once_with()

    def test_update_notifies_routers(self, monkeypatch, notify_mock):
        data = {"UM890-Strata": {"api_type": "openai", "base_url": "http://192.168.1.201:47832/v1", "api_key": "sk-old"}}
        saved = self._patch_storage(monkeypatch, data)
        result = dashboard_server.custom_providers_update("UM890-Strata", {"api_key": "sk-rotated"})
        assert result["ok"] is True
        assert saved["UM890-Strata"]["api_key"] == "sk-rotated"
        notify_mock.assert_called_once_with()

    def test_delete_notifies_routers(self, monkeypatch, notify_mock):
        data = {"UM890-Strata": {"api_type": "openai", "base_url": "http://192.168.1.201:47832/v1", "api_key": "sk-old"}}
        saved = self._patch_storage(monkeypatch, data)
        result = dashboard_server.custom_providers_delete("UM890-Strata")
        assert result["ok"] is True
        assert "UM890-Strata" not in saved
        notify_mock.assert_called_once_with()

    def test_reload_endpoint_notifies_routers(self, monkeypatch, notify_mock):
        fake_reloaded = {"UM890-Strata": {"api_type": "openai", "base_url": "http://x/v1", "api_key": "sk-x"}}
        monkeypatch.setattr(dashboard_server, "reload_custom_providers", MagicMock(return_value=fake_reloaded))
        result = dashboard_server.custom_providers_reload()
        assert result["ok"] is True
        notify_mock.assert_called_once_with()
