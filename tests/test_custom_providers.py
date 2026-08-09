from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings

# Remove stale routing state from previous runs before it gets loaded
_stale_state = Settings().config_path("routing_state.yaml")
if _stale_state.exists():
    _stale_state.unlink()

from runtime.orchestration.provider_router import (
    OpenAIAdapter, _CUSTOM_PROVIDERS, _load_custom_providers,
    adapter, reload_custom_providers,
)
from runtime.orchestration.routing_engine import (
    CAPABILITY_PROVIDER_SCORES, CLOUD_PROVIDERS,
    CLOUD_PROVIDER_ENDPOINTS, ROUTING_PROVIDERS,
    register_custom_providers, routing_engine,
    unregister_custom_providers,
)


def _with_temp_custom_providers(data: dict) -> Path:
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(data), encoding="utf-8")
    return tmp


# ── Settings ─────────────────────────────────────────────────────────

class TestSettingsLoadCustomProviders:

    def test_returns_empty_when_file_missing(self):
        with patch("config.settings.Settings.config_path", return_value=Path("/nonexistent/file.json")):
            s = Settings()
            assert s.load_custom_providers() == {}

    def test_returns_empty_on_invalid_json(self):
        tmp = Path(tempfile.mktemp(suffix=".json"))
        tmp.write_text("not json", encoding="utf-8")
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            assert s.load_custom_providers() == {}

    def test_returns_parsed_data(self):
        data = {"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}
        tmp = _with_temp_custom_providers(data)
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            assert s.load_custom_providers() == data

    def test_returns_empty_when_not_dict(self):
        tmp = _with_temp_custom_providers(["not", "a", "dict"])
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            assert s.load_custom_providers() == {}


class TestSettingsSaveCustomProviders:

    def test_writes_valid_json(self):
        tmp = Path(tempfile.mktemp(suffix=".json"))
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            data = {"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}
            s.save_custom_providers(data)
            assert json.loads(tmp.read_text(encoding="utf-8")) == data


# ── Provider Router ──────────────────────────────────────────────────

class TestLoadCustomProviders:

    def test_filters_valid_entries(self):
        data = {
            "agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"},
            "bad": {"api_type": "openai", "base_url": "", "api_key": ""},
            "not_dict": "string",
        }
        tmp = _with_temp_custom_providers(data)
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            with patch("runtime.orchestration.provider_router.settings", s):
                result = _load_custom_providers()
                assert "agnes" in result
                assert result["agnes"]["base_url"] == "https://api.agnes.ai/v1"
                assert "bad" not in result
                assert "not_dict" not in result

    def test_filters_non_openai_types(self):
        data = {
            "custom": {"api_type": "anthropic", "base_url": "https://api.anthropic.com/v1", "api_key": "sk-test"},
        }
        tmp = _with_temp_custom_providers(data)
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            with patch("runtime.orchestration.provider_router.settings", s):
                result = _load_custom_providers()
                assert "custom" not in result

    def test_skips_empty_name(self):
        data = {
            "": {"api_type": "openai", "base_url": "https://example.com/v1", "api_key": "sk-test"},
        }
        tmp = _with_temp_custom_providers(data)
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            with patch("runtime.orchestration.provider_router.settings", s):
                result = _load_custom_providers()
                assert "" not in result


class TestAdapterCustomProviders:

    def _patch_cp(self, data: dict):
        """Patch _CUSTOM_PROVIDERS and clear _CLOUD_ADAPTERS to avoid real config leakage."""
        from unittest.mock import patch
        return patch.dict("runtime.orchestration.provider_router._CUSTOM_PROVIDERS", data, clear=True)

    def test_returns_openai_adapter_for_custom_provider(self):
        with self._patch_cp({
            "agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-agnes"},
        }):
            with patch("runtime.orchestration.provider_router._cloud_pool", return_value=None):
                result = adapter("agnes")
                assert isinstance(result, OpenAIAdapter)
                assert result.base_url == "https://api.agnes.ai/v1"
                assert result.api_key == "sk-agnes"

    def test_raises_for_unknown_custom_provider(self):
        with self._patch_cp({}):
            with patch("runtime.orchestration.provider_router._cloud_pool", return_value=None):
                with pytest.raises(ValueError, match="Unsupported provider: unknown"):
                    adapter("unknown")

    def test_raises_for_custom_provider_without_api_key(self):
        with self._patch_cp({
            "nokey": {"api_type": "openai", "base_url": "https://example.com/v1", "api_key": ""},
        }):
            with patch("runtime.orchestration.provider_router._cloud_pool", return_value=None):
                with pytest.raises(Exception):
                    adapter("nokey")

    def test_still_resolves_builtin_providers(self):
        with patch.dict("runtime.orchestration.provider_router._CUSTOM_PROVIDERS", {}, clear=True):
            with patch("runtime.orchestration.provider_router._cloud_pool", return_value=None):
                with patch("runtime.orchestration.provider_router.OpenAIAdapter") as mock:
                    adapter("openai")
                    mock.assert_called_once()


class TestReloadCustomProviders:

    def test_reloads_and_updates_cache(self):
        data = {"test_prov": {"api_type": "openai", "base_url": "https://test.ai/v1", "api_key": "sk-test"}}
        tmp = _with_temp_custom_providers(data)
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            with patch("runtime.orchestration.provider_router.settings", s):
                with patch("runtime.orchestration.routing_engine.register_custom_providers") as mock_reg:
                    with patch("runtime.orchestration.routing_engine.unregister_custom_providers"):
                        with patch.dict("runtime.orchestration.provider_router._CUSTOM_PROVIDERS", {}, clear=True):
                            result = reload_custom_providers()
                            assert "test_prov" in result
                            mock_reg.assert_called_once_with(
                                ["test_prov"], {"test_prov": "https://test.ai/v1"}
                            )

    def test_reload_updates_cloud_adapters(self):
        import runtime.orchestration.provider_router as pr
        data = {"poolai": {"api_type": "openai", "base_url": "https://pool.ai/v1", "api_key": "sk-pool"}}
        tmp = _with_temp_custom_providers(data)
        with patch("config.settings.Settings.config_path", return_value=tmp):
            s = Settings()
            with patch("runtime.orchestration.provider_router.settings", s):
                with patch("runtime.orchestration.routing_engine.register_custom_providers"):
                    with patch("runtime.orchestration.routing_engine.unregister_custom_providers"):
                        with patch.dict("runtime.orchestration.provider_router._CUSTOM_PROVIDERS", {}, clear=True):
                            with patch.dict(pr._CLOUD_ADAPTERS, {}, clear=True):
                                reload_custom_providers()
                                assert "poolai" in pr._CLOUD_ADAPTERS
                                assert pr._CLOUD_ADAPTERS["poolai"] is OpenAIAdapter


# ── Routing Engine ───────────────────────────────────────────────────

class TestRegisterCustomProviders:

    def setup_method(self):
        import runtime.orchestration.provider_router as pr
        self._orig_cloud = list(CLOUD_PROVIDERS)
        self._orig_routing = list(ROUTING_PROVIDERS)
        self._orig_endpoints = dict(CLOUD_PROVIDER_ENDPOINTS)
        self._orig_scores = {k: dict(v) for k, v in CAPABILITY_PROVIDER_SCORES.items()}
        self._orig_cloud_adapters = dict(pr._CLOUD_ADAPTERS)
        self._orig_cred_pools = dict(pr._credential_pools)
        self._orig_prov_enabled = dict(routing_engine._provider_enabled)

    def teardown_method(self):
        import runtime.orchestration.provider_router as pr
        CLOUD_PROVIDERS[:] = self._orig_cloud
        ROUTING_PROVIDERS[:] = self._orig_routing
        CLOUD_PROVIDER_ENDPOINTS.clear()
        CLOUD_PROVIDER_ENDPOINTS.update(self._orig_endpoints)
        CAPABILITY_PROVIDER_SCORES.clear()
        CAPABILITY_PROVIDER_SCORES.update(self._orig_scores)
        pr._CLOUD_ADAPTERS.clear()
        pr._CLOUD_ADAPTERS.update(self._orig_cloud_adapters)
        pr._credential_pools.clear()
        pr._credential_pools.update(self._orig_cred_pools)
        routing_engine._provider_enabled.clear()
        routing_engine._provider_enabled.update(self._orig_prov_enabled)
        # remove state file leaks from previous runs
        state_path = routing_engine._state_path
        if state_path.exists():
            state_path.unlink(missing_ok=True)

    def test_registers_provider_in_all_lists(self):
        register_custom_providers(["testai"], {"testai": "https://test.ai/v1"})
        assert "testai" in CLOUD_PROVIDERS
        assert "testai" in ROUTING_PROVIDERS
        assert CLOUD_PROVIDER_ENDPOINTS["testai"] == ("", "", "https://test.ai/v1")
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            assert "testai" in cap_scores
            assert cap_scores["testai"] == cap_scores.get("openai", 85)
        assert routing_engine._provider_enabled.get("testai") is True

    def test_registers_idempotent(self):
        register_custom_providers(["testai"], {"testai": "https://test.ai/v1"})
        count_before = len(CLOUD_PROVIDERS)
        register_custom_providers(["testai"], {"testai": "https://test.ai/v2"})
        assert len(CLOUD_PROVIDERS) == count_before

    def test_unregisters_provider(self):
        register_custom_providers(["testai"], {"testai": "https://test.ai/v1"})
        unregister_custom_providers(["testai"])
        assert "testai" not in CLOUD_PROVIDERS
        assert "testai" not in ROUTING_PROVIDERS
        assert "testai" not in CLOUD_PROVIDER_ENDPOINTS
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            assert "testai" not in cap_scores
        assert routing_engine._provider_enabled.get("testai") is False

    def test_unregister_does_not_remove_builtin(self):
        register_custom_providers(["openai"], {"openai": "https://custom.openai.com/v1"})
        unregister_custom_providers(["openai"])
        assert "openai" in CLOUD_PROVIDERS
        assert "openai" in ROUTING_PROVIDERS
        assert "openai" in CLOUD_PROVIDER_ENDPOINTS
        assert routing_engine._provider_enabled.get("openai") is not False

    def test_route_custom_provider_case_insensitive(self):
        register_custom_providers(["OpenCode"], {"OpenCode": "https://opencode.ai/zen/v1"})
        try:
            registry = {"models": [{"name": "big-pickle", "provider": "OpenCode", "capabilities": ["chat", "thinking", "tools"]}]}
            decision = routing_engine.route("big-pickle", registry_models=registry["models"])
            assert decision.provider == "OpenCode"
            assert decision.model == "big-pickle"
            candidates = {c.provider for c in decision.candidates}
            assert "OpenCode" in candidates
        finally:
            unregister_custom_providers(["OpenCode"])

    def test_canonical_provider_name_case_insensitive(self):
        from runtime.orchestration.provider_router import canonical_provider_name
        register_custom_providers(["OpenCode"], {"OpenCode": "https://opencode.ai/zen/v1"})
        try:
            assert canonical_provider_name("opencode") == "OpenCode"
            assert canonical_provider_name("OpenCode") == "OpenCode"
        finally:
            unregister_custom_providers(["OpenCode"])




# ── Dashboard API ────────────────────────────────────────────────────

class TestDashboardAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from dashboard.dashboard_server import app
        return TestClient(app)

    def test_custom_providers_list_empty(self, client):
        with patch("config.settings.Settings.load_custom_providers", return_value={}):
            resp = client.get("/api/custom-providers")
            assert resp.status_code == 200
            data = resp.json()
            assert data["providers"] == {}

    def test_custom_providers_list_with_data(self, client):
        data = {
            "agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-secret"},
        }
        with patch("config.settings.Settings.load_custom_providers", return_value=data):
            resp = client.get("/api/custom-providers")
            assert resp.status_code == 200
            body = resp.json()
            assert "agnes" in body["providers"]
            assert body["providers"]["agnes"]["has_key"] is True
            assert "api_key" not in body["providers"]["agnes"]

    def test_custom_providers_create(self, client):
        with patch("config.settings.Settings.load_custom_providers", return_value={}):
            with patch("config.settings.Settings.save_custom_providers") as mock_save:
                with patch("dashboard.dashboard_server.reload_custom_providers") as mock_reload:
                    resp = client.post("/api/custom-providers", json={
                        "name": "agnes", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test",
                    })
                    assert resp.status_code == 200
                    assert resp.json()["name"] == "agnes"
                    mock_save.assert_called_once()
                    mock_reload.assert_called_once()

    def test_custom_providers_create_appends_v1(self, client):
        with patch("config.settings.Settings.load_custom_providers", return_value={}):
            with patch("config.settings.Settings.save_custom_providers") as mock_save:
                with patch("dashboard.dashboard_server.reload_custom_providers"):
                    resp = client.post("/api/custom-providers", json={
                        "name": "test", "base_url": "https://api.test.com", "api_key": "sk-test",
                    })
                    assert resp.status_code == 200
                    saved = mock_save.call_args[0][0]
                    assert "https://api.test.com/v1" in str(saved)

    def test_custom_providers_create_duplicate(self, client):
        with patch("config.settings.Settings.load_custom_providers",
                   return_value={"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}):
            resp = client.post("/api/custom-providers", json={
                "name": "agnes", "base_url": "https://api.agnes.ai/v2", "api_key": "sk-other",
            })
            assert resp.status_code == 409

    def test_custom_providers_create_missing_fields(self, client):
        with patch("config.settings.Settings.load_custom_providers", return_value={}):
            resp = client.post("/api/custom-providers", json={"name": "", "base_url": "", "api_key": ""})
            assert resp.status_code == 400

    def test_custom_providers_update_base_url(self, client):
        orig = {"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-old"}}
        with patch("config.settings.Settings.load_custom_providers", return_value=orig):
            with patch("config.settings.Settings.save_custom_providers") as mock_save:
                with patch("dashboard.dashboard_server.reload_custom_providers"):
                    resp = client.put("/api/custom-providers/agnes", json={"base_url": "https://api.agnes.ai/v2"})
                    assert resp.status_code == 200
                    saved = mock_save.call_args[0][0]
                    assert saved["agnes"]["base_url"] == "https://api.agnes.ai/v2/v1"

    def test_custom_providers_update_not_found(self, client):
        with patch("config.settings.Settings.load_custom_providers", return_value={}):
            resp = client.put("/api/custom-providers/nonexistent", json={"base_url": "https://x.com"})
            assert resp.status_code == 404

    def test_custom_providers_delete(self, client):
        with patch("config.settings.Settings.load_custom_providers",
                   return_value={"agnes": {"api_type": "openai", "base_url": "https://api.agnes.ai/v1", "api_key": "sk-test"}}):
            with patch("config.settings.Settings.save_custom_providers"):
                with patch("dashboard.dashboard_server.reload_custom_providers"):
                    resp = client.delete("/api/custom-providers/agnes")
                    assert resp.status_code == 200

    def test_custom_providers_delete_not_found(self, client):
        with patch("config.settings.Settings.load_custom_providers", return_value={}):
            resp = client.delete("/api/custom-providers/nonexistent")
            assert resp.status_code == 404

    def test_custom_providers_reload(self, client):
        with patch("dashboard.dashboard_server.reload_custom_providers", return_value={"test": {}}):
            resp = client.post("/api/custom-providers/reload")
            assert resp.status_code == 200
            assert resp.json()["ok"] is True
