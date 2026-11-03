from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.asr_admin_router as asr_admin_module
from runtime.asr.asr_manager import ASRConfigError


@pytest.fixture
def client(monkeypatch):
    manager = MagicMock()
    manager.snapshot.return_value = {
        "state": "ready",
        "active": {"target_id": "local:cuda:1", "provider": "faster_whisper", "model": "large-v3", "loaded_at": "2026-11-03T00:00:00+00:00"},
        "requested": {"target_id": "local:cuda:1", "provider": "faster_whisper", "model": "large-v3", "loaded_at": "2026-11-03T00:00:00+00:00"},
        "error": None,
        "version": 3,
    }
    manager.list_targets.return_value = {
        "local": [{"id": "local:cuda:0", "name": "GPU0", "memory_free_mb": 1000}, {"id": "local:cpu", "name": "CPU", "memory_free_mb": None}],
        "nodes": [{"id": "node:m4", "reachable": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}],
        "nodes_unreachable": False,
    }
    monkeypatch.setattr(asr_admin_module, "asr_manager", manager)
    app = FastAPI()
    app.include_router(asr_admin_module.router)
    return TestClient(app), manager


def test_get_config_shape(client):
    c, _ = client
    resp = c.get("/api/asr/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "ready"
    assert body["version"] == 3
    assert body["targets"]["local"][0]["id"] == "local:cuda:0"
    assert body["targets"]["nodes"][0]["id"] == "node:m4"
    assert body["engines"]["faster_whisper"] == ["large-v3", "medium", "small"]
    assert body["engines"]["qwen_asr"] == ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"]


def test_post_config_returns_202_loading(client):
    c, manager = client
    manager.reload.return_value = {"state": "loading", "active": None, "requested": None, "error": None, "version": 3}
    resp = c.post("/api/asr/config", json={"target_id": "local:cuda:0", "provider": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"})
    assert resp.status_code == 202
    assert resp.json()["state"] == "loading"
    manager.reload.assert_called_once_with("local:cuda:0", provider="qwen_asr", model="Qwen/Qwen3-ASR-1.7B")


def test_post_config_validation_error_maps_status(client):
    c, manager = client
    manager.reload.side_effect = ASRConfigError("device_index 9 is not available on this host", details={"valid_gpu_ids": [0, 1]})
    resp = c.post("/api/asr/config", json={"target_id": "local:cuda:9"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "device_index 9" in detail["message"]
    assert detail["valid_gpu_ids"] == [0, 1]


def test_post_config_conflict_maps_409(client):
    c, manager = client
    manager.reload.side_effect = ASRConfigError("an ASR reload is already in progress", status_code=409)
    resp = c.post("/api/asr/config", json={"target_id": "local:cpu"})
    assert resp.status_code == 409


def test_post_config_missing_target_id_passes_empty_and_maps_400(client):
    c, manager = client
    manager.reload.side_effect = ASRConfigError("target_id is required")
    resp = c.post("/api/asr/config", json={})
    assert resp.status_code == 400
    manager.reload.assert_called_once_with("", provider=None, model=None)


def test_post_config_non_string_provider_400_without_reload(client):
    c, manager = client
    resp = c.post("/api/asr/config", json={"target_id": "local:cpu", "provider": 123})
    assert resp.status_code == 400
    manager.reload.assert_not_called()


def test_post_config_non_string_model_400_without_reload(client):
    c, manager = client
    resp = c.post("/api/asr/config", json={"target_id": "local:cpu", "model": ["large-v3"]})
    assert resp.status_code == 400
    manager.reload.assert_not_called()
