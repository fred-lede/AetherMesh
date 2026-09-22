from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from dashboard import dashboard_server as dash
from runtime.orchestration import model_references
from runtime.orchestration import model_registry_store as store


@pytest.fixture()
def models_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.yaml"
    monkeypatch.setattr(store, "models_path", lambda: path)
    store.reload_models()
    return path


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(dash, "_auth_credentials_configured", lambda: True)
    monkeypatch.setattr(dash, "_session_auth_valid", lambda request: True)
    monkeypatch.setattr(dash, "_require_admin", lambda request: None)
    return TestClient(dash.app)


def test_capabilities_endpoint_lists_vocabulary(client: TestClient):
    resp = client.get("/api/models/capabilities")
    assert resp.status_code == 200
    values = {item["value"] for item in resp.json()["capabilities"]}
    assert "image_gen" in values and "video" in values


def test_list_models_classifies_local_and_cloud(client: TestClient, models_file: Path):
    store.save_models([
        {"name": "loc", "provider": "ollama", "worker_bindings": [{"node_id": "node-01", "port": 11434}], "capabilities": ["chat"]},
        {"name": "cld", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]},
    ])
    resp = client.get("/api/models")
    assert resp.status_code == 200
    by_name = {m["name"]: m for m in resp.json()["models"]}
    assert by_name["loc"]["category"] == "local"
    assert by_name["cld"]["category"] == "cloud"
    assert by_name["loc"]["workers"] == ["node-01:11434"]


def test_create_model(client: TestClient, models_file: Path):
    resp = client.post("/api/models", json={"name": "new", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]})
    assert resp.status_code == 200
    assert any(m["name"] == "new" for m in store.get_models())


def test_create_model_duplicate_returns_409(client: TestClient, models_file: Path):
    store.save_models([{"name": "dup", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}])
    resp = client.post("/api/models", json={"name": "dup", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]})
    assert resp.status_code == 409


def test_create_model_validation_returns_400(client: TestClient, models_file: Path):
    resp = client.post("/api/models", json={"provider": "openai"})
    assert resp.status_code == 400


def test_delete_missing_returns_404(client: TestClient, models_file: Path):
    resp = client.delete("/api/models/nope")
    assert resp.status_code == 404


def test_update_rename_without_references(client: TestClient, models_file: Path):
    store.save_models([{"name": "old", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}])
    resp = client.put("/api/models/old", json={"name": "new", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]})
    assert resp.status_code == 200
    assert any(m["name"] == "new" for m in store.get_models())
    assert not any(m["name"] == "old" for m in store.get_models())


def test_update_rename_with_references_requires_flag(client: TestClient, models_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rules = tmp_path / "routing_rules.yaml"
    rules.write_text(yaml.safe_dump({"aliases": {"fast": "old"}}), encoding="utf-8")
    monkeypatch.setattr(model_references, "_rules_path", lambda: rules)
    store.save_models([{"name": "old", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}])
    resp = client.put("/api/models/old", json={"name": "new", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["references"][0]["path"] == "aliases.fast"


def test_update_rename_updates_references_with_flag(client: TestClient, models_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rules = tmp_path / "routing_rules.yaml"
    rules.write_text(yaml.safe_dump({"aliases": {"fast": "old"}}), encoding="utf-8")
    monkeypatch.setattr(model_references, "_rules_path", lambda: rules)
    store.save_models([{"name": "old", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}])
    resp = client.put(
        "/api/models/old?update_references=true",
        json={"name": "new", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]},
    )
    assert resp.status_code == 200
    assert yaml.safe_load(rules.read_text(encoding="utf-8"))["aliases"]["fast"] == "new"


def test_update_rename_only_keeps_references(client: TestClient, models_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rules = tmp_path / "routing_rules.yaml"
    rules.write_text(yaml.safe_dump({"aliases": {"fast": "old"}}), encoding="utf-8")
    monkeypatch.setattr(model_references, "_rules_path", lambda: rules)
    store.save_models([{"name": "old", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}])
    resp = client.put(
        "/api/models/old?rename_only=true",
        json={"name": "new", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]},
    )
    assert resp.status_code == 200
    assert yaml.safe_load(rules.read_text(encoding="utf-8"))["aliases"]["fast"] == "old"
    assert any(m["name"] == "new" for m in store.get_models())


def test_providers_endpoint(client: TestClient):
    resp = client.get("/api/models/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert "ollama" in body["local"]
    assert "openai" in body["cloud"]


def test_reload_endpoint(client: TestClient, models_file: Path):
    store.save_models([{"name": "a", "provider": "openai", "worker_ports": [], "capabilities": ["chat"]}])
    resp = client.post("/api/models/reload")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
