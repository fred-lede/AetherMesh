from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from providers.base import ProviderError
from router import decision_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(decision_router.router)
    return TestClient(app)


def test_missing_model_returns_400():
    resp = _client().post("/v1/systemone", json={"state": "x", "questions": {}})
    assert resp.status_code == 400


def test_routes_to_ollama_adapter():
    adapter = MagicMock()
    adapter.systemone.return_value = {"model": "nimble:9b", "answers": {"intent": {"type": "choice", "choice": "book"}}}
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", {"base_url": "http://w:11434"})), \
         patch("router.decision_router.provider_router.adapter", return_value=adapter):
        resp = _client().post("/v1/systemone", json={"model": "nimble:9b", "state": "x", "questions": {"intent": {}}})
    assert resp.status_code == 200
    assert resp.json()["answers"]["intent"]["choice"] == "book"
    adapter.systemone.assert_called_once()
    assert adapter.systemone.call_args.args[0]["state"] == "x"


def test_unsupported_provider_returns_501():
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("agnes", None)):
        resp = _client().post("/v1/systemone", json={"model": "x", "state": "y", "questions": {}})
    assert resp.status_code == 501


def test_upstream_error_is_preserved():
    adapter = MagicMock()
    adapter.systemone.side_effect = ProviderError("questions must contain 1-64 fields")
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", {"base_url": "http://w:11434"})), \
         patch("router.decision_router.provider_router.adapter", return_value=adapter):
        resp = _client().post("/v1/systemone", json={"model": "m", "state": "s", "questions": {}})
    assert resp.status_code == 502
    assert "questions must contain" in resp.json()["detail"]


def test_no_worker_returns_400():
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", None)):
        resp = _client().post("/v1/systemone", json={"model": "ghost:1b", "state": "x", "questions": {}})
    assert resp.status_code == 400


def test_null_model_returns_400():
    resp = _client().post("/v1/systemone", json={"model": None, "state": "x", "questions": {}})
    assert resp.status_code == 400


def test_state_object_is_forwarded_uncoerced():
    adapter = MagicMock()
    adapter.systemone.return_value = {"answers": {}}
    state = {"a": [1, 2], "b": {"c": "d"}}
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", {"base_url": "http://w:11434"})), \
         patch("router.decision_router.provider_router.adapter", return_value=adapter):
        _client().post("/v1/systemone", json={"model": "nimble:9b", "state": state, "questions": {}})
    assert adapter.systemone.call_args.args[0]["state"] == state


def test_upstream_400_status_is_preserved():
    adapter = MagicMock()
    adapter.systemone.side_effect = ProviderError("questions must contain 1-64 fields", status_code=400)
    with patch("router.decision_router.model_registry_store.get_models", return_value=[]), \
         patch("router.decision_router.provider_router.resolve_provider", return_value=("ollama", {"base_url": "http://w:11434"})), \
         patch("router.decision_router.provider_router.adapter", return_value=adapter):
        resp = _client().post("/v1/systemone", json={"model": "m", "state": "s", "questions": {}})
    assert resp.status_code == 400
    assert "questions must contain" in resp.json()["detail"]
