from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import node.asr_worker as asr_worker


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(asr_worker, "_adapter", None)
    app = FastAPI()
    app.include_router(asr_worker.app.router)
    return TestClient(app)


def _engine(transcribe_result=None, side_effect=None):
    engine = MagicMock()
    engine.transcribe.return_value = transcribe_result if transcribe_result is not None else {"text": "hi"}
    if side_effect is not None:
        engine.transcribe.side_effect = side_effect
    return engine


def test_health_reports_unloaded_engine(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "asr-worker"
    assert body["model_loaded"] is False


def test_transcription_lazy_loads_and_returns_text(client, monkeypatch):
    engine = _engine({"text": "hello from node"})
    monkeypatch.setattr(asr_worker, "_build_adapter", lambda: engine)
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"wav-bytes", "audio/wav")},
        data={"language": "zh"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello from node"}
    engine.transcribe.assert_called_once()
    assert engine.transcribe.call_args.args[0] == b"wav-bytes"
    assert engine.transcribe.call_args.kwargs.get("language") == "zh"


def test_transcription_reuses_loaded_engine(client, monkeypatch):
    engine = _engine({"text": "second"})
    monkeypatch.setattr(asr_worker, "_build_adapter", lambda: engine)
    client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    client.post("/v1/audio/transcriptions", files={"file": ("b.wav", b"y", "audio/wav")})
    assert engine.transcribe.call_count == 2
    assert asr_worker._adapter is engine


def test_engine_build_failure_maps_503(client, monkeypatch):
    def boom():
        raise RuntimeError("qwen-asr not installed")

    monkeypatch.setattr(asr_worker, "_build_adapter", boom)
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_transcription_engine_error_maps_provider_status(client, monkeypatch):
    from providers.asr_base import ASRProviderError

    engine = _engine(side_effect=ASRProviderError("bad audio", status_code=400))
    monkeypatch.setattr(asr_worker, "_build_adapter", lambda: engine)
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    assert resp.status_code == 400


def test_probe_asr_worker_healthy(monkeypatch):
    import node.node_agent as node_agent_module

    resp = MagicMock(ok=True)
    resp.json.return_value = {"engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}
    monkeypatch.setattr(node_agent_module.requests, "get", lambda *a, **k: resp)
    monkeypatch.setenv("AIIH_ASR_PORT", "9450")
    assert node_agent_module._probe_asr_worker() == {
        "port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B",
    }


def test_probe_asr_worker_down(monkeypatch):
    import node.node_agent as node_agent_module

    def boom(*a, **k):
        raise node_agent_module.requests.ConnectionError("refused")

    monkeypatch.setattr(node_agent_module.requests, "get", boom)
    probe = node_agent_module._probe_asr_worker()
    assert probe == {"port": 9450, "healthy": False, "engine": None, "model": None}


def test_probe_asr_worker_bad_port_env_degrades(monkeypatch):
    import node.node_agent as node_agent_module

    def _fail(*a, **k):
        raise AssertionError("requests.get must not be called when AIIH_ASR_PORT is non-numeric")

    monkeypatch.setattr(node_agent_module.requests, "get", _fail)
    monkeypatch.setenv("AIIH_ASR_PORT", "not-a-number")
    assert node_agent_module._probe_asr_worker() == {"port": 9450, "healthy": False, "engine": None, "model": None}


def test_node_agent_payload_includes_asr_worker(monkeypatch):
    import node.node_agent as node_agent_module

    monkeypatch.setattr(node_agent_module.NodeAgent, "discover_workers", lambda self: [])
    monkeypatch.setattr(node_agent_module.NodeAgent, "_get_heavy_telemetry", lambda self, workers: ([], {}))
    monkeypatch.setattr(node_agent_module, "_probe_asr_worker", lambda: {"port": 9450, "healthy": False, "engine": None, "model": None})
    payload = node_agent_module.NodeAgent().payload()
    assert payload["metadata"]["asr_worker"] == {"port": 9450, "healthy": False, "engine": None, "model": None}


def test_launcher_defines_asr_worker_service():
    from runtime.launcher.launcher import SERVICE_DEFS

    entry = next(svc for svc in SERVICE_DEFS if svc["name"] == "asr_worker")
    assert entry["module"] == "node.asr_worker:app"
    assert entry["port_env"] == "AIIH_ASR_PORT"
    assert entry["port_default"] == 9450
