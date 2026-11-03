# tests/test_remote_asr_adapter.py
from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

import providers.remote_asr_adapter as remote_module
from providers.asr_base import ASRProviderError
from providers.remote_asr_adapter import RemoteASRAdapter, pcm16_to_wav


def _response(ok=True, status_code=200, payload=None, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    if ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = remote_module.requests.HTTPError(f"status {status_code}")
    return resp


def test_pcm16_to_wav_roundtrip():
    pcm = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    wav_bytes = pcm16_to_wav(pcm)
    assert wav_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(4) == pcm


def test_stream_contract_transcribes_via_http(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["url"] = url
        posted["files"] = files
        posted["data"] = data
        posted["headers"] = headers
        return _response(payload={"text": "hello remote"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    samples = np.full(16000, 0.01, dtype=np.float32)
    segments, info = adapter._model.transcribe(samples, language="zh")
    assert segments[0].text == "hello remote"
    assert posted["url"] == "http://10.0.0.9:9450/v1/audio/transcriptions"
    name, content, content_type = posted["files"]["file"]
    assert name == "audio.wav"
    assert content[:4] == b"RIFF"
    assert content_type == "audio/wav"
    assert posted["data"] == {"language": "zh"}


def test_stream_contract_empty_language_omits_form_field(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["data"] = data
        return _response(payload={"text": "hi"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    adapter._model.transcribe(np.zeros(1600, dtype=np.float32), language="")
    assert posted["data"] == {}


def test_token_sent_as_bearer(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["headers"] = headers
        return _response(payload={"text": "hi"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450", token="secret-token")
    adapter._model.transcribe(np.zeros(1600, dtype=np.float32))
    assert posted["headers"] == {"Authorization": "Bearer secret-token"}


def test_adapter_transcribe_forwards_bytes_verbatim(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["files"] = files
        posted["data"] = data
        return _response(payload={"text": "echo"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    result = adapter.transcribe(b"RAW-UPLOAD-BYTES", language="en")
    assert result == {"text": "echo"}
    assert posted["files"]["file"][1] == b"RAW-UPLOAD-BYTES"
    assert posted["data"] == {"language": "en"}


def test_http_error_maps_to_503_with_node_context(monkeypatch):
    monkeypatch.setattr(remote_module.requests, "post", lambda *a, **k: _response(ok=False, status_code=500, text="boom"))
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"x")
    assert excinfo.value.status_code == 503
    assert "n1" in str(excinfo.value)
    assert "http://10.0.0.9:9450" in str(excinfo.value)


def test_connection_error_maps_to_503(monkeypatch):
    def boom(*a, **k):
        raise remote_module.requests.ConnectionError("refused")

    monkeypatch.setattr(remote_module.requests, "post", boom)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter._model.transcribe(np.zeros(1600, dtype=np.float32))
    assert excinfo.value.status_code == 503
    assert "unreachable" in str(excinfo.value)


def test_http_400_from_worker_maps_to_400(monkeypatch):
    monkeypatch.setattr(remote_module.requests, "post", lambda *a, **k: _response(ok=False, status_code=400, text="bad request"))
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"x")
    assert excinfo.value.status_code == 400
    assert "n1" in str(excinfo.value)
    assert "HTTP 400" in str(excinfo.value)


def test_http_500_from_worker_maps_to_503(monkeypatch):
    monkeypatch.setattr(remote_module.requests, "post", lambda *a, **k: _response(ok=False, status_code=500, text="boom"))
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"x")
    assert excinfo.value.status_code == 503


def test_health_check_reachable(monkeypatch):
    monkeypatch.setattr(
        remote_module.requests, "get",
        lambda *a, **k: _response(payload={"engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B", "model_loaded": True}),
    )
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450/")
    health = adapter.health_check()
    assert health["reachable"] is True
    assert health["node_id"] == "n1"
    assert health["engine"] == "qwen_asr"
    assert health["model"] == "Qwen/Qwen3-ASR-1.7B"


def test_health_check_unreachable(monkeypatch):
    def boom(*a, **k):
        raise remote_module.requests.Timeout("timeout")

    monkeypatch.setattr(remote_module.requests, "get", boom)
    health = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450").health_check()
    assert health == {"reachable": False, "node_id": "n1", "base_url": "http://10.0.0.9:9450"}
