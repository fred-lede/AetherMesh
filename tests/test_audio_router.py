from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from providers.tts_base import TTSProviderError


@pytest.fixture
def mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.tts.return_value = b"fake-wav-bytes"
    adapter.list_voices.return_value = [
        {"voice_id": "v1", "name": "Voice 1", "language": "en",
         "created_at": "2026-01-01T00:00:00", "duration_seconds": 5.0},
    ]
    adapter.register_voice.return_value = {
        "voice_id": "new-id", "name": "New Voice", "language": "en",
        "created_at": "2026-06-29T00:00:00", "duration_seconds": 3.0,
    }
    adapter.delete_voice.return_value = True
    return adapter


@pytest.fixture
def app(mock_adapter: MagicMock) -> FastAPI:
    with patch("router.audio_router.settings.tts_enabled", True):
        with patch("router.audio_router.get_adapter", return_value=mock_adapter):
            from router.audio_router import router
            app = FastAPI()
            app.include_router(router)
            yield app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


AUDIO_MODULE = "router.audio_router"
ROUTER_MODULE = "runtime.orchestration.provider_router"


class TestAudioSpeech:
    def test_tts_success(self, client: TestClient, mock_adapter: MagicMock) -> None:
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "Hello world",
            "voice": "v1",
        })
        assert resp.status_code == 200
        assert resp.content == b"fake-wav-bytes"
        assert "audio/wav" in resp.headers.get("content-type", "")

    def test_tts_requires_input(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={"voice": "v1"})
        assert resp.status_code == 422

    def test_tts_requires_voice(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={"input": "hello"})
        assert resp.status_code == 422

    def test_tts_with_format(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "Hi",
            "voice": "v1",
            "response_format": "wav",
        })
        assert resp.status_code == 200

    def test_tts_with_language(self, client: TestClient) -> None:
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "您好",
            "voice": "v1",
            "language": "zh-tw",
        })
        assert resp.status_code == 200

    def test_tts_returns_404_for_unknown_voice(
        self, client: TestClient, mock_adapter: MagicMock
    ) -> None:
        mock_adapter.tts.side_effect = TTSProviderError("voice not found", status_code=404)
        resp = client.post("/v1/audio/speech", json={
            "model": "xtts-v2",
            "input": "hello",
            "voice": "nonexistent",
        })
        assert resp.status_code == 404


class TestVoicesAPI:
    def test_list_voices(self, client: TestClient) -> None:
        resp = client.get("/v1/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Voice 1"

    def test_register_voice(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/voices",
            data={"name": "New Voice", "language": "en"},
            files={"file": ("ref.wav", b"fake-audio-data", "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "New Voice"

    def test_register_voice_missing_name(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/voices",
            files={"file": ("ref.wav", b"data", "audio/wav")},
        )
        assert resp.status_code == 422

    def test_delete_voice(self, client: TestClient) -> None:
        resp = client.delete("/v1/voices/v1")
        assert resp.status_code == 204

    def test_delete_voice_not_found(
        self, client: TestClient, mock_adapter: MagicMock
    ) -> None:
        mock_adapter.delete_voice.return_value = False
        resp = client.delete("/v1/voices/nonexistent")
        assert resp.status_code == 404


@pytest.fixture
def mock_asr_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.transcribe.return_value = {"text": "hello world"}
    return adapter


@pytest.fixture
def asr_app(mock_asr_adapter: MagicMock) -> FastAPI:
    with patch("router.audio_router.settings.tts_enabled", False):
        with patch("router.audio_router.settings.asr_enabled", True):
            with patch("router.audio_router.get_adapter", return_value=mock_asr_adapter):
                from router.audio_router import router
                app = FastAPI()
                app.include_router(router)
                yield app


@pytest.fixture
def asr_client(asr_app: FastAPI) -> TestClient:
    return TestClient(asr_app)


class TestASR:
    def test_transcribe_success(self, asr_client: TestClient, mock_asr_adapter: MagicMock) -> None:
        resp = asr_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake-wav-data", "audio/wav")},
            data={"model": "whisper-large-v3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "hello world"

    def test_translate_success(self, asr_client: TestClient, mock_asr_adapter: MagicMock) -> None:
        resp = asr_client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", b"fake-wav-data", "audio/wav")},
            data={"model": "whisper-large-v3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "hello world"

    def test_transcribe_missing_file(self, asr_client: TestClient) -> None:
        resp = asr_client.post("/v1/audio/transcriptions", data={"model": "whisper-large-v3"})
        assert resp.status_code == 422

    def test_asr_disabled(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"data", "audio/wav")},
        )
        assert resp.status_code == 503

    def test_transcribe_adapter_error(self, asr_client: TestClient, mock_asr_adapter: MagicMock) -> None:
        from providers.asr_base import ASRProviderError
        mock_asr_adapter.transcribe.side_effect = ASRProviderError("bad audio", status_code=400)
        resp = asr_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"bad", "audio/wav")},
        )
        assert resp.status_code == 400
        assert "bad audio" in resp.text


class TestAdapterInitFailure:
    def test_tts_adapter_unavailable(self) -> None:
        with patch("router.audio_router.settings.tts_enabled", True):
            with patch("router.audio_router.get_adapter", return_value=None):
                from router.audio_router import router
                app = FastAPI()
                app.include_router(router)
                c = TestClient(app)
                resp = c.post("/v1/audio/speech", json={
                    "model": "xtts-v2", "input": "hi", "voice": "v1",
                })
                assert resp.status_code == 503
                assert "unavailable" in resp.text.lower()

    def test_asr_adapter_unavailable(self) -> None:
        with patch("router.audio_router.settings.tts_enabled", False):
            with patch("router.audio_router.settings.asr_enabled", True):
                with patch("router.audio_router.get_adapter", return_value=None):
                    from router.audio_router import router
                    app = FastAPI()
                    app.include_router(router)
                    c = TestClient(app)
                    resp = c.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"data", "audio/wav")},
                    )
                    assert resp.status_code == 503
                    assert "unavailable" in resp.text.lower()


class TestProviderRouterResilience:
    def test_asr_factory_cooldown(self) -> None:
        from runtime.orchestration import provider_router as pr
        import importlib
        importlib.reload(pr)
        pr._asr_adapter = None
        pr._asr_adapter_failed_at = 0.0
        with patch.object(pr.settings, "asr_enabled", True):
            with patch.object(pr, "FasterWhisperAdapter", side_effect=RuntimeError("CUDA error")):
                result = pr._get_asr_adapter()
                assert result is None
                assert pr._asr_adapter_failed_at > 0
                pr._ADAPTER_RETRY_COOLDOWN = 9999.0
                result2 = pr._get_asr_adapter()
                assert result2 is None
                pr._ADAPTER_RETRY_COOLDOWN = 30.0
        pr._asr_adapter = None
        pr._asr_adapter_failed_at = 0.0

    def test_tts_factory_cooldown(self) -> None:
        from runtime.orchestration import provider_router as pr
        import importlib
        importlib.reload(pr)
        pr._tts_adapter = None
        pr._tts_adapter_failed_at = 0.0
        with patch.object(pr.settings, "tts_enabled", True):
            with patch.object(pr, "XTTSAdapter", side_effect=RuntimeError("CUDA error")):
                result = pr._get_tts_adapter()
                assert result is None
                assert pr._tts_adapter_failed_at > 0
                pr._ADAPTER_RETRY_COOLDOWN = 9999.0
                result2 = pr._get_tts_adapter()
                assert result2 is None
                pr._ADAPTER_RETRY_COOLDOWN = 30.0
        pr._tts_adapter = None
        pr._tts_adapter_failed_at = 0.0
