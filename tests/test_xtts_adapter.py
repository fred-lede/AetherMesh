from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from providers.tts_base import TTSProviderError


@pytest.fixture(autouse=True)
def _mock_multiprocessing() -> dict[str, MagicMock]:
    """Mock multiprocessing so no real subprocess/queue is created."""
    with patch("providers.xtts_adapter.multiprocessing") as mock_mp:
        ctx = MagicMock()
        mock_mp.get_context.return_value = ctx

        _call_count: list[int] = [0]
        req_q = MagicMock()
        resp_q = MagicMock()

        def _queue_side_effect(*args, **kwargs):
            v = _call_count[0]
            _call_count[0] += 1
            if v % 2 == 0:
                return req_q
            return resp_q

        ctx.Queue.side_effect = _queue_side_effect

        process_mocks: list[MagicMock] = []

        def _process_side_effect(*args, **kwargs):
            p = MagicMock()
            p.is_alive.return_value = True
            process_mocks.append(p)
            return p
        ctx.Process.side_effect = _process_side_effect

        yield {
            "ctx": ctx,
            "request_queue": req_q,
            "response_queue": resp_q,
            "process_mocks": process_mocks,
            "mock_mp": mock_mp,
        }


@pytest.fixture
def voices_dir() -> Path:
    d = Path(tempfile.mkdtemp()) / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def adapter(
    _mock_multiprocessing: dict[str, MagicMock],
    voices_dir: Path,
) -> MagicMock:
    from providers.xtts_adapter import XTTSAdapter
    return XTTSAdapter(
        model_name="test-model",
        device="cpu",
        voices_dir=str(voices_dir),
    )


class TestXTTSAdapterInit:
    def test_provider_name(self, adapter: MagicMock) -> None:
        assert adapter.provider_name == "xtts"

    def test_voices_dir_created(self, adapter: MagicMock, voices_dir: Path) -> None:
        assert voices_dir.is_dir()

    def test_no_model_in_adapter(self, adapter: MagicMock) -> None:
        assert not hasattr(adapter, "_model")

    def test_worker_not_started_at_init(self, adapter: MagicMock) -> None:
        assert adapter._process is None
        assert adapter._request_queue is None
        assert adapter._response_queue is None


class TestWorkerLifecycle:
    def test_ensure_worker_starts_process(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        adapter._ensure_worker()
        _mock_multiprocessing["ctx"].Process.assert_called_once()
        assert len(_mock_multiprocessing["process_mocks"]) == 1
        _mock_multiprocessing["process_mocks"][0].start.assert_called_once()

    def test_ensure_worker_reuses_alive_process(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        adapter._ensure_worker()
        prev_proc = adapter._process
        adapter._ensure_worker()
        assert adapter._process is prev_proc
        assert len(_mock_multiprocessing["process_mocks"]) == 1
        _mock_multiprocessing["process_mocks"][0].start.assert_called_once()

    def test_ensure_worker_dead_process_restarts(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        adapter._ensure_worker()
        # Simulate: adapter detects dead process and cleans up
        adapter._process = None
        adapter._ensure_worker()
        assert _mock_multiprocessing["ctx"].Process.call_count == 2

    def test_ensure_worker_detects_dead_process(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        adapter._ensure_worker()
        assert len(_mock_multiprocessing["process_mocks"]) == 1
        adapter._process.is_alive.return_value = False
        prev = adapter._process
        adapter._ensure_worker()
        # Should have created a new process
        assert _mock_multiprocessing["ctx"].Process.call_count == 2

    def test_ensure_worker_cooldown_prevents_restart(
        self, adapter: MagicMock
    ) -> None:
        with patch("providers.xtts_adapter.time.monotonic", return_value=100.0):
            adapter._worker_last_failure = 90.0
            adapter._worker_cooldown = 30.0
            with pytest.raises(TTSProviderError) as exc:
                adapter._ensure_worker()
            assert exc.value.status_code == 503
            assert "cooldown" in str(exc.value)

    def test_start_worker_dies_immediately_raises(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        dead_proc = MagicMock()
        dead_proc.is_alive.return_value = False
        _mock_multiprocessing["ctx"].Process.side_effect = lambda *a, **kw: dead_proc
        with pytest.raises(TTSProviderError) as exc:
            adapter._start_worker()
        assert exc.value.status_code == 503
        assert "CUDA" in str(exc.value) or "died" in str(exc.value)


class TestSendWorkerRequest:
    def test_send_and_receive(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok", b"wav-data")
        result = adapter._send_worker_request({"type": "tts", "text": "hi"})
        assert result == b"wav-data"
        _mock_multiprocessing["request_queue"].put.assert_called_once()

    def test_worker_error_raises(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("error", "something broke")
        with pytest.raises(TTSProviderError) as exc:
            adapter._send_worker_request({"type": "tts"})
        assert exc.value.status_code == 500
        assert "something broke" in str(exc.value)

    def test_worker_dead_during_wait_raises_503(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        adapter._ensure_worker()
        assert adapter._process is not None
        is_alive_calls: list[bool] = [True, False]
        adapter._process.is_alive.side_effect = lambda: is_alive_calls.pop(0)
        _mock_multiprocessing["response_queue"].get.side_effect = Exception("timeout")
        with pytest.raises(TTSProviderError) as exc:
            adapter._send_worker_request({"type": "tts"})
        assert exc.value.status_code == 503

    def test_response_timeout_raises_504(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        adapter._ensure_worker()
        assert adapter._process is not None
        adapter._process.is_alive.return_value = True
        _mock_multiprocessing["response_queue"].get.side_effect = Exception("timeout")
        with pytest.raises(TTSProviderError) as exc:
            adapter._send_worker_request({"type": "tts"})
        assert exc.value.status_code == 504


class TestXTTSAdapterTTS:
    def test_tts_returns_bytes(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok", b"wav-data")
        result = adapter.tts({
            "voice": "test-id",
            "input": "Hello world",
            "language": "en",
        })
        assert isinstance(result, bytes)

    def test_tts_sends_correct_request(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok", b"wav-data")
        adapter.tts({"voice": "v1", "input": "hi", "language": "en", "speed": 1.5})
        sent = _mock_multiprocessing["request_queue"].put.call_args[0][0]
        assert sent["type"] == "tts"
        assert sent["text"] == "hi"
        assert sent["voice_id"] == "v1"
        assert sent["language"] == "en"
        assert sent["speed"] == 1.5

    def test_tts_applies_speed(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok", b"wav-data")
        applied: list[float] = []

        def fake_speed(data: bytes, speed: float) -> bytes:
            applied.append(speed)
            return data

        adapter._apply_speed = fake_speed  # type: ignore[method-assign]
        adapter.tts({"voice": "x", "input": "hi", "speed": 1.5})
        assert applied == [1.5]

    def test_tts_no_speed_does_not_call_apply(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok", b"wav-data")
        called = False

        def fake_speed(*args: object) -> bytes:
            nonlocal called
            called = True
            return b""

        adapter._apply_speed = fake_speed  # type: ignore[method-assign]
        adapter.tts({"voice": "x", "input": "hi"})
        assert not called

    def test_tts_worker_error_wrapped(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("error", "inference crash")
        with pytest.raises(TTSProviderError) as exc:
            adapter.tts({"voice": "x", "input": "hi"})
        assert exc.value.status_code == 500

    def test_tts_language_detection_from_meta(
        self, adapter: MagicMock, voices_dir: Path,
        _mock_multiprocessing: dict[str, MagicMock],
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok", b"wav-data")
        vp = voices_dir / "test-id"
        vp.mkdir(parents=True, exist_ok=True)
        (vp / "meta.json").write_text('{"language":"ja"}')
        adapter.tts({"voice": "test-id", "input": "konnichiwa"})
        sent = _mock_multiprocessing["request_queue"].put.call_args[0][0]
        assert sent["language"] == "ja"


class TestXTTSAdapterVoiceCRUD:
    def test_register_and_list(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok",)
        meta = adapter.register_voice(
            name="test-voice",
            audio_data=b"fake-wav-data",
            language="en",
        )
        assert "voice_id" in meta
        assert meta["name"] == "test-voice"
        assert meta["language"] == "en"

        voices = adapter.list_voices()
        assert any(v["voice_id"] == meta["voice_id"] for v in voices)

    def test_register_sends_to_worker(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok",)
        adapter.register_voice(name="v", audio_data=b"data", language="en")
        sent = _mock_multiprocessing["request_queue"].put.call_args[0][0]
        assert sent["type"] == "register"
        assert "audio_path" in sent
        assert "embedding_path" in sent

    def test_register_worker_error_cleans_up(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("error", "encoding failed")
        with pytest.raises(TTSProviderError) as exc:
            adapter.register_voice(name="fail", audio_data=b"data", language="en")
        assert exc.value.status_code == 422

    def test_delete_voice(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok",)
        meta = adapter.register_voice(name="del-me", audio_data=b"data")
        assert adapter.delete_voice(meta["voice_id"]) is True
        assert adapter.delete_voice("nonexistent") is False

    def test_update_voice(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok",)
        meta = adapter.register_voice(name="old", audio_data=b"data", language="en")
        adapter.update_voice(meta["voice_id"], name="new", language="ja")
        updated = adapter.list_voices()
        v = next(v for v in updated if v["voice_id"] == meta["voice_id"])
        assert v["name"] == "new"
        assert v["language"] == "ja"

    def test_health_check(
        self, adapter: MagicMock, _mock_multiprocessing: dict[str, MagicMock]
    ) -> None:
        health = adapter.health_check()
        assert health["provider"] == "xtts"
        assert "device" in health
        assert "worker_alive" in health
        assert "voices_count" in health

    def test_health_check_worker_dead(
        self, adapter: MagicMock
    ) -> None:
        health = adapter.health_check()
        assert health["worker_alive"] is False


class TestTrimReference:
    def test_no_trim_when_under_limit(
        self, adapter: MagicMock, voices_dir: Path
    ) -> None:
        ref = voices_dir / "v" / "reference.wav"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_bytes(b"fake-wav")
        with patch("providers.xtts_adapter.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "5.0\n"
            mock_run.return_value.returncode = 0
            duration = adapter._trim_reference(ref)
        assert duration == 5.0
        assert ref.exists()  # not deleted

    def test_trims_when_over_limit(
        self, adapter: MagicMock, voices_dir: Path
    ) -> None:
        ref = voices_dir / "v" / "reference.wav"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_bytes(b"fake-wav")
        with patch("providers.xtts_adapter.subprocess.run") as mock_run:
            def _side(cmd, *args, **kwargs):
                if "ffprobe" in str(cmd):
                    m = MagicMock()
                    m.stdout = "30.0\n"
                    m.returncode = 0
                    return m
                if "ffmpeg" in str(cmd):
                    trimmed = Path(cmd[-1])
                    trimmed.parent.mkdir(parents=True, exist_ok=True)
                    trimmed.write_bytes(b"trimmed-wav")
                    m = MagicMock()
                    m.returncode = 0
                    return m
                return MagicMock()
            mock_run.side_effect = _side
            duration = adapter._trim_reference(ref)
        assert duration == 10.0

    def test_register_voice_trims_long_audio(
        self, adapter: MagicMock, voices_dir: Path,
        _mock_multiprocessing: dict[str, MagicMock],
    ) -> None:
        _mock_multiprocessing["response_queue"].get.return_value = ("ok",)
        with patch("providers.xtts_adapter.subprocess.run") as mock_run:
            def _side(cmd, *args, **kwargs):
                if "ffprobe" in str(cmd):
                    m = MagicMock()
                    m.stdout = "30.0\n"
                    m.returncode = 0
                    return m
                if "ffmpeg" in str(cmd):
                    trimmed = Path(cmd[-1])
                    trimmed.parent.mkdir(parents=True, exist_ok=True)
                    trimmed.write_bytes(b"trimmed-wav")
                    m = MagicMock()
                    m.returncode = 0
                    return m
                return MagicMock()
            mock_run.side_effect = _side
            meta = adapter.register_voice(name="long", audio_data=b"data" * 5000, language="en")
        assert meta["duration_seconds"] == 10.0
