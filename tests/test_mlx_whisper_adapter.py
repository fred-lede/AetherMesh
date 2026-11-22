# tests/test_mlx_whisper_adapter.py
from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import pytest

import providers.mlx_whisper_adapter as mlx_module
from providers.asr_base import ASRProviderError
from providers.mlx_whisper_adapter import MLXWhisperAdapter, _resolve_repo


class _TempFile:
    def __init__(self, path: str) -> None:
        self.name = path
        self._handle = open(path, "wb")

    def write(self, data: bytes) -> None:
        self._handle.write(data)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _stub_mlx(monkeypatch, result: dict | None = None, error: Exception | None = None):
    mlx_whisper = MagicMock()
    if error is not None:
        mlx_whisper.transcribe.side_effect = error
    else:
        mlx_whisper.transcribe.return_value = result or {"text": "hello there"}

    holder_module = MagicMock()
    holder_module.ModelHolder.get_model.return_value = object()

    monkeypatch.setitem(sys.modules, "mlx_whisper", mlx_whisper)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", holder_module)
    monkeypatch.setattr(mlx_module, "_worker", None)
    return mlx_whisper, holder_module, sys.modules["mlx_whisper.transcribe"].ModelHolder


def _stub_tempfile(monkeypatch, tmp_path):
    monkeypatch.setattr(mlx_module.tempfile, "NamedTemporaryFile", lambda **kwargs: _TempFile(str(tmp_path / "a.wav")))
    return tmp_path / "a.wav"


def test_resolve_repo_maps_short_names():
    assert _resolve_repo("large-v3") == "mlx-community/whisper-large-v3-mlx"
    assert _resolve_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert _resolve_repo("tiny") == "mlx-community/whisper-tiny"
    assert _resolve_repo("medium") == "mlx-community/whisper-medium-mlx"
    assert _resolve_repo("") == "mlx-community/whisper-large-v3-mlx"


def test_resolve_repo_keeps_explicit_repo_or_local_path(tmp_path):
    assert _resolve_repo("mlx-community/whisper-base") == "mlx-community/whisper-base"
    local = tmp_path / "whisper-large-v3"
    local.mkdir()
    assert _resolve_repo(str(local)) == str(local)


def test_adapter_warms_model_holder_with_fp16(monkeypatch):
    mlx_whisper, _, holder = _stub_mlx(monkeypatch)
    adapter = MLXWhisperAdapter(model_name="large-v3-turbo", compute_type="float16")

    holder.get_model.assert_called_once()
    repo, dtype = holder.get_model.call_args.args
    assert repo == "mlx-community/whisper-large-v3-turbo"
    assert dtype is sys.modules["mlx.core"].float16
    assert mlx_whisper.transcribe.call_count == 0
    assert adapter.health_check() == {
        "provider": "mlx_whisper",
        "model_loaded": True,
        "model": "large-v3-turbo",
        "device": "mps",
    }


def test_adapter_float32_compute_type(monkeypatch):
    _, _, holder = _stub_mlx(monkeypatch)
    MLXWhisperAdapter(model_name="small", compute_type="float32")

    repo, dtype = holder.get_model.call_args.args
    assert repo == "mlx-community/whisper-small-mlx"
    assert dtype is sys.modules["mlx.core"].float32


def test_transcribe_returns_text_and_cleans_temp_file(monkeypatch, tmp_path):
    mlx_whisper, _, _ = _stub_mlx(monkeypatch, result={"text": "  see you!  "})
    audio_path = _stub_tempfile(monkeypatch, tmp_path)

    adapter = MLXWhisperAdapter(model_name="tiny")
    assert adapter.transcribe(audio=b"RIFFfake", language="th") == {"text": "see you!"}
    assert not audio_path.exists()

    args, kwargs = mlx_whisper.transcribe.call_args
    assert args[0] == str(audio_path)
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-tiny"
    assert kwargs["task"] == "transcribe"
    assert kwargs["language"] == "th"
    assert kwargs["fp16"] is True
    assert "initial_prompt" not in kwargs


def test_transcribe_passes_prompt_and_task(monkeypatch, tmp_path):
    mlx_whisper, _, _ = _stub_mlx(monkeypatch)
    _stub_tempfile(monkeypatch, tmp_path)

    adapter = MLXWhisperAdapter(model_name="tiny")
    adapter.transcribe(audio=b"RIFFfake", task="translate", language="ja", prompt="terms", temperature=0.4)

    kwargs = mlx_whisper.transcribe.call_args.kwargs
    assert kwargs["task"] == "translate"
    assert kwargs["language"] == "ja"
    assert kwargs["initial_prompt"] == "terms"
    assert kwargs["temperature"] == 0.4


@pytest.mark.parametrize(
    ("error", "status_code"),
    [(ValueError("bad audio"), 400), (RuntimeError("mlx crashed"), 503)],
)
def test_transcribe_errors_map_to_provider_status(monkeypatch, tmp_path, error, status_code):
    _stub_mlx(monkeypatch, error=error)
    audio_path = _stub_tempfile(monkeypatch, tmp_path)

    adapter = MLXWhisperAdapter(model_name="tiny")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(audio=b"RIFFfake")
    assert excinfo.value.status_code == status_code
    assert not audio_path.exists()


def test_streaming_model_returns_segments_and_info(monkeypatch, tmp_path):
    mlx_whisper, _, _ = _stub_mlx(
        monkeypatch,
        result={"text": "hello there", "segments": [{"text": " hello", "start": 0.0, "end": 1.25}], "language": "th"},
    )

    adapter = MLXWhisperAdapter(model_name="tiny")
    segments, info = adapter._model.transcribe([1.0, 2.0, 3.0], language="th", task="transcribe")

    assert [(seg.text, seg.start, seg.end) for seg in segments] == [(" hello", 0.0, 1.25)]
    assert info.language == "th"
    assert mlx_whisper.transcribe.call_args.args[0] == [1.0, 2.0, 3.0]
    assert mlx_whisper.transcribe.call_args.kwargs["path_or_hf_repo"] == "mlx-community/whisper-tiny"


def test_streaming_model_language_defaults_to_requested_one(monkeypatch, tmp_path):
    _stub_mlx(monkeypatch, result={"text": "ok", "segments": [{"text": " ok", "start": 0.0, "end": 0.5}]})

    adapter = MLXWhisperAdapter(model_name="tiny")
    _, info = adapter._model.transcribe([1.0, 2.0], language="ja")
    assert info.language == "ja"


def test_all_mlx_calls_run_on_the_worker_thread(monkeypatch, tmp_path):
    mlx_whisper, _, _ = _stub_mlx(monkeypatch)
    _stub_tempfile(monkeypatch, tmp_path)

    threads: set[str] = set()

    def record(*args, **kwargs):
        threads.add(threading.current_thread().name)
        return {"text": "ok"}

    mlx_whisper.transcribe.side_effect = record

    first = MLXWhisperAdapter(model_name="tiny")
    second = MLXWhisperAdapter(model_name="tiny")
    first.transcribe(audio=b"RIFFfake")
    second.transcribe(audio=b"RIFFfake")
    first._model.transcribe([1.0, 2.0], language="en")

    assert threads == {"mlx-whisper"}
    assert mlx_module._worker is not None
    assert mlx_module._worker._thread.name == "mlx-whisper"
