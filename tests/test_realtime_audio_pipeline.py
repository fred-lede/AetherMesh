from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from runtime.realtime.audio_pipeline import RealtimeAudioPipeline, resample_pcm16_24k_to_16k


def _pcm24(samples: int = 2400) -> bytes:
    return np.full(samples, 1000, dtype=np.int16).tobytes()


def _model(text: str = "hello world") -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = ([SimpleNamespace(text=text)], SimpleNamespace(language="en"))
    return model


def test_resample_pcm16_24k_to_16k_converts_duration_without_changing_level():
    result = resample_pcm16_24k_to_16k(_pcm24())
    samples = np.frombuffer(result, dtype=np.int16)
    assert len(samples) == 1600
    assert np.all(samples == 1000)


@pytest.mark.asyncio
async def test_audio_chunk_emits_source_and_translated_text_deltas():
    model = _model()
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好世界"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    assert events == [
        {"type": "session.input_transcript.delta", "delta": "hello world"},
        {"type": "session.output_transcript.delta", "delta": "你好世界"},
    ]
    assert service.handle_chat.call_args.args[0]["model"] == "local-chat-model"
    assert "zh" in service.handle_chat.call_args.args[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_close_flushes_audio_below_the_streaming_window():
    model = _model("last words")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "最後一句"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    await pipeline.add_audio(_pcm24(4800))
    events = await pipeline.close()
    assert events == [
        {"type": "session.input_transcript.delta", "delta": "last words"},
        {"type": "session.output_transcript.delta", "delta": "最後一句"},
    ]


@pytest.mark.asyncio
async def test_translation_without_choices_preserves_source_text_and_emits_error():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": []}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    assert events[0] == {"type": "session.input_transcript.delta", "delta": "hello world"}
    assert events[1]["type"] == "error"
    assert events[1]["error"]["message"] == "translation returned no choices"


@pytest.mark.asyncio
async def test_translation_provider_failure_preserves_source_and_emits_error():
    service = MagicMock()
    service.handle_chat.side_effect = RuntimeError("provider unavailable")
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    assert events[0]["type"] == "session.input_transcript.delta"
    assert events[1]["type"] == "error"
    assert events[1]["error"]["message"] == "provider unavailable"


@pytest.mark.asyncio
async def test_source_delta_is_emitted_before_translation_and_survives_failure():
    emitted = []

    async def emit(event):
        emitted.append(event)

    service = MagicMock()

    def fail_translation(_payload):
        assert emitted[0]["type"] == "session.input_transcript.delta"
        raise RuntimeError("provider unavailable")

    service.handle_chat.side_effect = fail_translation
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24(), emit=emit)
    assert emitted[0]["delta"] == "hello world"
    assert emitted[1]["type"] == "error"
