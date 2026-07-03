from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest

from providers.streaming_asr import StreamingASR


@pytest.fixture
def mock_model() -> MagicMock:
    m = MagicMock()
    seg = MagicMock()
    seg.text = " hello world "
    seg.language = "en"
    info = MagicMock()
    info.language = "en"
    m.transcribe.return_value = ([seg], info)
    return m


@pytest.fixture
def stream(mock_model: MagicMock) -> StreamingASR:
    return StreamingASR(
        model=mock_model,
        language="",
        interim=False,
        window_seconds=0.25,
        idle_timeout=0.1,
        vad_threshold=0.0,
    )


def _pcm(duration_sec: float = 0.3) -> bytes:
    samples = int(16000 * duration_sec)
    data = (np.sin(2 * np.pi * 440 * np.arange(samples) / 16000) * 3000).astype(np.int16)
    return data.tobytes()


class TestStreamingASR:
    async def test_add_audio_appends_to_buffer(self, stream: StreamingASR) -> None:
        data = _pcm(0.1)
        await stream.add_audio(data)
        assert len(stream._buffer) == len(data)

    async def test_transcribe_if_ready_returns_empty_when_not_enough_audio(
        self, stream: StreamingASR,
    ) -> None:
        await stream.add_audio(_pcm(0.05))
        result = await stream.transcribe_if_ready()
        assert result == []

    async def test_transcribe_if_ready_triggers_on_enough_audio(
        self, stream: StreamingASR, mock_model: MagicMock,
    ) -> None:
        await stream.add_audio(_pcm(0.3))
        results = await stream.transcribe_if_ready()
        assert len(results) == 1
        assert results[0]["type"] == "transcript"
        assert results[0]["text"] == "hello world"
        assert results[0]["is_final"] is True
        assert results[0]["seg"] == 1
        assert results[0]["language"] == "en"

    async def test_transcribe_if_ready_skips_below_vad(
        self, mock_model: MagicMock,
    ) -> None:
        s = StreamingASR(model=mock_model, window_seconds=0.25, vad_threshold=0.5)
        await s.add_audio(_pcm(0.3))
        results = await s.transcribe_if_ready()
        assert results == []

    async def test_transcribe_if_ready_returns_empty_when_closed(
        self, stream: StreamingASR,
    ) -> None:
        await stream.add_audio(_pcm(0.3))
        stream._closed = True
        results = await stream.transcribe_if_ready()
        assert results == []

    async def test_flush_returns_remaining_audio(
        self, stream: StreamingASR, mock_model: MagicMock,
    ) -> None:
        await stream.add_audio(_pcm(0.3))
        results = await stream.flush()
        assert len(results) == 1
        assert results[0]["text"] == "hello world"
        assert stream._closed is True

    async def test_flush_returns_empty_for_no_audio(self, stream: StreamingASR) -> None:
        results = await stream.flush()
        assert results == []

    async def test_set_language(self, stream: StreamingASR) -> None:
        stream.set_language("zh")
        assert stream._language == "zh"

    async def test_seg_counter_increments(
        self, stream: StreamingASR, mock_model: MagicMock,
    ) -> None:
        seg2 = MagicMock()
        seg2.text = "second segment"
        seg2.language = "en"
        mock_model.transcribe.return_value = ([seg2], MagicMock(language="en"))
        await stream.add_audio(_pcm(0.3))
        await stream.transcribe_if_ready()
        assert stream._seg_counter == 1
        await stream.add_audio(_pcm(0.3))
        mock_model.transcribe.return_value = ([seg2], MagicMock(language="en"))
        r2 = await stream.transcribe_if_ready()
        assert r2[0]["seg"] == 2

    async def test_idle_timeout_triggers_transcription(
        self, stream: StreamingASR, mock_model: MagicMock,
    ) -> None:
        await stream.add_audio(_pcm(0.05))
        stream._last_audio_time = 0
        results = await stream.transcribe_if_ready()
        assert len(results) == 1

    async def test_small_audio_returns_empty(
        self, stream: StreamingASR,
    ) -> None:
        await stream.add_audio(b"\x00\x00" * 100)
        results = await stream.transcribe_if_ready()
        assert results == []

    async def test_concurrent_add_and_transcribe(
        self, stream: StreamingASR, mock_model: MagicMock,
    ) -> None:
        async def adder() -> None:
            for _ in range(5):
                await stream.add_audio(_pcm(0.1))
                await asyncio.sleep(0.01)

        async def transcriber() -> None:
            for _ in range(3):
                await stream.transcribe_if_ready()
                await asyncio.sleep(0.02)

        await asyncio.gather(adder(), transcriber())
        assert len(stream._buffer) > 0
