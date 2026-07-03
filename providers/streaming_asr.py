from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("providers.streaming_asr")


class StreamingASR:
    """Streaming ASR with ring buffer, VAD, and sliding window Whisper.

    Audio format: 16 kHz, 16-bit signed PCM, mono.
    """

    def __init__(
        self,
        model: Any,
        language: str = "",
        interim: bool = False,
        task: str = "transcribe",
        window_seconds: float = 2.5,
        idle_timeout: float = 0.8,
        vad_threshold: float = 0.01,
    ) -> None:
        self._model = model
        self._language = language
        self._interim = interim
        self._task = task
        self._window_samples = int(window_seconds * 16000)
        self._idle_timeout = idle_timeout
        self._vad_threshold = vad_threshold

        self._buffer = bytearray()
        self._processed_up_to = 0
        self._last_audio_time = 0.0
        self._seg_counter = 0
        self._closed = False

        self._lock = asyncio.Lock()

    async def add_audio(self, pcm_bytes: bytes) -> None:
        async with self._lock:
            self._buffer.extend(pcm_bytes)
            self._last_audio_time = time.monotonic()

    def set_language(self, language: str) -> None:
        self._language = language

    async def transcribe_if_ready(self) -> list[dict[str, Any]]:
        async with self._lock:
            if self._closed:
                return []
            new_bytes = len(self._buffer) - self._processed_up_to
            new_samples = new_bytes // 2
            idle = time.monotonic() - self._last_audio_time
            if new_samples < self._window_samples and idle < self._idle_timeout:
                return []
            if new_bytes < 320:
                return []
            chunk = bytes(self._buffer[self._processed_up_to :])
            self._processed_up_to = len(self._buffer)
        return await self._transcribe(chunk)

    async def flush(self) -> list[dict[str, Any]]:
        async with self._lock:
            remaining = bytes(self._buffer[self._processed_up_to :])
            self._buffer.clear()
            self._processed_up_to = 0
            self._closed = True
        if len(remaining) < 320:
            return []
        return await self._transcribe(remaining)

    async def _transcribe(self, audio_bytes: bytes) -> list[dict[str, Any]]:
        if len(audio_bytes) < 320:
            return []
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < self._vad_threshold:
            return []
        kwargs: dict[str, Any] = {"task": self._task}
        if self._language:
            kwargs["language"] = self._language
        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            None, lambda: self._model.transcribe(samples, **kwargs)
        )
        seg_list = list(segments) if segments is not None else []
        results: list[dict[str, Any]] = []
        for seg in seg_list:
            self._seg_counter += 1
            lang = getattr(seg, "language", None) or getattr(info, "language", None) or self._language
            results.append({
                "type": "transcript",
                "text": seg.text.strip(),
                "is_final": True,
                "seg": self._seg_counter,
                "language": lang or "",
            })
        return results
