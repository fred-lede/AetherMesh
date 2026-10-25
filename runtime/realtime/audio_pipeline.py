from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import numpy as np

from providers.streaming_asr import StreamingASR


def resample_pcm16_24k_to_16k(pcm: bytes) -> bytes:
    if len(pcm) % 2:
        raise ValueError("PCM16 audio must contain complete samples")
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    output_count = (len(samples) * 2 + 1) // 3
    positions = np.arange(output_count, dtype=np.float64) * 1.5
    positions = np.minimum(positions, len(samples) - 1)
    resampled = np.interp(positions, np.arange(len(samples)), samples)
    return np.clip(np.rint(resampled), -32768, 32767).astype("<i2").tobytes()


class RealtimeAudioPipeline:
    def __init__(
        self,
        *,
        asr_model: Any,
        service: Any,
        model: str,
        target_language: str,
        window_seconds: float = 2.5,
    ) -> None:
        self._service = service
        self._model = model
        self._target_language = target_language
        self._stream = StreamingASR(
            model=asr_model,
            interim=False,
            window_seconds=window_seconds,
        )

    async def add_audio(
        self,
        pcm_24k: bytes,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        pcm_16k = resample_pcm16_24k_to_16k(pcm_24k)
        await self._stream.add_audio(pcm_16k)
        transcripts = await self._stream.transcribe_if_ready()
        return await self._translate_transcripts(transcripts, emit)

    async def close(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        transcripts = await self._stream.flush()
        return await self._translate_transcripts(transcripts, emit)

    async def _translate_transcripts(
        self,
        transcripts: list[dict[str, Any]],
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        async def publish(event: dict[str, Any]) -> None:
            if emit is None:
                events.append(event)
            else:
                await emit(event)

        for transcript in transcripts:
            source = str(transcript.get("text") or "").strip()
            if not source:
                continue
            await publish({"type": "session.input_transcript.delta", "delta": source})
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Translate the following speech into {self._target_language}. Return only the translation.",
                    },
                    {"role": "user", "content": source},
                ],
                "stream": False,
                "temperature": 0,
                "max_output_tokens": 512,
            }
            try:
                result = await asyncio.to_thread(self._service.handle_chat, payload)
                choices = result.get("choices") or []
                if not choices:
                    raise RuntimeError("translation returned no choices")
                translated = (choices[0].get("message") or {}).get("content")
                if not isinstance(translated, str):
                    raise RuntimeError("translation returned no text")
            except Exception as exc:
                await publish({"type": "error", "error": {"type": "server_error", "message": str(exc)}})
                continue
            if translated:
                await publish({"type": "session.output_transcript.delta", "delta": translated})
        return events
