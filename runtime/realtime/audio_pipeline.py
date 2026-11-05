from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import numpy as np

from providers.streaming_asr import StreamingASR
from runtime.realtime.openai_events import (
    input_buffer_committed_event,
    speech_started_event,
    speech_stopped_event,
)
from runtime.realtime.turn_detector import TurnDetector


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
        instructions: str = "",
        idle_timeout: float = 0.8,
    ) -> None:
        self._service = service
        self._model = model
        self._target_language = target_language
        self._instructions = str(instructions or "").strip()
        self._stream = StreamingASR(
            model=asr_model,
            interim=False,
            window_seconds=window_seconds,
            idle_timeout=idle_timeout,
        )
        self._detector = TurnDetector()
        self._turn = 0
        self._turn_item_id = ""

    async def add_audio(
        self,
        pcm_24k: bytes,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        events = await self._feed_detector(pcm_24k, emit)
        pcm_16k = resample_pcm16_24k_to_16k(pcm_24k)
        await self._stream.add_audio(pcm_16k)
        transcripts = await self._stream.transcribe_if_ready()
        events.extend(await self._translate_transcripts(transcripts, emit, is_final=False))
        return events

    async def close(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        events = await self._end_speaking_turn(emit)
        transcripts = await self._stream.flush()
        events.extend(await self._translate_transcripts(transcripts, emit, is_final=True))
        return events

    async def flush_idle(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        transcripts = await self._stream.transcribe_if_ready()
        return await self._translate_transcripts(transcripts, emit, is_final=True)

    async def _publish_all(
        self,
        events: list[dict[str, Any]],
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        for event in events:
            if emit is None:
                continue
            await emit(event)

    async def _feed_detector(
        self,
        pcm_24k: bytes,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for speech_event in self._detector.feed(pcm_24k):
            if speech_event == "start":
                self._turn += 1
                self._turn_item_id = f"item_in_{self._turn}"
                events.append(speech_started_event(self._turn_item_id))
            else:
                item_id = self._turn_item_id or f"item_in_{self._turn + 1}"
                self._turn_item_id = ""
                events.append(speech_stopped_event(item_id))
                events.append(input_buffer_committed_event(item_id))
        await self._publish_all(events, emit)
        return events

    async def _end_speaking_turn(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> list[dict[str, Any]]:
        if not self._detector.speaking:
            return []
        self._detector.end_turn()
        item_id = self._turn_item_id or f"item_in_{self._turn + 1}"
        self._turn_item_id = ""
        events = [speech_stopped_event(item_id), input_buffer_committed_event(item_id)]
        await self._publish_all(events, emit)
        return events

    async def _translate_transcripts(
        self,
        transcripts: list[dict[str, Any]],
        emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        is_final: bool = False,
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
            seg = int(transcript.get("seg", 0))
            await publish({
                "type": "session.input_transcript.delta",
                "delta": source,
                "text": source,
                "seg": seg,
                "is_final": is_final,
            })
            if self._instructions:
                system_prompt = self._instructions
            else:
                system_prompt = f"Translate the following speech into {self._target_language}. Return only the translation."
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
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
                await publish({
                    "type": "error",
                    "error": {"type": "server_error", "message": str(exc)},
                    "seg": seg,
                    "is_final": is_final,
                })
                continue
            if translated:
                await publish({
                    "type": "session.output_transcript.delta",
                    "delta": translated,
                    "text": translated,
                    "seg": seg,
                    "is_final": is_final,
                })
        return events
