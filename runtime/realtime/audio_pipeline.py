from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Awaitable, Callable

import numpy as np

from providers.streaming_asr import StreamingASR
from runtime.realtime.openai_events import (
    input_buffer_committed_event,
    speech_started_event,
    speech_stopped_event,
)
from runtime.realtime.turn_detector import TurnDetector

# How many earlier segments the translator may look at. Two is what the local pipeline
# uses, and it is enough to resolve a pronoun without dragging the whole talk in.
CONTEXT_SEGMENTS = 2

# How many terminology pairs one segment may carry. The service asks for at most 1000
# overall; a single segment needs far fewer to be useful.
MAX_TERMS_PER_SEGMENT = 8

# The languages the realtime endpoint names by code. A model asked to translate "into zh"
# does worse than one told "into Chinese", so the code is turned into words here. The names
# are the ones the clients themselves display, so both sides say the same thing.
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "ar": "Arabic",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "ur": "Urdu",
    "nb": "Norwegian",
    "sv": "Swedish",
    "da": "Danish",
    "he": "Hebrew",
    "fi": "Finnish",
    "pl": "Polish",
    "is": "Icelandic",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "yue": "Cantonese",
    "el": "Greek",
    "af": "Afrikaans",
    "ast": "Asturian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "et": "Estonian",
    "gl": "Galician",
    "gu": "Gujarati",
    "hr": "Croatian",
    "hu": "Hungarian",
    "jv": "Javanese",
    "kk": "Kazakh",
    "kn": "Kannada",
    "ky": "Kyrgyz",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sw": "Swahili",
    "tg": "Tajik",
    "uk": "Ukrainian",
}


def language_label(code: str) -> str:
    """The target language in words, with the code kept for the ones it helps with."""
    code = str(code or "").strip()
    if not code:
        return "the target language"
    name = LANGUAGE_NAMES.get(code.lower())
    if not name:
        return code
    return name if name.lower() == code.lower() else f"{name} ({code})"


def translation_instruction(target_language: str) -> str:
    """The default system prompt: one segment, natural wording, nothing but the translation."""
    return (
        f"Translate the following speech into {language_label(target_language)}. "
        "The input may be a fragment from a live conversation. Translate only the current "
        "segment; use the recent context only to understand it. Preserve its meaning and "
        "tone, and use natural wording. Output only the translation, with no explanation "
        "or quotation marks."
    )


def translation_user_text(source: str, context: list[str], terms: dict[str, str]) -> str:
    """The per-segment text: terminology, then the context, then the segment itself."""
    parts: list[str] = []
    usable = [
        (str(s).strip(), str(t).strip())
        for s, t in list(terms.items())[:MAX_TERMS_PER_SEGMENT]
        if str(s).strip() and str(t).strip()
    ]
    if usable:
        lines = "\n".join(f"{source} => {target}" for source, target in usable)
        parts.append(f"Terminology for current segment:\n{lines}\n")
    if context:
        joined = "\n".join(context)
        parts.append(f"Recent context (for meaning only; do not translate it):\n{joined}\n")
    parts.append(f"Current segment to translate:\n{source}")
    return "\n".join(parts)


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
        terms: dict[str, str] | None = None,
    ) -> None:
        self._service = service
        self._model = model
        self._target_language = target_language
        self._instructions = str(instructions or "").strip()
        self._terms = {str(k): str(v) for k, v in (terms or {}).items()}
        # The segments before this one, so a fragment can be read in context.
        self._context: deque[str] = deque(maxlen=CONTEXT_SEGMENTS)
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
                system_prompt = translation_instruction(self._target_language)
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": translation_user_text(source, list(self._context), self._terms)},
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
            # The segment joins the context whether or not it translated: the next one is
            # read against what was actually said, not against what the model made of it.
            self._context.append(source)
        return events
