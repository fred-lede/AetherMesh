from __future__ import annotations

import base64
import binascii
import time
import uuid
from typing import Any

MAX_AUDIO_FRAME_BYTES = 1_048_576



def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_item_text(item: dict[str, Any]) -> str | None:
    if not isinstance(item, dict) or item.get("type") != "message":
        return None
    parts: list[str] = []
    for part in item.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif isinstance(part, str):
            parts.append(part)
    return " ".join(parts).strip() if parts else None


class RealtimeSession:
    def __init__(self, model: str = "") -> None:
        self.session = {
            "id": f"realtime_{uuid.uuid4().hex[:24]}",
            "model": model or "",
            "instructions": "",
            "modalities": ["text"],
            "tools": [],
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "target_language": "",
            "transcription_model": "",
        }
        self.items: list[dict[str, Any]] = []
        self.audio_buffer = bytearray()
        self.audio_closed = False

    def handle_event(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = data.get("type")
        if not isinstance(event_type, str) or not event_type:
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "event type must be a string"}}]
        if event_type == "session.update":
            return [self._apply_session_update(data)]
        if event_type == "conversation.item.create":
            return [self._add_item(data)]
        if event_type == "ping":
            return [{"type": "pong", "time": _now_ms()}]
        if event_type == "session.input_audio_buffer.append":
            return self._append_audio(data)
        if event_type == "session.close":
            self.audio_closed = True
            return []
        if event_type in {"input_audio_buffer.append", "input_audio_buffer.commit"}:
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "audio input is not supported"}}]
        return []

    def _append_audio(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        encoded = data.get("audio")
        if not isinstance(encoded, str):
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "audio must be base64-encoded PCM16"}}]
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "audio must be valid base64"}}]
        if len(pcm) % 2:
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "audio must contain complete PCM16 samples"}}]
        if len(pcm) > MAX_AUDIO_FRAME_BYTES:
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "audio frame exceeds 1 MiB; split audio across append events"}}]
        if self.audio_closed:
            return [{"type": "error", "event_id": data.get("event_id"), "error": {"type": "invalid_request_error", "message": "audio session is already closed"}}]
        self.audio_buffer.extend(pcm)
        return []

    def take_audio(self) -> bytes:
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio

    def _apply_session_update(self, data: dict[str, Any]) -> dict[str, Any]:
        updates = data.get("session") or {}
        if not isinstance(updates, dict):
            return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.update requires a 'session' object"}}
        for key in ("model", "instructions", "temperature", "max_output_tokens"):
            if key in updates:
                self.session[key] = updates[key]
        audio = updates.get("audio")
        if audio is not None:
            if not isinstance(audio, dict):
                return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.audio must be an object"}}
            input_audio = audio.get("input") or {}
            if not isinstance(input_audio, dict):
                return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.audio.input must be an object"}}
            transcription = input_audio.get("transcription") or {}
            if not isinstance(transcription, dict):
                return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.audio.input.transcription must be an object"}}
            transcription_model = transcription.get("model")
            if transcription_model is not None:
                if not isinstance(transcription_model, str) or not transcription_model.strip():
                    return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.audio.input.transcription.model must be a non-empty string"}}
                self.session["transcription_model"] = transcription_model.strip()
            output = audio.get("output") or {}
            if not isinstance(output, dict):
                return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.audio.output must be an object"}}
            language = output.get("language")
            if language is not None:
                if not isinstance(language, str) or not language.strip():
                    return {"type": "error", "error": {"type": "invalid_request_error", "message": "session.audio.output.language must be a non-empty string"}}
                self.session["target_language"] = language.strip()
        if "modalities" in updates and isinstance(updates["modalities"], list):
            self.session["modalities"] = updates["modalities"]
        if "tools" in updates and isinstance(updates["tools"], list):
            self.session["tools"] = updates["tools"]
        return {"type": "session.updated", "session": dict(self.session)}

    def _add_item(self, data: dict[str, Any]) -> dict[str, Any]:
        item = data.get("item") or {}
        if not isinstance(item, dict):
            return {"type": "error", "error": {"type": "invalid_request_error", "message": "conversation.item.create requires an 'item' object"}}
        if item.get("type") == "message" and isinstance(item.get("content"), list):
            stored = dict(item)
            stored["id"] = stored.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
            self.items.append(stored)
            return {"type": "conversation.item.created", "item": stored}
        if item.get("type") in {"function_call", "function_call_output"}:
            stored = dict(item)
            stored["id"] = stored.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
            self.items.append(stored)
            return {"type": "conversation.item.created", "item": stored}
        return {"type": "error", "error": {"type": "invalid_request_error", "message": f"unsupported item type '{item.get('type')}'"}}

    def build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        instructions = str(self.session.get("instructions") or "").strip()
        if instructions:
            messages.append({"role": "system", "content": instructions})
        for item in self.items:
            text = _extract_item_text(item)
            if text is None:
                continue
            role = "assistant" if item.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": text})
        return messages
