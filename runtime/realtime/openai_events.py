from __future__ import annotations

import uuid
from typing import Any

ANCHOR_PURPOSE = "anchor"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def _event_id() -> str:
    return new_event_id()


def input_item_created_event(item_id: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.created",
        "event_id": _event_id(),
        "item": {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_audio", "transcript": None}],
        },
    }


def transcription_completed_event(item_id: str, transcript: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


def input_item_added_event(item_id: str, previous_item_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "conversation.item.added",
        "event_id": _event_id(),
        "previous_item_id": previous_item_id,
        "item": {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [],
        },
    }


def assistant_item_added_event(item_id: str, previous_item_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "conversation.item.added",
        "event_id": _event_id(),
        "previous_item_id": previous_item_id,
        "item": {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    }


def speech_started_event(item_id: str, audio_start_ms: int = 0) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.speech_started",
        "event_id": _event_id(),
        "item_id": item_id,
        "audio_start_ms": audio_start_ms,
    }


def speech_stopped_event(item_id: str, audio_end_ms: int = 0) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.speech_stopped",
        "event_id": _event_id(),
        "item_id": item_id,
        "audio_end_ms": audio_end_ms,
    }


def input_buffer_committed_event(item_id: str) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.committed",
        "event_id": _event_id(),
        "item_id": item_id,
        "previous_item_id": None,
    }


def audio_transcript_delta_event(text: str, *, response_id: str, item_id: str) -> dict[str, Any]:
    return {
        "type": "response.output_audio_transcript.delta",
        "event_id": _event_id(),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": text,
    }


def audio_transcript_done_event(text: str, *, response_id: str, item_id: str) -> dict[str, Any]:
    return {
        "type": "response.output_audio_transcript.done",
        "event_id": _event_id(),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": text,
    }


def response_sequence_for_text(
    text: str,
    *,
    response_id: str,
    item_id: str,
    conversation_id: str,
    previous_item_id: str | None = None,
    audio_transcripts: bool = False,
) -> list[dict[str, Any]]:
    if audio_transcripts:
        delta: dict[str, Any] = audio_transcript_delta_event(text, response_id=response_id, item_id=item_id)
        done: dict[str, Any] = audio_transcript_done_event(text, response_id=response_id, item_id=item_id)
    else:
        delta = {
            "type": "response.output_text.delta",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }
        done = {
            "type": "response.output_text.done",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        }
    return [
        {
            "type": "response.created",
            "event_id": _event_id(),
            "response": {
                "id": response_id,
                "conversation_id": conversation_id,
                "metadata": {},
                "status": "in_progress",
            },
        },
        {
            "type": "response.output_item.added",
            "event_id": _event_id(),
            "response_id": response_id,
            "output_index": 0,
            "item": {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
        },
        assistant_item_added_event(item_id, previous_item_id),
        {
            "type": "conversation.item.created",
            "event_id": _event_id(),
            "previous_item_id": previous_item_id,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "text", "text": ""},
        },
        delta,
        done,
        {
            "type": "response.output_item.done",
            "event_id": _event_id(),
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "text", "text": text}],
            },
        },
        {
            "type": "response.done",
            "event_id": _event_id(),
            "response": {
                "id": response_id,
                "conversation_id": conversation_id,
                "metadata": {},
                "status": "completed",
            },
        },
    ]


def anchor_response_sequence(response_id: str, metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    echoed = dict(metadata) if isinstance(metadata, dict) else {}
    return [
        {
            "type": "response.created",
            "event_id": _event_id(),
            "response": {
                "id": response_id,
                "conversation_id": None,
                "metadata": echoed,
                "status": "in_progress",
            },
        },
        {
            "type": "response.done",
            "event_id": _event_id(),
            "response": {
                "id": response_id,
                "conversation_id": None,
                "metadata": echoed,
                "status": "completed",
            },
        },
    ]


def is_anchor_request(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    if response.get("conversation") == "none":
        return True
    metadata = response.get("metadata")
    return isinstance(metadata, dict) and metadata.get("purpose") == ANCHOR_PURPOSE
