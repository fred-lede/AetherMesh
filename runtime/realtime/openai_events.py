from __future__ import annotations

import uuid
from typing import Any

ANCHOR_PURPOSE = "anchor"


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def transcription_completed_event(item_id: str, transcript: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": _event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
    }


def response_sequence_for_text(
    text: str,
    *,
    response_id: str,
    item_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
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
        {
            "type": "response.output_text.delta",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_text.done",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
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
