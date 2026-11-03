from __future__ import annotations

from runtime.realtime.openai_events import (
    anchor_response_sequence,
    is_anchor_request,
    response_sequence_for_text,
    transcription_completed_event,
)


def test_transcription_completed_event_shape():
    event = transcription_completed_event("item_1", "hello there")
    assert event["type"] == "conversation.item.input_audio_transcription.completed"
    assert event["item_id"] == "item_1"
    assert event["transcript"] == "hello there"
    assert event["content_index"] == 0


def test_response_sequence_matches_openai_realtime_contract():
    events = response_sequence_for_text(
        "你好",
        response_id="resp_1",
        item_id="msg_1",
        conversation_id="conv_local",
    )
    assert [e["type"] for e in events] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.done",
    ]
    created = events[0]["response"]
    assert created["id"] == "resp_1"
    assert created["conversation_id"] == "conv_local"
    assert created["metadata"].get("request") is None, "proactive responses must not claim a client request"
    assert created["status"] == "in_progress"

    added = events[1]
    assert added["response_id"] == "resp_1"
    assert added["item"]["id"] == "msg_1"
    assert added["item"]["role"] == "assistant"
    assert added["item"]["type"] == "message"

    delta = events[2]
    assert delta["response_id"] == "resp_1"
    assert delta["item_id"] == "msg_1"
    assert delta["delta"] == "你好"

    done = events[3]
    assert done["response_id"] == "resp_1"
    assert done["item_id"] == "msg_1"
    assert done["text"] == "你好"

    last = events[4]["response"]
    assert last["id"] == "resp_1"
    assert last["status"] == "completed"
    assert last["conversation_id"] == "conv_local"


def test_is_anchor_request_detects_anchor_markers():
    assert is_anchor_request({"conversation": "none", "metadata": {"purpose": "anchor"}})
    assert is_anchor_request({"metadata": {"purpose": "anchor"}})
    assert not is_anchor_request({"metadata": {"request": "sokuji_9"}})
    assert not is_anchor_request({})
    assert not is_anchor_request(None)


def test_anchor_response_sequence_is_out_of_band_and_echoes_metadata():
    metadata = {"purpose": "anchor"}
    events = anchor_response_sequence("resp_anchor", metadata)
    assert [e["type"] for e in events] == ["response.created", "response.done"]
    created = events[0]["response"]
    assert created["id"] == "resp_anchor"
    assert created["conversation_id"] is None
    assert created["metadata"] == {"purpose": "anchor"}
    done = events[1]["response"]
    assert done["id"] == "resp_anchor"
    assert done["conversation_id"] is None
    assert done["metadata"] == {"purpose": "anchor"}
    assert done["status"] == "completed"
