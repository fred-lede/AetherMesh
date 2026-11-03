from __future__ import annotations

from runtime.realtime.openai_events import (
    anchor_response_sequence,
    input_item_created_event,
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
        previous_item_id="item_1",
    )
    assert [e["type"] for e in events] == [
        "response.created",
        "response.output_item.added",
        "conversation.item.created",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.done",
    ]
    for event in events:
        assert event.get("event_id"), f"missing event_id on {event['type']}"
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

    item_created = events[2]
    assert item_created["item"]["id"] == "msg_1"
    assert item_created["item"]["role"] == "assistant"
    assert item_created["previous_item_id"] == "item_1"

    part_added = events[3]
    assert part_added["item_id"] == "msg_1"
    assert part_added["response_id"] == "resp_1"
    assert part_added["part"]["type"] == "text"

    delta = events[4]
    assert delta["response_id"] == "resp_1"
    assert delta["item_id"] == "msg_1"
    assert delta["delta"] == "你好"

    done = events[5]
    assert done["response_id"] == "resp_1"
    assert done["item_id"] == "msg_1"
    assert done["text"] == "你好"

    item_done = events[6]
    assert item_done["item"]["id"] == "msg_1"
    assert item_done["item"]["status"] == "completed"
    assert item_done["item"]["content"] == [{"type": "text", "text": "你好"}]

    last = events[7]["response"]
    assert last["id"] == "resp_1"
    assert last["status"] == "completed"
    assert last["conversation_id"] == "conv_local"


def test_input_item_created_precedes_transcription_completed():
    created = input_item_created_event("item_1")
    assert created["type"] == "conversation.item.created"
    assert created["item"]["id"] == "item_1"
    assert created["item"]["role"] == "user"
    assert created["item"]["type"] == "message"
    assert created["item"]["status"] == "completed"
    assert created["item"]["content"] == [{"type": "input_audio", "transcript": None}]
    assert created.get("event_id")


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
