from __future__ import annotations

from runtime.realtime.openai_events import (
    anchor_response_sequence,
    assistant_item_added_event,
    audio_transcript_delta_event,
    audio_transcript_done_event,
    input_buffer_committed_event,
    input_item_added_event,
    input_item_created_event,
    is_anchor_request,
    response_sequence_for_text,
    speech_started_event,
    speech_stopped_event,
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
        "conversation.item.added",
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

    item_added = events[2]
    assert item_added["item"]["id"] == "msg_1"
    assert item_added["item"]["role"] == "assistant"
    assert item_added["previous_item_id"] == "item_1"

    item_created = events[3]
    assert item_created["item"]["id"] == "msg_1"
    assert item_created["item"]["role"] == "assistant"
    assert item_created["previous_item_id"] == "item_1"

    part_added = events[4]
    assert part_added["item_id"] == "msg_1"
    assert part_added["response_id"] == "resp_1"
    assert part_added["part"]["type"] == "text"

    delta = events[5]
    assert delta["response_id"] == "resp_1"
    assert delta["item_id"] == "msg_1"
    assert delta["delta"] == "你好"

    done = events[6]
    assert done["response_id"] == "resp_1"
    assert done["item_id"] == "msg_1"
    assert done["text"] == "你好"

    item_done = events[7]
    assert item_done["item"]["id"] == "msg_1"
    assert item_done["item"]["status"] == "completed"
    assert item_done["item"]["content"] == [{"type": "text", "text": "你好"}]

    last = events[8]["response"]
    assert last["id"] == "resp_1"
    assert last["status"] == "completed"
    assert last["conversation_id"] == "conv_local"


def test_response_sequence_audio_transcripts_variant_swaps_transcript_frames():
    events = response_sequence_for_text(
        "你好",
        response_id="resp_1",
        item_id="msg_1",
        conversation_id="conv_local",
        previous_item_id="item_1",
        audio_transcripts=True,
    )
    types = [e["type"] for e in events]
    assert types == [
        "response.created",
        "response.output_item.added",
        "conversation.item.added",
        "conversation.item.created",
        "response.content_part.added",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
        "response.output_item.done",
        "response.done",
    ]
    delta = events[5]
    assert delta["response_id"] == "resp_1"
    assert delta["item_id"] == "msg_1"
    assert delta["delta"] == "你好"
    done = events[6]
    assert done["response_id"] == "resp_1"
    assert done["item_id"] == "msg_1"
    assert done["transcript"] == "你好"
    assert "text" not in done, "audio transcript done must not duplicate the text field"


def test_speech_event_shapes():
    started = speech_started_event("item_in_1", audio_start_ms=250)
    assert started["type"] == "input_audio_buffer.speech_started"
    assert started["item_id"] == "item_in_1"
    assert started["audio_start_ms"] == 250
    assert started.get("event_id")

    stopped = speech_stopped_event("item_in_1", audio_end_ms=1750)
    assert stopped["type"] == "input_audio_buffer.speech_stopped"
    assert stopped["item_id"] == "item_in_1"
    assert stopped["audio_end_ms"] == 1750
    assert stopped.get("event_id")


def test_input_buffer_committed_event_shape():
    event = input_buffer_committed_event("item_in_1")
    assert event["type"] == "input_audio_buffer.committed"
    assert event["item_id"] == "item_in_1"
    assert event["previous_item_id"] is None
    assert event.get("event_id")


def test_input_item_added_event_shape():
    event = input_item_added_event("item_1")
    assert event["type"] == "conversation.item.added"
    assert event["previous_item_id"] is None
    assert event["item"]["id"] == "item_1"
    assert event["item"]["role"] == "user"
    assert event["item"]["type"] == "message"
    assert event["item"]["status"] == "completed"
    assert event["item"]["content"] == []
    assert event.get("event_id")


def test_assistant_item_added_event_shape():
    event = assistant_item_added_event("msg_1", "item_1")
    assert event["type"] == "conversation.item.added"
    assert event["previous_item_id"] == "item_1"
    assert event["item"]["id"] == "msg_1"
    assert event["item"]["role"] == "assistant"
    assert event["item"]["type"] == "message"
    assert event["item"]["status"] == "in_progress"
    assert event.get("event_id")


def test_audio_transcript_delta_and_done_event_shapes():
    delta = audio_transcript_delta_event("你好", response_id="resp_1", item_id="msg_1")
    assert delta["type"] == "response.output_audio_transcript.delta"
    assert delta["response_id"] == "resp_1"
    assert delta["item_id"] == "msg_1"
    assert delta["output_index"] == 0
    assert delta["content_index"] == 0
    assert delta["delta"] == "你好"
    assert delta.get("event_id")

    done = audio_transcript_done_event("你好", response_id="resp_1", item_id="msg_1")
    assert done["type"] == "response.output_audio_transcript.done"
    assert done["response_id"] == "resp_1"
    assert done["item_id"] == "msg_1"
    assert done["transcript"] == "你好"
    assert done.get("event_id")


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
