from __future__ import annotations

from runtime.realtime.realtime_session import RealtimeSession, _extract_item_text
from router.realtime_router import _chunk_text


def test_session_created_defaults():
    session = RealtimeSession()
    assert session.session["modalities"] == ["text"]
    assert session.session["model"] == ""
    assert session.items == []


def test_session_update():
    session = RealtimeSession(model="m1")
    events = session.handle_event(
        {"type": "session.update", "session": {"model": "m2", "instructions": "be brief", "temperature": 0.2}}
    )
    assert events[0]["type"] == "session.updated"
    assert session.session["model"] == "m2"
    assert session.session["instructions"] == "be brief"
    assert session.session["temperature"] == 0.2


def test_conversation_item_create_text():
    session = RealtimeSession()
    events = session.handle_event(
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello"}]}}
    )
    assert events[0]["type"] == "conversation.item.created"
    assert session.items[0]["content"][0]["text"] == "hello"


def test_conversation_item_create_unsupported():
    session = RealtimeSession()
    events = session.handle_event({"type": "conversation.item.create", "item": {"type": "audio"}})
    assert events[0]["type"] == "error"
    assert session.items == []


def test_audio_buffer_not_supported():
    session = RealtimeSession()
    events = session.handle_event({"type": "input_audio_buffer.append"})
    assert events[0]["type"] == "error"


def test_ping_pong():
    session = RealtimeSession()
    events = session.handle_event({"type": "ping"})
    assert events[0]["type"] == "pong"
    assert "time" in events[0]


def test_build_messages_with_instructions():
    session = RealtimeSession(model="m")
    session.handle_event({"type": "session.update", "session": {"instructions": "be brief"}})
    session.handle_event({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hi"}]}})
    messages = session.build_messages()
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_multiple_turns():
    session = RealtimeSession()
    for role, text in [("user", "q1"), ("assistant", "a1"), ("user", "q2")]:
        session.handle_event(
            {"type": "conversation.item.create", "item": {"type": "message", "role": role, "content": [{"type": "text", "text": text}]}}
        )
    messages = session.build_messages()
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [m["content"] for m in messages] == ["q1", "a1", "q2"]


def test_build_messages_ignores_non_text_items():
    session = RealtimeSession()
    session.handle_event({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "real"}]}})
    session.handle_event({"type": "conversation.item.create", "item": {"type": "function_call_output", "output": "x"}})
    messages = session.build_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "real"


def test_extract_item_text_mixed_parts():
    item = {"type": "message", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _extract_item_text(item) == "a b"


def test_build_messages_empty():
    session = RealtimeSession()
    assert session.build_messages() == []


def test_chunk_text():
    assert _chunk_text("one two three four", 2) == ["one two", "three four"]
    assert _chunk_text("", 2) == [""]
