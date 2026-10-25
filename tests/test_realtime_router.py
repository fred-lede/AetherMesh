from __future__ import annotations

import base64
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

import router.realtime_router as realtime_router


class FakeASRModel:
    def transcribe(self, samples, **kwargs):
        return [SimpleNamespace(text="hello there")], SimpleNamespace(language="en")


class FakeWebSocket:
    def __init__(self, events, *, api_key="test-key", model="local-chat-model"):
        self.headers = {"authorization": f"Bearer {api_key}"}
        self.query_params = {"model": model}
        self.events = iter(events)
        self.sent = []
        self.close_code = None

    async def accept(self):
        pass

    async def receive(self):
        try:
            event = next(self.events)
        except StopIteration:
            raise WebSocketDisconnect(code=1000)
        if isinstance(event, tuple) and event[0] == "delay":
            import asyncio

            await asyncio.sleep(event[1])
            return await self.receive()
        if isinstance(event, str):
            return {"text": event}
        return {"text": json.dumps(event)}

    async def send_json(self, event):
        self.sent.append(event)

    async def close(self, code=1000):
        self.close_code = code


@pytest.fixture
def realtime_dependencies(monkeypatch):
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}

    async def verify(_key):
        return True

    monkeypatch.setattr(realtime_router, "_verify_ws_api_key", verify)
    monkeypatch.setattr(realtime_router.settings, "asr_enabled", True)
    monkeypatch.setattr(realtime_router, "get_adapter", lambda _name: SimpleNamespace(_model=FakeASRModel()), raising=False)
    from runtime.orchestration import openai_handler

    monkeypatch.setattr(openai_handler, "RouterService", lambda: service)
    return service


@pytest.mark.asyncio
async def test_realingo_audio_append_emits_source_and_target_deltas(realtime_dependencies):
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    events = [event for event in ws.sent if event["type"].endswith("transcript.delta")]
    assert [event["type"] for event in events] == [
        "session.input_transcript.delta",
        "session.output_transcript.delta",
    ]
    assert [event["delta"] for event in events] == ["hello there", "你好"]
    assert not any(event["type"] == "response.audio.delta" for event in ws.sent)
    assert ws.sent[-1]["type"] == "session.closed"
    assert ws.close_code == 1000


@pytest.mark.asyncio
async def test_session_close_flushes_last_audio_before_closed(realtime_dependencies):
    pcm = np.full(24_000, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert [event["type"] for event in ws.sent[-3:]] == [
        "session.input_transcript.delta",
        "session.output_transcript.delta",
        "session.closed",
    ]


@pytest.mark.asyncio
async def test_invalid_json_returns_error_and_session_remains_usable(realtime_dependencies):
    ws = FakeWebSocket(["{", {"type": "session.close"}])
    await realtime_router.realtime_websocket(ws)
    assert ws.sent[1]["type"] == "error"
    assert ws.sent[-1]["type"] == "session.closed"


@pytest.mark.asyncio
async def test_non_string_event_type_returns_recoverable_error(realtime_dependencies):
    ws = FakeWebSocket([{"type": []}, {"type": "session.close"}])
    await realtime_router.realtime_websocket(ws)
    assert ws.sent[1]["type"] == "error"
    assert "type must be a string" in ws.sent[1]["error"]["message"]


@pytest.mark.asyncio
async def test_session_update_after_audio_starts_is_rejected(realtime_dependencies):
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        ("delay", 0.05),
        {"type": "session.update", "session": {"audio": {"output": {"language": "es"}}}},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any("cannot change after audio input starts" in event.get("error", {}).get("message", "") for event in ws.sent)


@pytest.mark.asyncio
async def test_response_cancel_acknowledges_without_ending_session(realtime_dependencies):
    ws = FakeWebSocket([{"type": "response.cancel"}, {"type": "session.close"}])
    await realtime_router.realtime_websocket(ws)
    assert ws.sent[1]["type"] == "response.cancelled"
    assert ws.sent[-1]["type"] == "session.closed"


@pytest.mark.asyncio
async def test_response_cancel_discards_queued_audio_translation(realtime_dependencies):
    service = realtime_dependencies
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "response.cancel"},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event["type"] == "response.cancelled" for event in ws.sent)
    service.handle_chat.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_cleans_up_queued_audio(realtime_dependencies):
    service = realtime_dependencies
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
    ])
    await realtime_router.realtime_websocket(ws)
    assert not any(event["type"].endswith("transcript.delta") for event in ws.sent)
    service.handle_chat.assert_not_called()


@pytest.mark.asyncio
async def test_full_audio_queue_returns_error_and_keeps_reading_cancel(realtime_dependencies):
    pcm = np.full(480, 1000, dtype=np.int16).tobytes()
    events = [{"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}}]
    events.extend(
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        for _ in range(9)
    )
    events.extend([{"type": "response.cancel"}, {"type": "session.close"}])
    ws = FakeWebSocket(events)
    await realtime_router.realtime_websocket(ws)
    assert any("audio processing queue is full" in event.get("error", {}).get("message", "") for event in ws.sent)
    assert any(event["type"] == "response.cancelled" for event in ws.sent)


@pytest.mark.asyncio
async def test_cancelled_translation_does_not_overlap_new_work(realtime_dependencies):
    service = realtime_dependencies
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    calls = 0

    def slow_translation(_payload):
        nonlocal active, maximum_active, calls
        with lock:
            active += 1
            calls += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.15)
        with lock:
            active -= 1
        return {"choices": [{"message": {"content": "你好"}}]}

    service.handle_chat.side_effect = slow_translation
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        ("delay", 0.05),
        {"type": "response.cancel"},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        ("delay", 0.2),
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert maximum_active == 1
    assert calls == 2
    assert any("cancelled audio processing is still stopping" in event.get("error", {}).get("message", "") for event in ws.sent)


@pytest.mark.asyncio
async def test_asr_disabled_returns_error(realtime_dependencies, monkeypatch):
    monkeypatch.setattr(realtime_router.settings, "asr_enabled", False)
    pcm = np.full(24_000, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event.get("error", {}).get("message") == "ASR is not enabled" for event in ws.sent)


@pytest.mark.asyncio
async def test_unavailable_asr_adapter_returns_error(realtime_dependencies, monkeypatch):
    monkeypatch.setattr(realtime_router, "get_adapter", lambda _name: None)
    pcm = np.full(24_000, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event.get("error", {}).get("message") == "ASR adapter unavailable" for event in ws.sent)


@pytest.mark.asyncio
async def test_response_cancel_marks_active_text_response_cancelled(realtime_dependencies):
    service = realtime_dependencies
    service.handle_chat.side_effect = lambda _payload: __import__("time").sleep(0.1) or {"choices": []}
    ws = FakeWebSocket([
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello"}]}},
        {"type": "response.create"},
        {"type": "response.cancel"},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event.get("response", {}).get("status") == "cancelled" for event in ws.sent)
    assert any(event["type"] == "response.cancelled" for event in ws.sent)


@pytest.mark.asyncio
async def test_second_response_create_is_rejected_while_one_is_active(realtime_dependencies):
    service = realtime_dependencies
    service.handle_chat.side_effect = lambda _payload: __import__("time").sleep(0.1) or {"choices": []}
    ws = FakeWebSocket([
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello"}]}},
        {"type": "response.create"},
        {"type": "response.create"},
        {"type": "response.cancel"},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event.get("error", {}).get("message") == "a response is already in progress" for event in ws.sent)


@pytest.mark.asyncio
async def test_cancel_during_text_deltas_emits_cancelled_completion(realtime_dependencies):
    service = realtime_dependencies
    service.handle_chat.return_value = {"choices": [{"message": {"content": "word " * 500}}]}
    ws = FakeWebSocket([
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello"}]}},
        {"type": "response.create"},
        ("delay", 0.05),
        {"type": "response.cancel"},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event["type"] == "response.text.delta" for event in ws.sent)
    assert any(event.get("response", {}).get("status") == "cancelled" for event in ws.sent)


@pytest.mark.asyncio
async def test_text_response_create_still_works(realtime_dependencies):
    ws = FakeWebSocket([
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello"}]}},
        {"type": "response.create"},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    event_types = [event["type"] for event in ws.sent]
    assert "response.text.delta" in event_types
    assert "response.done" in event_types
    assert event_types[-1] == "session.closed"


@pytest.mark.asyncio
async def test_bearer_auth_failure_closes_socket(monkeypatch):
    async def reject(_key):
        return False

    monkeypatch.setattr(realtime_router, "_verify_ws_api_key", reject)
    ws = FakeWebSocket([], api_key="wrong")
    await realtime_router.realtime_websocket(ws)
    assert ws.sent[0]["error"]["type"] == "authentication_error"
    assert ws.close_code == 4001
