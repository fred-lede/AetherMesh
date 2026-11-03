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
    def __init__(self):
        self.samples_seen = 0

    def transcribe(self, samples, **kwargs):
        self.samples_seen += len(samples)
        return [SimpleNamespace(text="hello there")], SimpleNamespace(language="en")


class VersionedASRManager:
    def __init__(self) -> None:
        self._version = 0
        self.models: list[FakeASRModel] = [FakeASRModel()]

    def version(self) -> int:
        return self._version

    def get_adapter(self):
        return SimpleNamespace(_model=self.models[-1])


class FakeWebSocket:
    def __init__(self, events, *, api_key="test-key", model="local-chat-model", protocols=None):
        self.headers = {"authorization": f"Bearer {api_key}"}
        if protocols:
            self.headers["sec-websocket-protocol"] = ", ".join(protocols)
        self.query_params = {"model": model}
        self.scope = {"subprotocols": list(protocols or [])}
        self.events = iter(events)
        self.sent = []
        self.close_code = None
        self.accepted_subprotocol = None

    async def accept(self, subprotocol=None):
        self.accepted_subprotocol = subprotocol

    async def receive(self):
        try:
            event = next(self.events)
        except StopIteration:
            raise WebSocketDisconnect(code=1000)
        if isinstance(event, tuple) and event[0] == "delay":
            import asyncio

            await asyncio.sleep(event[1])
            return await self.receive()
        if isinstance(event, tuple) and event[0] == "call":
            event[1]()
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
    manager = VersionedASRManager()
    monkeypatch.setattr(realtime_router, "asr_manager", manager, raising=False)
    asr_model = manager.models[0]
    from runtime.orchestration import openai_handler

    monkeypatch.setattr(openai_handler, "RouterService", lambda: service)
    service.asr_model = asr_model
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
    flushed_types = [event["type"] for event in ws.sent if event["type"] in (
        "session.input_transcript.delta",
        "session.output_transcript.delta",
        "response.created",
        "response.done",
    )]
    assert flushed_types == [
        "session.input_transcript.delta",
        "session.output_transcript.delta",
        "response.created",
        "response.done",
    ]
    assert ws.sent[-1]["type"] == "session.closed"


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
async def test_audio_queue_applies_backpressure_without_dropping_frames(realtime_dependencies):
    service = realtime_dependencies
    pcm = np.full(2_400, 1000, dtype=np.int16).tobytes()
    events = [{"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}}]
    events.extend(
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
        for _ in range(130)
    )
    events.append({"type": "session.close"})
    ws = FakeWebSocket(events)
    await realtime_router.realtime_websocket(ws)
    assert not any("audio processing queue is full" in event.get("error", {}).get("message", "") for event in ws.sent)
    assert service.asr_model.samples_seen == 130 * 1600
    assert any(event["type"] == "session.input_transcript.delta" for event in ws.sent)
    assert any(event["type"] == "session.output_transcript.delta" for event in ws.sent)
    assert ws.sent[-1]["type"] == "session.closed"


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
    monkeypatch.setattr(realtime_router.asr_manager, "get_adapter", lambda: None)
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


@pytest.mark.asyncio
async def test_pipeline_built_once_from_manager_adapter(realtime_dependencies, monkeypatch):
    manager = realtime_router.asr_manager
    calls: list[int] = []
    orig_get_adapter = manager.get_adapter

    def spy():
        calls.append(1)
        return orig_get_adapter()

    monkeypatch.setattr(manager, "get_adapter", spy)

    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any(event["type"] == "session.input_transcript.delta" for event in ws.sent)
    assert len(calls) == 1, "pipeline must be built once from the manager adapter, not per frame"


@pytest.mark.asyncio
async def test_asr_version_change_rebuilds_pipeline_with_new_model(realtime_dependencies, monkeypatch):
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    manager = realtime_router.asr_manager
    calls: list[int] = []
    orig_get_adapter = manager.get_adapter

    def spy():
        calls.append(1)
        return orig_get_adapter()

    monkeypatch.setattr(manager, "get_adapter", spy)
    new_model = FakeASRModel()

    def switch_engine():
        manager.models.append(new_model)
        manager._version += 1

    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        ("delay", 0.05),
        ("call", switch_engine),
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    events = [event for event in ws.sent if event["type"] == "session.input_transcript.delta"]
    assert len(events) == 3
    assert manager.models[0].samples_seen > 0
    assert new_model.samples_seen > 0
    assert len(calls) == 2, "pipeline must be rebuilt once at the switch, then reused for later frames"
    assert ws.sent[-1]["type"] == "session.closed"
    assert ws.close_code == 1000


@pytest.mark.asyncio
async def test_disconnect_logs_lifetime_and_reason(realtime_dependencies, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    pcm = np.full(24_000, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
    ])
    await realtime_router.realtime_websocket(ws)
    disconnect_logs = [r.getMessage() for r in caplog.records if "Realtime websocket" in r.getMessage()]
    assert disconnect_logs, f"expected a disconnect log, got none"
    log_line = disconnect_logs[-1]
    assert "lifetime" in log_line, f"disconnect log should report lifetime, got: {log_line!r}"


@pytest.mark.asyncio
async def test_subprotocol_key_authenticates_and_realtime_is_echoed(realtime_dependencies, monkeypatch):
    seen: dict[str, str] = {}

    async def verify(key):
        seen["key"] = key
        return key == "secret-1"

    monkeypatch.setattr(realtime_router, "_verify_ws_api_key", verify)
    ws = FakeWebSocket(
        [{"type": "session.close"}],
        api_key="",
        protocols=["realtime", "openai-insecure-api-key.secret-1"],
    )
    await realtime_router.realtime_websocket(ws)
    assert seen["key"] == "secret-1"
    assert ws.accepted_subprotocol == "realtime"
    assert ws.sent[-1]["type"] == "session.closed"


@pytest.mark.asyncio
async def test_instructions_only_session_passes_gate_via_standard_append(realtime_dependencies):
    service = realtime_dependencies
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    instructions = "Translate the speech into English. Return only the translation."
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"instructions": instructions}},
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert not any(event["type"] == "error" for event in ws.sent)
    assert any(event["type"] == "session.output_transcript.delta" for event in ws.sent)
    payload = service.handle_chat.call_args.args[0]
    assert payload["messages"][0] == {"role": "system", "content": instructions}


@pytest.mark.asyncio
async def test_audio_input_without_target_language_or_instructions_is_rejected(realtime_dependencies):
    pcm = np.full(24_000, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert any("before audio input" in event.get("error", {}).get("message", "") for event in ws.sent)


@pytest.mark.asyncio
async def test_audio_stream_also_emits_openai_transcription_and_response_events(realtime_dependencies):
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert ws.sent[0].get("event_id"), "session.created must carry an event_id"
    types = [event["type"] for event in ws.sent]
    assert "conversation.item.input_audio_transcription.completed" in types
    completed = next(e for e in ws.sent if e["type"] == "conversation.item.input_audio_transcription.completed")
    assert completed["transcript"] == "hello there"
    assert completed["item_id"] == "item_1"
    assert completed.get("event_id")
    order = [
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.done",
    ]
    indices = [types.index(t) for t in order]
    assert indices == sorted(indices), f"canonical event order broken: {[t for t in types if t in order]}"
    input_item = next(e for e in ws.sent if e["type"] == "conversation.item.created" and e["item"]["role"] == "user")
    assert input_item["item"]["id"] == "item_1"
    assistant_item = next(e for e in ws.sent if e["type"] == "conversation.item.created" and e["item"]["role"] == "assistant")
    assert assistant_item["item"]["id"] == "msg_1"
    assert assistant_item["previous_item_id"] == "item_1"
    out_delta = next(e for e in ws.sent if e["type"] == "response.output_text.delta")
    assert out_delta["delta"] == "你好"
    assert out_delta["item_id"] == "msg_1"
    out_done = next(e for e in ws.sent if e["type"] == "response.output_text.done")
    assert out_done["text"] == "你好"


@pytest.mark.asyncio
async def test_anchor_response_create_replies_out_of_band_without_llm(realtime_dependencies):
    service = realtime_dependencies
    ws = FakeWebSocket([
        {"type": "response.create", "response": {"conversation": "none", "output_modalities": ["text"], "metadata": {"purpose": "anchor"}}},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    service.handle_chat.assert_not_called()
    created = next(e for e in ws.sent if e["type"] == "response.created")
    assert created["response"]["conversation_id"] is None
    assert created["response"]["metadata"] == {"purpose": "anchor"}
    done = next(e for e in ws.sent if e["type"] == "response.done")
    assert done["response"]["conversation_id"] is None
    assert done["response"]["metadata"] == {"purpose": "anchor"}


@pytest.mark.asyncio
async def test_response_create_echoes_metadata_request_and_emits_output_text_deltas(realtime_dependencies):
    ws = FakeWebSocket([
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}},
        {"type": "response.create", "response": {"metadata": {"request": "sokuji_7"}}},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    created = next(e for e in ws.sent if e["type"] == "response.created")
    assert created["response"]["metadata"] == {"request": "sokuji_7"}
    assert created["response"]["conversation_id"]
    assert created.get("event_id"), "response.created must carry an event_id"
    assistant_items = [e for e in ws.sent if e["type"] == "conversation.item.created" and e["item"]["role"] == "assistant"]
    assert assistant_items, "assistant item must be announced via conversation.item.created"
    assert assistant_items[0].get("event_id")
    assert assistant_items[0]["previous_item_id"], "assistant item must name the user item it follows"
    done = next(e for e in ws.sent if e["type"] == "response.done")
    assert done["response"]["metadata"] == {"request": "sokuji_7"}
    assert done["response"]["status"] == "completed"
    delta = next(e for e in ws.sent if e["type"] == "response.output_text.delta")
    assert delta["delta"] == "你好"
    assert delta.get("event_id")
    assert any(e["type"] == "response.text.delta" for e in ws.sent), "legacy dialect deltas must be kept"
