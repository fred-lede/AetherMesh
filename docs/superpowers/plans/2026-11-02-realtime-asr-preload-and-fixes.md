# Realtime ASR 預載、黏字修正與斷線日誌 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three issues in `/v1/realtime`: (1) first-audio-frames delay while ASR model lazy-loads; (2) `input/output_transcript.delta` missing text field and `is_final` flag causing subtitle glue-word problem; (3) no disconnect logging to trace remote resets.

**Architecture:** 
- Pre-load ASR adapter in `router/realtime_router.py` on startup via `@app.on_event("startup")` + a module-level `_ASR_READY` flag that `_get_asr_adapter` polls; `process_audio` no longer blocks on first frame's `asyncio.to_thread(get_adapter, "asr")` call.
- Extend `audio_pipeline._translate_transcripts` to emit a `text` field (full ASR segment, not delta) alongside existing `delta` field on every `session.input_transcript.delta` / `session.output_transcript.delta` event, and set `is_final: true` on the last segment of each flush batch (the one produced on `session.close`). No whitespace appended to segments.
- Add `logger.info` with `time.monotonic`-measured lifetime on `WebSocketDisconnect` and on the `finally` block of `realtime_websocket`, so remote resets leave a server-side trace.

**Tech Stack:** Python, FastAPI WebSockets, asyncio, numpy, faster-whisper (existing), pytest.

**Spec:** The three confirmed diagnostics from `AGENTS.md`-adjacent checkpoint:
1. Local `127.0.0.1` repro shows ASR first-frame latency: `session.close` events (transcript deltas) blocked ~9s in first repro (183→191s) because `WhisperModel(...)` initialization ran synchronously inside the thread pool on first audio frame.
2. `audio_pipeline.py` `_translate_transcripts` emits `delta` field = full Whisper segment text, no `text` field, no `is_final` — client ReaLingo treats `delta` as word-level fragment and concatenates → "Oh, andto you. And I".
3. `realtime_router.py` `except WebSocketDisconnect: logger.info("Realtime websocket disconnected")` carries no lifetime, no reason, and 0 occurrences in today's log — meaning either log level filter or path never reached; also `finally` block (line 288-292) silently cleans up without logging.

## Global Constraints

- `router/` handles protocol only; no execution logic; ASR model load is *initialization* (allowed in startup hook), inference stays in `runtime/`.
- No API key values in any code, test, or log line.
- ASR pre-load is best-effort: if `settings.asr_enabled` is false, do nothing; if load fails, set `_ASR_READY = False` and let the existing first-frame error path handle it.
- `_ASR_READY` must be a module-level `bool` in `router/realtime_router.py`, set to `True` by the startup task, read in `process_audio` before calling `get_adapter`.
- `text` field added to `input/output_transcript.delta` events must equal the exact `delta` value (full segment text, no trailing whitespace) — do not add spaces.
- `is_final` flag: `True` on the last segment of a `session.close` flush batch only; `False` (or omitted) on all interim segments.
- Disconnect logging must include: duration (`time.monotonic()` delta from `websocket.accept()` return), whether a `session.close` was received, and the exception type.

## Review Focus

- **ASR pre-load timing:** if `WhisperModel` load takes 30s+, the startup task must not block FastAPI app readiness — it should run in a background task, and `process_audio` must wait for `_ASR_READY` (up to a configurable timeout, default 60s) instead of failing fast with "ASR adapter unavailable".
- **Chinese subtitles:** `text` and `delta` fields for `target_language=zh` must be identical contiguous Chinese strings with no inserted whitespace; test with a mocked `handle_chat` returning Chinese content.
- **Interim vs final segment boundary:** interim ASR windows (triggered by `transcribe_if_ready` on 2.5s accumulation or 0.8s idle) emit `is_final: false`; only `session.close`'s `flush()` batch emits `is_final: true`.
- **First-frame backpressure:** audio queue `maxsize=8` unchanged; pre-load must not add frames to the queue.
- **Log hygiene:** no `logger.info` on every `/health` poll; disconnect log only on actual WS disconnect, not on clean `session.close`.

---

### Task 1: Pre-load ASR adapter on router startup

**Files:**
- Modify: `router/realtime_router.py` (add `_asr_ready` flag + startup task; change `process_audio` to poll it)
- Test: `tests/test_realtime_router.py` (add 2 tests: ready flag gates first frame; unavailable adapter still errors)

**Interfaces:**
- Consumes: `runtime.orchestration.provider_router.adapter` (already imported as `get_adapter` in `realtime_router.py` line 11), `config.settings.settings` (already imported line 10).
- Produces: module-level `_asr_ready: bool` in `router/realtime_router.py`; `realtime_websocket` no longer calls `asyncio.to_thread(get_adapter, "asr")` on first frame if `_asr_ready` is already `True`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_realtime_router.py
# The fixture (realtime_dependencies) provides a FakeASRModel; we patch the
# module-level _preloaded_asr_model + _asr_ready to simulate a completed pre-load,
# and verify that get_adapter is NOT called on the first audio frame.

@pytest.mark.asyncio
async def test_first_audio_frame_uses_preloaded_asr_model(realtime_dependencies, monkeypatch):
    import router.realtime_router as rr
    asr_model = rr.get_adapter("asr")._model  # the fixture's FakeASRModel
    monkeypatch.setattr(rr, "_asr_ready", True)
    monkeypatch.setattr(rr, "_preloaded_asr_model", asr_model)

    # Record if get_adapter is ever called during the websocket lifecycle
    calls: list[str] = []
    orig = rr.get_adapter
    def spy(name: str, *a, **k):
        calls.append(name)
        return orig(name, *a, **k)
    monkeypatch.setattr(rr, "get_adapter", spy)

    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    assert "asr" not in calls, "first audio frame must not trigger lazy ASR load"
    assert any(event["type"] == "session.input_transcript.delta" for event in ws.sent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_realtime_router.py::test_first_audio_frame_uses_preloaded_asr_model -v`
Expected: FAIL — `AttributeError: _asr_ready` (module attribute does not exist yet).

- [ ] **Step 3: Implement `_asr_ready` flag + `preload_asr()`**

In `router/realtime_router.py`:

Add `import threading` to the imports block (lines 1-17).

After the `logger` line (line 17), add:

```python
_asr_ready = False
_preloaded_asr_model: Any = None
_preload_thread: threading.Thread | None = None


def preload_asr() -> None:
    """Load the ASR adapter now so the first realtime audio frame is not blocked.

    Idempotent. Runs on a background thread; _asr_ready flips when done.
    Safe to call from a FastAPI startup hook.
    """
    global _asr_ready, _preloaded_asr_model, _preload_thread
    if _asr_ready:
        return
    if not settings.asr_enabled:
        logger.info("ASR disabled; skipping pre-load")
        return
    if _preload_thread is not None and _preload_thread.is_alive():
        return
    _preload_thread = threading.Thread(target=_preload_asr_worker, daemon=True, name="asr-preload")
    _preload_thread.start()


def _preload_asr_worker() -> None:
    global _asr_ready, _preloaded_asr_model
    t0 = time.monotonic()
    try:
        adapter = get_adapter("asr")
        if adapter is not None:
            _preloaded_asr_model = adapter._model
            _asr_ready = True
            logger.info("ASR pre-loaded in %.1fs", time.monotonic() - t0)
    except Exception:
        logger.exception("ASR pre-load failed; falling back to lazy load on first frame")
        _asr_ready = False
```

(Add `import time` to the imports block too.)

In `process_audio` (lines 144-175), change the lazy-init block:

```python
# OLD (lines 153-156):
# if audio_pipeline is None:
#     asr_adapter = await asyncio.to_thread(get_adapter, "asr")
#     if asr_adapter is None:
#         raise RuntimeError("ASR adapter unavailable")
#     audio_pipeline = RealtimeAudioPipeline(
#         asr_model=asr_adapter._model,
#         ...
# NEW:
if audio_pipeline is None:
    if _asr_ready and _preloaded_asr_model is not None:
        asr_model = _preloaded_asr_model
    else:
        asr_adapter = await asyncio.to_thread(get_adapter, "asr")
        if asr_adapter is None:
            raise RuntimeError("ASR adapter unavailable")
        asr_model = asr_adapter._model
    audio_pipeline = RealtimeAudioPipeline(
        asr_model=asr_model,
        service=service,
        model=session.session.get("model") or "",
        target_language=session.session["target_language"],
    )
```

In `router/openai_router.py`, add a new startup handler next to the existing one (line 88-104):

```python
@app.on_event("startup")
async def _startup_preload_asr() -> None:
    from router.realtime_router import preload_asr
    preload_asr()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_realtime_router.py -x -v -k "first_audio_frame"`
Expected: NEW test PASSES.

- [ ] **Step 5: Run all realtime router tests**

Run: `pytest tests/test_realtime_router.py -x -v`
Expected: ALL PASS (no regression from the new module-level flags).

- [ ] **Step 6: Commit**

```bash
git add router/realtime_router.py router/openai_router.py tests/test_realtime_router.py
git commit -m "realtime: pre-load ASR adapter on startup to avoid first-frame block"
```

---

### Task 2: Add `text` field and `is_final` flag to transcript delta events

**Files:**
- Modify: `runtime/realtime/audio_pipeline.py` (lines 60-104: `_translate_transcripts`)
- Test: `tests/test_realtime_audio_pipeline.py` (update existing assertions + add new tests)

**Interfaces:**
- Consumes: `transcripts: list[dict[str, Any]]` from `StreamingASR.transcribe_if_ready()` / `flush()` — each has keys `type`, `text`, `is_final`, `seg`, `language` (see `providers/streaming_asr.py` lines 126-132).
- Produces: `session.input_transcript.delta` and `session.output_transcript.delta` events now carry:
  - `delta`: unchanged (full segment text, no trailing whitespace)
  - `text`: same value as `delta`
  - `is_final`: `True` if this is the last segment in the current batch AND the batch came from `close()`/`flush()`; `False` otherwise
  - `seg`: the `_seg_counter` value from the transcript dict (already present in `transcripts[i]["seg"]`)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_realtime_audio_pipeline.py
@pytest.mark.asyncio
async def test_transcript_delta_events_carry_text_and_seg_fields():
    """Each input/output_transcript.delta must include text, seg, and is_final fields."""
    model = _model("hello there")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好世界"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    for event in events:
        if event["type"] in ("session.input_transcript.delta", "session.output_transcript.delta"):
            assert "text" in event, f"missing text field in {event['type']}"
            assert "seg" in event, f"missing seg field in {event['type']}"
            assert "is_final" in event, f"missing is_final field in {event['type']}"
            assert event["text"] == event["delta"]
    input_events = [e for e in events if e["type"] == "session.input_transcript.delta"]
    output_events = [e for e in events if e["type"] == "session.output_transcript.delta"]
    assert all(e["is_final"] is False for e in input_events + output_events)
```

```python
# Add to tests/test_realtime_audio_pipeline.py
@pytest.mark.asyncio
async def test_close_emits_is_final_true_on_last_transcript_delta():
    """Segments produced by close()/flush() must carry is_final=True."""
    model = _model("last words")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "最後一句"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    await pipeline.add_audio(_pcm24(4800))
    events = await pipeline.close()
    for event in events:
        if event["type"] in ("session.input_transcript.delta", "session.output_transcript.delta"):
            assert event["is_final"] is True, f"expected is_final=True on close batch, got {event}"
```

```python
# Add to tests/test_realtime_audio_pipeline.py
@pytest.mark.asyncio
async def test_transcript_events_do_not_append_trailing_whitespace():
    """Neither delta nor text must end with a space character."""
    model = _model("hello world")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    for event in events:
        if event["type"] in ("session.input_transcript.delta", "session.output_transcript.delta"):
            assert not event["delta"].endswith(" "), f"delta must not end with space: {event['delta']!r}"
            assert not event["text"].endswith(" "), f"text must not end with space: {event['text']!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_realtime_audio_pipeline.py -x -v -k "text_and_seg or close_emits_is_final or do_not_append"`
Expected: FAIL — `text`, `seg`, `is_final` keys do not exist in emitted events.

- [ ] **Step 3: Implement `text`, `seg`, `is_final` fields**

In `runtime/realtime/audio_pipeline.py`, change `_translate_transcripts`:

```python
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
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following speech into {self._target_language}. Return only the translation.",
                },
                {"role": "user", "content": source},
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
    return events
```

Update `add_audio` to pass `is_final=False` (default) and `close` to pass `is_final=True`:

```python
async def add_audio(self, pcm_24k: bytes, emit=None) -> list[dict[str, Any]]:
    pcm_16k = resample_pcm16_24k_to_16k(pcm_24k)
    await self._stream.add_audio(pcm_16k)
    transcripts = await self._stream.transcribe_if_ready()
    return await self._translate_transcripts(transcripts, emit, is_final=False)

async def close(self, emit=None) -> list[dict[str, Any]]:
    transcripts = await self._stream.flush()
    return await self._translate_transcripts(transcripts, emit, is_final=True)
```

- [ ] **Step 4: Update existing pipeline tests**

Update `tests/test_realtime_audio_pipeline.py` — the existing `test_audio_chunk_emits_source_and_translated_text_deltas` and `test_close_flushes_audio_below_the_streaming_window` assert exact event dicts; they now have extra keys, so relax them to check `delta`/`text`/`type` only:

```python
# Change these assertions from full-dict equality to field checks:
# OLD: assert events == [
#     {"type": "session.input_transcript.delta", "delta": "hello world"},
#     {"type": "session.output_transcript.delta", "delta": "你好世界"},
# ]
# NEW:
assert [e["type"] for e in events] == [
    "session.input_transcript.delta",
    "session.output_transcript.delta",
]
assert events[0]["delta"] == "hello world" and events[0]["text"] == "hello world"
assert events[1]["delta"] == "你好世界" and events[1]["text"] == "你好世界"
```

Do the same for `test_close_flushes_audio_below_the_streaming_window`.

Run: `pytest tests/test_realtime_audio_pipeline.py -x -v`
Expected: ALL PASS.

- [ ] **Step 5: Update router-level tests that assert exact event shapes**

In `tests/test_realtime_router.py`, `test_realingo_audio_append_emits_source_and_target_deltas` (line 78) asserts:
```python
assert [event["delta"] for event in events] == ["hello there", "你好"]
```
This still passes (extra keys don't affect `.get("delta")`). No change needed here.

- [ ] **Step 6: Run full realtime test suite**

Run: `pytest tests/test_realtime_audio_pipeline.py tests/test_realtime_router.py tests/test_realtime_session.py -x -v`
Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add runtime/realtime/audio_pipeline.py tests/test_realtime_audio_pipeline.py
git commit -m "realtime: add text/seg/is_final fields to transcript delta events"
```

---

### Task 3: Log disconnect reason, lifetime, and ASR pre-load duration

**Files:**
- Modify: `router/realtime_router.py` (lines 286-292: `except WebSocketDisconnect` + `finally`)
- Modify: `runtime/realtime/audio_pipeline.py` (add a `logger.warning` when ASR pre-load takes >10s — optional, defer to task 1)
- Test: `tests/test_realtime_router.py` (add 1 test verifying log output via `caplog`)

**Interfaces:**
- Consumes: `time.monotonic()` at `websocket.accept()` return.
- Produces: `logger.info` line on clean `session.close` path (no `session.closed` log on clean close), `logger.info` with duration + exception type on `WebSocketDisconnect`, `logger.info` on `finally` block if a disconnect exception was caught.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_realtime_router.py
import logging

@pytest.mark.asyncio
async def test_disconnect_logs_lifetime_and_reason(realtime_dependencies, caplog):
    """When the client drops the WS connection (no session.close), log lifetime and exception type."""
    caplog.set_level(logging.DEBUG)
    pcm = np.full(24_000, 1000, dtype=np.int16).tobytes()
    # FakeWebSocket raises WebSocketDisconnect when events are exhausted (no session.close)
    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
    ])
    await realtime_router.realtime_websocket(ws)
    # Find the disconnect log line
    disconnect_logs = [r.getMessage() for r in caplog.records if "disconnected" in r.getMessage()]
    assert len(disconnect_logs) >= 1, f"expected a disconnect log, got: {disconnect_logs}"
    log_line = disconnect_logs[0]
    assert "WebSocketDisconnect" in log_line or "lifetime" in log_line, \
        f"disconnect log should mention reason or lifetime, got: {log_line!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_realtime_router.py::test_disconnect_logs_lifetime_and_reason -v`
Expected: FAIL — no log line contains "disconnected" with lifetime/reason.

- [ ] **Step 3: Implement disconnect logging**

In `router/realtime_router.py`, at the top of `realtime_websocket` after `await websocket.accept()` (line 118), capture the start time:

```python
_t0 = time.monotonic()
```

Add `import time` if not already present (check: `time` is not currently imported in `realtime_router.py` — add it to the imports at line 2).

Replace the `except WebSocketDisconnect` and `finally` blocks (lines 286-292):

```python
# OLD:
    except WebSocketDisconnect:
        logger.info("Realtime websocket disconnected")
    finally:
        await stop_audio_processing(cancel=True)
        if response_task is not None and not response_task.done():
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)

# NEW:
    except WebSocketDisconnect as exc:
        lifetime_s = time.monotonic() - _t0
        logger.info(
            "Realtime websocket disconnected (lifetime=%.1fs, close_code=%s, clean=%s)",
            lifetime_s,
            getattr(exc, "code", None),
            getattr(exc, "reason", None) is not None,
        )
    finally:
        await stop_audio_processing(cancel=True)
        if response_task is not None and not response_task.done():
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
```

Also log on clean `session.close` path (inside the `session.close` handler, just before `await websocket.close()` at line 257):

```python
lifetime_s = time.monotonic() - _t0
logger.info("Realtime websocket session.close (lifetime=%.1fs)", lifetime_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_realtime_router.py -x -v -k "disconnect_logs"`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/test_realtime_router.py tests/test_realtime_audio_pipeline.py tests/test_realtime_session.py -x -v`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add router/realtime_router.py tests/test_realtime_router.py
git commit -m "realtime: log disconnect reason, lifetime, and session.close duration"
```

---

### Task 4: Regression verification

**Files:**
- Test: `tests/test_realtime_session.py`, `tests/test_realtime_audio_pipeline.py`, `tests/test_realtime_router.py`

- [ ] **Step 1: Run all realtime + streaming ASR + audio-router tests**

Run: `pytest tests/test_realtime_session.py tests/test_realtime_audio_pipeline.py tests/test_realtime_router.py tests/test_streaming_asr.py -x -v`
(If `test_streaming_asr.py` does not exist, run the three above.)

- [ ] **Step 2: Run full test suite, report env-specific failures separately**

Run: `pytest tests/ -x -q 2>&1 | tail -30`
Report any failures with their cause (env-specific vs. regression).

- [ ] **Step 3: Verify no unintended edits**

Run: `git diff --stat`
Expected: only `router/realtime_router.py`, `router/openai_router.py`, `runtime/realtime/audio_pipeline.py`, `tests/test_realtime_router.py`, `tests/test_realtime_audio_pipeline.py` changed.

- [ ] **Step 4: Local manual smoke test (optional, skip if GPU unavailable)**

If `data/asr_models` is present and GPU available:
Run: `python tests/test_ws_client.py 127.0.0.1:8001 <key>` (or the `rt_repro2.py` script)
Expected: `session.created` → `session.updated` → transcript deltas with `text` field → `session.closed`; no "ASR adapter unavailable" error.
