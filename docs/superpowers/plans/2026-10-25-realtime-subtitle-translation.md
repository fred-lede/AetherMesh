# ReaLingo Realtime Subtitle Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ReaLingo-compatible live source and translated subtitles to `/v1/realtime` without returning speech audio.

**Architecture:** Keep WebSocket event handling in `router/realtime_router.py`, store validated session settings and audio state in `RealtimeSession`, and isolate PCM-to-ASR plus chat translation in a small runtime Realtime pipeline. Use existing configured ASR and `RouterService`; do not add provider dependencies.

**Tech Stack:** Python, FastAPI WebSockets, asyncio, NumPy, existing Faster Whisper and `RouterService` adapters, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-realtime-subtitle-translation-design.md`

## Global Constraints

- Router handles protocol only; runtime handles execution.
- Blocking provider calls must not run on the event loop.
- Keep 24 kHz mono PCM16 input and ReaLingo event names unchanged.
- Limit each decoded audio append to 1 MiB and the per-session pending queue to eight frames.
- Use the WebSocket `model` query parameter as the AetherMesh translation model.
- Reject session configuration changes after accepted audio begins.
- Do not emit TTS or `response.audio.delta` in the first release.
- Preserve existing text Realtime event behavior.

## Review Focus

- Invalid JSON, event shapes, base64, odd-length PCM, and appends over 1 MiB must produce a recoverable protocol error.
- `session.close` must flush and deliver final transcript/translation before `session.closed`.
- Cancellation or disconnect must not leak per-session tasks or audio buffers.
- Cancellation must suppress stale results and avoid overlapping provider work within one session.
- Emit the source transcript before waiting for chat translation; retain it if translation fails.
- One slow ASR/translation session must not block other WebSockets or the event loop.
- Existing text session events must remain compatible.

---

### Task 1: Realtime session audio contract

**Files:**
- Modify: `runtime/realtime/realtime_session.py`
- Test: `tests/test_realtime_session.py`

**Interfaces:**
- Consumes: ReaLingo `session.update`, `session.input_audio_buffer.append`, and `session.close` JSON events.
- Produces: validated target language/audio config and protocol errors while keeping legacy text events intact.

- [ ] Add failing tests for the nested ReaLingo target-language update, valid audio append shape, malformed payload, and close state.
- [ ] Run the focused tests and confirm they fail because these Realtime audio fields/events are unsupported.
- [ ] Implement minimal session state and validation for base64 audio, target language, and close without changing text behavior.
- [ ] Run `pytest tests/test_realtime_session.py -x -v` and confirm it passes.

### Task 2: Local subtitle pipeline

**Files:**
- Create: `runtime/realtime/audio_pipeline.py`
- Test: `tests/test_realtime_audio_pipeline.py`

**Interfaces:**
- Consumes: decoded 24 kHz mono PCM16 frames, configured ASR adapter, target language, and chat service.
- Produces: source and translated text deltas, plus a close-time flush result.

- [ ] Add failing tests for 24-to-16 kHz conversion, incremental ASR-to-translation events, close-time flush, and translation provider errors.
- [ ] Run the focused tests and confirm they fail because the pipeline does not exist.
- [ ] Implement a per-session bounded pipeline using existing ASR and chat providers; keep blocking work off the event loop.
- [ ] Run `pytest tests/test_realtime_audio_pipeline.py -x -v` and confirm it passes.

### Task 3: WebSocket event integration

**Files:**
- Modify: `router/realtime_router.py`
- Test: `tests/test_realtime_router.py`

**Interfaces:**
- Consumes: validated audio events from `RealtimeSession` and transcript events from the pipeline.
- Produces: ReaLingo transcript deltas, errors, final completion, and `session.closed`; existing text events remain supported.

- [ ] Add failing WebSocket tests for Bearer authentication, `session.update`, audio append, source/target deltas, ASR disabled/unavailable errors, close flush ordering, cancel, malformed messages, disconnect cleanup, and legacy `response.create`.
- [ ] Run the focused tests and confirm the audio cases fail against the current route.
- [ ] Implement the audio-session lifecycle and cleanup while preserving the current text response path.
- [ ] Run `pytest tests/test_realtime_router.py tests/test_realtime_session.py tests/test_realtime_audio_pipeline.py -x -v` and confirm they pass.

### Task 4: Regression verification

**Files:**
- Test: `tests/test_realtime_session.py`, `tests/test_realtime_audio_pipeline.py`, `tests/test_realtime_router.py`

- [ ] Run all Realtime, StreamingASR, and audio-router tests.
- [ ] Run the repository test suite and report any environment-specific failures separately.
- [ ] Inspect the final diff to verify no audio output/TTS changes and no unrelated edits.
