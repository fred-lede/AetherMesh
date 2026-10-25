# ReaLingo Realtime Subtitle Translation

## Intent

Let ReaLingo use AetherMesh `/v1/realtime` for live source transcription and target-language subtitles over the OpenAI-compatible WebSocket contract.

## Goals

- Authenticate WebSocket connections with `Authorization: Bearer <key>`.
- Accept `session.update` with ReaLingo's audio transcription and target-language settings.
- Accept base64 24 kHz mono PCM16 audio appends, emit source and translated transcript deltas, and flush the final input on `session.close` before `session.closed`.
- Keep each WebSocket session isolated and responsive to cancellation, malformed messages, and disconnects.
- Preserve the existing text Realtime flow.

## Non-goals

- Returning speech audio or emitting `response.audio.delta` in this first version.
- Proxying requests to an upstream Realtime service.
- Changing ReaLingo or adding provider configuration.

## Approaches

- **Local ASR, chat translation, and TTS:** reuses AetherMesh providers and is local-first, but adds audio conversion and speech synthesis work.
- **Upstream Realtime proxy:** delegates recognition and translation but requires an upstream WebSocket service and credentials, and adds proxy/error-mapping complexity.
- **Local subtitle pipeline (selected):** uses AetherMesh ASR and chat translation but omits TTS for the first release. It satisfies the selected subtitle scope with the existing runtime.

## Architecture and constraints

The router owns WebSocket protocol and event formatting. A focused runtime Realtime audio pipeline owns PCM conversion, ASR buffering, and translation. ASR and chat provider calls, including lazy ASR segment iteration, must not block the asyncio event loop. The server emits source text as soon as ASR returns it, then emits target `session.*_transcript.delta` events; a translation failure must not discard a successful source transcript. On close, flush accepted audio before `session.closed`. No audio delta is emitted.

Each append is limited to 1 MiB of decoded PCM16 and queued work is bounded, so the client should split long audio into smaller append events. The ReaLingo `model` URL parameter selects the AetherMesh chat model used for translation; it must name a model AetherMesh can route. Session settings become immutable after the first accepted audio append. Cancellation suppresses queued/in-flight output; synchronous provider work already running may finish in its worker thread before the same session accepts new audio.

## Acceptance

Verify Bearer authentication, session update parsing, valid and invalid audio appends, ongoing source/target deltas, final flush ordering, cancellation and disconnect cleanup, bounded per-session work, and unchanged text Realtime behavior.

## Builder decisions

Keep the current text route intact and add only the protocol fields and dispatch needed for ReaLingo. Use configured AetherMesh ASR and chat providers; the client-supplied transcription model name is accepted as protocol metadata, while ASR provider selection remains controlled by AetherMesh settings. Use the WebSocket `model` query value as the translation chat model.
