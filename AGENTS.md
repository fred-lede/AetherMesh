# AetherMesh — Agent Context

## Project Identity
AetherMesh (v5.0.0) is a **local-first AI Runtime OS Kernel** — multi-provider, multi-GPU, agent-oriented. Presents OpenAI/Anthropic-compatible APIs while handling routing, execution, tool loops, memory, security, and orchestration.

## Architecture (3 strict layers)
- **`router/`** — Protocol only (FastAPI routes + format conversion). No execution logic.
- **`runtime/`** — Execution only (no FastAPI, no format conversion). Core kernel.
- **`providers/`** — Provider only (API wrappers for Ollama, OpenAI, Gemini, NVIDIA NIM, Ollama Cloud).

## Code Conventions
- **Python**: 4-space indent, `from __future__ import annotations` at top, type hints on all public APIs. No docstrings/comments unless the task explicitly requires them.
- **Imports**: standard lib → third-party → internal, groups separated by blank line, absolute imports.
- **Tests**: `pytest` with `asyncio_mode = auto`. Heavy use of `MagicMock`/`monkeypatch`/`patch`. No external dependencies. File name: `tests/test_<module>.py`.
- **Config**: `config/settings.py` — `Settings` dataclass loaded from `.env` + YAML. Use `settings.<field>`, never `settings.get()`.
- **Logging**: module-level `logger = logging.getLogger("<path>")`. Use structured logging with `%s` formatting, not f-strings.

## Key Files & Locations

| Area | File | Notes |
|---|---|---|
| Chat completions | `runtime/orchestration/openai_handler.py` | 1566 lines. `RouterService` class with `handle_chat`, `handle_streaming_chat`, `handle_responses`, `handle_streaming_responses`. Memory wired in all success/error paths. |
| Tool loop | `runtime/responses/tool_loop.py` | Multi-turn tool loop for non-OpenAI providers. |
| DAG engine | `runtime/orchestration/graph_executor.py` | Async DAG runner with parallel groups. |
| Agent orchestration | `runtime/agents/agent_loop.py` | `AgentLoop` class — 4 handler factories, `GraphExecutor`-backed. |
| Routing | `runtime/orchestration/provider_router.py` | `ROUTE_PREFIXES`, `provider_for_model`, `resolve_provider`. |
| Intelligence | `runtime/intelligence/execution_selector.py` | `rerank()` for provider scoring. |
| Memory | `runtime/memory/episodic_memory.py` | `record(**kwargs)` — accepts `model`, `provider`, `duration_ms`, `success`, `token_count`, `error`, `session_id`, `task_summary`. |
| Multi-agent | `runtime/multi_agent/coordinator.py` | `delegate()`, `fan_out()`, `orchestrate()`. |
| Settings | `config/settings.py` | `Settings` dataclass. Key methods: `model_registry()`, `model_alias_prefix()`, `strip_model_route_prefix()`, `resolve_model_alias()`. |
| SSE streaming | `router/streaming_router.py` | `ensure_ascii=False` (Chinese text). |
| Adapters | `providers/ollama_adapter.py`, `openai_adapter.py`, `gemini_adapter.py`, `nvidia_nim_adapter.py`, `ollama_cloud_adapter.py` | All have `chat()`, `stream()`, `responses()`. |
| Protocol | `router/anthropic/messages_adapter.py` | Anthropic `v1/messages` endpoint. |
| Protocol | `router/openai/chat_adapter.py` | OpenAI `v1/chat/completions` endpoint. |
| Tool exec | `runtime/tools/tool_executor.py` | `ToolExecutor.execute()` — sync. Use `asyncio.to_thread` when calling from async. Optional `sandbox_manager` param. |
| Sandbox | `runtime/security/sandbox/` | `SandboxProfile`, `PlatformSandbox`, `MacSandbox`, `LinuxSandbox`, `SandboxManager`. 21 tests. |
| Test helpers | `tests/test_agent_loop.py` | Pattern: handler factories are module-level lambdas. GraphExecutor with registered handlers. |

## Test Commands
```bash
pytest tests/ -x -v              # single file, stop on first fail
pytest tests/test_agent_loop.py -x -v
pytest tests/test_orchestration.py -x -v -k "memory"
```

296 tests passing. 25 pre-existing env-specific failures: 22 in `test_dashboard_auth.py`, 1 in `test_capabilities.py` (IP mismatch), 2 in `test_security.py` (`AIIH_API_KEY` env var). 3 skipped (Linux-only sandbox tests on macOS).

## Critical Gotchas
- **`asyncio.to_thread`** required for all synchronous adapter calls (`adapter.chat()`, `ToolExecutor.execute()`, `eval()`).
- **Tool format**: OpenAI nested (`function.wrapper`). Normalize with `_ensure_openai_tools()` for providers that accept flat format.
- **Routing**: `routing_engine.route()` returns `RoutingDecision` with `.provider`, `.worker`, `.candidates`. Always strip prefix via `settings.strip_model_route_prefix()` before matching.
- **GPU workers**: streaming path must `_finalize_request()` in `finally` to release workers.
- **Memory recording**: call `memory_manager.episodic.record()` before raising exceptions in error paths so failures are captured.
- **`eval()` in conditional**: restricted `__builtins__` — safe for DAG condition predicates only.
- **Server tools**: `messages_adapter.py` evaluates policy first; converter drops server tools for OpenAI providers.
- **Ollama**: `JSONDecodeError` in stream lines — log and skip, don't crash.

## Recent Work (May 2026)
- Memory `record()` wired into `openai_handler.py` chat + streaming paths (all success/error/fallback paths).
- `AgentLoop` rewritten as functional DAG executor with 4 async handlers (`llm_call`, `tool_call`, `conditional`, `agent_call`).
- Fixed: Settings `.get()` → attribute access, NVIDIA NIM tool format, SSE Chinese escaping, model prefix routing, Ollama `tool_choice`, worker leak, streaming tool loop (double call, dropped events, missing SSE events).
- Fixed: `from_content_with_thinking()` in non-streaming Anthropic path, dead code removal, metadata passthrough, empty choices guard, max_turns data loss, Gemini fake → real streaming, Gemini tool call parsing.
- Fixed: Auth bypass for `/api/metrics/` paths (401 on localhost metrics endpoints). `/health`, `/docs`, `/openapi.json`, `/.well-known/` also bypass auth now.
- Fixed: Python 3.14 `UnboundLocalError` in `tool_executor.py` (removed `import asyncio.tasks` inside function body).
- Execution Environment Abstraction: 5 commits — `SandboxProfile`, `PlatformSandbox` ABC, `MacSandbox`, `LinuxSandbox`, `SandboxManager` + ToolExecutor/Settings integration. 21 new tests.
- 6 new E2E tests for streaming tool loop, 5 for AgentLoop, 2 for memory wiring.

## What's NOT Done (Roadmap)
- Browser Tools (Playwright builtin) — low priority, `web_fetch` covers most needs
- Skills System — ABI layer exists, this would add declarative registration/discovery
- Execution Environment Abstraction (process sandbox) — DONE. `runtime/security/sandbox/` with `SandboxProfile`, `PlatformSandbox` ABC, `MacSandbox` (subprocess + RLIMIT), `LinuxSandbox` (subprocess + unshare NEWNET), `SandboxManager` (policy vs process dispatch). `ToolExecutor` accepts optional `sandbox_manager`. `Settings.sandbox_profiles` + `sandbox_manager()` factory.
- Vector Search / RAG — low priority, TF-IDF semantic memory sufficient
- End-to-end test for `/v1/responses` with tools (Phase 22 item)
