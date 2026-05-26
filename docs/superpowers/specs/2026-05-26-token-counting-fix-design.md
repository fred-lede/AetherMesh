# Token Counting Fix — Design Spec

**Date**: 2026-05-26
**Status**: Approved

## Problem

Dashboard Provider TOKENS 顯示全為 0。根因：`openai_handler.py` 從未呼叫 `request_metrics.record_request()`，Dashboard 讀的 `request_metrics` 只從 Anthropic 路徑收到資料。同時存在 11 個相關 Gap，涵蓋 adapter、handler、memory、DB 四層。

## Root Cause Map

```
Provider Response → Adapter (extract usage)
                  → Handler (_record_metrics)
                     ├→ request_metrics  (Dashboard 讀這個)
                     ├→ DB token_usage   (user_id != None 才寫)
                     └→ episodic memory  (session 統計)
```

所有 OpenAI-path 請求在 handler 層沒連到 `request_metrics`，所以 Dashboard 全部為 0。

## Approach: `_record_metrics()` Unified Helper

擴展現有 `_record_token_usage()` 為 `_record_metrics()`，一次呼叫寫入三個目標。

### Signature

```python
def _record_metrics(
    self,
    *,
    model: str,
    provider: str,
    endpoint: str,
    streaming: bool,
    request_id: str = "",
    duration_ms: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    user_id: int | None = None,
    api_key_id: int | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
```

### Behavior

1. **Always**: `request_metrics.record_request(RequestRecord(...))`
2. **Always**: `memory_manager.episodic.record(token_count={"prompt_tokens": input_tokens, "completion_tokens": output_tokens}, ...)`
3. **When `user_id is not None`**: call existing `_record_token_usage()` (DB write)

### Why not just add `request_metrics.record_request()` calls?

Three separate write targets (request_metrics, DB, episodic) need the same data. A unified helper prevents:
- Forgetting one target in some path
- Divergent token extraction logic per target
- Inconsistent error/success handling

## Gap Fix Details

### GAP 1: Gemini adapter returns empty `usage`

**File**: `providers/gemini_adapter.py`

**Non-streaming `chat()`**: `_to_chat_completion()` currently hardcodes `"usage": {}`.
Fix: extract from `data.get("usageMetadata", {})`:
- `promptTokenCount` → `prompt_tokens`
- `candidatesTokenCount` → `completion_tokens`
- Sum → `total_tokens`

**Streaming `stream()`**: final chunk includes `usageMetadata`. Extract same fields, emit `usage` dict in the final yield.

### GAP 2: OpenAI handler never calls `request_metrics.record_request()`

**File**: `runtime/orchestration/openai_handler.py`

Replace all `_record_token_usage()` calls and bare `memory_manager.episodic.record()` calls with `_record_metrics()`. This applies to:

- `handle_chat`: success path (L131-147), fallback success paths (L172, L221), error paths
- `handle_streaming_chat`: success path (L409-426), error paths (L311, L332, L364, L380, L391)
- `handle_responses`: all 3 success branches (L524, L553, L567), error paths
- `handle_streaming_responses`: all paths

### GAP 3 & 4: Anthropic streaming `input_tokens=0` and output estimated

**File**: `runtime/orchestration/streaming.py`

`stream_anthropic_with_metrics()` currently:
- `input_tokens=0` always
- `output_tokens=max(1, total_output_tokens // 4)` (char count estimate)

Fix: extract from final chunk `item.get("usage")` at line 207 (already available in `stream_anthropic()`):
- `usage.get("prompt_tokens", 0)` for input_tokens
- `usage.get("completion_tokens", 0)` for output_tokens (fallback to char//4)

Need to pass these values back from `stream_anthropic()` to `stream_anthropic_with_metrics()`. Options:
- Return via a mutable container (e.g., `stats: dict` passed into `stream_anthropic()`)
- Use a closure/shared state

Chosen: pass `stats: dict` into `stream_anthropic()`, populated at finish_reason block.

### GAP 5: Episodic memory missing `token_count`

Solved by `_record_metrics()` — always includes `token_count` parameter.

### GAP 6: Fallback success paths don't record tokens

Solved by `_record_metrics()` — fallback paths now call `_record_metrics()` with usage from fallback response.

### GAP 7: `_record_token_usage` skipped when `user_id=None`

`_record_metrics()` always writes to `request_metrics` and `episodic` regardless of `user_id`. DB write remains conditional on `user_id`.

### GAP 8: Anthropic streaming no episodic record

**File**: `runtime/orchestration/streaming.py`

Add `memory_manager.episodic.record()` in `stream_anthropic_with_metrics()` finally block, after extracting real token counts from stats dict.

### GAP 9: Anthropic fallback success missing episodic record

**File**: `router/anthropic/messages_adapter.py`

Add `memory_manager.episodic.record()` in fallback success paths (currently only `_record_token_usage` is called).

### GAP 10: AgentLoop never records `token_count`

**File**: `runtime/agents/agent_loop.py`

In `llm_call` handler, extract usage from response and pass to `memory_manager.episodic.record()`.

### GAP 11: Ollama Cloud streaming missing `total_tokens`

**File**: `providers/ollama_cloud_adapter.py`

Add `"total_tokens": pe + ec` to streaming usage dict (matches local Ollama adapter pattern).

## Files Changed

| File | Changes |
|------|---------|
| `runtime/orchestration/openai_handler.py` | New `_record_metrics()`, replace all `_record_token_usage` + bare episodic calls |
| `runtime/orchestration/streaming.py` | Pass `stats` dict, extract real tokens, add episodic record |
| `providers/gemini_adapter.py` | Extract `usageMetadata` → `usage` in chat + stream |
| `providers/ollama_cloud_adapter.py` | Add `total_tokens` to streaming usage |
| `router/anthropic/messages_adapter.py` | Add episodic record in fallback success paths |
| `runtime/agents/agent_loop.py` | Add `token_count` to episodic record in llm_call |

## Test Strategy

- `tests/test_orchestration.py`: verify `_record_metrics` writes to all 3 targets
- `tests/test_gemini_adapter.py` (new): verify `usageMetadata` → `usage` conversion
- `tests/test_streaming.py` (or extend existing): verify Anthropic streaming records real input_tokens
- Verify `request_metrics.get_provider_metrics()` returns non-zero tokens after OpenAI-path requests
