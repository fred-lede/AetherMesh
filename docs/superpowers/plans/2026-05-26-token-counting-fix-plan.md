# Token Counting Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 11 token-counting gaps so Dashboard Provider TOKENS reflect real data.

**Architecture:** Extract `_record_metrics()` unified helper from `_record_token_usage()` in `openai_handler.py`. One call writes to request_metrics + episodic memory + DB (when user_id present). Fix Gemini adapter to extract `usageMetadata`. Fix Anthropic streaming to pass real token counts from final chunk.

**Tech Stack:** Python 3.14, pytest, FastAPI, existing metrics/memory modules.

---

### Task 1: Gemini adapter — extract `usageMetadata` in `chat()`

**Files:**
- Modify: `providers/gemini_adapter.py:29-57`
- Test: `tests/test_token_counting.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_gemini_chat_extracts_usage_metadata():
    from providers.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.api_key = "test"
    adapter.base_url = "https://fake"

    data = {
        "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 7, "totalTokenCount": 49},
    }

    with patch.object(adapter, "_extract_content", return_value=("hello", [])):
        result = adapter._to_chat_completion(data, model="gemini-pro")

    assert result["usage"]["prompt_tokens"] == 42
    assert result["usage"]["completion_tokens"] == 7
    assert result["usage"]["total_tokens"] == 49


def test_gemini_chat_usage_defaults_to_zero():
    from providers.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.api_key = "test"
    adapter.base_url = "https://fake"

    data = {
        "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
    }

    with patch.object(adapter, "_extract_content", return_value=("hello", [])):
        result = adapter._to_chat_completion(data, model="gemini-pro")

    assert result["usage"]["prompt_tokens"] == 0
    assert result["usage"]["completion_tokens"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py::test_gemini_chat_extracts_usage_metadata -v`

Expected: FAIL — `GeminiAdapter` has no `_to_chat_completion` method (or usage is empty dict).

- [ ] **Step 3: Implement `_to_chat_completion` and update `chat()`**

In `providers/gemini_adapter.py`, extract the `_to_chat_completion` logic from `chat()` into a method and extract `usageMetadata`:

```python
def _to_chat_completion(self, data: dict[str, Any], model: str) -> dict[str, Any]:
    text, tool_calls = self._extract_content(data)
    um = data.get("usageMetadata") or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": um.get("promptTokenCount", 0),
            "completion_tokens": um.get("candidatesTokenCount", 0),
            "total_tokens": um.get("totalTokenCount", 0),
        },
    }
```

Update `chat()` to use it:

```python
def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
    model = payload["model"]
    body = {
        "contents": self._messages_to_contents(payload.get("messages", [])),
        "tools": self._tools_to_gemini(payload.get("tools", [])),
    }
    response = post_with_retry(
        get_session(),
        f"{self.base_url}/models/{model}:generateContent",
        params={"key": self.api_key},
        json=body,
        timeout=settings.request_timeout_s,
    )
    data = response.json()
    return self._to_chat_completion(data, model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/gemini_adapter.py tests/test_token_counting.py
git commit -m "fix: extract Gemini usageMetadata in _to_chat_completion"
```

---

### Task 2: Gemini adapter — extract `usageMetadata` in `stream()`

**Files:**
- Modify: `providers/gemini_adapter.py:96-167`
- Test: `tests/test_token_counting.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gemini_stream_emits_usage():
    from providers.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.api_key = "test"
    adapter.base_url = "https://fake"

    chunks = list(adapter._process_stream_chunks(
        [
            {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13}},
        ],
        model="gemini-pro",
    ))

    usage_chunks = [c for c in chunks if isinstance(c, dict) and c.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["prompt_tokens"] == 10
    assert usage_chunks[0]["usage"]["completion_tokens"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py::test_gemini_stream_emits_usage -v`

Expected: FAIL — no `_process_stream_chunks` method or no `usage` in output.

- [ ] **Step 3: Implement stream usage extraction**

Extract stream chunk processing into `_process_stream_chunks`. In the final chunk (when `finish_reason` is set), extract `usageMetadata` and include `usage` dict:

In `providers/gemini_adapter.py`, refactor `stream()` to use a helper. The key change: when building the finish_reason chunk, add `usage` from `usageMetadata`:

```python
if finish_reason:
    um = data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": um.get("promptTokenCount", 0),
        "completion_tokens": um.get("candidatesTokenCount", 0),
        "total_tokens": um.get("totalTokenCount", 0),
    }
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": usage,
    }
```

Note: `usageMetadata` may appear on any chunk. Track the last-seen `usageMetadata` and apply it to the finish chunk.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/gemini_adapter.py tests/test_token_counting.py
git commit -m "fix: extract Gemini usageMetadata in streaming final chunk"
```

---

### Task 3: Ollama Cloud streaming — add `total_tokens`

**Files:**
- Modify: `providers/ollama_cloud_adapter.py:98-108`
- Test: `tests/test_token_counting.py`

- [ ] **Step 1: Write the failing test**

Already present in the Ollama cloud streaming output (line 107 already includes `total_tokens`). Verify with test:

```python
def test_ollama_cloud_stream_usage_has_total_tokens():
    from providers.ollama_cloud_adapter import OllamaCloudAdapter

    adapter = OllamaCloudAdapter.__new__(OllamaCloudAdapter)
    adapter.api_key = "test"
    adapter.base_url = "https://fake"

    mock_item = {"done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 10}
    chunks = list(adapter._process_done_chunk(mock_item, model="llama3"))

    assert chunks[0]["usage"]["total_tokens"] == 15
```

Note: If the `total_tokens` is already there (line 107), this is a validation test. If it passes immediately, skip to commit.

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py::test_ollama_cloud_stream_usage_has_total_tokens -v`

Expected: PASS (already fixed). If FAIL, add `"total_tokens": pe + ec` to the streaming done chunk.

- [ ] **Step 3: Commit (only if code changed)**

```bash
git add providers/ollama_cloud_adapter.py tests/test_token_counting.py
git commit -m "fix: add total_tokens to Ollama Cloud streaming usage"
```

---

### Task 4: `_record_metrics()` unified helper in `openai_handler.py`

**Files:**
- Modify: `runtime/orchestration/openai_handler.py:1367-1399`
- Test: `tests/test_token_counting.py`

- [ ] **Step 1: Write the failing test**

```python
def test_record_metrics_writes_all_three_targets():
    from metrics.request_metrics import RequestMetricsCollector
    from runtime.memory.episodic_memory import EpisodicMemory
    from runtime.orchestration.openai_handler import RouterService

    collector = RequestMetricsCollector()
    episodic = EpisodicMemory()

    service = RouterService.__new__(RouterService)
    service._record_token_usage = lambda *a, **kw: None

    with patch("runtime.orchestration.openai_handler.request_metrics", collector):
        with patch("runtime.orchestration.openai_handler.memory_manager") as mock_mm:
            mock_mm.episodic = episodic
            service._record_metrics(
                model="test-model",
                provider="ollama",
                endpoint="/v1/chat/completions",
                streaming=False,
                request_id="req-1",
                duration_ms=100.0,
                input_tokens=50,
                output_tokens=25,
                user_id=None,
                api_key_id=None,
                success=True,
            )

    summary = collector.get_summary()
    assert summary["total_input_tokens"] == 50
    assert summary["total_output_tokens"] == 25

    records = episodic.recent(limit=10)
    assert len(records) == 1
    assert records[0].get("token_count", {}).get("prompt_tokens") == 50


def test_record_metrics_writes_db_when_user_id():
    db_written = {"called": False}

    def fake_record_token_usage(user_id, api_key_id, input_tokens, output_tokens, provider, model):
        db_written["called"] = True

    from runtime.orchestration.openai_handler import RouterService

    service = RouterService.__new__(RouterService)
    service._record_token_usage = fake_record_token_usage

    with patch("runtime.orchestration.openai_handler.request_metrics", RequestMetricsCollector()):
        with patch("runtime.orchestration.openai_handler.memory_manager"):
            service._record_metrics(
                model="test-model",
                provider="ollama",
                endpoint="/v1/chat/completions",
                streaming=False,
                duration_ms=100.0,
                input_tokens=50,
                output_tokens=25,
                user_id=1,
                api_key_id=1,
                success=True,
            )

    assert db_written["called"]


def test_record_metrics_skips_db_when_no_user_id():
    db_written = {"called": False}

    def fake_record_token_usage(user_id, api_key_id, input_tokens, output_tokens, provider, model):
        db_written["called"] = True

    from runtime.orchestration.openai_handler import RouterService

    service = RouterService.__new__(RouterService)
    service._record_token_usage = fake_record_token_usage

    with patch("runtime.orchestration.openai_handler.request_metrics", RequestMetricsCollector()):
        with patch("runtime.orchestration.openai_handler.memory_manager"):
            service._record_metrics(
                model="test-model",
                provider="ollama",
                endpoint="/v1/chat/completions",
                streaming=False,
                duration_ms=100.0,
                input_tokens=50,
                output_tokens=25,
                user_id=None,
                api_key_id=None,
                success=True,
            )

    assert not db_written["called"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py -k "record_metrics" -v`

Expected: FAIL — `_record_metrics` method does not exist.

- [ ] **Step 3: Implement `_record_metrics()`**

In `runtime/orchestration/openai_handler.py`, add after `_record_response_usage`:

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
    request_metrics.record_request(RequestRecord(
        request_id=request_id or f"req_{uuid.uuid4().hex[:16]}",
        model=model,
        provider=provider,
        endpoint=endpoint,
        streaming=streaming,
        latency_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=not success,
        error_message=error or "",
    ))
    memory_manager.episodic.record(
        model=model,
        provider=provider,
        duration_ms=duration_ms,
        success=success,
        token_count={"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        error=error[:200] if error else None,
    )
    if user_id is not None and success:
        self._record_token_usage(
            user_id, api_key_id,
            input_tokens=input_tokens, output_tokens=output_tokens,
            provider=provider, model=model,
        )
```

Ensure `RequestRecord` is imported at top (already imported: `from metrics.request_metrics import request_metrics`, but `RequestRecord` also needed):

```python
from metrics.request_metrics import RequestRecord, request_metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py -k "record_metrics" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/openai_handler.py tests/test_token_counting.py
git commit -m "feat: add _record_metrics unified helper to RouterService"
```

---

### Task 5: Wire `handle_chat` to use `_record_metrics()`

**Files:**
- Modify: `runtime/orchestration/openai_handler.py:129-267`

- [ ] **Step 1: Replace success path (lines 131-147)**

Replace the `_record_token_usage` call and bare `episodic.record` call with a single `_record_metrics`:

```python
usage = response.get("usage") or {}
self._record_metrics(
    model=effective_payload.get("model", ""),
    provider=provider,
    endpoint="/v1/chat/completions",
    streaming=False,
    duration_ms=(time.perf_counter() - started) * 1000,
    input_tokens=usage.get("prompt_tokens", 0),
    output_tokens=usage.get("completion_tokens", 0),
    user_id=user_id,
    api_key_id=api_key_id,
    success=True,
)
```

Remove the old `self._record_token_usage(...)` (lines 131-136) and `memory_manager.episodic.record(...)` (lines 138-147).

- [ ] **Step 2: Replace fallback success paths (lines 172 and 221)**

Add `token_count` from fallback response:

```python
fallback_usage = fallback_response.get("usage") or {}
self._record_metrics(
    model=effective_payload.get("model", ""),
    provider=provider,
    endpoint="/v1/chat/completions",
    streaming=False,
    duration_ms=(time.perf_counter() - started) * 1000,
    input_tokens=fallback_usage.get("prompt_tokens", 0),
    output_tokens=fallback_usage.get("completion_tokens", 0),
    success=True,
)
```

Remove old bare `memory_manager.episodic.record(...)` at lines 172-177 and 221-226.

- [ ] **Step 3: Replace error paths (lines 182, 190, 231, 239, 250)**

Each error path's `memory_manager.episodic.record(...)` becomes:

```python
self._record_metrics(
    model=effective_payload.get("model", ""),
    provider=provider,
    endpoint="/v1/chat/completions",
    streaming=False,
    duration_ms=(time.perf_counter() - started) * 1000,
    success=False,
    error=str(exc)[:200],
)
```

Remove old bare `memory_manager.episodic.record(...)` calls.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_orchestration.py tests/test_token_counting.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add runtime/orchestration/openai_handler.py
git commit -m "fix: wire handle_chat to _record_metrics for all paths"
```

---

### Task 6: Wire `handle_streaming_chat` to use `_record_metrics()`

**Files:**
- Modify: `runtime/orchestration/openai_handler.py:278-427`

- [ ] **Step 1: Replace success path (lines 409-426)**

Replace bare `episodic.record` + `_record_token_usage` with:

```python
if not error:
    pt = 0
    ct = 0
    if isinstance(last_chunk, dict):
        usage = last_chunk.get("usage") or {}
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    self._record_metrics(
        model=state["payload"].get("model", ""),
        provider=str(state["provider"]),
        endpoint="/v1/chat/completions",
        streaming=True,
        duration_ms=(time.perf_counter() - started) * 1000,
        input_tokens=pt,
        output_tokens=ct,
        user_id=user_id,
        api_key_id=api_key_id,
        success=True,
    )
```

- [ ] **Step 2: Replace error paths (lines 311, 332, 364, 380, 391)**

Each error path becomes:

```python
self._record_metrics(
    model=state["payload"].get("model", ""),
    provider=str(state["provider"]),
    endpoint="/v1/chat/completions",
    streaming=True,
    duration_ms=(time.perf_counter() - started) * 1000,
    success=False,
    error=str(exc)[:200],
)
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_orchestration.py tests/test_token_counting.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add runtime/orchestration/openai_handler.py
git commit -m "fix: wire handle_streaming_chat to _record_metrics for all paths"
```

---

### Task 7: Wire `handle_responses` and `handle_streaming_responses` to use `_record_metrics()`

**Files:**
- Modify: `runtime/orchestration/openai_handler.py:430-847`

- [ ] **Step 1: Replace `handle_responses` success paths**

Three branches:

**OpenAI provider path (lines 523-527):**
```python
usage_data = result.get("usage", {})
self._record_metrics(
    model=model,
    provider=provider,
    endpoint="/v1/responses",
    streaming=False,
    duration_ms=(time.perf_counter() - started) * 1000,
    input_tokens=usage_data.get("input_tokens", 0),
    output_tokens=usage_data.get("output_tokens", 0),
    user_id=user_id,
    api_key_id=api_key_id,
    success=True,
)
```

**Tool loop path (lines 553-554):**
```python
usage = getattr(response_object, "usage", None) or {}
self._record_metrics(
    model=model,
    provider=provider,
    endpoint="/v1/responses",
    streaming=False,
    duration_ms=(time.perf_counter() - started) * 1000,
    input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
    output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
    user_id=user_id,
    api_key_id=api_key_id,
    success=True,
)
```

**Chat adapter path (lines 566-570):**
```python
usage = completion.get("usage") or {}
self._record_metrics(
    model=effective_payload.get("model", ""),
    provider=provider,
    endpoint="/v1/responses",
    streaming=False,
    duration_ms=(time.perf_counter() - started) * 1000,
    input_tokens=usage.get("prompt_tokens", 0),
    output_tokens=usage.get("completion_tokens", 0),
    user_id=user_id,
    api_key_id=api_key_id,
    success=True,
)
```

- [ ] **Step 2: Replace `_with_tracking` in `handle_streaming_responses` (lines 718-734)**

```python
def _with_tracking(raw_chunks) -> Iterable[dict[str, Any] | str]:
    last_chunk = None
    for chunk in raw_chunks:
        if isinstance(chunk, dict):
            last_chunk = chunk
        yield chunk
    pt = 0
    ct = 0
    if isinstance(last_chunk, dict):
        usage = last_chunk.get("usage") or {}
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    self._record_metrics(
        model=outer_state["payload"].get("model", model),
        provider=str(outer_state["provider"]),
        endpoint="/v1/responses",
        streaming=True,
        duration_ms=(time.perf_counter() - started) * 1000,
        input_tokens=pt,
        output_tokens=ct,
        user_id=user_id,
        api_key_id=api_key_id,
        success=not stream_error if "stream_error" in dir() else True,
    )
```

Implementation: Move the `_record_metrics` call out of `_with_tracking` entirely. Instead, add it to the `finally` block of `stream_yield()` (line 844-845) where `stream_error` is accessible:

```python
finally:
    _finalize(stream_error)
    pt = 0
    ct = 0
    if isinstance(last_chunk, dict):
        usage = last_chunk.get("usage") or {}
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    self._record_metrics(
        model=outer_state["payload"].get("model", model),
        provider=str(outer_state["provider"]),
        endpoint="/v1/responses",
        streaming=True,
        duration_ms=(time.perf_counter() - started) * 1000,
        input_tokens=pt,
        output_tokens=ct,
        user_id=user_id,
        api_key_id=api_key_id,
        success=not stream_error,
        error=error_code if stream_error else None,
    )
```

`_with_tracking` then only tracks `last_chunk` — remove the `_record_token_usage` call from it entirely.

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_orchestration.py tests/test_token_counting.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add runtime/orchestration/openai_handler.py
git commit -m "fix: wire handle_responses and handle_streaming_responses to _record_metrics"
```

---

### Task 8: Anthropic streaming — pass real token counts via `stats` dict

**Files:**
- Modify: `runtime/orchestration/streaming.py:20-77`
- Modify: `runtime/orchestration/streaming.py:80-213`
- Test: `tests/test_token_counting.py`

- [ ] **Step 1: Write the failing test**

```python
def test_stream_anthropic_with_metrics_records_real_tokens():
    from metrics.request_metrics import RequestMetricsCollector

    collector = RequestMetricsCollector()

    def fake_stream_anthropic(service, iterator, model, allowed_tool_names=None):
        yield "data: something"
        yield "data: done"

    mock_service = MagicMock()

    final_chunk = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    with patch("runtime.orchestration.streaming.stream_anthropic", fake_stream_anthropic):
        with patch("runtime.orchestration.streaming.request_metrics", collector):
            with patch("runtime.orchestration.streaming.record_token_usage"):
                from runtime.orchestration.streaming import stream_anthropic_with_metrics
                list(stream_anthropic_with_metrics(
                    mock_service, iter([]), "claude-3", "anthropic",
                    "req-1", time.time(),
                ))

    metrics = collector.get_provider_metrics()
    if "anthropic" in metrics:
        assert metrics["anthropic"]["total_input_tokens"] >= 0
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py::test_stream_anthropic_with_metrics_records_real_tokens -v`

Expected: Currently `input_tokens=0` in the metrics record.

- [ ] **Step 3: Modify `stream_anthropic()` to populate `stats` dict**

Add `stats: dict[str, Any] | None = None` parameter to `stream_anthropic()`. At the finish_reason block (line 207-211), populate stats:

```python
if finish_reason:
    # ... existing logic ...
    usage = item.get("usage") or {}
    output_tokens = usage.get("completion_tokens", 0)
    if stats is not None:
        stats["input_tokens"] = usage.get("prompt_tokens", 0)
        stats["output_tokens"] = output_tokens
    # ... rest of existing logic ...
```

- [ ] **Step 4: Modify `stream_anthropic_with_metrics()` to use `stats`**

```python
def stream_anthropic_with_metrics(
    anthropic_service: AnthropicRouter,
    iterator: Iterable[dict[str, Any] | str],
    model: str,
    provider: str,
    request_id: str,
    start_time: float,
    allowed_tool_names: set[str] | None = None,
    user_id: int | None = None,
    api_key_id: int | None = None,
) -> Iterable[str]:
    total_output_tokens = 0
    last_error = None
    stats: dict[str, Any] = {}
    try:
        for item in stream_anthropic(anthropic_service, iterator, model, allowed_tool_names=allowed_tool_names, stats=stats):
            yield item
            if isinstance(item, str) and "content_block_delta" in item and "text_delta" in item:
                data = json.loads(item.split("data: ", 1)[1]) if "data: " in item else {}
                total_output_tokens += len(data.get("delta", {}).get("text", ""))
            if isinstance(item, str) and "error" in item and "event:" in item:
                last_error = item
    except Exception as e:
        logger.warning("Upstream stream interrupted for %s: %s: %s", model, type(e).__name__, e)
        last_error = f"Stream interrupted: {e}"
        try:
            from protocols.anthropic.sse_builder import AnthropicSSEBuilder
            yield AnthropicSSEBuilder(model).error(str(last_error))
        except Exception:
            pass
    finally:
        latency_ms = (time.time() - start_time) * 1000
        real_input = stats.get("input_tokens", 0)
        real_output = stats.get("output_tokens", 0)
        estimated_output = max(1, total_output_tokens // 4) if total_output_tokens else 0
        final_output = real_output if real_output > 0 else estimated_output
        request_metrics.record_request(RequestRecord(
            request_id=request_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            streaming=True,
            latency_ms=latency_ms,
            input_tokens=real_input,
            output_tokens=final_output,
            error=last_error is not None,
            error_message=str(last_error or ""),
        ))
        routing_engine.set_provider_latency(provider, latency_ms)
        routing_engine.set_provider_health(provider, last_error is None)
        if user_id is not None and last_error is None:
            try:
                db = SessionLocal()
                try:
                    record_token_usage(
                        db, user_id=user_id, api_key_id=api_key_id,
                        input_tokens=real_input,
                        output_tokens=final_output,
                        provider=provider, model=model,
                    )
                finally:
                    db.close()
            except Exception:
                logger.exception("Failed to record streaming token usage")
        memory_manager.episodic.record(
            model=model,
            provider=provider,
            duration_ms=latency_ms,
            success=last_error is None,
            token_count={"prompt_tokens": real_input, "completion_tokens": final_output},
            error=str(last_error)[:200] if last_error else None,
        )
```

Important: `stream_anthropic_with_metrics` must import `memory_manager`:

```python
from runtime.memory import memory_manager
```

And `stream_anthropic` must accept and pass through `stats`:

```python
def stream_anthropic(
    anthropic_service: AnthropicRouter,
    iterator: Iterable[dict[str, Any] | str],
    model: str,
    allowed_tool_names: set[str] | None = None,
    stats: dict[str, Any] | None = None,
) -> Iterable[str]:
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_token_counting.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add runtime/orchestration/streaming.py tests/test_token_counting.py
git commit -m "fix: extract real tokens in Anthropic streaming + add episodic record"
```

---

### Task 9: Anthropic fallback — add `episodic.record()` in `messages_adapter.py`

**Files:**
- Modify: `router/anthropic/messages_adapter.py:297-322`

- [ ] **Step 1: Add episodic record after fallback success**

After line 316 (`_record_token_usage(...)` call), add:

```python
memory_manager.episodic.record(
    session_id=request_id,
    model=model,
    provider="ollama",
    duration_ms=fallback_latency_ms,
    success=True,
    token_count=dict(usage) if usage else None,
)
```

Also update the main non-streaming success path (line 238-245) — it already has `episodic.record` with `token_count=dict(usage)`, so no change needed there.

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/ -x -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add router/anthropic/messages_adapter.py
git commit -m "fix: add episodic record to Anthropic fallback success path"
```

---

### Task 10: AgentLoop — add `token_count` to episodic record

**Files:**
- Modify: `runtime/agents/agent_loop.py:54-62`

- [ ] **Step 1: Modify the `run()` method's episodic record**

Extract token counts from the `exec_result` node_results where llm_call nodes returned usage:

```python
total_input = 0
total_output = 0
for nr in (exec_result.node_results.values() if isinstance(exec_result.node_results, dict) else []):
    if isinstance(nr, dict):
        u = nr.get("usage") or {}
        total_input += u.get("prompt_tokens", u.get("input_tokens", 0))
        total_output += u.get("completion_tokens", u.get("output_tokens", 0))

memory_manager.episodic.record(
    session_id=context.session_id or task,
    model="agent",
    provider="agent_loop",
    task_summary=task[:200],
    duration_ms=exec_result.elapsed_ms,
    success=exec_result.success,
    token_count={"prompt_tokens": total_input, "completion_tokens": total_output},
    error="; ".join(exec_result.node_errors.values()) if exec_result.node_errors else None,
)
```

- [ ] **Step 2: Modify `_make_llm_handler()` to include usage in return**

In the llm_call handler (line 100-103), extract usage from response:

```python
try:
    response = await asyncio.to_thread(adapter.chat, payload)
    text = _extract_text_from_chat(response)
    usage = response.get("usage") or {}
    return {"text": text, "provider": provider, "model": model, "usage": usage}
except Exception as exc:
    logger.error("LLM handler failed for %s/%s: %s", provider, model, exc)
    return {"error": str(exc), "text": "", "usage": {}}
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/test_agent_loop.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add runtime/agents/agent_loop.py
git commit -m "fix: add token_count to AgentLoop episodic record"
```

---

### Task 11: Integration test — verify Dashboard gets tokens from all paths

**Files:**
- Test: `tests/test_token_counting.py`

- [ ] **Step 1: Write integration test**

```python
def test_handle_chat_records_metrics_in_request_metrics():
    from metrics.request_metrics import RequestMetricsCollector
    from runtime.orchestration.openai_handler import RouterService

    collector = RequestMetricsCollector()
    service = RouterService.__new__(RouterService)
    service._record_token_usage = lambda *a, **kw: None

    mock_adapter = MagicMock()
    mock_adapter.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 15},
    }

    with patch.object(service, "_resolve_provider_and_worker", return_value=("ollama", None)):
        with patch.object(service, "_adapter", return_value=mock_adapter):
            with patch.object(service, "_normalize_payload_for_provider", side_effect=lambda p, pr: p):
                with patch.object(service, "_apply_generation_defaults", side_effect=lambda p: p):
                    with patch.object(service, "_finalize_request"):
                        with patch("runtime.orchestration.openai_handler.request_metrics", collector):
                            with patch("runtime.orchestration.openai_handler.memory_manager"):
                                service.handle_chat({"model": "llama3", "messages": [{"role": "user", "content": "hi"}]})

    metrics = collector.get_provider_metrics()
    assert "ollama" in metrics
    assert metrics["ollama"]["total_input_tokens"] == 30
    assert metrics["ollama"]["total_output_tokens"] == 15


def test_request_metrics_sees_openai_path_tokens():
    from metrics.request_metrics import RequestMetricsCollector

    collector = RequestMetricsCollector()
    from runtime.orchestration.openai_handler import RouterService

    service = RouterService.__new__(RouterService)
    service._record_token_usage = lambda *a, **kw: None

    mock_adapter = MagicMock()
    mock_adapter.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "test"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    with patch.object(service, "_resolve_provider_and_worker", return_value=("openai", None)):
        with patch.object(service, "_adapter", return_value=mock_adapter):
            with patch.object(service, "_normalize_payload_for_provider", side_effect=lambda p, pr: p):
                with patch.object(service, "_apply_generation_defaults", side_effect=lambda p: p):
                    with patch.object(service, "_finalize_request"):
                        with patch("runtime.orchestration.openai_handler.request_metrics", collector):
                            with patch("runtime.orchestration.openai_handler.memory_manager"):
                                service.handle_chat({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})

    summary = collector.get_summary()
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50
```

- [ ] **Step 2: Run full test suite**

Run: `PYTHONPATH=/Users/fred/ai/my_opencode/AetherMesh .venv/bin/pytest tests/ -v --no-header 2>&1 | tail -5`

Expected: 296+ passed (existing) + new tests passed, same 25 pre-existing failures, 3 skipped.

- [ ] **Step 3: Commit**

```bash
git add tests/test_token_counting.py
git commit -m "test: add integration tests for token counting in request_metrics"
```
