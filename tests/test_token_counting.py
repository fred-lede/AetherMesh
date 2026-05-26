from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from metrics.request_metrics import RequestMetricsCollector, RequestRecord


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
    assert result["usage"]["total_tokens"] == 0


def test_gemini_stream_emits_usage_in_final_chunk():
    from providers.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.api_key = "test"
    adapter.base_url = "https://fake"

    raw_lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"totalTokenCount":13}}',
    ]

    with patch.object(adapter, "_messages_to_contents", return_value=[]):
        with patch.object(adapter, "_tools_to_gemini", return_value=[]):
            with patch("providers.gemini_adapter.post_with_retry") as mock_post:
                mock_response = MagicMock()
                mock_response.iter_lines.return_value = iter(raw_lines)
                mock_post.return_value = mock_response
                chunks = list(adapter.stream({"model": "gemini-pro", "messages": []}))

    finish_chunks = [c for c in chunks if isinstance(c, dict) and c.get("choices", [{}])[0].get("finish_reason") == "stop"]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["usage"]["prompt_tokens"] == 10
    assert finish_chunks[0]["usage"]["completion_tokens"] == 3
    assert finish_chunks[0]["usage"]["total_tokens"] == 13


def test_ollama_cloud_stream_usage_has_total_tokens():
    from providers.ollama_cloud_adapter import OllamaCloudAdapter

    adapter = OllamaCloudAdapter.__new__(OllamaCloudAdapter)
    adapter.api_key = "test"
    adapter.base_url = "https://fake"

    raw_lines = [
        '{"message":{"content":"hi"},"done":false}',
        '{"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":10}',
    ]

    with patch("providers.ollama_cloud_adapter.post_with_retry") as mock_post:
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(raw_lines)
        mock_post.return_value = mock_response
        chunks = list(adapter.stream({"model": "llama3", "messages": []}))

    finish_chunks = [c for c in chunks if isinstance(c, dict) and c.get("usage")]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["usage"]["prompt_tokens"] == 5
    assert finish_chunks[0]["usage"]["completion_tokens"] == 10
    assert finish_chunks[0]["usage"]["total_tokens"] == 15


def test_record_metrics_writes_all_three_targets():
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
    assert records[0].token_count is not None
    assert records[0].token_count.get("prompt_tokens") == 50


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


def test_handle_chat_records_metrics_in_request_metrics():
    collector = RequestMetricsCollector()
    from runtime.orchestration.openai_handler import RouterService

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


def test_stream_anthropic_with_metrics_records_real_tokens():
    from runtime.orchestration.streaming import stream_anthropic_with_metrics
    from runtime.orchestration.anthropic_converter import AnthropicRouter

    collector = RequestMetricsCollector()
    mock_service = MagicMock(spec=AnthropicRouter)
    mock_service._looks_like_tool_status_text = MagicMock(return_value=False)
    mock_service._tool_call_allowed = MagicMock(return_value=True)
    mock_service._log_suppressed_tool_call = MagicMock()
    mock_service.tool_call_normalizer = MagicMock()
    mock_service.tool_call_normalizer.from_text = MagicMock(return_value=[])

    final_chunk = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    def fake_stream_anthropic(service, iterator, model, allowed_tool_names=None, stats=None):
        if stats is not None:
            stats["input_tokens"] = 100
            stats["output_tokens"] = 50
        yield "data: fake"

    with patch("runtime.orchestration.streaming.stream_anthropic", fake_stream_anthropic):
        with patch("runtime.orchestration.streaming.request_metrics", collector):
            with patch("runtime.orchestration.streaming.record_token_usage"):
                with patch("runtime.orchestration.streaming.memory_manager"):
                    list(stream_anthropic_with_metrics(
                        mock_service, iter([]), "claude-3", "anthropic", "req-1", time.time(),
                    ))

    metrics = collector.get_provider_metrics()
    assert "anthropic" in metrics
    assert metrics["anthropic"]["total_input_tokens"] == 100
    assert metrics["anthropic"]["total_output_tokens"] == 50
