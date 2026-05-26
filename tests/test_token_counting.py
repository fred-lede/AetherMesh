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
