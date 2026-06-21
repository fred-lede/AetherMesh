from __future__ import annotations

from unittest.mock import MagicMock, patch

from providers.ollama_adapter import OllamaAdapter


def test_stream_uses_configured_read_timeout() -> None:
    response = MagicMock(status_code=200)

    with (
        patch("providers.ollama_adapter.get_session") as get_session,
        patch.object(OllamaAdapter, "_chat_payload", return_value={"model": "test"}),
        patch("providers.ollama_adapter.settings.request_timeout_s", 30),
        patch("providers.ollama_adapter.settings.stream_read_timeout_s", 1800),
    ):
        get_session.return_value.post.return_value = response
        adapter = OllamaAdapter("http://127.0.0.1:11434")
        adapter._post_chat_with_retry({"model": "test"}, stream=True)

    assert get_session.return_value.post.call_args.kwargs["timeout"] == (30, 1800)
