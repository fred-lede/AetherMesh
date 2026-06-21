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
        patch("providers.ollama_adapter.settings.ollama_evict_on_model_switch", False),
    ):
        get_session.return_value.post.return_value = response
        adapter = OllamaAdapter("http://127.0.0.1:11434")
        adapter._post_chat_with_retry({"model": "test"}, stream=True)

    assert get_session.return_value.post.call_args.kwargs["timeout"] == (30, 1800)


def test_model_switch_unloads_different_loaded_model() -> None:
    ps_response = MagicMock(ok=True)
    ps_response.json.return_value = {"models": [{"name": "gemma4:31b-it-qat"}]}
    unload_response = MagicMock(ok=True)

    with (
        patch("providers.ollama_adapter.get_session") as get_session,
        patch("providers.ollama_adapter.settings.ollama_evict_on_model_switch", True),
        patch.object(OllamaAdapter, "_wait_for_model_switch_turn", return_value=True),
    ):
        get_session.return_value.get.return_value = ps_response
        get_session.return_value.post.return_value = unload_response
        adapter = OllamaAdapter(
            "http://127.0.0.1:31435",
            worker={"worker_id": "node-01:31435", "assignment_id": "current"},
        )
        adapter._prepare_model_switch("qwen3-coder:30b")

    get_session.return_value.post.assert_called_once_with(
        "http://127.0.0.1:31435/api/generate",
        json={"model": "gemma4:31b-it-qat", "keep_alive": 0},
        timeout=120,
    )


def test_model_switch_keeps_requested_model_loaded() -> None:
    ps_response = MagicMock(ok=True)
    ps_response.json.return_value = {"models": [{"name": "qwen3-coder:30b"}]}

    with (
        patch("providers.ollama_adapter.get_session") as get_session,
        patch("providers.ollama_adapter.settings.ollama_evict_on_model_switch", True),
    ):
        get_session.return_value.get.return_value = ps_response
        adapter = OllamaAdapter("http://127.0.0.1:31436")
        adapter._prepare_model_switch("qwen3-coder:30b")

    get_session.return_value.post.assert_not_called()


def test_model_switch_waits_for_older_assignment() -> None:
    older_assignment = MagicMock(ok=True)
    older_assignment.json.return_value = {
        "workers": [{
            "worker_id": "node-01:31437",
            "metadata": {"active_assignments": {"older": 1.0, "current": 2.0}},
        }],
    }
    current_assignment = MagicMock(ok=True)
    current_assignment.json.return_value = {
        "workers": [{
            "worker_id": "node-01:31437",
            "metadata": {"active_assignments": {"current": 2.0}},
        }],
    }

    with (
        patch("providers.ollama_adapter.get_session") as get_session,
        patch("providers.ollama_adapter.settings.ollama_evict_on_model_switch", True),
        patch("providers.ollama_adapter.time.sleep") as sleep,
    ):
        get_session.return_value.get.side_effect = [older_assignment, current_assignment]
        adapter = OllamaAdapter(
            "http://127.0.0.1:31437",
            worker={"worker_id": "node-01:31437", "assignment_id": "current"},
        )
        assert adapter._wait_for_model_switch_turn()

    sleep.assert_called_once_with(1)
