from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import requests


from runtime.orchestration.model_context import (
    configured_context_length,
    fetch_context_length,
    resolve_context_length,
    resolve_effective_max_context,
)


def test_configured_context_length_from_registry():
    registry = {"models": [{"name": "my-model", "context_length": 32000}]}
    with patch("config.settings.Settings.model_registry", return_value=registry):
        assert configured_context_length("my-model") == 32000


def test_configured_context_length_missing():
    registry = {"models": [{"name": "my-model"}]}
    with patch("config.settings.Settings.model_registry", return_value=registry):
        assert configured_context_length("my-model") is None


def test_configured_context_length_invalid():
    registry = {"models": [{"name": "my-model", "context_length": "abc"}]}
    with patch("config.settings.Settings.model_registry", return_value=registry):
        assert configured_context_length("my-model") is None


def test_resolve_context_length_prefers_configured():
    registry = {"models": [{"name": "my-model", "context_length": 50000}]}
    with patch("config.settings.Settings.model_registry", return_value=registry):
        assert resolve_context_length("my-model") == 50000


def test_resolve_effective_max_context_model_first():
    registry = {"models": [{"name": "my-model", "context_length": 200000}]}
    with patch("config.settings.Settings.model_registry", return_value=registry):
        assert resolve_effective_max_context("openai", "my-model") == 200000


def test_resolve_effective_max_context_falls_back_to_provider():
    assert resolve_effective_max_context("gemini") == 200000
    assert resolve_effective_max_context("openai") == 128000


def test_resolve_effective_max_context_unknown_provider():
    assert resolve_effective_max_context("nope") == 8192


@patch("runtime.orchestration.model_context.requests.post")
def test_fetch_ollama_uses_api_show(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"context_length": 8192}
    mock_post.return_value = mock_resp
    assert fetch_context_length("llama3", provider="ollama", base_url="http://x:11434") == 8192
    mock_post.assert_called_once_with(
        "http://x:11434/api/show", json={"model": "llama3"}, timeout=10.0
    )


@patch("runtime.orchestration.model_context.requests.post")
def test_fetch_ollama_falls_back_to_parameters(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"parameters": {"num_ctx": 4096}}
    mock_post.return_value = mock_resp
    assert fetch_context_length("llama3", provider="ollama", base_url="http://x:11434") == 4096


@patch("runtime.orchestration.model_context.requests.post")
def test_fetch_ollama_uses_model_info_context_length(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"model_info": {"gemma4.context_length": 131072}}
    mock_post.return_value = mock_resp
    assert fetch_context_length("gemma4:e2b", provider="ollama", base_url="http://x:11434") == 131072


@patch("runtime.orchestration.model_context.requests.post")
def test_fetch_ollama_model_info_prefers_arch_over_general(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "model_info": {"general.context_length": 4096, "qwen3.context_length": 262144}
    }
    mock_post.return_value = mock_resp
    assert fetch_context_length("qwen3.8:27b", provider="ollama", base_url="http://x:11434") == 262144


@patch("runtime.orchestration.model_context.requests.post")
def test_fetch_ollama_request_failure(mock_post):
    mock_post.side_effect = requests.RequestException("boom")
    assert fetch_context_length("llama3", provider="ollama", base_url="http://x:11434") is None


@patch("runtime.orchestration.model_context.requests.get")
def test_fetch_cloud_uses_models_context_window(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "my-model", "context_window": 200000}]
    }
    mock_get.return_value = mock_resp
    assert fetch_context_length("my-model", provider="openai", base_url="https://api.openai.com/v1", api_key="sk") == 200000


@patch("runtime.orchestration.model_context.requests.get")
def test_fetch_cloud_multiple_field_fallback(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "my-model", "max_model_len": 65536}]
    }
    mock_get.return_value = mock_resp
    assert fetch_context_length("my-model", provider="nvidia_nim", base_url="https://x/v1") == 65536


@patch("runtime.orchestration.model_context.requests.get")
def test_fetch_cloud_model_not_in_list(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "other", "context_window": 128000}]}
    mock_get.return_value = mock_resp
    assert fetch_context_length("my-model", provider="openai", base_url="https://x/v1") is None