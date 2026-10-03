from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from providers.base import ProviderError
from providers.ollama_adapter import OllamaAdapter


def test_systemone_posts_and_returns_json():
    adapter = OllamaAdapter("http://x:11434")
    resp = MagicMock(ok=True)
    resp.json.return_value = {"model": "nimble:9b", "answers": {"intent": {"type": "choice", "choice": "book"}}}
    with patch("providers.ollama_adapter.get_session") as factory:
        factory.return_value.post.return_value = resp
        out = adapter.systemone({"model": "nimble:9b", "state": "x", "questions": {"intent": {"type": "choice"}}})
    assert out["answers"]["intent"]["choice"] == "book"
    assert factory.return_value.post.call_args.args[0] == "http://x:11434/v1/systemone"


def test_systemone_raises_on_error():
    adapter = OllamaAdapter("http://x:11434")
    resp = MagicMock(ok=False)
    resp.text = "model is required"
    resp.status_code = 400
    with patch("providers.ollama_adapter.get_session") as factory:
        factory.return_value.post.return_value = resp
        with pytest.raises(ProviderError):
            adapter.systemone({"model": "nimble:9b"})
