from __future__ import annotations

import pytest

from config.settings import settings
from runtime.orchestration.openai_handler import RouterService


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = {
        "models": [
            {"name": "muse-glimmer:30b", "capabilities": ["chat", "thinking", "tools", "vision"]},
            {"name": "hermes:8b", "capabilities": ["chat", "tools"]},
        ]
    }
    monkeypatch.setattr(settings, "default_max_tokens", 1024)
    monkeypatch.setattr(type(settings), "model_registry", lambda self: registry)


def test_thinking_model_no_max_tokens_gets_default() -> None:
    service = RouterService()
    payload = {"model": "muse-glimmer:30b", "messages": [{"role": "user", "content": "hi"}]}
    out = service._apply_generation_defaults(payload)
    assert out["max_tokens"] == 1024


def test_thinking_model_explicit_max_tokens_preserved() -> None:
    service = RouterService()
    payload = {"model": "muse-glimmer:30b", "max_tokens": 200, "messages": []}
    out = service._apply_generation_defaults(payload)
    assert out["max_tokens"] == 200


def test_thinking_model_max_completion_tokens_preserved() -> None:
    service = RouterService()
    payload = {"model": "muse-glimmer:30b", "max_completion_tokens": 64, "messages": []}
    out = service._apply_generation_defaults(payload)
    assert "max_tokens" not in out


def test_non_thinking_model_no_default_injected() -> None:
    service = RouterService()
    payload = {"model": "hermes:8b", "messages": []}
    out = service._apply_generation_defaults(payload)
    assert "max_tokens" not in out


def test_disabled_default_no_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "default_max_tokens", 0)
    service = RouterService()
    payload = {"model": "muse-glimmer:30b", "messages": []}
    out = service._apply_generation_defaults(payload)
    assert "max_tokens" not in out
