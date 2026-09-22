from __future__ import annotations

from typing import Any

import pytest

from runtime.orchestration import openai_handler
import runtime.orchestration.model_registry_store as store


class _SettingsStub:
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self.calls = 0

    def model_registry(self) -> Any:
        self.calls += 1
        return self._registry


def test_ensure_registry_refreshes_on_mtime_change(monkeypatch):
    service = openai_handler.RouterService.__new__(openai_handler.RouterService)
    service.registry = [{"name": "old"}]
    service._registry_mtime = 1.0
    stub = _SettingsStub([{"name": "new"}])
    monkeypatch.setattr(openai_handler, "settings", stub)
    monkeypatch.setattr(store, "models_mtime", lambda: 2.0)
    service._ensure_registry()
    assert service.registry == [{"name": "new"}]


def test_ensure_registry_keeps_cache_when_unchanged(monkeypatch):
    service = openai_handler.RouterService.__new__(openai_handler.RouterService)
    service.registry = [{"name": "old"}]
    service._registry_mtime = 5.0
    stub = _SettingsStub([{"name": "new"}])
    monkeypatch.setattr(openai_handler, "settings", stub)
    monkeypatch.setattr(store, "models_mtime", lambda: 5.0)
    service._ensure_registry()
    assert service.registry == [{"name": "old"}]
    assert stub.calls == 0


def test_anthropic_ensure_registry_refreshes(monkeypatch):
    from runtime.orchestration import anthropic_converter

    service = anthropic_converter.AnthropicRouter.__new__(anthropic_converter.AnthropicRouter)
    service.registry = {"models": [{"name": "old"}]}
    service._registry_mtime = 1.0
    stub = _SettingsStub({"models": [{"name": "new"}]})
    monkeypatch.setattr(anthropic_converter, "settings", stub)
    monkeypatch.setattr(store, "models_mtime", lambda: 2.0)
    service._ensure_registry()
    assert service.registry == {"models": [{"name": "new"}]}
