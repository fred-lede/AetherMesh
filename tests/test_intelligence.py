from __future__ import annotations

from runtime.intelligence.provider_scoring import ProviderCapabilityRegistry, ProviderCapabilities, ScoringContext
from runtime.intelligence.execution_selector import ExecutionSelector


def test_registry_has_defaults() -> None:
    registry = ProviderCapabilityRegistry()
    assert registry.get_capabilities("ollama") is not None
    assert registry.get_capabilities("openai") is not None


def test_registry_register_and_score() -> None:
    registry = ProviderCapabilityRegistry()
    registry.register_provider("test-p", ProviderCapabilities(chat=True, tools=True))
    caps = registry.get_capabilities("test-p")
    assert caps is not None
    assert caps.chat is True
    registry.unregister_provider("test-p")
    assert registry.get_capabilities("test-p") is None


def test_get_best_provider() -> None:
    registry = ProviderCapabilityRegistry()
    ctx = ScoringContext(model="test-model")
    best = registry.get_best_provider(["chat"], ctx)
    assert best is not None
    assert best.provider in registry._providers
    assert best.score > 0


def test_selector_reranks_preserves_decision() -> None:
    from runtime.orchestration.routing_engine import RoutingDecision, RouteCandidate
    selector = ExecutionSelector()
    decision = RoutingDecision(
        provider="ollama", model="test-model", worker=None, score=10.0,
        candidates=[RouteCandidate(provider="ollama", model="test-model", score=10.0)],
        rules_applied=[],
    )
    result = selector.rerank(decision, model="test-model", session_id="s1")
    assert result is not None
    assert result.provider == "ollama"
    assert result.score >= 10.0
