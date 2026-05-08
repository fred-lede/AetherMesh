from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("intelligence.scoring")


@dataclass
class ProviderCapabilities:
    chat: bool = True
    thinking: bool = False
    tools: bool = False
    vision: bool = False
    audio: bool = False
    embeddings: bool = False
    max_context: int = 8192
    supports_streaming: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    reliability_score: float = 1.0


@dataclass
class ScoringContext:
    required_capabilities: list[str] = field(default_factory=list)
    estimated_input_tokens: int | None = None
    has_tools: bool = False
    has_vision: bool = False
    has_thinking: bool = False
    session_id: str | None = None
    model: str = ""
    message_count: int = 0
    previous_provider: str | None = None


@dataclass
class ScoredProvider:
    provider: str
    model: str
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


class ProviderCapabilityRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderCapabilities] = {}
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        self.register_provider("ollama", ProviderCapabilities(
            chat=True, thinking=True, tools=True, vision=True, audio=True,
            embeddings=True, max_context=32768, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        ))
        self.register_provider("openai", ProviderCapabilities(
            chat=True, thinking=True, tools=True, vision=True, audio=False,
            embeddings=True, max_context=128000, cost_per_1k_input=0.015, cost_per_1k_output=0.06,
        ))
        self.register_provider("nvidia_nim", ProviderCapabilities(
            chat=True, thinking=True, tools=True, vision=True, audio=False,
            embeddings=False, max_context=128000, cost_per_1k_input=0.01, cost_per_1k_output=0.04,
        ))
        self.register_provider("gemini", ProviderCapabilities(
            chat=True, thinking=True, tools=True, vision=True, audio=True,
            embeddings=True, max_context=200000, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        ))
        self.register_provider("ollama_cloud", ProviderCapabilities(
            chat=True, thinking=True, tools=True, vision=True, audio=False,
            embeddings=False, max_context=32000, cost_per_1k_input=0.005, cost_per_1k_output=0.015,
        ))

    def register_provider(self, name: str, caps: ProviderCapabilities) -> None:
        self._providers[name] = caps
        logger.debug("Registered provider %s with caps=%s", name, caps)

    def unregister_provider(self, name: str) -> None:
        self._providers.pop(name, None)

    def get_capabilities(self, name: str) -> ProviderCapabilities | None:
        return self._providers.get(name)

    def get_best_provider(
        self, required: list[str], context: ScoringContext
    ) -> ScoredProvider | None:
        best: ScoredProvider | None = None
        for name in self._providers:
            caps = self._providers[name]
            score = self._score_provider(name, caps, required, context)
            if best is None or score > best.score:
                best = ScoredProvider(
                    provider=name, model=context.model,
                    score=score, breakdown={},
                )
        return best

    def score_provider(
        self, name: str, required: list[str], context: ScoringContext
    ) -> float:
        caps = self._providers.get(name)
        if not caps:
            return 0.0
        return self._score_provider(name, caps, required, context)

    def _score_provider(
        self, name: str, caps: ProviderCapabilities,
        required: list[str], context: ScoringContext,
    ) -> float:
        breakdown: dict[str, float] = {}
        score = 50.0

        cap_match = self._capability_match(name, caps, required)
        score += cap_match * 0.25
        breakdown["capability_match"] = cap_match

        if context.estimated_input_tokens and context.estimated_input_tokens > caps.max_context:
            ctx_penalty = -20.0
            score += ctx_penalty
            breakdown["context_penalty"] = ctx_penalty

        if context.session_id and context.previous_provider == name:
            score += 5.0
            breakdown["session_affinity"] = 5.0

        if caps.reliability_score < 0.8:
            reliability_penalty = -10.0 * (1.0 - caps.reliability_score)
            score += reliability_penalty
            breakdown["reliability_penalty"] = reliability_penalty

        if caps.cost_per_1k_input > 0 and context.estimated_input_tokens:
            estimated_cost = caps.cost_per_1k_input * context.estimated_input_tokens / 1000
            cost_penalty = min(estimated_cost * 5, -10.0)
            score += cost_penalty
            breakdown["cost_penalty"] = cost_penalty

        score = max(0.0, min(100.0, score))
        breakdown["total"] = score
        return score

    @staticmethod
    def _capability_match(
        name: str, caps: ProviderCapabilities, required: list[str]
    ) -> float:
        from runtime.orchestration.routing_engine import CAPABILITY_PROVIDER_SCORES
        if not required:
            return 50.0
        total = 0.0
        for cap in required:
            cap_scores = CAPABILITY_PROVIDER_SCORES.get(cap, {})
            total += cap_scores.get(name, 0.0)
        return total / len(required)


provider_capability_registry = ProviderCapabilityRegistry()
