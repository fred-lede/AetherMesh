from __future__ import annotations

import logging
import time
from typing import Any

from runtime.orchestration.routing_engine import (
    CLOUD_PROVIDERS,
    ROUTING_PROVIDERS,
    RouteCandidate,
    RoutingDecision,
    routing_engine,
)
from runtime.intelligence.provider_scoring import (
    ScoringContext,
    provider_capability_registry,
)

logger = logging.getLogger("intelligence.selector")


class ExecutionSelector:
    """Post-processes routing decisions with live intelligence signals.

    Adds scoring dimensions the routing engine doesn't consider:
    - Warm model availability
    - Session affinity / provider continuity
    - Historical reliability beyond current health
    - Context window fit
    """

    def rerank(
        self,
        decision: RoutingDecision,
        model: str = "",
        required_capabilities: list[str] | None = None,
        has_tools: bool = False,
        session_id: str | None = None,
        estimated_input_tokens: int | None = None,
    ) -> RoutingDecision:
        if not decision.candidates:
            return decision

        context = ScoringContext(
            required_capabilities=required_capabilities or [],
            estimated_input_tokens=estimated_input_tokens,
            has_tools=has_tools,
            model=model,
            session_id=session_id,
            previous_provider=self._previous_provider(session_id),
        )

        adjusted: list[RouteCandidate] = []
        for c in decision.candidates:
            score = c.score

            warm_bonus = self._warm_bonus(c.provider, c.model)
            score += warm_bonus

            affinity_bonus = self._affinity_bonus(c.provider, context)
            score += affinity_bonus

            reliability_adj = self._reliability_adjustment(c.provider)
            score += reliability_adj

            context_penalty = self._context_penalty(c.provider, context)
            score += context_penalty

            adjusted.append(RouteCandidate(
                provider=c.provider,
                model=c.model,
                score=max(0.0, round(score, 1)),
                latency_ms=c.latency_ms,
                healthy=c.healthy,
                reason=c.reason,
            ))

        adjusted.sort(key=lambda c: c.score, reverse=True)
        best = adjusted[0]

        return RoutingDecision(
            provider=best.provider,
            model=best.model,
            worker=decision.worker,
            score=best.score,
            candidates=adjusted,
            rules_applied=decision.rules_applied + ["execution_selector_rerank"],
        )

    def score_breakdown(
        self,
        decision: RoutingDecision,
        model: str = "",
        required_capabilities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for c in decision.candidates:
            row: dict[str, Any] = {
                "provider": c.provider,
                "model": c.model,
                "base_score": c.score,
                "healthy": c.healthy,
                "latency_ms": c.latency_ms,
                "warm_bonus": self._warm_bonus(c.provider, c.model),
                "affinity_bonus": 0.0,
                "reliability_adjustment": self._reliability_adjustment(c.provider),
                "adjusted_score": 0.0,
            }
            row["adjusted_score"] = max(0.0, round(
                c.score + row["warm_bonus"] + row["affinity_bonus"] + row["reliability_adjustment"], 1
            ))
            rows.append(row)
        rows.sort(key=lambda r: r["adjusted_score"], reverse=True)
        return rows

    def _warm_bonus(self, provider: str, model: str) -> float:
        if provider != "ollama":
            return 0.0
        try:
            from runtime.gpu.warm_pool import warm_pool
            warm_models = warm_pool.warm_models()
            for entry in warm_models:
                if entry.model_name == model and time.time() < entry.kept_warm_until:
                    logger.debug("Warm bonus +10 for %s/%s", provider, model)
                    return 10.0
        except ImportError:
            pass
        return 0.0

    def _affinity_bonus(self, provider: str, context: ScoringContext) -> float:
        if not context.session_id or not context.previous_provider:
            return 0.0
        if context.previous_provider == provider:
            logger.debug("Affinity bonus +5 for %s (session %s)", provider, context.session_id)
            return 5.0
        return 0.0

    def _reliability_adjustment(self, provider: str) -> float:
        health = routing_engine._provider_health.get(provider)
        if health is False:
            return -15.0
        return 0.0

    def _context_penalty(self, provider: str, context: ScoringContext) -> float:
        if not context.estimated_input_tokens:
            return 0.0
        caps = provider_capability_registry.get_capabilities(provider)
        if not caps:
            return 0.0
        if context.estimated_input_tokens > caps.max_context:
            return -20.0
        return 0.0

    @staticmethod
    def _previous_provider(session_id: str | None) -> str | None:
        if not session_id:
            return None
        try:
            from runtime.sessions.session_store import session_store
            session = session_store.get(session_id)
            if session and hasattr(session, "metadata") and isinstance(session.metadata, dict):
                return session.metadata.get("last_provider")
        except ImportError:
            pass
        return None


execution_selector = ExecutionSelector()
