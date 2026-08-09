from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import requests
import yaml

from config.settings import settings
from cluster.load_balancer import choose_best_worker


logger = logging.getLogger("routing_engine")


CAPABILITY_PROVIDER_SCORES = {
    "chat": {"ollama": 100, "openai": 95, "nvidia_nim": 90, "ollama_cloud": 85, "gemini": 90},
    "thinking": {"openai": 95, "nvidia_nim": 90, "ollama": 85, "ollama_cloud": 80, "gemini": 88},
    "tools": {"openai": 98, "ollama": 85, "nvidia_nim": 88, "ollama_cloud": 82, "gemini": 85},
    "vision": {"openai": 95, "gemini": 98, "ollama": 80, "ollama_cloud": 85, "nvidia_nim": 88},
    "audio": {"ollama": 90, "openai": 85, "gemini": 92, "nvidia_nim": 70, "ollama_cloud": 75},
    "embeddings": {"ollama": 90, "openai": 95, "gemini": 88, "nvidia_nim": 85, "ollama_cloud": 60},
}

ROUTING_PROVIDERS = ["ollama", "openai", "gemini", "nvidia_nim", "ollama_cloud"]
CLOUD_PROVIDERS = ["openai", "gemini", "nvidia_nim", "ollama_cloud"]
CLOUD_PROVIDER_ENDPOINTS = {
    "openai": ("OPENAI_API_BASE", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "gemini": ("GEMINI_API_BASE", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta"),
    "nvidia_nim": ("NVIDIA_NIM_API_BASE", "NVIDIA_NIM_API_KEY", "https://integrate.api.nvidia.com/v1"),
    "ollama_cloud": ("OLLAMA_CLOUD_API_BASE", "OLLAMA_CLOUD_API_KEY", "https://ollama.com"),
}


_CUSTOM_PROVIDER_NAMES: set[str] = set()
_BUILTIN_CLOUD_PROVIDERS: frozenset = frozenset(CLOUD_PROVIDERS)
_BUILTIN_ROUTING_PROVIDERS: frozenset = frozenset(ROUTING_PROVIDERS)


def canonical_provider(name: str) -> str:
    lowered = str(name).lower()
    for provider in ROUTING_PROVIDERS:
        if provider.lower() == lowered:
            return provider
    return str(name)


def register_custom_providers(names: list[str], name_base_urls: dict[str, str]) -> None:
    for name in names:
        if name not in CLOUD_PROVIDERS:
            CLOUD_PROVIDERS.append(name)
        if name not in ROUTING_PROVIDERS:
            ROUTING_PROVIDERS.append(name)
        if name not in CLOUD_PROVIDER_ENDPOINTS:
            base_url = name_base_urls.get(name, "")
            CLOUD_PROVIDER_ENDPOINTS[name] = ("", "", base_url)
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            if name not in cap_scores:
                cap_scores[name] = cap_scores.get("openai", 85)
    for name in names:
        if name not in routing_engine._provider_enabled:
            routing_engine.set_provider_enabled(name, True)


def unregister_custom_providers(names: list[str]) -> None:
    for name in names:
        if name in _BUILTIN_CLOUD_PROVIDERS or name in _BUILTIN_ROUTING_PROVIDERS:
            continue
        CLOUD_PROVIDERS[:] = [p for p in CLOUD_PROVIDERS if p != name]
        ROUTING_PROVIDERS[:] = [p for p in ROUTING_PROVIDERS if p != name]
        CLOUD_PROVIDER_ENDPOINTS.pop(name, None)
        for cap_scores in CAPABILITY_PROVIDER_SCORES.values():
            cap_scores.pop(name, None)
        routing_engine.set_provider_enabled(name, False)


@dataclass
class RouteCandidate:
    provider: str
    model: str
    score: float
    latency_ms: float = 0.0
    healthy: bool = True
    reason: str = ""


@dataclass
class RoutingDecision:
    provider: str
    model: str
    worker: dict[str, Any] | None
    score: float
    candidates: list[RouteCandidate]
    rules_applied: list[str]
    timestamp: float = field(default_factory=time.time)


class ModelRoutingEngine:
    """Decides which backend provider to route a request to based on capabilities,
    provider health, latency history, and model availability."""

    def __init__(self) -> None:
        self._lock = __import__("threading").RLock()
        self._provider_health: dict[str, bool] = {}
        self._provider_latency: dict[str, float] = {}
        self._provider_cooldown_until: dict[str, float] = {}
        self._provider_cooldown_reason: dict[str, str] = {}
        self._provider_enabled: dict[str, bool] = {
            provider: True for provider in ROUTING_PROVIDERS
        }
        self._worker_health_cache: dict[str, tuple[float, bool]] = {}
        self._worker_health_cache_ttl = 15
        self._workers_cache: list[dict[str, Any]] = []
        self._workers_cache_at: float = 0.0
        self._workers_cache_ttl = 10
        self._provider_credentials: dict[str, bool] = self._check_provider_credentials()
        self._model_overrides: dict[str, str] = {}
        self._routing_rules: list[dict[str, Any]] = []
        self._state_path: Path = settings.config_path("routing_state.yaml")
        self._audit_path: Path = settings.config_path("routing_audit.jsonl")
        self._load_config()
        self._load_state()

    def _load_config(self) -> None:
        path = settings.config_path("routing_rules.yaml")
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._model_overrides = {str(k): str(v) for k, v in cfg.get("model_overrides", {}).items()}
        self._routing_rules = cfg.get("rules", [])

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        with self._state_path.open("r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}

        provider_enabled = state.get("provider_enabled", {})
        if isinstance(provider_enabled, dict):
            for provider, enabled in provider_enabled.items():
                if provider and isinstance(provider, str):
                    self._provider_enabled[str(provider)] = bool(enabled)

        model_overrides = state.get("model_overrides", {})
        if isinstance(model_overrides, dict):
            self._model_overrides.update({str(k): str(v) for k, v in model_overrides.items()})

    def _save_state_locked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "provider_enabled": dict(self._provider_enabled),
            "model_overrides": dict(self._model_overrides),
            "updated_at": time.time(),
        }
        tmp_path = self._state_path.with_name(f"{self._state_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(state, f, sort_keys=True, allow_unicode=False)
        os.replace(tmp_path, self._state_path)

    def _append_audit_locked(self, action: str, actor: str, details: dict[str, Any]) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": time.time(),
            "actor": actor or "system",
            "action": action,
            "details": details,
        }
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")

    @staticmethod
    def _check_provider_credentials() -> dict[str, bool]:
        result: dict[str, bool] = {}
        for provider, (_, api_key_env, _) in CLOUD_PROVIDER_ENDPOINTS.items():
            result[provider] = bool(os.getenv(api_key_env, "").strip())
        for provider in ROUTING_PROVIDERS:
            result.setdefault(provider, True)
        custom_path = settings.config_path("custom_providers.json")
        if custom_path.exists():
            try:
                custom_data = json.loads(custom_path.read_text(encoding="utf-8"))
                if isinstance(custom_data, dict):
                    for name, cfg in custom_data.items():
                        if isinstance(cfg, dict) and str(cfg.get("api_key", "")).strip():
                            result[name] = True
            except (OSError, json.JSONDecodeError):
                pass
        return result

    def set_provider_health(self, provider: str, healthy: bool) -> None:
        with self._lock:
            self._provider_health[provider] = healthy
            if healthy:
                self._provider_cooldown_until.pop(provider, None)
                self._provider_cooldown_reason.pop(provider, None)

    def set_provider_failure(
        self,
        provider: str,
        *,
        code: str = "provider_error",
        message: str = "",
        cooldown_s: int | None = None,
    ) -> None:
        with self._lock:
            self._provider_health[provider] = False
            cooldown_codes = {
                "model_not_found",
                "provider_overloaded",
                "provider_rate_limited",
                "provider_timeout",
                "provider_unreachable",
            }
            if code not in cooldown_codes:
                return
            duration = settings.provider_cooldown_s if cooldown_s is None else cooldown_s
            if duration <= 0:
                return
            self._provider_cooldown_until[provider] = time.time() + duration
            self._provider_cooldown_reason[provider] = message or code

    def set_provider_latency(self, provider: str, latency_ms: float) -> None:
        with self._lock:
            self._provider_latency[provider] = latency_ms

    def _provider_cooldown_remaining(self, provider: str) -> float:
        until = self._provider_cooldown_until.get(provider, 0.0)
        remaining = until - time.time()
        if remaining <= 0:
            self._provider_cooldown_until.pop(provider, None)
            self._provider_cooldown_reason.pop(provider, None)
            return 0.0
        return remaining

    def _provider_available_locked(self, provider: str, rules_applied: list[str]) -> bool:
        if not self._provider_enabled.get(provider, False):
            return False
        remaining = self._provider_cooldown_remaining(provider)
        if remaining > 0:
            rules_applied.append(f"provider_cooldown {provider} {int(remaining)}s")
            return False
        return True

    def set_provider_enabled(self, provider: str, enabled: bool, actor: str = "system") -> None:
        with self._lock:
            previous = self._provider_enabled.get(provider)
            self._provider_enabled[provider] = enabled
            self._save_state_locked()
            self._append_audit_locked(
                "provider_enabled_changed",
                actor,
                {"provider": provider, "enabled": enabled, "previous_enabled": previous},
            )

    def set_local_only_mode(self, enabled: bool, actor: str = "system") -> None:
        with self._lock:
            previous = dict(self._provider_enabled)
            if enabled:
                self._provider_enabled["ollama"] = True
                for provider in CLOUD_PROVIDERS:
                    self._provider_enabled[provider] = False
            else:
                for provider in ROUTING_PROVIDERS:
                    self._provider_enabled[provider] = True
            self._save_state_locked()
            self._append_audit_locked(
                "local_only_mode_changed",
                actor,
                {
                    "enabled": enabled,
                    "previous_provider_enabled": previous,
                    "provider_enabled": dict(self._provider_enabled),
                },
            )

    def set_model_override(self, model: str, provider: str, actor: str = "system") -> None:
        with self._lock:
            previous = self._model_overrides.get(model)
            self._model_overrides[model] = provider
            self._save_state_locked()
            self._append_audit_locked(
                "model_override_set",
                actor,
                {"model": model, "provider": provider, "previous_provider": previous},
            )

    def clear_model_override(self, model: str, actor: str = "system") -> None:
        with self._lock:
            previous = self._model_overrides.pop(model, None)
            self._save_state_locked()
            self._append_audit_locked(
                "model_override_cleared",
                actor,
                {"model": model, "previous_provider": previous},
            )

    def get_all_overrides(self) -> dict[str, str]:
        with self._lock:
            return dict(self._model_overrides)

    def get_audit_events(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self._audit_path.exists():
            return []
        with self._lock:
            try:
                lines = self._audit_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
        events: list[dict[str, Any]] = []
        for line in lines[-max(1, limit):]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events

    def route(
        self,
        model: str,
        required_capabilities: list[str] | None = None,
        registry_models: list[dict[str, Any]] | None = None,
        request_payload: dict[str, Any] | None = None,
        tool_requirement: bool = False,
    ) -> RoutingDecision:
        with self._lock:
            clean_model = settings.resolve_model_alias(model)

            rules_applied: list[str] = []
            if clean_model != model:
                rules_applied.append(f"model_alias {model} -> {clean_model}")
            candidates: list[RouteCandidate] = []
            registry_models = registry_models or settings.model_registry().get("models", [])
            capabilities = self._infer_capabilities(clean_model, registry_models)
            required_capabilities = required_capabilities or capabilities

            override = self._model_overrides.get(model) or self._model_overrides.get(clean_model)
            if override:
                rules_applied.append(f"model_override -> {override}")
                worker = None
                routed_model = clean_model
                if override == "ollama":
                    worker = self._worker_for_model(model, clean_model, registry_models)
                    if worker is None:
                        fallback = self._local_model_fallback(required_capabilities, registry_models)
                        if fallback:
                            routed_model = str(fallback.get("name", clean_model))
                            worker = self._worker_for_model(routed_model, routed_model, registry_models)
                            rules_applied.append(f"local_model_fallback {clean_model} -> {routed_model}")
                return RoutingDecision(
                    provider=override,
                    model=routed_model,
                    worker=worker,
                    score=100.0,
                    candidates=[],
                    rules_applied=rules_applied,
                )

            registry_match = self._find_in_registry(clean_model, registry_models)
            if registry_match:
                provider = canonical_provider(str(registry_match.get("provider", "ollama")))
                bindings = registry_match.get("worker_bindings", [])
                worker = None
                if provider == "ollama" and bindings:
                    result = self._first_healthy_binding(bindings)
                    if result:
                        worker = result
                if not self._model_supports_required(registry_match, required_capabilities):
                    rules_applied.append(
                        "registry_match_missing_capabilities "
                        f"{clean_model}: {','.join(required_capabilities)}"
                    )
                elif self._provider_available_locked(provider, rules_applied):
                    candidates.append(RouteCandidate(
                        provider=provider, model=clean_model, score=100.0,
                        healthy=self._provider_health.get(provider, True),
                        reason="registry_match",
                    ))

            for cap in required_capabilities:
                cap_scores = CAPABILITY_PROVIDER_SCORES.get(cap, {})
                for provider, base_score in cap_scores.items():
                    if not self._provider_available_locked(provider, rules_applied):
                        continue
                    if provider in CLOUD_PROVIDERS and not self._provider_credentials.get(provider, True):
                        rules_applied.append(f"credential_missing {provider}")
                        continue
                    if not self._provider_health.get(provider, True):
                        base_score *= 0.3
                        rules_applied.append(f"health_penalty for {provider}")
                    latency = self._provider_latency.get(provider, 0)
                    latency_penalty = min(latency / 100, 0.3)
                    final_score = base_score * (1 - latency_penalty)

                    gpu_pressure = self._gpu_pressure_score(provider)
                    if gpu_pressure > 0:
                        gpu_penalty = min(gpu_pressure / 100, 0.25)
                        final_score *= 1 - gpu_penalty
                        rules_applied.append(f"gpu_pressure {provider}: {gpu_pressure:.1f}%")

                    is_cloud = provider in CLOUD_PROVIDERS
                    if is_cloud:
                        cost_penalty = self._cloud_cost_penalty(provider)
                        final_score *= 1 - cost_penalty
                        rules_applied.append(f"cost_penalty {provider}: {cost_penalty:.2f}")

                    if tool_requirement and provider not in CLOUD_PROVIDERS:
                        final_score *= 1.05
                        rules_applied.append(f"tool_affinity {provider}")

                    existing = next((c for c in candidates if c.provider == provider), None)
                    if existing:
                        existing.score = max(existing.score, final_score)
                    else:
                        candidates.append(RouteCandidate(
                            provider=provider, model=clean_model, score=round(final_score, 1),
                            latency_ms=latency,
                            healthy=self._provider_health.get(provider, True),
                            reason=f"capability:{cap}",
                        ))
                    rules_applied.append(f"capability_score {cap} -> {provider}")

            registry_has_healthy_match = any(
                c.reason == "registry_match" and c.healthy for c in candidates
            )
            for rule in self._routing_rules:
                if registry_has_healthy_match:
                    rules_applied.append("rules_skipped_registry_match")
                    break
                rule_model = rule.get("model", "")
                if rule_model and not (
                    rule_model in (model, clean_model)
                    or fnmatch(model, rule_model)
                    or fnmatch(clean_model, rule_model)
                ):
                    continue
                rule_provider = rule.get("provider")
                if rule_provider:
                    rule_enabled = rule.get("enabled", True)
                    if not rule_enabled:
                        for c in candidates:
                            if c.provider == rule_provider and c.reason != "registry_match":
                                c.score *= 0.1
                        rules_applied.append(f"rule_disable {rule_provider}")
                    else:
                        for c in candidates:
                            if c.provider == rule_provider and c.reason != "registry_match":
                                c.score *= rule.get("priority_boost", 1.0)
                        rules_applied.append(f"rule_boost {rule_provider} x{rule.get('priority_boost', 1.0)}")

            candidates.sort(key=lambda c: c.score, reverse=True)
            healthy_candidates = [c for c in candidates if c.healthy]
            best = healthy_candidates[0] if healthy_candidates else (candidates[0] if candidates else None)

            if not best:
                best = RouteCandidate(provider="ollama", model=clean_model, score=0.0, reason="fallback")

            selected_model = self._find_in_registry(best.model, registry_models)
            if selected_model is not None and not self._model_supports_required(selected_model, required_capabilities):
                fallback = self._local_model_fallback(required_capabilities, registry_models)
                if fallback:
                    previous_model = best.model
                    best = RouteCandidate(
                        provider="ollama",
                        model=str(fallback.get("name", clean_model)),
                        score=max(best.score, 75.0),
                        latency_ms=best.latency_ms,
                        healthy=best.healthy,
                        reason="capability_fallback",
                    )
                    rules_applied.append(f"capability_fallback {previous_model} -> {best.model}")
                else:
                    rules_applied.append(
                        "capability_missing_no_fallback "
                        f"{best.model}: {','.join(required_capabilities)}"
                    )

            worker = None
            if best.provider == "ollama":
                if best.model not in (model, clean_model):
                    worker = self._worker_for_model(best.model, best.model, registry_models)
                else:
                    worker = self._worker_for_model(model, clean_model, registry_models)
                selected_model = self._find_in_registry(best.model, registry_models)
                selected_missing_caps = (
                    selected_model is not None
                    and not self._model_supports_required(selected_model, required_capabilities)
                )
                if worker is None or selected_missing_caps:
                    fallback = self._local_model_fallback(required_capabilities, registry_models)
                    if fallback:
                        best = RouteCandidate(
                            provider="ollama",
                            model=str(fallback.get("name", clean_model)),
                            score=max(best.score, 75.0),
                            latency_ms=best.latency_ms,
                            healthy=best.healthy,
                            reason="local_model_fallback",
                        )
                        worker = self._worker_for_model(best.model, best.model, registry_models)
                        rules_applied.append(f"local_model_fallback {clean_model} -> {best.model}")
                    else:
                        rules_applied.append("no_ollama_worker_available")

            return RoutingDecision(
                provider=best.provider,
                model=best.model,
                worker=worker,
                score=best.score,
                candidates=candidates,
                rules_applied=rules_applied,
            )

    def _gpu_pressure_score(self, provider: str) -> float:
        if provider != "ollama":
            return 0.0
        try:
            from runtime.gpu.vram_scheduler import vram_scheduler
            usage = vram_scheduler.gpu_usage_summary() if hasattr(vram_scheduler, "gpu_usage_summary") else {}
            if isinstance(usage, dict):
                queue_total = sum(
                    len(q.get("queue", []))
                    for q in usage.values() if isinstance(q, dict)
                )
                return min(queue_total * 10, 100.0)
        except Exception:
            pass
        return 0.0

    def _cloud_cost_penalty(self, provider: str) -> float:
        cost_factors = {
            "nvidia_nim": 0.15,
            "openai": 0.12,
            "gemini": 0.08,
            "ollama_cloud": 0.10,
        }
        return cost_factors.get(provider, 0.05)

    def _worker_for_model(
        self,
        model: str,
        clean_model: str,
        registry_models: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for m in registry_models:
            if m.get("name") in (model, clean_model):
                bindings = m.get("worker_bindings", [])
                if bindings:
                    result = self._first_healthy_binding(bindings)
                    if result:
                        return result
                return None
        return None

    def _get_workers(self) -> list[dict[str, Any]]:
        now = time.time()
        if now - self._workers_cache_at < self._workers_cache_ttl:
            return self._workers_cache
        try:
            resp = requests.get(f"{settings.control_plane_url}/cluster/workers", timeout=3)
            if resp.ok:
                data = resp.json()
                self._workers_cache = data.get("workers", [])
                self._workers_cache_at = now
            else:
                logger.warning("Failed to fetch workers: HTTP %s", resp.status_code)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch workers: %s", exc)
        return self._workers_cache

    def _first_healthy_binding(self, bindings: list[dict[str, Any]]) -> dict[str, Any] | None:
        workers = self._get_workers()
        if workers:
            binding_urls = []
            for b in bindings:
                base_url = settings.worker_base_url(b)
                if base_url:
                    binding_urls.append(base_url)
            candidates = [w for w in workers if w.get("base_url") in binding_urls]
            if candidates:
                best = choose_best_worker(candidates, max_queue_size=settings.max_worker_queue_size)
                if best:
                    logger.debug("load_balancer selected %s (queue=%s, util=%s%%)",
                                 best.get("base_url"), best.get("queue_size"), best.get("gpu_utilization"))
                    return {"base_url": best["base_url"]}
        for b in bindings:
            base_url = settings.worker_base_url(b)
            if base_url and self._probe_worker(base_url):
                return {"base_url": base_url}
        return None

    def _probe_worker(self, base_url: str) -> bool:
        now = time.time()
        cached = self._worker_health_cache.get(base_url)
        if cached and now - cached[0] < self._worker_health_cache_ttl:
            return cached[1]
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=2)
            alive = resp.ok
        except requests.RequestException:
            alive = False
        self._worker_health_cache[base_url] = (now, alive)
        if not alive:
            logger.warning("Worker %s unreachable, skipping", base_url)
        return alive

    def _local_model_fallback(
        self,
        required_capabilities: list[str],
        registry_models: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        required = set(required_capabilities or ["chat"])
        for model in registry_models:
            if str(model.get("provider", "ollama")).lower() != "ollama":
                continue
            if not model.get("worker_bindings"):
                continue
            capabilities = set(model.get("capabilities", []))
            if required.issubset(capabilities):
                return model
        for model in registry_models:
            if str(model.get("provider", "ollama")).lower() == "ollama" and model.get("worker_bindings"):
                return model
        return None

    def _model_supports_required(
        self,
        model: dict[str, Any],
        required_capabilities: list[str],
    ) -> bool:
        required = set(required_capabilities or ["chat"])
        capabilities = set(model.get("capabilities", []))
        return required.issubset(capabilities)

    def _infer_capabilities(self, model: str, registry_models: list[dict[str, Any]]) -> list[str]:
        for m in registry_models:
            if m.get("name") in (model,):
                caps = m.get("capabilities", [])
                if caps:
                    return caps
        caps = ["chat"]
        if any(kw in model.lower() for kw in ("vision", "vl", "vision-")):
            caps.append("vision")
        if any(kw in model.lower() for kw in ("thinking", "reason", "o1", "o3", "deepseek")):
            caps.append("thinking")
        if any(kw in model.lower() for kw in ("audio", "gemma")):
            caps.append("audio")
        if any(kw in model.lower() for kw in ("tool", "function")):
            caps.append("tools")
        return caps

    def _find_in_registry(self, model: str, registry_models: list[dict[str, Any]]) -> dict[str, Any] | None:
        for m in registry_models:
            if m.get("name") == model:
                return m
        return None

    def _fallback_status(self) -> dict[str, Any]:
        configured_model = settings.ollama_fallback_model()
        configured_base_url = settings.ollama_fallback_base_url()
        registry_models = settings.model_registry().get("models", [])
        resolved = None
        source = "auto"

        if configured_model:
            configured = self._find_in_registry(configured_model, registry_models)
            if configured and str(configured.get("provider", "ollama")).lower() == "ollama":
                resolved = configured
                source = "configured"

        if resolved is None:
            resolved = self._local_model_fallback(["chat", "tools"], registry_models)

        worker = None
        resolved_model = ""
        if resolved:
            resolved_model = str(resolved.get("name", ""))
            if configured_base_url:
                worker = {"base_url": configured_base_url}
            else:
                worker = self._worker_for_model(resolved_model, resolved_model, registry_models)

        return {
            "ollama_default_model": configured_model,
            "ollama_base_url": configured_base_url,
            "resolved_model": resolved_model,
            "source": source if configured_model else "auto",
            "worker": worker,
        }

    def _model_alias_status(self) -> list[dict[str, Any]]:
        prefix = settings.model_alias_prefix()
        registry_models = settings.model_registry().get("models", [])
        aliases = settings.model_alias_entries()
        rows: list[dict[str, Any]] = []

        for alias, target in aliases.items():
            model = self._find_in_registry(target, registry_models)
            provider = str(model.get("provider", "")) if model else ""
            worker = self._worker_for_model(target, target, registry_models) if model else None
            worker_label = ""
            if worker and worker.get("base_url"):
                worker_label = str(worker["base_url"])
            elif provider in CLOUD_PROVIDER_ENDPOINTS:
                worker = self._cloud_adapter_worker(provider)
                worker_label = f"cloud adapter @ {worker['base_url']}"
                if not worker.get("credential_configured"):
                    worker_label = f"{worker_label} (API key missing)"
            capabilities = model.get("capabilities", []) if model else []
            prefixed_alias = f"{prefix}/{alias}" if prefix else alias
            rows.append(
                {
                    "alias": alias,
                    "prefixed_alias": prefixed_alias,
                    "gateway_model": f"anthropic/{prefixed_alias}",
                    "target_model": target,
                    "provider": provider or "unknown",
                    "worker": worker,
                    "worker_label": worker_label or "not configured",
                    "capabilities": capabilities,
                    "configured": model is not None,
                }
            )

        return rows

    def _cloud_adapter_worker(self, provider: str) -> dict[str, Any]:
        if provider in CLOUD_PROVIDER_ENDPOINTS:
            base_url_env, api_key_env, default_base_url = CLOUD_PROVIDER_ENDPOINTS[provider]
            base_url = os.getenv(base_url_env, default_base_url).rstrip("/")
            if api_key_env:
                configured = bool(os.getenv(api_key_env, "").strip())
            else:
                configured = False
                custom_path = settings.config_path("custom_providers.json")
                if custom_path.exists():
                    try:
                        custom_data = json.loads(custom_path.read_text(encoding="utf-8"))
                        if isinstance(custom_data, dict) and provider in custom_data:
                            cfg = custom_data[provider]
                            if isinstance(cfg, dict) and str(cfg.get("api_key", "")).strip():
                                configured = True
                    except (OSError, json.JSONDecodeError):
                        pass
            return {
                "kind": "cloud_adapter",
                "base_url": base_url,
                "credential_configured": configured,
            }
        return {"kind": "cloud_adapter", "base_url": "", "credential_configured": False}

    def get_routing_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "providers": {
                    provider: {
                        "enabled": self._provider_enabled.get(provider, False),
                        "healthy": self._provider_health.get(provider, True),
                        "latency_ms": self._provider_latency.get(provider, 0),
                        "cooldown_remaining_s": round(self._provider_cooldown_remaining(provider), 1),
                        "cooldown_reason": self._provider_cooldown_reason.get(provider, ""),
                    }
                    for provider in ROUTING_PROVIDERS
                },
                "model_overrides": dict(self._model_overrides),
                "local_only": self._provider_enabled.get("ollama", False)
                and all(not self._provider_enabled.get(provider, False) for provider in CLOUD_PROVIDERS),
                "rules_count": len(self._routing_rules),
                "state_path": str(self._state_path),
                "audit_path": str(self._audit_path),
                "audit_events": self.get_audit_events(limit=12),
                "fallback": self._fallback_status(),
                "model_aliases": self._model_alias_status(),
            }


routing_engine = ModelRoutingEngine()
