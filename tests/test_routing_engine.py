from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest
import requests

from runtime.orchestration.routing_engine import ModelRoutingEngine


# ── helpers ─────────────────────────────────────────────────────────

def _make_worker(
    *,
    worker_id: str = "w1",
    base_url: str = "http://192.168.1.10:11434",
    gpu_name: str = "RTX 5090",
    queue_size: int = 0,
    status: str = "healthy",
    gpu_utilization: float = 10.0,
    ps_models: list[str] | None = None,
) -> dict:
    return {
        "worker_id": worker_id,
        "base_url": base_url,
        "gpu_name": gpu_name,
        "queue_size": queue_size,
        "status": status,
        "gpu_utilization": gpu_utilization,
        "ps_models": ps_models or [],
    }


def _mock_requests_get(url: str, **kwargs):
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200

    if url.endswith("/cluster/workers"):
        resp.json.return_value = {
            "workers": [
                _make_worker(
                    worker_id="node-01:11434",
                    base_url="http://192.168.1.200:11434",
                    gpu_name="RTX 5090",
                    ps_models=["gemma4:31b"],
                ),
                _make_worker(
                    worker_id="node-p40-01:11434",
                    base_url="http://192.168.1.123:11434",
                    gpu_name="Tesla P40",
                    ps_models=["gemma4:e4b"],
                ),
            ]
        }
    elif url.endswith("/api/tags"):
        resp.ok = True
        resp.json.return_value = {"models": [{"name": "gemma4:31b"}]}
    else:
        resp.ok = False
        resp.status_code = 404

    return resp


# ── _probe_worker ──────────────────────────────────────────────────

def test_probe_worker_success() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {}

    with patch("requests.get") as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"models": [{"name": "gemma4:31b"}]}

        result = engine._probe_worker("http://192.168.1.200:11434")

        assert result is True
        mock_get.assert_called_once_with(
            "http://192.168.1.200:11434/api/tags", timeout=2
        )


def test_probe_worker_failure() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {}

    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("connection refused")

        result = engine._probe_worker("http://192.168.1.200:11434")

        assert result is False


def test_probe_worker_not_ok() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {}

    with patch("requests.get") as mock_get:
        mock_get.return_value.ok = False
        mock_get.return_value.status_code = 503

        result = engine._probe_worker("http://192.168.1.200:11434")

        assert result is False


def test_probe_worker_cache_hit() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {"http://192.168.1.200:11434": (time.time(), True)}

    with patch("requests.get") as mock_get:
        result = engine._probe_worker("http://192.168.1.200:11434")

        assert result is True
        mock_get.assert_not_called()


def test_probe_worker_cache_expired() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {
        "http://192.168.1.200:11434": (time.time() - 30, True)
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"models": []}

        result = engine._probe_worker("http://192.168.1.200:11434")

        assert result is True
        mock_get.assert_called_once()


# ── _get_workers ───────────────────────────────────────────────────

def test_get_workers_success() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache = []
    engine._workers_cache_at = 0.0

    with patch("requests.get") as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {
            "workers": [
                {"worker_id": "node-01:11434", "base_url": "http://192.168.1.200:11434"},
            ]
        }

        workers = engine._get_workers()

        assert len(workers) == 1
        assert workers[0]["worker_id"] == "node-01:11434"


def test_get_workers_error_returns_empty() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache = []
    engine._workers_cache_at = 0.0

    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("timeout")

        workers = engine._get_workers()

        assert workers == []


def test_get_workers_cache() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache = [{"worker_id": "cached"}]
    engine._workers_cache_at = time.time() + 9999

    with patch("requests.get") as mock_get:
        workers = engine._get_workers()

        assert workers == [{"worker_id": "cached"}]
        mock_get.assert_not_called()


def test_get_workers_not_ok() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache = []
    engine._workers_cache_at = 0.0

    with patch("requests.get") as mock_get:
        mock_get.return_value.ok = False
        mock_get.return_value.status_code = 500

        workers = engine._get_workers()

        assert workers == []


# ── _first_healthy_binding ─────────────────────────────────────────

def test_first_healthy_binding_no_bindings() -> None:
    engine = ModelRoutingEngine()
    result = engine._first_healthy_binding([])
    assert result is None


def test_first_healthy_binding_load_balancer_picks_best() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0

    bindings = [
        {"worker_id": "node-p40-01:11434", "host": "192.168.1.123", "port": 11434},
        {"worker_id": "node-01:11434", "host": "192.168.1.200", "port": 11434},
    ]

    with patch("requests.get") as mock_get:
        mock_get.side_effect = _mock_requests_get

        result = engine._first_healthy_binding(bindings)

        assert result is not None
        assert "base_url" in result


def test_first_healthy_binding_direct_probe_fallback() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache = []
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {}

    bindings = [
        {"host": "192.168.1.200", "port": 11434},
    ]

    with patch("requests.get") as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"models": []}

        result = engine._first_healthy_binding(bindings)

        assert result is not None
        assert "base_url" in result


def test_first_healthy_binding_empty_bindings() -> None:
    engine = ModelRoutingEngine()
    result = engine._first_healthy_binding([])
    assert result is None


# ── _worker_for_model ──────────────────────────────────────────────

def test_worker_for_model_found_in_registry() -> None:
    registry_models = [
        {"name": "gemma4:31b", "provider": "ollama",
         "worker_bindings": [{"host": "192.168.1.200", "port": 11434}]},
    ]
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0

    with patch("requests.get") as mock_get:
        mock_get.side_effect = _mock_requests_get

        result = engine._worker_for_model(
            "gemma4:31b", "gemma4:31b", registry_models
        )

        assert result is not None
        assert "base_url" in result


def test_worker_for_model_not_found() -> None:
    engine = ModelRoutingEngine()
    result = engine._worker_for_model("nonexistent:99b", "nonexistent:99b", [])
    assert result is None


def test_worker_for_model_no_registry_models() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0

    with patch("requests.get") as mock_get:
        mock_get.side_effect = _mock_requests_get

        result = engine._worker_for_model("gemma4:31b", "gemma4:31b", [])

        assert result is None


# ── route with multi-binding and load balancer ────────────────────

def test_route_basic_with_mocked_workers() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0

    with patch("requests.get") as mock_get:
        mock_get.side_effect = _mock_requests_get

        decision = engine.route(
            model="gemma4:31b",
            registry_models=[
                {
                    "name": "gemma4:31b",
                    "provider": "ollama",
                    "capabilities": ["chat"],
                    "worker_bindings": [
                        {"host": "192.168.1.200", "port": 11434},
                        {"host": "192.168.1.123", "port": 11434},
                    ],
                }
            ],
        )

        assert decision is not None
        assert decision.provider == "ollama"
        assert decision.model == "gemma4:31b"
        assert decision.worker is not None
        assert "base_url" in decision.worker


def test_route_falls_back_to_other_binding() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0
    engine._worker_health_cache = {}

    call_count = 0

    def mock_side_effect(url: str, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if url.endswith("/cluster/workers"):
            resp.ok = True
            resp.json.return_value = {
                "workers": [
                    {"worker_id": "w1", "base_url": "http://192.168.1.123:11434",
                     "gpu_name": "Tesla P40", "queue_size": 0, "status": "healthy",
                     "gpu_utilization": 5.0, "ps_models": ["gemma4:e4b"]},
                ]
            }
        elif "192.168.1.123" in url and url.endswith("/api/tags"):
            resp.ok = True
            resp.json.return_value = {"models": [{"name": "gemma4:e4b"}]}
        else:
            resp.ok = False
        return resp

    with patch("requests.get") as mock_get:
        mock_get.side_effect = mock_side_effect

        decision = engine.route(
            model="gemma4:e4b",
            registry_models=[
                {
                    "name": "gemma4:e4b",
                    "provider": "ollama",
                    "capabilities": ["chat"],
                    "worker_bindings": [
                        {"host": "192.168.1.200", "port": 11434},
                        {"host": "192.168.1.123", "port": 11434},
                    ],
                }
            ],
        )

        assert decision is not None
        assert decision.worker is not None
        assert "192.168.1.123" in decision.worker.get("base_url", "")


def test_route_with_no_available_workers() -> None:
    engine = ModelRoutingEngine()
    engine._workers_cache_at = 0.0

    def mock_no_workers(url: str, **kwargs):
        resp = MagicMock()
        if url.endswith("/api/tags"):
            resp.ok = False  # direct probe also fails
        else:
            resp.ok = True
            resp.json.return_value = {"workers": []}
        return resp

    with patch("requests.get") as mock_get:
        mock_get.side_effect = mock_no_workers

        decision = engine.route(
            model="gemma4:31b",
            registry_models=[
                {
                    "name": "gemma4:31b",
                    "provider": "ollama",
                    "capabilities": ["chat"],
                    "worker_bindings": [
                        {"host": "192.168.1.200", "port": 11434},
                    ],
                }
            ],
        )

        assert decision is not None
        assert decision.worker is None or decision.provider != "ollama"


def test_route_unknown_model_passthrough() -> None:
    engine = ModelRoutingEngine()
    decision = engine.route(
        model="nvidia_nim:deepseek-r1:671b",
        registry_models=[],
    )
    assert decision is not None
    assert isinstance(decision.provider, str)


# ── provider health management ─────────────────────────────────────

def test_set_provider_health() -> None:
    engine = ModelRoutingEngine()
    engine.set_provider_health("ollama", True)
    assert engine._provider_health.get("ollama") is True


def test_set_provider_health_clears_cooldown() -> None:
    engine = ModelRoutingEngine()
    engine._provider_cooldown_until["ollama"] = time.time() + 9999
    engine.set_provider_health("ollama", True)
    assert "ollama" not in engine._provider_cooldown_until


def test_set_provider_failure_sets_health_false() -> None:
    engine = ModelRoutingEngine()
    engine.set_provider_failure("ollama", code="provider_error", message="error")
    assert engine._provider_health.get("ollama") is False


def test_set_provider_failure_sets_cooldown_for_cloud() -> None:
    engine = ModelRoutingEngine()
    engine.set_provider_failure("openai", code="provider_overloaded", message="busy", cooldown_s=10)
    assert engine._provider_health.get("openai") is False
    assert engine._provider_cooldown_until.get("openai", 0) > 0
    assert engine._provider_cooldown_reason.get("openai") == "busy"


def test_set_provider_failure_sets_cooldown_for_ollama() -> None:
    engine = ModelRoutingEngine()
    engine.set_provider_failure("ollama", code="provider_overloaded", message="busy", cooldown_s=10)
    assert engine._provider_health.get("ollama") is False
    assert engine._provider_cooldown_until.get("ollama", 0) > 0
    assert engine._provider_cooldown_reason.get("ollama") == "busy"


def test_set_provider_latency() -> None:
    engine = ModelRoutingEngine()
    engine.set_provider_latency("ollama", 150.0)
    assert engine._provider_latency.get("ollama") == 150.0


# ── routing status ─────────────────────────────────────────────────

def test_get_routing_status() -> None:
    engine = ModelRoutingEngine()
    status = engine.get_routing_status()
    assert "providers" in status
    assert "local_only" in status
    assert "model_overrides" in status
    assert "fallback" in status
    assert "ollama" in status["providers"]
    for provider_key in ["enabled", "healthy", "latency_ms", "cooldown_remaining_s"]:
        assert provider_key in status["providers"]["ollama"]


# ── routing rules ──────────────────────────────────────────────────

def test_rule_glob_matches_prefix_pattern() -> None:
    engine = ModelRoutingEngine()
    engine._routing_rules = [
        {
            "name": "boost openai o1",
            "model": "o1-*",
            "provider": "openai",
            "priority_boost": 2.0,
            "enabled": True,
        }
    ]
    with patch.object(
        engine,
        "_provider_available_locked",
        return_value=True,
    ):
        engine._provider_credentials["openai"] = True
        decision = engine.route("o1-preview", registry_models=[])
        assert "rule_boost openai x2.0" in decision.rules_applied
        openai_cand = next((c for c in decision.candidates if c.provider == "openai"), None)
        assert openai_cand is not None
        assert openai_cand.score > 150


def test_rule_glob_does_not_match_different_model() -> None:
    engine = ModelRoutingEngine()
    engine._routing_rules = [
        {
            "name": "boost openai o1",
            "model": "o1-*",
            "provider": "openai",
            "priority_boost": 2.0,
            "enabled": True,
        }
    ]
    with patch.object(
        engine,
        "_provider_available_locked",
        return_value=True,
    ):
        engine._provider_credentials["openai"] = True
        decision = engine.route("gemma4:e2b", registry_models=[])
        assert "rule_boost openai x2.0" not in decision.rules_applied


def test_rule_skipped_when_registry_match_healthy() -> None:
    engine = ModelRoutingEngine()
    engine._routing_rules = [
        {
            "name": "boost ollama",
            "model": "*",
            "provider": "ollama",
            "priority_boost": 2.0,
            "enabled": True,
        }
    ]
    with patch.object(
        engine,
        "_provider_available_locked",
        return_value=True,
    ):
        registry = [{"name": "gpt-4.1-mini", "provider": "openai", "capabilities": ["chat"]}]
        decision = engine.route("gpt-4.1-mini", registry_models=registry)
        assert decision.provider == "openai"
        assert "rule_boost ollama x2.0" not in decision.rules_applied


# ── state file save retry (Windows file lock race) ─────────────────

def test_save_state_retries_on_permission_error(monkeypatch) -> None:
    engine = ModelRoutingEngine()
    real_replace = __import__("os").replace
    calls = {"n": 0}

    def flaky_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(32, "file in use")
        real_replace(src, dst)

    monkeypatch.setattr("os.replace", flaky_replace)
    engine.set_provider_enabled("ollama", True)
    assert calls["n"] == 3
    assert engine._state_path.exists()


def test_save_state_raises_after_all_retries(monkeypatch) -> None:
    engine = ModelRoutingEngine()

    def always_fail(src: str, dst: str) -> None:
        raise PermissionError(32, "file in use")

    monkeypatch.setattr("os.replace", always_fail)
    with pytest.raises(PermissionError):
        engine.set_provider_enabled("ollama", True)


# ── alias prefix stripping ─────────────────────────────────────────

def test_strip_model_route_prefix_strips_alias_prefix(monkeypatch) -> None:
    from config.settings import Settings, settings

    monkeypatch.setattr(Settings, "model_alias_prefix", lambda self: "AIIH")
    assert settings.strip_model_route_prefix("AIIH/muse-glimmer:30b") == "muse-glimmer:30b"
    assert settings.strip_model_route_prefix("anthropic/AIIH/muse-glimmer:30b") == "muse-glimmer:30b"
    assert settings.strip_model_route_prefix("openai/AIIH/gemma4:26b") == "gemma4:26b"
    assert settings.strip_model_route_prefix("openai/gpt-4.1-mini") == "gpt-4.1-mini"
    assert settings.resolve_model_alias("AIIH/muse-glimmer:30b") == "muse-glimmer:30b"


def test_alias_prefixed_model_routes_to_registry_not_fallback(monkeypatch) -> None:
    from config.settings import Settings, settings

    monkeypatch.setattr(Settings, "model_alias_prefix", lambda self: "AIIH")
    engine = ModelRoutingEngine()
    registry = [{
        "name": "muse-glimmer:30b",
        "provider": "ollama",
        "capabilities": ["chat", "thinking", "tools"],
        "worker_bindings": [{"node_id": "node-01", "gpu_index": 0}],
    }]
    with patch.object(engine, "_first_healthy_binding", return_value={"base_url": "http://192.168.1.200:11434"}):
        with patch.object(engine, "_provider_available_locked", return_value=True):
            decision = engine.route("AIIH/muse-glimmer:30b", registry_models=registry)
    assert decision.provider == "ollama"
    assert decision.model == "muse-glimmer:30b"
    assert decision.worker is not None
    assert "local_model_fallback" not in " ".join(decision.rules_applied)
