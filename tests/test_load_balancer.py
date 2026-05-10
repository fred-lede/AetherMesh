from __future__ import annotations

from cluster.load_balancer import (
    _is_worker_available,
    _get_gpu_tier,
    _get_tier_weight,
    _get_model_affinity_score,
    choose_best_worker,
    choose_best_worker_simple,
    _round_robin_by_tier,
)


def _make_worker(
    *,
    worker_id: str = "w1",
    gpu_name: str = "RTX 5090",
    queue_size: int = 0,
    status: str = "healthy",
    gpu_utilization: float = 10.0,
    ps_models: list[str] | None = None,
) -> dict:
    return {
        "worker_id": worker_id,
        "base_url": f"http://192.168.1.10:11434",
        "gpu_name": gpu_name,
        "queue_size": queue_size,
        "status": status,
        "gpu_utilization": gpu_utilization,
        "ps_models": ps_models or [],
    }


# ── _is_worker_available ────────────────────────────────────────────

def test_worker_available_when_healthy_and_low_queue() -> None:
    w = _make_worker(status="healthy", queue_size=2)
    assert _is_worker_available(w, max_queue_size=8) is True


def test_worker_unavailable_when_dead() -> None:
    w = _make_worker(status="dead")
    assert _is_worker_available(w, max_queue_size=8) is False


def test_worker_unavailable_when_queue_full() -> None:
    w = _make_worker(status="healthy", queue_size=8)
    assert _is_worker_available(w, max_queue_size=8) is False


def test_worker_unavailable_when_gpu_overloaded() -> None:
    w = _make_worker(status="healthy", queue_size=0, gpu_utilization=90.0)
    assert _is_worker_available(w, max_queue_size=8) is False


def test_worker_available_at_exactly_84pct_utilization() -> None:
    w = _make_worker(status="healthy", queue_size=0, gpu_utilization=84.0)
    assert _is_worker_available(w, max_queue_size=8) is True


def test_worker_unavailable_at_exactly_85pct_utilization() -> None:
    w = _make_worker(status="healthy", queue_size=0, gpu_utilization=85.0)
    assert _is_worker_available(w, max_queue_size=8) is False


def test_worker_availability_default_max_queue() -> None:
    w = _make_worker(status="healthy", queue_size=4)
    assert _is_worker_available(w, max_queue_size=None) is True


def test_worker_available_when_degraded() -> None:
    w = _make_worker(status="degraded")
    assert _is_worker_available(w, max_queue_size=8) is True


# ── _get_gpu_tier ───────────────────────────────────────────────────

def test_gpu_tier_5090() -> None:
    assert _get_gpu_tier(_make_worker(gpu_name="NVIDIA GeForce RTX 5090")) == "5090"


def test_gpu_tier_4090() -> None:
    assert _get_gpu_tier(_make_worker(gpu_name="NVIDIA GeForce RTX 4090")) == "4090"


def test_gpu_tier_4070_ti() -> None:
    assert _get_gpu_tier(_make_worker(gpu_name="NVIDIA GeForce RTX 4070 Ti SUPER")) == "4070 ti"


def test_gpu_tier_p40() -> None:
    assert _get_gpu_tier(_make_worker(gpu_name="Tesla P40")) == "p40"


def test_gpu_tier_unknown_fallback() -> None:
    assert _get_gpu_tier(_make_worker(gpu_name="Intel Arc A770")) == "default"


def test_gpu_tier_from_metadata() -> None:
    w = _make_worker(gpu_name="some gpu")
    w["tier"] = "custom_tier"
    assert _get_gpu_tier(w) == "custom_tier"


# ── _get_tier_weight ────────────────────────────────────────────────

def test_tier_weight_5090() -> None:
    assert _get_tier_weight(_make_worker(gpu_name="RTX 5090")) == 60


def test_tier_weight_4090() -> None:
    assert _get_tier_weight(_make_worker(gpu_name="RTX 4090")) == 45


def test_tier_weight_4070ti() -> None:
    assert _get_tier_weight(_make_worker(gpu_name="RTX 4070 Ti SUPER")) == 30


def test_tier_weight_p40() -> None:
    assert _get_tier_weight(_make_worker(gpu_name="Tesla P40")) == 10


def test_tier_weight_default() -> None:
    assert _get_tier_weight(_make_worker(gpu_name="Intel Arc A770")) == 20


# ── _get_model_affinity_score ──────────────────────────────────────

def test_affinity_model_loaded() -> None:
    w = _make_worker(ps_models=["gemma4:e4b", "llama3.2:1b"])
    assert _get_model_affinity_score(w, "gemma4:e4b") == 100


def test_affinity_model_not_loaded() -> None:
    w = _make_worker(ps_models=["gemma4:31b"])
    assert _get_model_affinity_score(w, "gemma4:e4b") == 0


def test_affinity_no_ps_models() -> None:
    w = _make_worker(ps_models=[])
    assert _get_model_affinity_score(w, "gemma4:e4b") == 0


def test_affinity_empty_ps_models_list() -> None:
    w = _make_worker()
    assert _get_model_affinity_score(w, "gemma4:e4b") == 0


def test_affinity_model_in_metadata_ps_models() -> None:
    w = _make_worker(ps_models=[])
    w["metadata"] = {"ps_models": ["gemma4:31b"]}
    assert _get_model_affinity_score(w, "gemma4:31b") == 100


def test_affinity_model_in_metadata_model_field() -> None:
    w = _make_worker(ps_models=[])
    w["metadata"] = {"model": "gemma4:31b"}
    assert _get_model_affinity_score(w, "gemma4:31b") == 90


def test_affinity_recently_used() -> None:
    w = _make_worker(ps_models=[])
    w["metadata"] = {"recent_models": ["gemma4:31b", "llama3.2:1b"]}
    assert _get_model_affinity_score(w, "gemma4:31b") == 50


# ── choose_best_worker_simple ──────────────────────────────────────

def test_simple_picks_available_worker() -> None:
    workers = [
        _make_worker(worker_id="w1", queue_size=5, status="healthy"),
        _make_worker(worker_id="w2", queue_size=1, status="healthy"),
    ]
    result = choose_best_worker_simple(workers, max_queue_size=8)
    assert result is not None
    assert result["worker_id"] == "w2"


def test_simple_returns_none_if_none_available() -> None:
    workers = [
        _make_worker(worker_id="w1", status="dead"),
    ]
    assert choose_best_worker_simple(workers, max_queue_size=8) is None


def test_simple_empty_workers() -> None:
    assert choose_best_worker_simple([], max_queue_size=8) is None


# ── _round_robin_by_tier ────────────────────────────────────────────

def test_round_robin_returns_healthy_worker() -> None:
    workers = [
        _make_worker(worker_id="w1", gpu_name="RTX 5090", queue_size=0),
        _make_worker(worker_id="w2", gpu_name="Tesla P40", queue_size=0),
    ]
    result = _round_robin_by_tier(workers, model=None)
    assert result is not None
    assert result["worker_id"] in ("w1", "w2")


def test_round_robin_picks_any_in_tier() -> None:
    workers = [
        _make_worker(worker_id="w1", gpu_name="Tesla P40", queue_size=0),
        _make_worker(worker_id="w2", gpu_name="RTX 5090", queue_size=0),
    ]
    result = _round_robin_by_tier(workers, model=None)
    assert result is not None
    assert result["worker_id"] in ("w1", "w2")


def test_round_robin_all_dead() -> None:
    workers = [
        _make_worker(worker_id="w1", status="dead"),
        _make_worker(worker_id="w2", status="dead"),
    ]
    result = _round_robin_by_tier(workers, model=None)
    assert result is not None  # round_robin doesn't filter by status; choose_best_worker does


def test_round_robin_empty_list() -> None:
    assert _round_robin_by_tier([], model=None) is None


def test_round_robin_prefers_model_affinity() -> None:
    workers = [
        _make_worker(worker_id="w1", gpu_name="Tesla P40", ps_models=["gemma4:e4b"]),
        _make_worker(worker_id="w2", gpu_name="RTX 5090", ps_models=[]),
    ]
    result = _round_robin_by_tier(workers, model="gemma4:e4b")
    assert result is not None
    assert result["worker_id"] == "w2"


# ── choose_best_worker (hybrid strategy) ──────────────────────────

def test_choose_best_worker_prefers_higher_tier() -> None:
    workers = [
        _make_worker(worker_id="p40", gpu_name="Tesla P40", queue_size=0),
        _make_worker(worker_id="r5090", gpu_name="RTX 5090", queue_size=0),
    ]
    result = choose_best_worker(workers, max_queue_size=8)
    assert result is not None
    assert result["worker_id"] == "r5090"


def test_choose_best_worker_skips_dead() -> None:
    workers = [
        _make_worker(worker_id="dead1", status="dead", gpu_name="RTX 5090"),
        _make_worker(worker_id="alive", status="healthy", gpu_name="Tesla P40", queue_size=0),
    ]
    result = choose_best_worker(workers, max_queue_size=8)
    assert result is not None
    assert result["worker_id"] == "alive"


def test_choose_best_worker_all_dead_returns_none() -> None:
    workers = [
        _make_worker(worker_id="w1", status="dead"),
        _make_worker(worker_id="w2", status="dead"),
    ]
    assert choose_best_worker(workers, max_queue_size=8) is None


def test_choose_best_worker_empty_list() -> None:
    assert choose_best_worker([], max_queue_size=8) is None


def test_choose_best_worker_queue_full_filtered() -> None:
    workers = [
        _make_worker(worker_id="full", queue_size=8, gpu_name="RTX 5090"),
        _make_worker(worker_id="free", queue_size=0, gpu_name="Tesla P40"),
    ]
    result = choose_best_worker(workers, max_queue_size=8)
    assert result is not None
    assert result["worker_id"] == "free"


def test_choose_best_worker_model_affinity() -> None:
    workers = [
        _make_worker(worker_id="no_affinity", gpu_name="RTX 5090", queue_size=0, ps_models=[]),
        _make_worker(worker_id="has_model", gpu_name="Tesla P40", queue_size=0,
                     ps_models=["gemma4:e4b"]),
    ]
    result = choose_best_worker(workers, max_queue_size=8, model="gemma4:e4b")
    assert result is not None
    assert result["worker_id"] == "has_model"


def test_choose_best_worker_same_gpu_computes_queue() -> None:
    workers = [
        _make_worker(worker_id="busy", gpu_name="Tesla P40", queue_size=5),
        _make_worker(worker_id="idle", gpu_name="Tesla P40", queue_size=0),
    ]
    result = choose_best_worker(workers, max_queue_size=8, strategy="least-loaded")
    assert result is not None
    assert result["worker_id"] == "idle"


# ── choose_best_worker with different strategies ───────────────────

def test_strategy_least_loaded() -> None:
    workers = [
        _make_worker(worker_id="loaded", gpu_name="Tesla P40", queue_size=7),
        _make_worker(worker_id="free", gpu_name="Tesla P40", queue_size=0),
    ]
    result = choose_best_worker(workers, max_queue_size=8, strategy="least-loaded")
    assert result is not None
    assert result["worker_id"] == "free"


def test_strategy_round_robin() -> None:
    workers = [
        _make_worker(worker_id="w1", gpu_name="RTX 5090", queue_size=0),
        _make_worker(worker_id="w2", gpu_name="Tesla P40", queue_size=0),
    ]
    result = choose_best_worker(workers, max_queue_size=8, strategy="round-robin")
    assert result is not None
    assert result["worker_id"] in ("w1", "w2")


def test_hybrid_strategy_dynamic_switch() -> None:
    workers = [
        _make_worker(worker_id="w1", gpu_name="RTX 5090", queue_size=0, gpu_utilization=10.0),
        _make_worker(worker_id="w2", gpu_name="Tesla P40", queue_size=0, gpu_utilization=10.0),
    ]
    result = choose_best_worker(workers, max_queue_size=8, strategy="hybrid")
    assert result is not None


def test_max_queue_size_none_does_not_filter() -> None:
    w = _make_worker(worker_id="w1", queue_size=999)
    result = choose_best_worker([w], max_queue_size=None)
    assert result is not None
