"""
Advanced load balancer with:
1. Tier-based weight distribution (5090 > 4070Ti > P40)
2. Model affinity (prefer workers with model already loaded)
3. Hybrid strategy (least-loaded when low, round-robin when high)
"""

from __future__ import annotations

import random
import time
from typing import Any


# GPU Tier weights (higher = better GPU gets more traffic)
TIER_WEIGHTS = {
    "5090": 60,      # High-tier: 60%
    "5090ti": 50,
    "5090 ti": 50,
    "4090": 45,
    "4080": 40,
    "4070ti": 30,    # Mid-tier: 30%
    "4070 ti": 30,
    "4070": 25,
    "a100": 35,
    "a6000": 30,
    "a5000": 25,
    "p40": 10,       # Low-tier: 10%
    "p100": 15,
    "v100": 20,
    "default": 20,
}


def _get_gpu_tier(worker: dict[str, Any]) -> str:
    """Get GPU tier from worker info."""
    name = str(worker.get("gpu_name", "")).lower()
    tier = worker.get("tier", "").lower()
    if tier:
        return tier
    # Extract tier from GPU name
    for key in TIER_WEIGHTS:
        if key in name:
            return key
    return "default"


def _get_tier_weight(worker: dict[str, Any]) -> int:
    """Get weight for this GPU tier."""
    tier = _get_gpu_tier(worker)
    return TIER_WEIGHTS.get(tier, TIER_WEIGHTS["default"])


def _get_model_affinity_score(worker: dict[str, Any], model: str) -> int:
    """
    Calculate model affinity score.
    Higher score = prefer this worker (model already loaded).
    """
    if not model:
        return 0

    # Check if model is in loaded models (ps_models)
    loaded_models = worker.get("ps_models", [])
    if model in loaded_models:
        return 100  # Perfect match

    # Check in other metadata
    metadata = worker.get("metadata", {})
    if isinstance(metadata, dict):
        if model in metadata.get("ps_models", []):
            return 100
        if metadata.get("model") == model:
            return 90

    # Check recently used models
    recently_used = metadata.get("recent_models", []) or []
    if model in recently_used:
        return 50

    return 0


def _is_worker_available(worker: dict[str, Any], max_queue_size: int | None) -> bool:
    """Check if worker is available for dispatch."""
    status = worker.get("status", "").lower()

    # Dead workers are not available
    if status == "dead":
        return False

    # Check queue capacity
    if max_queue_size and max_queue_size > 0:
        queue_size = int(worker.get("queue_size", 0))
        if queue_size >= max_queue_size:
            return False

    # Check GPU saturation (proactive diversion at 85%)
    gpu_util = float(worker.get("gpu_utilization", 0))
    if gpu_util >= 85:
        return False

    return True


def choose_best_worker(
    workers: list[dict[str, Any]],
    max_queue_size: int | None = None,
    model: str | None = None,
    strategy: str = "hybrid",
) -> dict | None:
    """
    Advanced worker selection with tier-based weights, model affinity, and hybrid strategy.

    Args:
        workers: List of worker dicts
        max_queue_size: Max queue size per worker
        model: Requested model name (for affinity)
        strategy: "least-loaded", "round-robin", or "hybrid" (default)

    Returns:
        Best worker dict or None
    """
    if not workers:
        return None

    # Filter available workers
    available = [w for w in workers if _is_worker_available(w, max_queue_size)]

    if not available:
        return None

    # Score each worker
    scored = []
    for worker in available:
        # Base score: lower is better
        gpu_util = float(worker.get("gpu_utilization", 100))
        queue_size = int(worker.get("queue_size", 0))

        # 1. Tier weight (higher tier = lower score = better)
        tier_weight = _get_tier_weight(worker)
        tier_score = 100 - tier_weight  # Invert: high weight -> low score

        # 2. Model affinity
        affinity = _get_model_affinity_score(worker, model)

        # 3. Queue score
        queue_score = queue_size

        # Combined score (lower is better)
        total_score = (
            tier_score * 0.3 +
            queue_score * 0.4 +
            affinity * (-1.0) * 0.3  # Negative because higher affinity is better
        )

        # Add slight random to prevent thundering herd
        total_score += random.uniform(-1, 1)

        scored.append((total_score, worker))

    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[0])

    # Strategy selection
    if strategy == "round-robin":
        # For round-robin, use tier-based distribution
        selected = _round_robin_by_tier(available, model)
    elif strategy == "least-loaded":
        # Pure least-loaded
        selected = scored[0][1] if scored else None
    else:  # hybrid
        # Hybrid: if low GPU utilization, use weight affinity; otherwise use round-robin
        avg_util = sum(float(w.get("gpu_utilization", 0)) for w in available) / len(available)
        if avg_util < 50:
            # Low load: use best score
            selected = scored[0][1] if scored else None
        else:
            # High load: distribute evenly
            selected = _round_robin_by_tier(available, model)

    return selected


def _round_robin_by_tier(
    workers: list[dict[str, Any]],
    model: str | None,
) -> dict[str,Any] | None:
    """Round-robin selection weighted by tier."""
    if not workers:
        return None

    # Separate by tier
    tiers = {"high": [], "mid": [], "low": [], "unknown": []}
    for w in workers:
        tier = _get_gpu_tier(w)
        if tier in ["5090", "5090ti", "4090", "4080"]:
            tiers["high"].append(w)
        elif tier in ["4070ti", "4070", "a100", "a6000"]:
            tiers["mid"].append(w)
        elif tier in ["p40", "p100", "v100"]:
            tiers["low"].append(w)
        else:
            tiers["unknown"].append(w)

    # Attempt high tier first
    for tier_group in [tiers["high"], tiers["mid"], tiers["unknown"], tiers["low"]]:
        if tier_group:
            # Shuffle for round-robin
            random.shuffle(tier_group)
            # Prefer worker with model affinity
            if model:
                for w in tier_group:
                    if _get_model_affinity_score(w, model) > 0:
                        return w
            # Return random from tier
            return tier_group[0]

    return None


# Legacy function for backward compatibility
def choose_best_worker_simple(
    workers: list[dict[str, Any]],
    max_queue_size: int | None = None,
) -> dict[str, Any] | None:
    """Simple worker selection (legacy compatibility)."""
    return choose_best_worker(workers, max_queue_size, model=None, strategy="hybrid")