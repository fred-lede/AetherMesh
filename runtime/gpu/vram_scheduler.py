from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("gpu.vram")


@dataclass
class VRAMProfile:
    total_mb: int
    used_mb: int = 0
    free_mb: int = 0

    @property
    def utilization_pct(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100


@dataclass
class GPUResource:
    gpu_id: str
    node_id: str
    worker_port: int
    vram: VRAMProfile
    model_loaded: str = ""
    queue_depth: int = 0
    temperature_c: float = 0.0
    power_w: float = 0.0
    healthy: bool = True
    tier: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


GPU_TIER_WEIGHTS: dict[str, int] = {
    "RTX 5090": 60,
    "RTX 4090": 50,
    "RTX 4070 Ti SUPER": 30,
    "RTX 4070 Ti": 30,
    "RTX 4070": 25,
    "Tesla P40": 10,
    "Tesla T4": 15,
    "Apple Silicon M4": 20,
    "Apple Silicon M3": 15,
    "Apple Silicon M2": 10,
    "unknown": 5,
}


def tier_for_gpu_name(name: str) -> str:
    for tier_key in GPU_TIER_WEIGHTS:
        if tier_key.lower() in name.lower():
            return tier_key
    return "unknown"


class VRAMScheduler:
    def __init__(self) -> None:
        self._gpus: dict[str, GPUResource] = {}

    def update_gpu(self, resource: GPUResource) -> None:
        key = f"{resource.node_id}:{resource.worker_port}"
        self._gpus[key] = resource

    def remove_gpu(self, node_id: str, worker_port: int) -> None:
        self._gpus.pop(f"{node_id}:{worker_port}", None)

    def best_gpu(self, required_vram_mb: int = 0, prefer_model: str = "") -> GPUResource | None:
        candidates = sorted(
            self._gpus.values(),
            key=lambda g: (
                - (g.vram.free_mb >= required_vram_mb) if required_vram_mb > 0 else 0,
                - GPU_TIER_WEIGHTS.get(g.tier, 0),
                g.queue_depth,
                g.vram.utilization_pct,
            ),
        )
        for gpu in candidates:
            if not gpu.healthy:
                continue
            if required_vram_mb > 0 and gpu.vram.free_mb < required_vram_mb:
                continue
            return gpu
        return None

    def all_gpus(self) -> list[GPUResource]:
        return list(self._gpus.values())


vram_scheduler = VRAMScheduler()
