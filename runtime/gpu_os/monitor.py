from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUDevice:
    device_id: str = ""
    name: str = ""
    total_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    utilization: float = 0.0
    temperature: float = 0.0
    models_loaded: list[str] = field(default_factory=list)
    healthy: bool = True

    @property
    def used_vram_gb(self) -> float:
        return self.total_vram_gb - self.available_vram_gb

    @property
    def vram_used_pct(self) -> float:
        if self.total_vram_gb <= 0:
            return 0.0
        return (self.used_vram_gb / self.total_vram_gb) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "total_vram_gb": self.total_vram_gb,
            "available_vram_gb": self.available_vram_gb,
            "used_vram_gb": self.used_vram_gb,
            "vram_used_pct": round(self.vram_used_pct, 1),
            "utilization": self.utilization,
            "temperature": self.temperature,
            "models_loaded": list(self.models_loaded),
            "healthy": self.healthy,
        }


class GPUManager:
    def __init__(self) -> None:
        self._devices: dict[str, GPUDevice] = {}

    def register_device(self, device: GPUDevice) -> None:
        self._devices[device.device_id] = device

    def unregister_device(self, device_id: str) -> None:
        self._devices.pop(device_id, None)

    def get_device(self, device_id: str) -> GPUDevice | None:
        return self._devices.get(device_id)

    def list_devices(self) -> list[GPUDevice]:
        return list(self._devices.values())

    def allocate_vram(self, device_id: str, amount_gb: float) -> bool:
        device = self._devices.get(device_id)
        if not device:
            return False
        if device.available_vram_gb < amount_gb:
            return False
        device.available_vram_gb -= round(amount_gb, 2)
        return True

    def release_vram(self, device_id: str, amount_gb: float) -> None:
        device = self._devices.get(device_id)
        if not device:
            return
        device.available_vram_gb = round(min(
            device.available_vram_gb + amount_gb, device.total_vram_gb
        ), 2)

    def set_utilization(self, device_id: str, pct: float) -> None:
        device = self._devices.get(device_id)
        if device:
            device.utilization = round(pct, 1)

    def refresh(self, ttl_seconds: float = 15.0) -> None:
        now = time.time()
        if getattr(self, "_last_refresh", 0) + ttl_seconds > now:
            return
        self._last_refresh = now
        try:
            from cluster.gpu_discovery import discover_gpus

            for gpu in discover_gpus():
                device_id = f"cuda:{gpu.get('id', 0)}"
                total_mb = int(gpu.get("memory", 0))
                existing = self._devices.get(device_id)
                if existing:
                    existing.utilization = gpu.get("utilization", 0.0)
                    existing.temperature = gpu.get("temperature", 0.0)
                else:
                    dev = GPUDevice(
                        device_id=device_id,
                        name=gpu.get("name", "Unknown GPU"),
                        total_vram_gb=round(total_mb / 1024.0, 1),
                        available_vram_gb=round(total_mb / 1024.0, 1),
                        utilization=gpu.get("utilization", 0.0),
                        temperature=gpu.get("temperature", 0.0),
                    )
                    self.register_device(dev)
        except Exception:
            pass

    def snapshot(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._devices.values()]

    def total_vram_gb(self) -> float:
        return sum(d.total_vram_gb for d in self._devices.values())

    def available_vram_gb(self) -> float:
        return sum(d.available_vram_gb for d in self._devices.values())


gpu_manager = GPUManager()


def _auto_register_gpus() -> None:
    try:
        from cluster.gpu_discovery import discover_gpus

        for gpu in discover_gpus():
            total_mb = int(gpu.get("memory", 0))
            dev = GPUDevice(
                device_id=f"cuda:{gpu.get('id', 0)}",
                name=gpu.get("name", "Unknown GPU"),
                total_vram_gb=round(total_mb / 1024.0, 1),
                available_vram_gb=round(total_mb / 1024.0, 1),
                utilization=gpu.get("utilization", 0.0),
                temperature=gpu.get("temperature", 0.0),
            )
            gpu_manager.register_device(dev)
    except Exception:
        pass  # GPU discovery may fail (no driver, no nvidia-smi, etc.)


_auto_register_gpus()
