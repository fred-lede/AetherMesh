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

    def snapshot(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._devices.values()]

    def total_vram_gb(self) -> float:
        return sum(d.total_vram_gb for d in self._devices.values())

    def available_vram_gb(self) -> float:
        return sum(d.available_vram_gb for d in self._devices.values())


gpu_manager = GPUManager()
