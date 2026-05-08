from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from runtime.gpu_os.monitor import gpu_manager


@dataclass
class LoadedModel:
    name: str = ""
    device_id: str = ""
    vram_gb: float = 0.0
    loaded_at: float = 0.0
    last_used_at: float = 0.0
    use_count: int = 0


class EvictionPolicy:
    LRU = "lru"
    FIFO = "fifo"

    @staticmethod
    def select(lru_key: tuple, fifo_key: tuple) -> str:
        return EvictionPolicy.LRU


class ModelScheduler:
    def __init__(self, max_vram_gb: float = 0.0) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._access_order: OrderedDict[str, float] = OrderedDict()
        self._load_order: list[str] = []
        self.max_vram_gb = max_vram_gb

    def load(
        self,
        model_name: str,
        vram_gb: float = 1.0,
        device_id: str = "",
    ) -> bool:
        if model_name in self._models:
            return True

        devices = gpu_manager.list_devices()
        target = device_id or (devices[0].device_id if devices else "")

        if not target or not gpu_manager.allocate_vram(target, vram_gb):
            freed = self._evict(vram_gb)
            if not freed:
                return False
            if not gpu_manager.allocate_vram(target, vram_gb):
                return False

        now = time.time()
        model = LoadedModel(
            name=model_name,
            device_id=target,
            vram_gb=vram_gb,
            loaded_at=now,
            last_used_at=now,
        )
        self._models[model_name] = model
        self._access_order[model_name] = now
        self._load_order.append(model_name)

        device = gpu_manager.get_device(target)
        if device and model_name not in device.models_loaded:
            device.models_loaded.append(model_name)

        return True

    def unload(self, model_name: str) -> None:
        model = self._models.pop(model_name, None)
        if not model:
            return
        self._access_order.pop(model_name, None)
        if model_name in self._load_order:
            self._load_order.remove(model_name)
        gpu_manager.release_vram(model.device_id, model.vram_gb)
        device = gpu_manager.get_device(model.device_id)
        if device and model_name in device.models_loaded:
            device.models_loaded.remove(model_name)

    def unload_all(self, device_id: str | None = None) -> None:
        for name in list(self._models.keys()):
            model = self._models[name]
            if device_id is None or model.device_id == device_id:
                self.unload(name)

    def access(self, model_name: str) -> None:
        model = self._models.get(model_name)
        if model:
            now = time.time()
            model.last_used_at = now
            model.use_count += 1
            self._access_order[model_name] = now

    def get_loaded_models(self) -> list[LoadedModel]:
        return list(self._models.values())

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._models

    def _evict(self, needed_gb: float) -> bool:
        candidates = sorted(
            self._models.items(),
            key=lambda x: x[1].last_used_at,
        )
        freed = 0.0
        for name, model in candidates:
            if freed >= needed_gb:
                break
            self.unload(name)
            freed += model.vram_gb
        return freed >= needed_gb

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_vram_gb": self.max_vram_gb,
            "loaded_count": len(self._models),
            "models": [
                {
                    "name": m.name,
                    "device_id": m.device_id,
                    "vram_gb": m.vram_gb,
                    "use_count": m.use_count,
                    "loaded_seconds_ago": round(time.time() - m.loaded_at, 1),
                }
                for m in self._models.values()
            ],
        }


model_scheduler = ModelScheduler()
