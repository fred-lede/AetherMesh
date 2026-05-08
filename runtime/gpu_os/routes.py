from __future__ import annotations

from fastapi import APIRouter

from runtime.gpu_os import gpu_manager, model_scheduler
from runtime.gpu_os.monitor import GPUDevice

gpu_router = APIRouter(prefix="/v1/gpu", tags=["gpu-os"])


@gpu_router.get("/status")
def gpu_status():
    return {
        "devices": gpu_manager.snapshot(),
        "scheduler": model_scheduler.snapshot(),
        "total_vram_gb": gpu_manager.total_vram_gb(),
        "available_vram_gb": gpu_manager.available_vram_gb(),
    }


@gpu_router.post("/models/load")
def gpu_load_model(model_name: str, vram_gb: float = 1.0, device_id: str = ""):
    ok = model_scheduler.load(model_name, vram_gb=vram_gb, device_id=device_id)
    if not ok:
        return {"ok": False, "error": "Failed to allocate VRAM"}
    return {"ok": True, "model": model_name, "device_id": device_id}


@gpu_router.post("/models/unload")
def gpu_unload_model(model_name: str):
    model_scheduler.unload(model_name)
    return {"ok": True, "model": model_name}


@gpu_router.post("/devices/register")
def gpu_register_device(device_id: str, name: str = "", total_vram_gb: float = 0.0):
    dev = GPUDevice(
        device_id=device_id,
        name=name or device_id,
        total_vram_gb=total_vram_gb,
        available_vram_gb=total_vram_gb,
    )
    gpu_manager.register_device(dev)
    return {"ok": True, "device_id": device_id}
