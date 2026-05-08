from __future__ import annotations

from runtime.gpu_os.monitor import GPUManager, GPUDevice
from runtime.gpu_os.scheduler import ModelScheduler
from runtime.gpu_os import gpu_manager as global_gpu_manager, model_scheduler as global_model_scheduler


def setup_function() -> None:
    global_gpu_manager._devices.clear()
    global_model_scheduler._models.clear()
    global_model_scheduler._access_order.clear()
    global_model_scheduler._load_order.clear()


def test_gpu_register_device() -> None:
    d = GPUDevice(device_id="cuda:0", name="A100", total_vram_gb=80.0, available_vram_gb=80.0)
    global_gpu_manager.register_device(d)
    assert global_gpu_manager.get_device("cuda:0") is d
    assert len(global_gpu_manager.list_devices()) == 1


def test_gpu_allocate_vram() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=80.0))
    assert global_gpu_manager.allocate_vram("cuda:0", 40.0) is True
    assert global_gpu_manager.get_device("cuda:0").available_vram_gb == 40.0


def test_gpu_allocate_insufficient() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=10.0, available_vram_gb=10.0))
    assert global_gpu_manager.allocate_vram("cuda:0", 20.0) is False


def test_gpu_release_vram() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=0.0))
    global_gpu_manager.release_vram("cuda:0", 40.0)
    assert global_gpu_manager.get_device("cuda:0").available_vram_gb == 40.0


def test_gpu_release_does_not_exceed_total() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=80.0))
    global_gpu_manager.release_vram("cuda:0", 999.0)
    assert global_gpu_manager.get_device("cuda:0").available_vram_gb == 80.0


def test_gpu_vram_properties() -> None:
    d = GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=40.0)
    assert d.used_vram_gb == 40.0
    assert d.vram_used_pct == 50.0


def test_gpu_snapshot() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", name="A100", total_vram_gb=80.0, available_vram_gb=80.0))
    s = global_gpu_manager.snapshot()
    assert len(s) == 1
    assert s[0]["device_id"] == "cuda:0"


def test_gpu_total_and_available() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=40.0))
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:1", total_vram_gb=80.0, available_vram_gb=80.0))
    assert global_gpu_manager.total_vram_gb() == 160.0
    assert global_gpu_manager.available_vram_gb() == 120.0


def test_scheduler_load_model() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=80.0))
    assert global_model_scheduler.load("gemma-7b", vram_gb=16.0, device_id="cuda:0") is True
    assert global_model_scheduler.is_loaded("gemma-7b") is True


def test_scheduler_unload_model() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=80.0))
    global_model_scheduler.load("gemma-7b", vram_gb=16.0, device_id="cuda:0")
    global_model_scheduler.unload("gemma-7b")
    assert global_model_scheduler.is_loaded("gemma-7b") is False


def test_scheduler_eviction_lru() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=10.0))
    global_model_scheduler.load("model-a", vram_gb=8.0, device_id="cuda:0")
    global_model_scheduler.load("model-b", vram_gb=8.0, device_id="cuda:0")
    assert not global_model_scheduler.is_loaded("model-a")
    assert global_model_scheduler.is_loaded("model-b")


def test_scheduler_access_tracking() -> None:
    global_gpu_manager.register_device(GPUDevice(device_id="cuda:0", total_vram_gb=80.0, available_vram_gb=80.0))
    global_model_scheduler.load("m1", vram_gb=16.0, device_id="cuda:0")
    global_model_scheduler.access("m1")
    models = global_model_scheduler.get_loaded_models()
    assert models[0].use_count == 1


def test_scheduler_snapshot() -> None:
    snap = global_model_scheduler.snapshot()
    assert "loaded_count" in snap
    assert "models" in snap
