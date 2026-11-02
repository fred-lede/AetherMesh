from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import cluster.gpu_discovery as gpu_discovery


def _completed(stdout: str) -> MagicMock:
    completed = MagicMock(returncode=0)
    completed.stdout = stdout
    return completed


def test_nvidia_discovery_parses_free_memory(monkeypatch):
    stdout = (
        "0, NVIDIA GeForce RTX 5090, 32607, 851, 5, 54, 14.50\n"
        "1, NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 2346, 40, 50, 22.00\n"
    )
    monkeypatch.setattr(gpu_discovery.subprocess, "run", lambda *a, **k: _completed(stdout))
    gpus = gpu_discovery.discover_gpus()
    assert gpus[0]["memory"] == 32607
    assert gpus[0]["memory_used_mb"] == 851
    assert gpus[0]["memory_free_mb"] == 31756
    assert gpus[1]["memory_free_mb"] == 14030
    assert gpus[0]["utilization"] == 5.0
    assert gpus[0]["power_watts"] == 14.5


def test_nvidia_discovery_six_field_variant(monkeypatch):
    stdout = "0, GPU X, 8192, 1000, 10, 50\n"
    monkeypatch.setattr(gpu_discovery.subprocess, "run", lambda *a, **k: _completed(stdout))
    gpus = gpu_discovery.discover_gpus()
    assert gpus[0]["memory_free_mb"] == 7192
    assert gpus[0]["power_watts"] == 0.0


def test_nvidia_discovery_free_memory_clamped_to_zero(monkeypatch):
    stdout = "0, GPU X, 8192, 9000, 10, 50\n"
    monkeypatch.setattr(gpu_discovery.subprocess, "run", lambda *a, **k: _completed(stdout))
    assert gpu_discovery.discover_gpus()[0]["memory_free_mb"] == 0


def test_discovery_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_discovery.subprocess, "run", boom)
    assert gpu_discovery.discover_gpus() == []


def test_macos_mps_reports_null_free_memory(monkeypatch):
    monkeypatch.setattr(gpu_discovery, "_read_macos_powermetrics", lambda: (10.0, 50.0, 5.0))
    fake_mem = SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3)
    monkeypatch.setattr(gpu_discovery.psutil, "virtual_memory", lambda: fake_mem)
    gpus = gpu_discovery._discover_macos_mps_gpu()
    assert gpus[0]["memory_free_mb"] is None
    assert gpus[0]["memory_used_mb"] == 8192
