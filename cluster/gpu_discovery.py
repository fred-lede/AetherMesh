from __future__ import annotations

import platform
import re
import subprocess
from typing import Any

import psutil


def discover_gpus() -> list[dict[str, Any]]:
    nvidia = _discover_nvidia_gpus()
    if nvidia:
        return nvidia

    if platform.system() == "Darwin":
        return _discover_macos_mps_gpu()

    return []


def _discover_nvidia_gpus() -> list[dict[str, Any]]:
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
    ]

    completed = None
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
            break
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue

    if completed is None:
        return []

    gpus: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in {5, 6}:
            continue

        gpu_id, name, memory_total, utilization, temperature = parts[:5]
        power_draw = parts[5] if len(parts) == 6 else "0"

        try:
            gpus.append(
                {
                    "id": int(gpu_id),
                    "name": name,
                    "memory": int(float(memory_total)),
                    "utilization": _safe_float(utilization),
                    "temperature": _safe_float(temperature),
                    "power_watts": _safe_float(power_draw),
                }
            )
        except ValueError:
            continue
    return gpus


def _discover_macos_mps_gpu() -> list[dict[str, Any]]:
    memory = psutil.virtual_memory()
    total_mb = int(memory.total / (1024 * 1024))
    used_mb = int((memory.total - memory.available) / (1024 * 1024))
    utilization, temperature, power_watts = _read_macos_powermetrics()

    return [
        {
            "id": 0,
            "name": "Apple Silicon GPU (MPS)",
            "memory": total_mb,
            "utilization": utilization,
            "temperature": temperature,
            "power_watts": power_watts,
            "metadata": {
                "memory_type": "unified",
                "unified_memory_used_mb": used_mb,
                "utilization_source": "powermetrics",
                "power_source": "powermetrics",
            },
        }
    ]


def _read_macos_powermetrics() -> tuple[float, float, float]:
    command = ["powermetrics", "-n", "1", "-i", "100", "--samplers", "gpu_power"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, PermissionError):
        return 0.0, 0.0, 0.0

    output = completed.stdout
    utilization = _extract_float(
        output,
        [
            r"GPU HW active residency:\s*([0-9.]+)%",
            r"GPU active residency:\s*([0-9.]+)%",
        ],
    )
    temperature = _extract_float(
        output,
        [
            r"GPU die temperature:\s*([0-9.]+)",
            r"GPU temperature:\s*([0-9.]+)",
            r"Temperature:\s*([0-9.]+).*GPU",
        ],
    )
    power_watts = _extract_power_watts(output)
    return utilization, temperature, power_watts


def _extract_power_watts(text: str) -> float:
    mw = _extract_float(text, [r"GPU power:\s*([0-9.]+)\s*mW"])
    if mw > 0:
        return mw / 1000.0
    return _extract_float(text, [r"GPU power:\s*([0-9.]+)\s*W"])


def _extract_float(text: str, patterns: list[str]) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0

