# GPU Scheduling

AetherMesh manages multiple GPU tiers for optimal workload distribution.

## Hardware Topology

| GPU | VRAM | Tier | Role |
|---|---|---|---|
| RTX 5090 | 32 GB | S | High-performance inference |
| RTX 4070 Ti SUPER | 16 GB | A | Mid-range inference |
| Tesla P40 (×N) | 24 GB each | B | High-capacity batch/background |
| Apple Silicon M4 | 64 GB (unified) | B+ | Large model pool |

## Scheduling Dimensions

| Dimension | Strategy |
|---|---|
| VRAM | Assign to GPU with sufficient free VRAM |
| Model locality | Prefer GPU with model already loaded |
| Queue depth | Divert when queue > threshold |
| GPU tier | Weighted distribution (S > A > B) |
| Latency history | Penalize high-latency workers |

## Components

| Module | Purpose |
|---|---|
| `runtime/gpu/vram_scheduler.py` | VRAM-aware GPU selection |
| `runtime/gpu/model_affinity.py` | Track model loading across workers |
| `runtime/gpu/warm_pool.py` | Keep models warm with keepalive |
