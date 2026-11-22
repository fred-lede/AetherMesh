# GPU Topology

## Supported Hardware

| GPU | VRAM | Tier | Use Case |
|-----|------|------|----------|
| NVIDIA RTX 5090 | 32 GB | S | High-performance primary |
| NVIDIA RTX 4070 Ti SUPER | 16 GB | A | Mid-range inference |
| NVIDIA Tesla P40 | 24 GB | B | High-capacity batch |
| Apple Silicon M4 | 64 GB (unified) | C | Large memory models |

> **Framework note:** Tesla P40 is Pascal (`sm_61`). PyTorch CUDA 13.x wheels and cu128
> builds from PyTorch 2.8 onward do not include Pascal kernels, so `torch`-based engines
> such as `qwen_asr` need an interpreter whose `torch.cuda.get_arch_list()` contains
> `sm_60`. See [Qwen3-ASR on Pascal GPUs](../README.md#qwen3-asr-on-pascal-gpus-tesla-p40--sm_61)
> for the `AIIH_QWEN_ASR_PYTHON` workaround.

## Scheduling Dimensions
The GPU runtime (`runtime/gpu/`) schedules models based on:

| Dimension | Source | Weight |
|-----------|--------|--------|
| VRAM | `vram_scheduler.py` | Required |
| Model locality | `model_affinity.py` | Prefers loaded models |
| Queue depth | Worker heartbeats | Load balancing |
| GPU tier | Hardware detection | Higher tier preferred |

## VRAM Scheduler
The `VramScheduler` tracks available VRAM per GPU and allocates model loads to GPUs with sufficient free memory. When memory is constrained, models may be unloaded from less active GPUs to make room.

## Model Affinity
`ModelAffinity` tracks which models are currently loaded on each GPU. The scheduler prefers keeping models loaded to avoid reload latency, using a least-recently-used eviction policy when memory pressure requires unloading.
