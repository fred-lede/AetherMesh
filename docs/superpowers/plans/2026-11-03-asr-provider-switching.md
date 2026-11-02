# ASR Provider 熱切換 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** faster-whisper 與 Qwen3-ASR 雙引擎並存，可於 dashboard `/admin/audio` 熱切換引擎/GPU/遠端叢集節點，不需重啟 router。

**Architecture:** `runtime/asr/asr_manager.py` 是唯一事實來源（狀態機 + 版本號 + 背景載入 + 原子替換 + 失敗回滾）；`router/asr_admin_router.py` 提供 `GET/POST /api/asr/config` 純協定層；dashboard 以 x-api-key proxy 轉發。遠端節點走既有 cluster 機制（node_agent 心跳攜帶 `metadata.asr_worker`，節點上跑新服務 `node/asr_worker.py`）。

**Tech Stack:** Python 3.13/3.14 + FastAPI + pytest（asyncio_mode=auto）；Qwen3-ASR 走 PyPI `qwen-asr`（transformers backend，vLLM 不用）；遠端用 OpenAI 相容 multipart HTTP。

**Spec:** `docs/superpowers/specs/2026-11-03-asr-provider-switching-design.md`（§1 目標、§2 程式碼地圖、§3 架構、§4 詳細設計、§5 錯誤政策、§6 測試計畫、§7 驗收、§8 Out of Scope）

## Global Constraints

- Python：4-space 縮排、檔案首行 `from __future__ import annotations`、公開 API 一律型別提示、**無 docstring/註解**（除非本計畫明確給出）。
- Imports：stdlib → 第三方 → 內部，三組之間空行分隔，一律絕對 import。
- Logging：模組層 `logger = logging.getLogger("<dotted.path>")`，訊息用 `%s`/`%.1f` 格式化，**不用 f-string**。
- Settings：只能屬性存取（`settings.asr_model`），禁止 `settings.get()`。
- 測試：pytest + `MagicMock`/`monkeypatch`/`patch`，**不依賴 GPU、網路、`qwen-asr` 套件、torch**；檔名 `tests/test_<module>.py`；`asyncio_mode = auto`（同步測試不必標 `@pytest.mark.asyncio`，async 測試標註亦可）。
- 所有從 async 呼叫同步 adapter 的路徑必須 `asyncio.to_thread(...)`（既有模式，不得破壞）。
- 設定檔寫入一律 `settings.save_*()` 的 tmp+`replace()` 原子模式；dashboard 不直接寫 `config/asr_config.json`。
- API key / token 不得出現在 log、錯誤訊息或瀏覽器端。
- 失敗政策（每個 task 都隱含）：新 adapter 載入失敗 → 舊 adapter 續用、version 不變、`state=failed`、`config/asr_config.json` 保留新選擇。
- Windows 為主要平台；所有路徑用 `pathlib`；subprocess 有 timeout。

## Review Focus

規格暗示但測試最易遺漏、對真實使用者最痛的輸入類別（每項已釘在所屬 task）：

1. **reload 進行中再收到切換請求** → 必須 409，不得併發兩個載入執行緒（Task 5 釘測試）。
2. **載入中途失敗（CUDA OOM）** → 舊 adapter 續用、version 不變、error 進 snapshot、設定檔保留新選擇（Task 5 釘測試）。
3. **realtime session 進行中切換引擎** → 下一個音訊幀必須用新模型重建 pipeline，不得用 stale model 或崩潰（Task 7 釘測試）。
4. **control-plane 不可達** → `list_targets()` 節點區塊標 `nodes_unreachable`，local 目標與既有 ASR 服務完全不受影響（Task 5 釘測試）。
5. **未認證存取 `/api/asr/config`** → dashboard 需 admin cookie（401/403），router 端由既有 API-key middleware 保護（Task 8/10 釘測試）。
6. **whisper 風格語言碼（"zh"/"en"）餵給 Qwen** → 必須映射為自然語言名（"Chinese"/"English"）或 None（自動偵測），不得原樣傳入（Task 3 釘測試）。

---

### Task 1: Settings — `asr_provider` 欄位 + `load/save_asr_config()`

**Files:**
- Modify: `config/settings.py`（欄位加在 L170 `asr_models_dir` 之後；方法加在 L452 `save_web_search_config` 之後）
- Test: `tests/test_settings.py`（新檔）

**Interfaces:**
- Consumes: 既有 `config_path()`、`save_web_search_config()` 的 tmp+replace 模式（L448-452）。
- Produces: `settings.asr_provider: str`（env `AIIH_ASR_PROVIDER`，預設 `"faster_whisper"`）；`settings.load_asr_config() -> dict[str, Any]`（不存在/損毀回 `{}`）；`settings.save_asr_config(data: dict[str, Any]) -> None`（tmp+replace 原子寫 `config/asr_config.json`）。Task 5 的 manager 與 Task 9 的 asr_worker 依賴這三者。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings.py
from __future__ import annotations

from config.settings import Settings, settings


def test_asr_provider_env_default(monkeypatch):
    monkeypatch.delenv("AIIH_ASR_PROVIDER", raising=False)
    assert Settings().asr_provider == "faster_whisper"


def test_asr_provider_env_override(monkeypatch):
    monkeypatch.setenv("AIIH_ASR_PROVIDER", "qwen_asr")
    assert Settings().asr_provider == "qwen_asr"


def test_load_asr_config_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    assert settings.load_asr_config() == {}


def test_asr_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    data = {"target_id": "local:cuda:1", "provider": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B", "updated_at": "2026-11-03T00:00:00+00:00"}
    settings.save_asr_config(data)
    assert settings.load_asr_config() == data
    assert (tmp_path / "asr_config.json").exists()
    assert not (tmp_path / "asr_config.tmp").exists()


def test_load_asr_config_corrupt_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    (tmp_path / "asr_config.json").write_text("{not json", encoding="utf-8")
    assert settings.load_asr_config() == {}


def test_load_asr_config_non_dict_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    (tmp_path / "asr_config.json").write_text("[1, 2]", encoding="utf-8")
    assert settings.load_asr_config() == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings.py -v`
Expected: FAIL（`Settings` 無 `asr_provider` 欄位 / `settings` 無 `load_asr_config` attribute error）

- [ ] **Step 3: Implement — `config/settings.py`**

在 `asr_models_dir` 欄位（L170）之後加一行欄位：

```python
    asr_provider: str = field(default_factory=lambda: os.getenv("AIIH_ASR_PROVIDER", "faster_whisper").strip().lower())
```

在 `save_web_search_config` 方法（L448-452）之後加兩個方法：

```python
    def load_asr_config(self) -> dict[str, Any]:
        path = self.config_path("asr_config.json")
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def save_asr_config(self, data: dict[str, Any]) -> None:
        path = self.config_path("asr_config.json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_settings.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat(config): asr_provider field + atomic load/save of asr_config.json"
```

---

### Task 2: `gpu_discovery` — 回傳 VRAM 餘量

**Files:**
- Modify: `cluster/gpu_discovery.py`（nvidia-smi 查詢字串 L23-34、解析 L53-78、MPS L81-102）
- Test: `tests/test_gpu_discovery.py`（新檔）

**Interfaces:**
- Consumes: 既有 `subprocess.run` + nvidia-smi CSV 解析結構。
- Produces: 每個 NVIDIA GPU dict 新增 `"memory_used_mb": int`、`"memory_free_mb": int`（= total − used，夾底 0）；既有鍵（`id`/`name`/`memory`/`utilization`/`temperature`/`power_watts`）不動。MPS 路徑新增 `"memory_used_mb": int`、`"memory_free_mb": None`（統一記憶體無法表達 VRAM 餘量）。Task 5 的 `list_targets()` 依賴 `memory_free_mb`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_discovery.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_discovery.py -v`
Expected: FAIL（`KeyError: 'memory_used_mb'`）

- [ ] **Step 3: Implement — `cluster/gpu_discovery.py`**

查詢字串（L23-34）加入 `memory.used`：

```python
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
    ]
```

解析迴圈（原 L58-77）改為 6/7 欄：

```python
        if len(parts) not in {6, 7}:
            continue

        gpu_id, name, memory_total, memory_used, utilization, temperature = parts[:6]
        power_draw = parts[6] if len(parts) == 7 else "0"

        try:
            total_mb = int(float(memory_total))
            used_mb = int(float(memory_used))
            gpus.append(
                {
                    "id": int(gpu_id),
                    "name": name,
                    "memory": total_mb,
                    "memory_used_mb": used_mb,
                    "memory_free_mb": max(0, total_mb - used_mb),
                    "utilization": _safe_float(utilization),
                    "temperature": _safe_float(temperature),
                    "power_watts": _safe_float(power_draw),
                }
            )
        except ValueError:
            continue
```

`_discover_macos_mps_gpu` 回傳 dict（原 L87-102）加兩鍵：

```python
            "memory_used_mb": used_mb,
            "memory_free_mb": None,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_discovery.py -v`
Expected: 5 passed

- [ ] **Step 5: 確認既有依賴不受影響**

Run: `pytest tests/ -k "gpu or node_agent or cluster" -q`
Expected: 全部 passed（`memory_used_mb` 為新增鍵，舊消費端如 `aggregate_gpus()` 只讀既有鍵，不受影響）

- [ ] **Step 6: Commit**

```bash
git add cluster/gpu_discovery.py tests/test_gpu_discovery.py
git commit -m "feat(cluster): gpu_discovery reports memory_used_mb/memory_free_mb"
```

---

### Task 3: `providers/qwen_asr_adapter.py` — Qwen3-ASR 引擎

**Files:**
- Create: `providers/qwen_asr_adapter.py`
- Test: `tests/test_qwen_asr_adapter.py`（新檔）

**Interfaces:**
- Consumes: `providers/asr_base.py` 的 `ASRProviderAdapter`（abstract `transcribe(audio: bytes, task, language, prompt, temperature, response_format) -> dict`）與 `ASRProviderError(message, status_code=500)`；PyPI `qwen-asr` 的 `Qwen3ASRModel.from_pretrained(path, dtype=..., device_map=..., cache_dir=...)` 與 `model.transcribe(audio=(np.ndarray, sr), language=str|None, return_time_stamps=False) -> [obj(.text, .language, .time_stamps)]`。
- Produces: `QwenASRAdapter(model_name="Qwen/Qwen3-ASR-1.7B", device="cuda", device_index=0, download_dir=None)`，`provider_name = "qwen_asr"`：
  - `._model` 為 `_QwenTranscriber`，提供 **串流契約** `.transcribe(samples_16k_float32, language="") -> (segments, info)`，`segments[0].text`/`info.language` 與 faster-whisper 一致（Task 5 的 manager、Task 7 的 realtime pipeline 依賴）。
  - `.transcribe(audio: bytes, ...) -> {"text": str}`（檔案契約；僅收 16kHz WAV，否則 `ASRProviderError(400)`）。
  - `.health_check() -> {"provider", "model_loaded", "device", "device_index"}`。
  - 套件缺失 → 建構子拋 `ASRProviderError(503)`，訊息含 `pip install -U qwen-asr`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qwen_asr_adapter.py
from __future__ import annotations

import io
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import providers.qwen_asr_adapter as qwen_module
from providers.asr_base import ASRProviderError
from providers.qwen_asr_adapter import QwenASRAdapter


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.fixture
def fake_qwen(monkeypatch):
    calls: list[dict] = []
    instance = SimpleNamespace(init_path=None, init_kwargs=None)
    instance.transcribe = lambda audio, language=None, return_time_stamps=False: (
        calls.append({"audio": audio, "language": language}) or [SimpleNamespace(text="你好世界", language="Chinese")]
    )
    fake = types.ModuleType("qwen_asr")

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            instance.init_path = path
            instance.init_kwargs = kwargs
            return instance

    fake.Qwen3ASRModel = FakeQwen3ASRModel
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    monkeypatch.setitem(sys.modules, "qwen_asr", fake)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return instance, calls


def _samples() -> np.ndarray:
    return np.full(16000, 0.01, dtype=np.float32)


def test_stream_contract_shape(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(model_name="Qwen/Qwen3-ASR-1.7B", device="cpu")
    segments, info = adapter._model.transcribe(_samples(), language="")
    assert segments[0].text == "你好世界"
    assert info.language == "Chinese"


def test_language_auto_detect_passes_none(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    adapter._model.transcribe(_samples(), language="")
    assert calls[0]["language"] is None


@pytest.mark.parametrize("code,expected", [("zh", "Chinese"), ("en", "English"), ("ja", "Japanese"), ("klingon", None)])
def test_language_code_mapping(fake_qwen, code, expected):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    adapter._model.transcribe(_samples(), language=code)
    assert calls[-1]["language"] == expected


def test_from_pretrained_device_and_cache(fake_qwen):
    instance, _ = fake_qwen
    QwenASRAdapter(model_name="Qwen/Qwen3-ASR-0.6B", device="cuda", device_index=1, download_dir="data/asr_models")
    assert instance.init_path == "Qwen/Qwen3-ASR-0.6B"
    assert instance.init_kwargs["device_map"] == "cuda:1"
    assert instance.init_kwargs["cache_dir"] == "data/asr_models"


def test_cpu_uses_float32(fake_qwen):
    instance, _ = fake_qwen
    QwenASRAdapter(device="cpu")
    assert instance.init_kwargs["device_map"] == "cpu"
    assert instance.init_kwargs["dtype"] == "float32"


def test_bytes_transcribe_returns_text(fake_qwen):
    instance, calls = fake_qwen
    adapter = QwenASRAdapter(device="cpu")
    result = adapter.transcribe(_wav_bytes(_samples()))
    assert result == {"text": "你好世界"}
    audio, sr = calls[0]["audio"]
    assert sr == 16000
    assert audio.dtype == np.float32


def test_bytes_rejects_non_wav(fake_qwen):
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"not a wav file at all")
    assert excinfo.value.status_code == 400


def test_bytes_rejects_wrong_sample_rate(fake_qwen):
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(_wav_bytes(_samples(), sample_rate=44100))
    assert excinfo.value.status_code == 400
    assert "16kHz" in str(excinfo.value)


def test_inference_error_wrapped_as_503(fake_qwen):
    instance, _ = fake_qwen
    instance.transcribe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuda exploded"))
    adapter = QwenASRAdapter(device="cpu")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(_wav_bytes(_samples()))
    assert excinfo.value.status_code == 503


def test_missing_package_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    with pytest.raises(ASRProviderError) as excinfo:
        QwenASRAdapter(device="cpu")
    assert "pip install" in str(excinfo.value)
    assert excinfo.value.status_code == 503


def test_health_check_shape(fake_qwen):
    adapter = QwenASRAdapter(model_name="m", device="cuda", device_index=2)
    assert adapter.health_check() == {"provider": "qwen_asr", "model_loaded": True, "device": "cuda", "device_index": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qwen_asr_adapter.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'providers.qwen_asr_adapter'`）

- [ ] **Step 3: Implement — `providers/qwen_asr_adapter.py`**

```python
from __future__ import annotations

import io
import logging
import wave
from typing import Any

import numpy as np

from providers.asr_base import ASRProviderAdapter, ASRProviderError

logger = logging.getLogger("providers.qwen_asr_adapter")

_LANGUAGE_CODES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
}


class _QwenSegment:
    def __init__(self, text: str, language: str = "") -> None:
        self.text = text
        self.language = language


class _QwenInfo:
    def __init__(self, language: str = "") -> None:
        self.language = language


class _QwenTranscriber:
    def __init__(self, model_name: str, device: str, device_index: int, download_dir: str | None) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRProviderError(
                "qwen-asr is not installed; run: pip install -U qwen-asr",
                status_code=503,
            ) from exc
        on_cuda = device.strip().lower() == "cuda"
        kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16 if on_cuda else torch.float32,
            "device_map": f"cuda:{device_index}" if on_cuda else "cpu",
        }
        if download_dir:
            kwargs["cache_dir"] = download_dir
        self._model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)

    def transcribe(self, samples: np.ndarray, language: str = "") -> tuple[list[Any], Any]:
        qwen_language = _LANGUAGE_CODES.get(language.strip().lower()) if language else None
        results = self._model.transcribe(
            audio=(np.asarray(samples, dtype=np.float32), 16000),
            language=qwen_language,
            return_time_stamps=False,
        )
        first = results[0]
        text = str(getattr(first, "text", "") or "")
        detected = str(getattr(first, "language", "") or "")
        return [_QwenSegment(text, detected)], _QwenInfo(detected)


class QwenASRAdapter(ASRProviderAdapter):
    provider_name = "qwen_asr"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        device_index: int = 0,
        download_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._device_index = device_index
        self._download_dir = download_dir
        self._model = self._load_model()

    def _load_model(self) -> _QwenTranscriber:
        return _QwenTranscriber(self._model_name, self._device, self._device_index, self._download_dir)

    @staticmethod
    def _decode_wav(audio: bytes) -> np.ndarray:
        if audio[:4] != b"RIFF":
            raise ASRProviderError("qwen_asr file input supports 16kHz WAV only", status_code=400)
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
        except wave.Error as exc:
            raise ASRProviderError("qwen_asr file input supports 16kHz WAV only", status_code=400) from exc
        if sample_rate != 16000:
            raise ASRProviderError(f"qwen_asr file input must be 16kHz WAV (got {sample_rate}Hz)", status_code=400)
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            raise ASRProviderError(f"unsupported WAV sample width {sampwidth}", status_code=400)
        samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if sampwidth == 1:
            samples = (samples - 128.0) / 128.0
        elif sampwidth == 2:
            samples /= 32768.0
        elif sampwidth == 4:
            samples /= 2147483648.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples

    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        try:
            samples = self._decode_wav(audio)
            segments, _info = self._model.transcribe(samples, language=language)
            return {"text": segments[0].text}
        except ASRProviderError:
            raise
        except Exception as exc:
            raise ASRProviderError(f"ASR transcription failed: {exc}", status_code=503) from exc

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model is not None,
            "device": self._device,
            "device_index": self._device_index,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_qwen_asr_adapter.py -v`
Expected: 14 passed（`test_language_code_mapping` 參數化 ×4）

- [ ] **Step 5: Commit**

```bash
git add providers/qwen_asr_adapter.py tests/test_qwen_asr_adapter.py
git commit -m "feat(providers): Qwen3-ASR adapter (transformers backend, whisper-compatible contract)"
```

---

### Task 4: `providers/remote_asr_adapter.py` — 叢集節點客戶端

**Files:**
- Create: `providers/remote_asr_adapter.py`
- Test: `tests/test_remote_asr_adapter.py`（新檔）

**Interfaces:**
- Consumes: `providers/asr_base.py`；`requests`（multipart POST / GET）。
- Produces: `RemoteASRAdapter(node_id: str, base_url: str, timeout_s: float = 10.0, token: str | None = None)`，`provider_name = "remote_asr"`：
  - `._model` 為 `_RemoteTranscriber`，提供 **串流契約** `.transcribe(samples_16k_float32, language="") -> (segments, info)`（內部 float32→int16→WAV→multipart POST）。
  - `.transcribe(audio: bytes, ...) -> {"text": str}`（檔案契約：**原樣轉發**上傳 bytes，由遠端引擎自行解碼）。
  - `.health_check() -> {"reachable": bool, "node_id", "base_url", "engine"?, "model"?}`（GET `{base_url}/health`）。
  - 逾時/連線失敗/HTTP 錯誤 → `ASRProviderError(503)`，訊息含 `node_id` 與 `base_url`。
  - 模組層 `pcm16_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes`。
  - `token` 有值時所有請求帶 `Authorization: Bearer <token>`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_asr_adapter.py
from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

import providers.remote_asr_adapter as remote_module
from providers.asr_base import ASRProviderError
from providers.remote_asr_adapter import RemoteASRAdapter, pcm16_to_wav


def _response(ok=True, status_code=200, payload=None, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    if ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = remote_module.requests.HTTPError(f"status {status_code}")
    return resp


def test_pcm16_to_wav_roundtrip():
    pcm = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    wav_bytes = pcm16_to_wav(pcm)
    assert wav_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(4) == pcm


def test_stream_contract_transcribes_via_http(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["url"] = url
        posted["files"] = files
        posted["data"] = data
        posted["headers"] = headers
        return _response(payload={"text": "hello remote"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    samples = np.full(16000, 0.01, dtype=np.float32)
    segments, info = adapter._model.transcribe(samples, language="zh")
    assert segments[0].text == "hello remote"
    assert posted["url"] == "http://10.0.0.9:9450/v1/audio/transcriptions"
    name, content, content_type = posted["files"]["file"]
    assert name == "audio.wav"
    assert content[:4] == b"RIFF"
    assert content_type == "audio/wav"
    assert posted["data"] == {"language": "zh"}


def test_stream_contract_empty_language_omits_form_field(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["data"] = data
        return _response(payload={"text": "hi"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    adapter._model.transcribe(np.zeros(1600, dtype=np.float32), language="")
    assert posted["data"] == {}


def test_token_sent_as_bearer(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["headers"] = headers
        return _response(payload={"text": "hi"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450", token="secret-token")
    adapter._model.transcribe(np.zeros(1600, dtype=np.float32))
    assert posted["headers"] == {"Authorization": "Bearer secret-token"}


def test_adapter_transcribe_forwards_bytes_verbatim(monkeypatch):
    posted = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        posted["files"] = files
        posted["data"] = data
        return _response(payload={"text": "echo"})

    monkeypatch.setattr(remote_module.requests, "post", fake_post)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    result = adapter.transcribe(b"RAW-UPLOAD-BYTES", language="en")
    assert result == {"text": "echo"}
    assert posted["files"]["file"][1] == b"RAW-UPLOAD-BYTES"
    assert posted["data"] == {"language": "en"}


def test_http_error_maps_to_503_with_node_context(monkeypatch):
    monkeypatch.setattr(remote_module.requests, "post", lambda *a, **k: _response(ok=False, status_code=500, text="boom"))
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter.transcribe(b"x")
    assert excinfo.value.status_code == 503
    assert "n1" in str(excinfo.value)
    assert "http://10.0.0.9:9450" in str(excinfo.value)


def test_connection_error_maps_to_503(monkeypatch):
    def boom(*a, **k):
        raise remote_module.requests.ConnectionError("refused")

    monkeypatch.setattr(remote_module.requests, "post", boom)
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450")
    with pytest.raises(ASRProviderError) as excinfo:
        adapter._model.transcribe(np.zeros(1600, dtype=np.float32))
    assert excinfo.value.status_code == 503
    assert "unreachable" in str(excinfo.value)


def test_health_check_reachable(monkeypatch):
    monkeypatch.setattr(
        remote_module.requests, "get",
        lambda *a, **k: _response(payload={"engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B", "model_loaded": True}),
    )
    adapter = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450/")
    health = adapter.health_check()
    assert health["reachable"] is True
    assert health["node_id"] == "n1"
    assert health["engine"] == "qwen_asr"
    assert health["model"] == "Qwen/Qwen3-ASR-1.7B"


def test_health_check_unreachable(monkeypatch):
    def boom(*a, **k):
        raise remote_module.requests.Timeout("timeout")

    monkeypatch.setattr(remote_module.requests, "get", boom)
    health = RemoteASRAdapter(node_id="n1", base_url="http://10.0.0.9:9450").health_check()
    assert health == {"reachable": False, "node_id": "n1", "base_url": "http://10.0.0.9:9450"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_remote_asr_adapter.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'providers.remote_asr_adapter'`）

- [ ] **Step 3: Implement — `providers/remote_asr_adapter.py`**

```python
from __future__ import annotations

import io
import logging
import os
import wave
from typing import Any

import numpy as np
import requests

from providers.asr_base import ASRProviderAdapter, ASRProviderError

logger = logging.getLogger("providers.remote_asr_adapter")


def pcm16_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buf.getvalue()


class _RemoteSegment:
    def __init__(self, text: str) -> None:
        self.text = text
        self.language = ""


class _RemoteInfo:
    language = ""


class _RemoteTranscriber:
    def __init__(self, node_id: str, base_url: str, token: str | None, timeout_s: float) -> None:
        self._node_id = node_id
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _fail(self, message: str) -> ASRProviderError:
        return ASRProviderError(f"asr worker node={self._node_id} at {self._base_url} {message}", status_code=503)

    def transcribe_file(self, audio: bytes, language: str = "") -> str:
        files = {"file": ("upload.bin", audio, "application/octet-stream")}
        data = {"language": language} if language else {}
        return self._post(files, data)

    def transcribe(self, samples: np.ndarray, language: str = "", **kwargs) -> tuple[list[Any], Any]:
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2").tobytes()
        data = {"language": language} if language else {}
        text = self._post({"file": ("audio.wav", pcm16_to_wav(pcm16), "audio/wav")}, data)
        return [_RemoteSegment(text)], _RemoteInfo()

    def _post(self, files: dict[str, Any], data: dict[str, str]) -> str:
        try:
            response = requests.post(
                f"{self._base_url}/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=self._headers(),
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            raise self._fail(f"unreachable: {exc}") from exc
        if not response.ok:
            raise self._fail(f"returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise self._fail("returned a non-JSON response") from exc
        return str(payload.get("text") or "")

    def health_check(self) -> dict[str, Any]:
        base = {"reachable": False, "node_id": self._node_id, "base_url": self._base_url}
        try:
            response = requests.get(f"{self._base_url}/health", headers=self._headers(), timeout=3)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return base
        return {
            "reachable": True,
            "node_id": self._node_id,
            "base_url": self._base_url,
            "engine": payload.get("engine"),
            "model": payload.get("model"),
            "model_loaded": payload.get("model_loaded"),
        }


class RemoteASRAdapter(ASRProviderAdapter):
    provider_name = "remote_asr"

    def __init__(self, node_id: str, base_url: str, timeout_s: float = 10.0, token: str | None = None) -> None:
        self._node_id = node_id
        self._base_url = base_url.rstrip("/")
        if token is None:
            token = os.getenv("AIIH_ASR_WORKER_TOKEN") or None
        self._model = _RemoteTranscriber(node_id, base_url, token=token, timeout_s=timeout_s)

    def transcribe(
        self,
        audio: bytes,
        task: str = "transcribe",
        language: str = "",
        prompt: str = "",
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> dict[str, Any]:
        try:
            text = self._model.transcribe_file(audio, language=language)
            return {"text": text}
        except ASRProviderError:
            raise
        except Exception as exc:
            raise ASRProviderError(f"ASR transcription failed: {exc}", status_code=503) from exc

    def health_check(self) -> dict[str, Any]:
        return self._model.health_check()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_remote_asr_adapter.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add providers/remote_asr_adapter.py tests/test_remote_asr_adapter.py
git commit -m "feat(providers): remote ASR adapter for cluster asr_worker nodes"
```

---

### Task 5: `runtime/asr/asr_manager.py` — 狀態機與熱切換核心

**Files:**
- Create: `runtime/asr/__init__.py`（空檔案）
- Create: `runtime/asr/asr_manager.py`
- Test: `tests/test_asr_manager.py`（新檔）

**Interfaces:**
- Consumes: Task 1 的 `settings.load_asr_config()/save_asr_config()/asr_provider/asr_enabled/asr_model/asr_device/asr_device_index/asr_models_dir/asr_compute_type/control_plane_url`；Task 2 的 `discover_gpus()`（含 `memory_free_mb`）；Task 3/4 的兩個新 adapter；既有 `providers/faster_whisper_adapter.py`；control-plane `GET {settings.control_plane_url}/cluster/nodes`（回 `{"nodes": [{node_id, ip, metadata: {asr_worker: {port, healthy, engine, model}}}]}`）。
- Produces（Task 6/7/8/10 依賴）:
  - `ASRTarget(target_id: str, kind: str, node_id: str | None, base_url: str | None, device_index: int | None)`；`kind ∈ {"local_cuda", "local_cpu", "node"}`。
  - `ASRConfigError(message: str, status_code: int = 400, details: Any = None)`，屬性 `.message/.status_code/.details`。
  - `LOCAL_ENGINES: dict[str, list[str]]`（`faster_whisper → [large-v3, medium, small]`、`qwen_asr → [Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B]`）。
  - `ASRManager(builders=None, dispose=None, persist=None)`；模組單例 `asr_manager = ASRManager()`。
  - 方法：`snapshot() -> {"state", "active", "requested", "error", "version"}`；`get_adapter() -> Any | None`（disabled 或從未載入成功回 `None`；loading 期間回**舊** adapter）；`version() -> int`；`preload() -> None`（只在 `idle` 狀態生效；`asr_enabled=false` → `state="disabled"`）；`reload(target_id: str, provider: str | None, model: str | None) -> snapshot dict`（進行中 → `ASRConfigError(409)`；驗證失敗 → `ASRConfigError(400/404/503)`）；`resolve_target(target_id) -> ASRTarget`；`list_targets() -> {"local": [...], "nodes": [...], "nodes_unreachable": bool}`。
  - builder 簽名：`(target: ASRTarget, provider: str, model: str) -> adapter`，以 `target.kind` 為 key（`DEFAULT_BUILDERS`）；dispose 簽名：`(adapter) -> None`；persist 簽名：`(config: dict) -> None`（預設 `settings.save_asr_config`）。

- [ ] **Step 1: Create empty package marker**

```bash
mkdir runtime/asr
```

建立空檔案 `runtime/asr/__init__.py`（0 bytes）。

- [ ] **Step 2: Write the failing test**

```python
# tests/test_asr_manager.py
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

import runtime.asr.asr_manager as asr_manager_module
from runtime.asr.asr_manager import ASRConfigError, ASRManager


def _fake_adapter(name: str = "fake") -> MagicMock:
    return MagicMock(provider_name=name)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(asr_manager_module.settings, "asr_enabled", True)
    monkeypatch.setattr(asr_manager_module.settings, "asr_provider", "faster_whisper")
    monkeypatch.setattr(asr_manager_module.settings, "asr_model", "large-v3")
    monkeypatch.setattr(asr_manager_module.settings, "asr_device", "cuda")
    monkeypatch.setattr(asr_manager_module.settings, "asr_device_index", 0)
    monkeypatch.setattr(asr_manager_module.settings, "load_asr_config", lambda: {})
    monkeypatch.setattr(asr_manager_module, "discover_gpus", lambda: [
        {"id": 0, "name": "GPU0", "memory_free_mb": 1000},
        {"id": 1, "name": "GPU1", "memory_free_mb": 2000},
    ])
    monkeypatch.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [])
    return monkeypatch


def _make_manager(builders: dict):
    persisted: list[dict] = []
    manager = ASRManager(builders=builders, dispose=MagicMock(), persist=persisted.append)
    return manager, persisted


def _wait_state(manager: ASRManager, state: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.snapshot()["state"] == state:
            return
        time.sleep(0.01)
    pytest.fail(f"state never reached '{state}': {manager.snapshot()}")


def _both_builders(builder):
    return {"local_cuda": builder, "local_cpu": builder}


def test_initial_state_idle():
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda data: None)
    snapshot = manager.snapshot()
    assert snapshot == {"state": "idle", "active": None, "requested": None, "error": None, "version": 0}
    assert manager.get_adapter() is None


def test_preload_disabled(env):
    env.setattr(asr_manager_module.settings, "asr_enabled", False)
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    manager.preload()
    assert manager.snapshot()["state"] == "disabled"
    assert manager.get_adapter() is None


def test_preload_success(env):
    adapter = _fake_adapter("a")
    manager, persisted = _make_manager(_both_builders(lambda t, p, m: adapter))
    manager.preload()
    _wait_state(manager, "ready")
    snapshot = manager.snapshot()
    assert snapshot["version"] == 1
    assert manager.get_adapter() is adapter
    assert snapshot["active"]["target_id"] == "local:cuda:0"
    assert snapshot["active"]["model"] == "large-v3"
    assert snapshot["active"]["provider"] == "faster_whisper"
    assert persisted == []


def test_preload_uses_persisted_config(env):
    env.setattr(asr_manager_module.settings, "load_asr_config", lambda: {
        "target_id": "local:cpu", "provider": "faster_whisper", "model": "small",
    })
    seen = {}

    def builder(target, provider, model):
        seen["args"] = (target.target_id, target.kind, provider, model)
        return _fake_adapter("b")

    manager, _ = _make_manager(_both_builders(builder))
    manager.preload()
    _wait_state(manager, "ready")
    assert seen["args"] == ("local:cpu", "local_cpu", "faster_whisper", "small")


def test_preload_failure_marks_failed(env):
    def builder(target, provider, model):
        raise RuntimeError("CUDA out of memory")

    manager, _ = _make_manager(_both_builders(builder))
    manager.preload()
    _wait_state(manager, "failed")
    snapshot = manager.snapshot()
    assert "CUDA out of memory" in snapshot["error"]
    assert manager.get_adapter() is None
    assert snapshot["version"] == 0


def test_reload_success_swaps_and_disposes_old(env):
    adapter_a, adapter_b = _fake_adapter("a"), _fake_adapter("b")
    builders = {"local_cuda": lambda t, p, m: adapter_a, "local_cpu": lambda t, p, m: adapter_b}
    manager, persisted = _make_manager(builders)
    manager.preload()
    _wait_state(manager, "ready")

    snapshot = manager.reload("local:cpu", provider="qwen_asr", model="Qwen/Qwen3-ASR-1.7B")
    assert snapshot["state"] == "loading"
    _wait_state(manager, "ready")

    snapshot = manager.snapshot()
    assert snapshot["version"] == 2
    assert manager.get_adapter() is adapter_b
    assert snapshot["active"]["provider"] == "qwen_asr"
    assert snapshot["active"]["target_id"] == "local:cpu"
    manager._dispose.assert_called_once_with(adapter_a)
    assert persisted[-1] == {
        "target_id": "local:cpu",
        "provider": "qwen_asr",
        "model": "Qwen/Qwen3-ASR-1.7B",
        "updated_at": persisted[-1]["updated_at"],
    }
    assert persisted[-1]["updated_at"]


def test_reload_failure_rolls_back(env):
    adapter_a = _fake_adapter("a")
    calls = {"n": 0}

    def builder(target, provider, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return adapter_a
        raise RuntimeError("CUDA out of memory")

    manager, persisted = _make_manager(_both_builders(builder))
    manager.preload()
    _wait_state(manager, "ready")
    manager.reload("local:cpu", provider="faster_whisper", model="medium")
    _wait_state(manager, "failed")

    snapshot = manager.snapshot()
    assert snapshot["version"] == 1
    assert manager.get_adapter() is adapter_a
    assert "CUDA out of memory" in snapshot["error"]
    assert snapshot["active"]["model"] == "large-v3"
    assert persisted[-1]["model"] == "medium"


def test_reload_concurrent_returns_409(env):
    adapter = _fake_adapter("a")
    gate = threading.Event()

    def slow_builder(target, provider, model):
        gate.wait(timeout=5)
        return adapter

    manager, _ = _make_manager(_both_builders(slow_builder))
    manager.reload("local:cuda:0", provider="faster_whisper", model="large-v3")
    _wait_state(manager, "loading")
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cpu", provider="faster_whisper", model="small")
    assert excinfo.value.status_code == 409
    gate.set()
    _wait_state(manager, "ready")


def test_reload_unknown_target_400(env):
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("warp-drive", provider=None, model=None)
    assert excinfo.value.status_code == 400


def test_reload_unknown_provider_400(env):
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cpu", provider="mozart", model=None)
    assert excinfo.value.status_code == 400
    assert "valid_providers" in excinfo.value.details


def test_reload_unknown_model_400(env):
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cpu", provider="qwen_asr", model="tiny-cn")
    assert excinfo.value.status_code == 400
    assert "Qwen/Qwen3-ASR-1.7B" in excinfo.value.details["valid_models"]


def test_reload_device_index_out_of_range_400(env):
    manager = ASRManager(builders=_both_builders(lambda t, p, m: _fake_adapter()), dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("local:cuda:9", provider="faster_whisper", model=None)
    assert excinfo.value.status_code == 400
    assert excinfo.value.details["valid_gpu_ids"] == [0, 1]


def test_reload_node_resolves_base_url_and_reports_worker_engine(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "m4", "ip": "10.1.1.5", "metadata": {"asr_worker": {"port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}}},
    ])
    built = {}
    remote = MagicMock(provider_name="remote")
    remote.health_check.return_value = {"reachable": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}

    def node_builder(target, provider, model):
        built["base_url"] = target.base_url
        built["node_id"] = target.node_id
        return remote

    manager, _ = _make_manager({"local_cuda": lambda t, p, m: None, "local_cpu": lambda t, p, m: None, "node": node_builder})
    manager.reload("node:m4", provider=None, model=None)
    _wait_state(manager, "ready")
    assert built["base_url"] == "http://10.1.1.5:9450"
    assert built["node_id"] == "m4"
    snapshot = manager.snapshot()
    assert snapshot["active"]["provider"] == "qwen_asr"
    assert snapshot["active"]["target_id"] == "node:m4"


def test_reload_node_without_worker_400(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "m4", "ip": "10.1.1.5", "metadata": {"asr_worker": {"port": 9450, "healthy": False}}},
    ])
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("node:m4", provider=None, model=None)
    assert excinfo.value.status_code == 400


def test_reload_node_unknown_node_404(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [{"node_id": "other", "ip": "1.2.3.4", "metadata": {}}])
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    with pytest.raises(ASRConfigError) as excinfo:
        manager.reload("node:ghost", provider=None, model=None)
    assert excinfo.value.status_code == 404


def test_reload_node_worker_health_unreachable_rolls_back(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "x", "ip": "10.9.9.9", "metadata": {"asr_worker": {"port": 9450, "healthy": True}}},
    ])
    remote = MagicMock(provider_name="remote_asr")
    remote.health_check.return_value = {"reachable": False}
    manager, _ = _make_manager({"node": lambda t, p, m: remote, "local_cuda": lambda t, p, m: None, "local_cpu": lambda t, p, m: None})
    manager.reload("node:x", provider=None, model=None)
    _wait_state(manager, "failed")
    assert manager.get_adapter() is None
    assert "unreachable" in manager.snapshot()["error"]


def test_list_targets_combines_sources(env):
    env.setattr(asr_manager_module, "_fetch_cluster_nodes", lambda: [
        {"node_id": "m4", "ip": "10.1.1.5", "metadata": {"asr_worker": {"port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}}},
        {"node_id": "bare", "ip": "10.1.1.6", "metadata": {}},
    ])
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    targets = manager.list_targets()
    assert targets["local"][0] == {"id": "local:cuda:0", "name": "GPU0", "memory_free_mb": 1000}
    assert targets["local"][-1] == {"id": "local:cpu", "name": "CPU", "memory_free_mb": None}
    by_id = {node["id"]: node for node in targets["nodes"]}
    assert by_id["node:m4"]["reachable"] is True
    assert by_id["node:m4"]["engine"] == "qwen_asr"
    assert by_id["node:bare"]["reachable"] is False
    assert targets["nodes_unreachable"] is False


def test_list_targets_control_plane_down_degrades(env):
    def boom():
        raise requests.ConnectionError("control-plane down")

    env.setattr(asr_manager_module, "_fetch_cluster_nodes", boom)
    manager = ASRManager(builders={}, dispose=MagicMock(), persist=lambda d: None)
    targets = manager.list_targets()
    assert targets["nodes_unreachable"] is True
    assert targets["nodes"] == []
    assert [t["id"] for t in targets["local"]] == ["local:cuda:0", "local:cuda:1", "local:cpu"]


def test_get_adapter_during_loading_returns_previous(env):
    adapter_a = _fake_adapter("a")
    gate = threading.Event()

    def slow_builder(target, provider, model):
        gate.wait(timeout=5)
        return _fake_adapter("b")

    builders = {"local_cuda": lambda t, p, m: adapter_a, "local_cpu": slow_builder}
    manager, _ = _make_manager(builders)
    manager.preload()
    _wait_state(manager, "ready")
    manager.reload("local:cpu", provider="faster_whisper", model="small")
    _wait_state(manager, "loading")
    assert manager.get_adapter() is adapter_a
    gate.set()
    _wait_state(manager, "ready")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_asr_manager.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'runtime.asr'`）

- [ ] **Step 4: Implement — `runtime/asr/asr_manager.py`**

```python
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from cluster.gpu_discovery import discover_gpus
from config.settings import settings

logger = logging.getLogger("runtime.asr.asr_manager")

LOCAL_ENGINES: dict[str, list[str]] = {
    "faster_whisper": ["large-v3", "medium", "small"],
    "qwen_asr": ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"],
}

_DEFAULT_MODELS: dict[str, str] = {
    "faster_whisper": "large-v3",
    "qwen_asr": "Qwen/Qwen3-ASR-1.7B",
}

_GPU_CACHE_TTL_S = 5.0


class ASRConfigError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


@dataclass
class ASRTarget:
    target_id: str
    kind: str
    node_id: str | None = None
    base_url: str | None = None
    device_index: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_local(target: ASRTarget, provider: str, model: str) -> Any:
    cuda = target.kind == "local_cuda"
    device_index = target.device_index if target.device_index is not None else 0
    if provider == "qwen_asr":
        from providers.qwen_asr_adapter import QwenASRAdapter

        return QwenASRAdapter(
            model_name=model,
            device="cuda" if cuda else "cpu",
            device_index=device_index,
            download_dir=settings.asr_models_dir,
        )
    from providers.faster_whisper_adapter import FasterWhisperAdapter

    return FasterWhisperAdapter(
        model_name=model,
        device="cuda" if cuda else "cpu",
        device_index=device_index,
        compute_type=settings.asr_compute_type if cuda else "int8",
        download_dir=settings.asr_models_dir,
    )


def _build_node(target: ASRTarget, provider: str, model: str) -> Any:
    from providers.remote_asr_adapter import RemoteASRAdapter

    return RemoteASRAdapter(
        node_id=target.node_id or "",
        base_url=target.base_url or "",
        token=os.getenv("AIIH_ASR_WORKER_TOKEN") or None,
    )


DEFAULT_BUILDERS: dict[str, Callable[[ASRTarget, str, str], Any]] = {
    "local_cuda": _build_local,
    "local_cpu": _build_local,
    "node": _build_node,
}


def default_dispose_adapter(adapter: Any) -> None:
    del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fetch_cluster_nodes() -> list[dict[str, Any]]:
    response = requests.get(f"{settings.control_plane_url}/cluster/nodes", timeout=3)
    response.raise_for_status()
    payload = response.json()
    nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    return [node for node in nodes if isinstance(node, dict)]


class ASRManager:
    def __init__(
        self,
        builders: dict[str, Callable[[ASRTarget, str, str], Any]] | None = None,
        dispose: Callable[[Any], None] | None = None,
        persist: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._builders = builders if builders is not None else DEFAULT_BUILDERS
        self._dispose = dispose if dispose is not None else default_dispose_adapter
        self._persist = persist if persist is not None else settings.save_asr_config
        self._lock = threading.RLock()
        self._state = "idle"
        self._active_adapter: Any = None
        self._active_info: dict[str, Any] | None = None
        self._requested_info: dict[str, Any] | None = None
        self._error: str | None = None
        self._version = 0
        self._gpu_cache: list[dict[str, Any]] = []
        self._gpu_cache_at = 0.0

    def version(self) -> int:
        with self._lock:
            return self._version

    def get_adapter(self) -> Any:
        with self._lock:
            if self._state == "disabled":
                return None
            return self._active_adapter

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "active": dict(self._active_info) if self._active_info else None,
                "requested": dict(self._requested_info) if self._requested_info else None,
                "error": self._error,
                "version": self._version,
            }

    def _gpus_cached(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if self._gpu_cache and now - self._gpu_cache_at < _GPU_CACHE_TTL_S:
                return self._gpu_cache
        try:
            gpus = discover_gpus()
        except Exception:
            gpus = []
        with self._lock:
            self._gpu_cache = gpus
            self._gpu_cache_at = now
        return gpus

    def resolve_target(self, target_id: str) -> ASRTarget:
        if not target_id or not isinstance(target_id, str):
            raise ASRConfigError("target_id is required")
        if target_id == "local:cpu":
            return ASRTarget(target_id=target_id, kind="local_cpu")
        if target_id.startswith("local:cuda:"):
            try:
                index = int(target_id.rsplit(":", 1)[1])
            except ValueError as exc:
                raise ASRConfigError(f"invalid target_id '{target_id}'") from exc
            return ASRTarget(target_id=target_id, kind="local_cuda", device_index=index)
        if target_id.startswith("node:"):
            node_id = target_id[len("node:"):]
            base_url = self._resolve_node_base_url(node_id)
            return ASRTarget(target_id=target_id, kind="node", node_id=node_id, base_url=base_url)
        raise ASRConfigError(
            f"unknown target_id '{target_id}'",
            details={"targets": [t["id"] for t in self.list_targets()["local"]]},
        )

    def _resolve_node_base_url(self, node_id: str) -> str:
        try:
            nodes = _fetch_cluster_nodes()
        except requests.RequestException as exc:
            raise ASRConfigError(f"control-plane unreachable: {exc}", status_code=503) from exc
        for node in nodes:
            if node.get("node_id") != node_id:
                continue
            metadata = node.get("metadata") or {}
            asr = metadata.get("asr_worker") or {}
            if not asr.get("healthy"):
                raise ASRConfigError(
                    f"node '{node_id}' has no reachable asr_worker",
                    details={"node_id": node_id},
                )
            ip = node.get("ip") or "127.0.0.1"
            return f"http://{ip}:{int(asr.get('port') or 9450)}"
        raise ASRConfigError(
            f"unknown node '{node_id}'",
            status_code=404,
            details={"known_nodes": [n.get("node_id") for n in nodes]},
        )

    def _validate(self, target: ASRTarget, provider: str, model: str) -> tuple[str, str]:
        if target.kind == "node":
            return "remote_asr", ""
        if provider not in LOCAL_ENGINES:
            raise ASRConfigError(
                f"unknown provider '{provider}'",
                details={"valid_providers": sorted(LOCAL_ENGINES)},
            )
        if not model:
            model = _DEFAULT_MODELS[provider]
        elif model not in LOCAL_ENGINES[provider]:
            raise ASRConfigError(
                f"unknown model '{model}' for provider '{provider}'",
                details={"valid_models": list(LOCAL_ENGINES[provider])},
            )
        if target.kind == "local_cuda":
            gpu_ids = [gpu.get("id") for gpu in self._gpus_cached()]
            if target.device_index not in gpu_ids:
                raise ASRConfigError(
                    f"device_index {target.device_index} is not available on this host",
                    details={"valid_gpu_ids": gpu_ids},
                )
        return provider, model

    def preload(self) -> None:
        with self._lock:
            if self._state != "idle":
                return
            if not settings.asr_enabled:
                self._state = "disabled"
                logger.info("ASR disabled; manager state=disabled")
                return
            config = settings.load_asr_config()
            if config:
                logger.info("ASR config loaded from asr_config.json: %s", config)
            target_id = str(config.get("target_id") or self._env_default_target_id())
            provider = str(config.get("provider") or settings.asr_provider).strip().lower()
            model = str(config.get("model") or "").strip()
            self._requested_info = {"target_id": target_id, "provider": provider, "model": model, "loaded_at": None}
            self._state = "loading"
        thread = threading.Thread(
            target=self._preload_worker,
            args=(target_id, provider, model),
            daemon=True,
            name="asr-preload",
        )
        thread.start()

    def _env_default_target_id(self) -> str:
        if settings.asr_device.strip().lower() == "cpu":
            return "local:cpu"
        return f"local:cuda:{settings.asr_device_index}"

    def _preload_worker(self, target_id: str, provider: str, model: str) -> None:
        try:
            target = self.resolve_target(target_id)
            provider, model = self._validate(target, provider, model)
        except ASRConfigError as exc:
            with self._lock:
                self._state = "failed"
                self._error = exc.message
            logger.error("ASR preload failed: %s", exc.message)
            return
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
            logger.exception("ASR preload failed")
            return
        self._load_worker(target, provider, model)

    def reload(self, target_id: str, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._state == "loading":
                raise ASRConfigError("an ASR reload is already in progress", status_code=409)
        resolved_provider = (provider or settings.asr_provider).strip().lower()
        resolved_model = (model or "").strip()
        target = self.resolve_target(target_id)
        resolved_provider, resolved_model = self._validate(target, resolved_provider, resolved_model)
        with self._lock:
            self._requested_info = {
                "target_id": target.target_id,
                "provider": resolved_provider,
                "model": resolved_model,
                "loaded_at": None,
            }
            self._state = "loading"
        try:
            self._persist({
                "target_id": target.target_id,
                "provider": resolved_provider,
                "model": resolved_model,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.exception("ASR config persist failed; continuing with reload")
        thread = threading.Thread(
            target=self._load_worker,
            args=(target, resolved_provider, resolved_model),
            daemon=True,
            name="asr-reload",
        )
        thread.start()
        return self.snapshot()

    def _load_worker(self, target: ASRTarget, provider: str, model: str) -> None:
        started = time.monotonic()
        try:
            builder = self._builders.get(target.kind)
            if builder is None:
                raise ASRConfigError(f"unsupported target kind '{target.kind}'")
            adapter = builder(target, provider, model)
            if target.kind == "node":
                health = adapter.health_check()
                if not health.get("reachable", True):
                    raise ASRConfigError(
                        f"asr worker on node '{target.node_id}' unreachable at {target.base_url}",
                        status_code=503,
                    )
                provider = str(health.get("engine") or "remote_asr")
                model = str(health.get("model") or "")
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
            logger.exception(
                "ASR load failed (target=%s provider=%s model=%s); keeping previous adapter",
                target.target_id, provider, model,
            )
            return
        with self._lock:
            previous = self._active_adapter
            self._active_adapter = adapter
            self._active_info = {
                "target_id": target.target_id,
                "provider": provider,
                "model": model,
                "loaded_at": _now_iso(),
            }
            self._requested_info = dict(self._active_info)
            self._version += 1
            self._state = "ready"
            self._error = None
        logger.info(
            "ASR adapter active target=%s provider=%s model=%s load=%.1fs version=%d",
            target.target_id, provider, model, time.monotonic() - started, self.version(),
        )
        if previous is not None:
            try:
                self._dispose(previous)
            except Exception:
                logger.exception("ASR adapter disposal failed")

    def list_targets(self) -> dict[str, Any]:
        gpus = self._gpus_cached()
        local = [
            {
                "id": f"local:cuda:{gpu.get('id')}",
                "name": gpu.get("name"),
                "memory_free_mb": gpu.get("memory_free_mb"),
            }
            for gpu in gpus
        ]
        local.append({"id": "local:cpu", "name": "CPU", "memory_free_mb": None})
        nodes: list[dict[str, Any]] = []
        nodes_unreachable = False
        try:
            cluster_nodes = _fetch_cluster_nodes()
        except requests.RequestException:
            nodes_unreachable = True
            cluster_nodes = []
        for node in cluster_nodes:
            metadata = node.get("metadata") or {}
            asr = metadata.get("asr_worker") or {}
            nodes.append({
                "id": f"node:{node.get('node_id')}",
                "reachable": bool(asr.get("healthy")),
                "engine": asr.get("engine"),
                "model": asr.get("model"),
            })
        return {"local": local, "nodes": nodes, "nodes_unreachable": nodes_unreachable}


asr_manager = ASRManager()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_asr_manager.py -v`
Expected: 19 passed

- [ ] **Step 6: Commit**

```bash
git add runtime/asr/__init__.py runtime/asr/asr_manager.py tests/test_asr_manager.py
git commit -m "feat(runtime): ASRManager state machine with hot-swap, rollback and cluster targets"
```

---

### Task 6: `provider_router` 委派 manager + dashboard `audio_status` 修正

**Files:**
- Modify: `runtime/orchestration/provider_router.py`（刪 L23-26 `FasterWhisperAdapter` try/except import；L189-192 dispatch 去除 `FasterWhisperAdapter` 檢查；L242-269 整段替換）
- Modify: `dashboard/dashboard_server.py:2243-2271`（`audio_status` 的 `_asr_adapter` import 與 ASR 區塊——Task 6 移除 `_asr_adapter` 後此 import 會 `ImportError`，必須同 commit 修正）
- Test: `tests/test_audio_router.py`（改寫 L229-246 的 `TestProviderRouterResilience.test_asr_factory_cooldown`）

**Interfaces:**
- Consumes: Task 5 的 `runtime.asr.asr_manager.asr_manager`（`.get_adapter() -> Any | None`）。
- Produces: `provider_router._get_asr_adapter() -> Any | None`（簽名不變；`adapter("asr")` dispatch 於 L192 呼叫它，僅移除 L190-191 的 `FasterWhisperAdapter is None` 前置檢查——import 已刪，保留會 NameError）。`module-level _asr_adapter/_asr_adapter_failed_at` 消失；`_ADAPTER_RETRY_COOLDOWN` 保留（TTS/image_gen 仍用）。dashboard `/api/audio/status` 的 `asr` 區塊改由 manager 快照供給，鍵 `enabled/model/device/device_index/compute_type` 保留，新增 `state/version/error`。

- [ ] **Step 1: 改寫測試（冷卻測試 → 委派測試）**

`tests/test_audio_router.py` 中 `class TestProviderRouterResilience` 的 `test_asr_factory_cooldown`（L230-246，操作 `pr._asr_adapter`/`pr._asr_adapter_failed_at` module globals）**整個方法刪除**，替換為：

```python
    def test_asr_adapter_delegates_to_manager(self) -> None:
        from runtime.asr import asr_manager as asr_manager_module
        from runtime.orchestration import provider_router as pr

        with patch.object(asr_manager_module.asr_manager, "get_adapter", return_value="sentinel-adapter"):
            assert pr._get_asr_adapter() == "sentinel-adapter"

    def test_asr_adapter_none_when_disabled(self) -> None:
        from runtime.asr import asr_manager as asr_manager_module
        from runtime.orchestration import provider_router as pr

        with patch.object(asr_manager_module.asr_manager, "get_adapter", return_value=None):
            assert pr._get_asr_adapter() is None
```

注意：同類別的 `test_tts_factory_cooldown`（L248 起）**保留不動**。檔案頂部若已有 `from unittest.mock import patch` 則沿用。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_router.py::TestProviderRouterResilience -v`
Expected: `test_asr_adapter_delegates_to_manager` **FAIL**（現行實作不認識 manager，回傳 None ≠ "sentinel-adapter"）；`test_asr_adapter_none_when_disabled` 可能 PASS（現行 disabled 路徑本就回 None）——以 delegate 測試轉紅為 TDD 訊號

- [ ] **Step 3: Implement — `provider_router.py`**

刪除 L23-26：

```python
try:
    from providers.faster_whisper_adapter import FasterWhisperAdapter
except ImportError:
    FasterWhisperAdapter = None  # type: ignore[assignment, misc]
```

將 L242-269 整段（`_asr_adapter`/`_asr_adapter_failed_at` globals + `def _get_asr_adapter()` 本體）替換為：

```python
def _get_asr_adapter() -> Any:
    from runtime.asr.asr_manager import asr_manager

    return asr_manager.get_adapter()
```

`adapter()` dispatch（L189-192）的 asr 分支改為（刪去引用已刪除 symbol 的檢查，manager 自行處理不可用情形）：

```python
    if provider == "asr":
        return _get_asr_adapter()
```

- [ ] **Step 4: Implement — `dashboard/dashboard_server.py` `audio_status`**

L2243 的 import 行：

```python
    from runtime.orchestration.provider_router import _tts_adapter, _asr_adapter
```

改為：

```python
    from runtime.asr.asr_manager import asr_manager
    from runtime.orchestration.provider_router import _tts_adapter
```

L2245 `asr_info` 宣告與 L2262-2271 的 `if settings.asr_enabled:` 區塊改為：

```python
    asr_info: dict[str, Any] = {"enabled": settings.asr_enabled}
    if settings.asr_enabled:
        snapshot = asr_manager.snapshot()
        asr_info["state"] = snapshot["state"]
        asr_info["version"] = snapshot["version"]
        if snapshot["error"]:
            asr_info["error"] = snapshot["error"]
        adapter = asr_manager.get_adapter()
        if adapter is not None:
            try:
                asr_info.update(adapter.health_check())
            except Exception:
                pass
        active = snapshot.get("active") or {}
        asr_info["model"] = active.get("model") or settings.asr_model
        asr_info["device"] = settings.asr_device
        asr_info["device_index"] = settings.asr_device_index
        asr_info["compute_type"] = settings.asr_compute_type
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_audio_router.py -v`
Expected: 全部 passed（含改寫後的 2 個新測試與既有 TTS cooldown 測試）

- [ ] **Step 6: 全套件回歸**

Run: `pytest tests/ -q`
Expected: 與基線一致（1042+ passed、16 個 pre-existing 環境失敗），**無新增失敗**。若 `tests/test_dashboard_auth.py` 以外出現新失敗，先修再 commit。

- [ ] **Step 7: Commit**

```bash
git add runtime/orchestration/provider_router.py dashboard/dashboard_server.py tests/test_audio_router.py
git commit -m "feat(runtime): _get_asr_adapter delegates to asr_manager; audio_status reads manager snapshot"
```

---

### Task 7: `realtime_router` — 跟隨 manager 版本重建 pipeline

**Files:**
- Modify: `router/realtime_router.py`（刪 L6 `import threading`、L14 `get_adapter` import、L24-58 preload 區塊；L184-206 `process_audio` pipeline 建構）
- Test: `tests/test_realtime_router.py`（fixture L58-74 改 manager；`FakeWebSocket` 加 `"call"` 事件；新增版本切換測試）

**Interfaces:**
- Consumes: Task 5 的 `asr_manager`（`.version() -> int`、`.get_adapter() -> Any | None`）。
- Produces: `router/realtime_router.preload_asr() -> None`（薄委派 `asr_manager.preload()`；`openai_router.py` L96-99 的 startup hook **不改**，繼續呼叫它）。`process_audio` 於每幀比對 `asr_manager.version()`，版本變更或 pipeline 不存在 → 重建 `RealtimeAudioPipeline(asr_model=adapter._model, ...)`。既有測試 fixture 改 patch `realtime_router.asr_manager`。

- [ ] **Step 1: 更新測試 fixture 與 FakeWebSocket**

`tests/test_realtime_router.py`：

(a) `FakeWebSocket.receive()`（L37-49）在 `"delay"` 分支之後加一個分支：

```python
        if isinstance(event, tuple) and event[0] == "call":
            event[1]()
            return await self.receive()
```

(b) 在 `FakeASRModel` 之後加 manager fake，並改寫 `realtime_dependencies` fixture（L58-74）：刪除 L69 的 `monkeypatch.setattr(realtime_router, "get_adapter", ...)` 行，改為：

```python
class VersionedASRManager:
    def __init__(self) -> None:
        self._version = 0
        self.models: list[FakeASRModel] = [FakeASRModel()]

    def version(self) -> int:
        return self._version

    def get_adapter(self):
        return SimpleNamespace(_model=self.models[-1])
```

fixture 內（L68 `asr_model = FakeASRModel()` 起改為）：

```python
    manager = VersionedASRManager()
    monkeypatch.setattr(realtime_router, "asr_manager", manager, raising=False)
    asr_model = manager.models[0]
```

（`service.asr_model = asr_model` 行保留。）

- [ ] **Step 2: 新增版本切換測試**

```python
@pytest.mark.asyncio
async def test_asr_version_change_rebuilds_pipeline_with_new_model(realtime_dependencies):
    pcm = np.full(24_000 * 3, 1000, dtype=np.int16).tobytes()
    manager = realtime_router.asr_manager
    new_model = FakeASRModel()

    def switch_engine():
        manager.models.append(new_model)
        manager._version += 1

    ws = FakeWebSocket([
        {"type": "session.update", "session": {"audio": {"output": {"language": "zh"}}}},
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        ("call", switch_engine),
        {"type": "session.input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.close"},
    ])
    await realtime_router.realtime_websocket(ws)
    events = [event for event in ws.sent if event["type"] == "session.input_transcript.delta"]
    assert len(events) == 2
    assert manager.models[0].samples_seen > 0
    assert new_model.samples_seen > 0
    assert ws.sent[-1]["type"] == "session.closed"
    assert ws.close_code == 1000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_realtime_router.py -v`
Expected: FAIL — `realtime_router` 沒有 `asr_manager` 屬性，fixture `monkeypatch.setattr(..., raising=False)` 雖不報錯，但 `process_audio` 仍走舊 `get_adapter`/`_asr_ready` 路徑，兩幀都由同一個 `models[0]` 轉錄 → 新測試的 `new_model.samples_seen > 0` 失敗

- [ ] **Step 4: Implement — `router/realtime_router.py`**

(a) 刪 L6 `import threading`（grep 確認檔內再無 `threading.` 使用——僅 preload 區塊使用）與 L14 `from runtime.orchestration.provider_router import adapter as get_adapter`。

(b) L14 刪除後，原位（import 分組內）補上 manager import：

```python
from runtime.asr.asr_manager import asr_manager
```

(c) L24-58（`_asr_ready`/`_preloaded_asr_model`/`_preload_thread`/`preload_asr`/`_preload_asr_worker`）整段替換為薄委派（`router = APIRouter()` 在原檔 L22，保留不動）：

```python
def preload_asr() -> None:
    asr_manager.preload()
```

(d) `process_audio`（L184-206）改為：

```python
    async def process_audio(queue: asyncio.Queue[tuple[int, bytes] | None]) -> None:
        nonlocal audio_pipeline
        manager_version = asr_manager.version()
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            generation, pcm = item
            try:
                if audio_pipeline is None or asr_manager.version() != manager_version:
                    audio_pipeline = None
                    adapter = await asyncio.to_thread(asr_manager.get_adapter)
                    if adapter is None:
                        raise RuntimeError("ASR adapter unavailable")
                    audio_pipeline = RealtimeAudioPipeline(
                        asr_model=adapter._model,
                        service=service,
                        model=session.session.get("model") or "",
                        target_language=session.session["target_language"],
                    )
```

（`emit`/`add_audio`/例外處理等後續程式碼不動。）

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_realtime_router.py -v`
Expected: 全部 passed（既有 45 個 + 新增 1 個；fixture 已改 patch manager）

- [ ] **Step 6: Commit**

```bash
git add router/realtime_router.py tests/test_realtime_router.py
git commit -m "feat(realtime): rebuild ASR pipeline when asr_manager version changes"
```

---

### Task 8: `router/asr_admin_router.py` — 管理端點 + 註冊

**Files:**
- Create: `router/asr_admin_router.py`
- Modify: `router/openai_router.py`（import 區 L35 附近 + `app.include_router` 區 L127 之後）
- Test: `tests/test_asr_admin_router.py`（新檔）

**Interfaces:**
- Consumes: Task 5 的 `asr_manager`（`snapshot()/list_targets()/reload()/LOCAL_ENGINES`）與 `ASRConfigError(.message/.status_code/.details)`。
- Produces:
  - `GET /api/asr/config` → `200`：manager 快照 + `"targets": {"local": [...], "nodes": [...], "nodes_unreachable": bool}` + `"engines": {provider: [models]}`。
  - `POST /api/asr/config`（body `{"target_id": str, "provider": str|null, "model": str|null}`）→ `202`（snapshot，`state=loading`）；`ASRConfigError` 依 `.status_code` 轉為 HTTPException（400/404/409/503），detail 為 `{"message": ...}.update(details)`；`provider`/`model` 非 str → `400`。
  - 認證由 `runtime.security.middleware`（`add_security_middleware`，`openai_router.py` L57 已掛）對 `/api/*` 統一要求 API key，本 router 不自行驗證。
  - `openai_router.py` 以 `app.include_router(asr_admin_router)` **無條件**註冊（disabled 時 UI 仍可查狀態）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_asr_admin_router.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.asr_admin_router as asr_admin_module
from runtime.asr.asr_manager import ASRConfigError


@pytest.fixture
def client(monkeypatch):
    manager = MagicMock()
    manager.snapshot.return_value = {
        "state": "ready",
        "active": {"target_id": "local:cuda:1", "provider": "faster_whisper", "model": "large-v3", "loaded_at": "2026-11-03T00:00:00+00:00"},
        "requested": {"target_id": "local:cuda:1", "provider": "faster_whisper", "model": "large-v3", "loaded_at": "2026-11-03T00:00:00+00:00"},
        "error": None,
        "version": 3,
    }
    manager.list_targets.return_value = {
        "local": [{"id": "local:cuda:0", "name": "GPU0", "memory_free_mb": 1000}, {"id": "local:cpu", "name": "CPU", "memory_free_mb": None}],
        "nodes": [{"id": "node:m4", "reachable": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}],
        "nodes_unreachable": False,
    }
    monkeypatch.setattr(asr_admin_module, "asr_manager", manager)
    app = FastAPI()
    app.include_router(asr_admin_module.router)
    return TestClient(app), manager


def test_get_config_shape(client):
    c, _ = client
    resp = c.get("/api/asr/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "ready"
    assert body["version"] == 3
    assert body["targets"]["local"][0]["id"] == "local:cuda:0"
    assert body["targets"]["nodes"][0]["id"] == "node:m4"
    assert body["engines"]["faster_whisper"] == ["large-v3", "medium", "small"]
    assert body["engines"]["qwen_asr"] == ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"]


def test_post_config_returns_202_loading(client):
    c, manager = client
    manager.reload.return_value = {"state": "loading", "active": None, "requested": None, "error": None, "version": 3}
    resp = c.post("/api/asr/config", json={"target_id": "local:cuda:0", "provider": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"})
    assert resp.status_code == 202
    assert resp.json()["state"] == "loading"
    manager.reload.assert_called_once_with("local:cuda:0", provider="qwen_asr", model="Qwen/Qwen3-ASR-1.7B")


def test_post_config_validation_error_maps_status(client):
    c, manager = client
    manager.reload.side_effect = ASRConfigError("device_index 9 is not available on this host", details={"valid_gpu_ids": [0, 1]})
    resp = c.post("/api/asr/config", json={"target_id": "local:cuda:9"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "device_index 9" in detail["message"]
    assert detail["valid_gpu_ids"] == [0, 1]


def test_post_config_conflict_maps_409(client):
    c, manager = client
    manager.reload.side_effect = ASRConfigError("an ASR reload is already in progress", status_code=409)
    resp = c.post("/api/asr/config", json={"target_id": "local:cpu"})
    assert resp.status_code == 409


def test_post_config_missing_target_id_passes_empty_and_maps_400(client):
    c, manager = client
    manager.reload.side_effect = ASRConfigError("target_id is required")
    resp = c.post("/api/asr/config", json={})
    assert resp.status_code == 400
    manager.reload.assert_called_once_with("", provider=None, model=None)


def test_post_config_non_string_provider_400_without_reload(client):
    c, manager = client
    resp = c.post("/api/asr/config", json={"target_id": "local:cpu", "provider": 123})
    assert resp.status_code == 400
    manager.reload.assert_not_called()


def test_post_config_non_string_model_400_without_reload(client):
    c, manager = client
    resp = c.post("/api/asr/config", json={"target_id": "local:cpu", "model": ["large-v3"]})
    assert resp.status_code == 400
    manager.reload.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_admin_router.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'router.asr_admin_router'`）

- [ ] **Step 3: Implement — `router/asr_admin_router.py`**

```python
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from runtime.asr.asr_manager import ASRConfigError, LOCAL_ENGINES, asr_manager

router = APIRouter()


def _detail(exc: ASRConfigError) -> dict[str, Any]:
    detail: dict[str, Any] = {"message": exc.message}
    if exc.details is not None:
        detail.update(exc.details)
    return detail


@router.get("/api/asr/config")
def get_asr_config() -> dict[str, Any]:
    snapshot = asr_manager.snapshot()
    snapshot["targets"] = asr_manager.list_targets()
    snapshot["engines"] = {provider: list(models) for provider, models in LOCAL_ENGINES.items()}
    return snapshot


@router.post("/api/asr/config")
def post_asr_config(body: dict[str, Any] = Body(...)) -> JSONResponse:
    target_id = body.get("target_id")
    provider = body.get("provider")
    model = body.get("model")
    if target_id is not None and not isinstance(target_id, str):
        raise HTTPException(status_code=400, detail={"message": "target_id must be a string"})
    if provider is not None and not isinstance(provider, str):
        raise HTTPException(status_code=400, detail={"message": "provider must be a string"})
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail={"message": "model must be a string"})
    try:
        snapshot = asr_manager.reload(
            str(target_id or ""),
            provider=provider,
            model=model,
        )
    except ASRConfigError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_detail(exc)) from exc
    return JSONResponse(status_code=202, content=snapshot)
```

- [ ] **Step 4: 註冊進 `router/openai_router.py`**

import 區（L34 `from router.audit_router import ...` 附近）加：

```python
from router.asr_admin_router import router as asr_admin_router
```

`app.include_router(document_router)`（L127）之後加：

```python
app.include_router(asr_admin_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_asr_admin_router.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add router/asr_admin_router.py router/openai_router.py tests/test_asr_admin_router.py
git commit -m "feat(router): GET/POST /api/asr/config admin endpoints"
```

---

### Task 9: `node/asr_worker.py` — 遠端節點轉錄服務 + node_agent 探測 + launcher

**Files:**
- Create: `node/asr_worker.py`
- Modify: `node/node_agent.py`（模組層新增 `_probe_asr_worker()`；`NodeAgent.payload()` 的 `metadata` dict 加 `"asr_worker"` 鍵）
- Modify: `runtime/launcher/launcher.py`（`SERVICE_DEFS` 於 `worker_agent` 條目之後加 `asr_worker`）
- Test: `tests/test_asr_worker.py`（新檔，含 node_agent 探測與 launcher 條目測試）

**Interfaces:**
- Consumes: `settings.asr_provider/asr_model/asr_device/asr_device_index/asr_models_dir`（**遠端機器自己的** .env 決定引擎）；Task 3/4 adapter；`node_agent.payload()` 的 `metadata` 機制（control-plane 原樣保存並經 `GET /cluster/nodes` 回傳，Task 5 的 `_resolve_node_base_url` 消費）。
- Produces:
  - `node/asr_worker.py`：FastAPI `app`；`GET /health` → `{"status": "ok", "service": "asr-worker", "engine": <settings.asr_provider>, "model": <settings.asr_model>, "model_loaded": bool}`；`POST /v1/audio/transcriptions`（multipart：`file` 必填、`language` 表單選填）→ `{"text": str}`；引擎 lazy 載入（首次請求），模組層 `get_adapter()` 執行緒安全；啟動事件若 `AIIH_ASR_PRELOAD ∈ {1,true,yes}` 背景執行緒預載；引擎建構失敗 → `503`。
  - `node_agent._probe_asr_worker() -> {"port": int, "healthy": bool, "engine": str | None, "model": str | None}`（GET `http://127.0.0.1:{AIIH_ASR_PORT:-9450}/health`，2s timeout，任何例外 → `healthy=False`）；心跳 payload `metadata` 新增 `"asr_worker"` 鍵。
  - launcher 條目：`{"name": "asr_worker", "module": "node.asr_worker:app", "host": "0.0.0.0", "port_env": "AIIH_ASR_PORT", "port_default": 9450, "desc": "ASR transcription worker (OpenAI-compatible /v1/audio/transcriptions)"}`。預設啟用；主機不想跑時在 `config/services.json` 設 `{"asr_worker": {"enabled": false}}`（launcher 既有 `desired_enabled()` 機制，L339/L359-362）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_asr_worker.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import node.asr_worker as asr_worker


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(asr_worker, "_adapter", None)
    app = FastAPI()
    app.include_router(asr_worker.app.router)
    return TestClient(app)


def _engine(transcribe_result=None, side_effect=None):
    engine = MagicMock()
    engine.transcribe.return_value = transcribe_result if transcribe_result is not None else {"text": "hi"}
    if side_effect is not None:
        engine.transcribe.side_effect = side_effect
    return engine


def test_health_reports_unloaded_engine(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "asr-worker"
    assert body["model_loaded"] is False


def test_transcription_lazy_loads_and_returns_text(client, monkeypatch):
    engine = _engine({"text": "hello from node"})
    monkeypatch.setattr(asr_worker, "_build_adapter", lambda: engine)
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"wav-bytes", "audio/wav")},
        data={"language": "zh"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello from node"}
    engine.transcribe.assert_called_once()
    assert engine.transcribe.call_args.args[0] == b"wav-bytes"
    assert engine.transcribe.call_args.kwargs.get("language") == "zh"


def test_transcription_reuses_loaded_engine(client, monkeypatch):
    engine = _engine({"text": "second"})
    monkeypatch.setattr(asr_worker, "_build_adapter", lambda: engine)
    client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    client.post("/v1/audio/transcriptions", files={"file": ("b.wav", b"y", "audio/wav")})
    assert engine.transcribe.call_count == 2
    assert asr_worker._adapter is engine


def test_engine_build_failure_maps_503(client, monkeypatch):
    def boom():
        raise RuntimeError("qwen-asr not installed")

    monkeypatch.setattr(asr_worker, "_build_adapter", boom)
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_transcription_engine_error_maps_provider_status(client, monkeypatch):
    from providers.asr_base import ASRProviderError

    engine = _engine(side_effect=ASRProviderError("bad audio", status_code=400))
    monkeypatch.setattr(asr_worker, "_build_adapter", lambda: engine)
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    assert resp.status_code == 400


def test_probe_asr_worker_healthy(monkeypatch):
    import node.node_agent as node_agent_module

    resp = MagicMock(ok=True)
    resp.json.return_value = {"engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}
    monkeypatch.setattr(node_agent_module.requests, "get", lambda *a, **k: resp)
    monkeypatch.setenv("AIIH_ASR_PORT", "9450")
    assert node_agent_module._probe_asr_worker() == {
        "port": 9450, "healthy": True, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B",
    }


def test_probe_asr_worker_down(monkeypatch):
    import node.node_agent as node_agent_module

    def boom(*a, **k):
        raise node_agent_module.requests.ConnectionError("refused")

    monkeypatch.setattr(node_agent_module.requests, "get", boom)
    probe = node_agent_module._probe_asr_worker()
    assert probe == {"port": 9450, "healthy": False, "engine": None, "model": None}


def test_node_agent_payload_includes_asr_worker(monkeypatch):
    import node.node_agent as node_agent_module

    monkeypatch.setattr(node_agent_module.NodeAgent, "discover_workers", lambda self: [])
    monkeypatch.setattr(node_agent_module.NodeAgent, "_get_heavy_telemetry", lambda self, workers: ([], {}))
    monkeypatch.setattr(node_agent_module, "_probe_asr_worker", lambda: {"port": 9450, "healthy": False, "engine": None, "model": None})
    payload = node_agent_module.NodeAgent().payload()
    assert payload["metadata"]["asr_worker"] == {"port": 9450, "healthy": False, "engine": None, "model": None}


def test_launcher_defines_asr_worker_service():
    from runtime.launcher.launcher import SERVICE_DEFS

    entry = next(svc for svc in SERVICE_DEFS if svc["name"] == "asr_worker")
    assert entry["module"] == "node.asr_worker:app"
    assert entry["port_env"] == "AIIH_ASR_PORT"
    assert entry["port_default"] == 9450
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_worker.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'node.asr_worker'`；node_agent 無 `_probe_asr_worker`）

- [ ] **Step 3: Implement — `node/asr_worker.py`**

```python
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config.settings import settings

logger = logging.getLogger("node.asr_worker")

app = FastAPI(title="AetherMesh ASR Worker", version="5.0.0")

_adapter: Any = None
_adapter_lock = threading.Lock()


def _build_adapter() -> Any:
    if settings.asr_provider == "qwen_asr":
        from providers.qwen_asr_adapter import QwenASRAdapter

        return QwenASRAdapter(
            model_name=settings.asr_model if "/" in settings.asr_model else "Qwen/Qwen3-ASR-1.7B",
            device=settings.asr_device,
            device_index=settings.asr_device_index,
            download_dir=settings.asr_models_dir,
        )
    from providers.faster_whisper_adapter import FasterWhisperAdapter

    return FasterWhisperAdapter(
        model_name=settings.asr_model,
        device=settings.asr_device,
        device_index=settings.asr_device_index,
        compute_type=settings.asr_compute_type,
        download_dir=settings.asr_models_dir,
    )


def get_adapter() -> Any:
    global _adapter
    if _adapter is not None:
        return _adapter
    with _adapter_lock:
        if _adapter is None:
            logger.info(
                "ASR worker loading engine=%s model=%s device=%s index=%s",
                settings.asr_provider, settings.asr_model, settings.asr_device, settings.asr_device_index,
            )
            _adapter = _build_adapter()
            logger.info("ASR worker engine loaded")
    return _adapter


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "asr-worker",
        "engine": settings.asr_provider,
        "model": settings.asr_model,
        "model_loaded": _adapter is not None,
    }


@app.post("/v1/audio/transcriptions")
def transcriptions(file: UploadFile = File(...), language: str = Form("")) -> dict[str, Any]:
    try:
        adapter = get_adapter()
    except Exception as exc:
        logger.exception("ASR worker adapter load failed")
        raise HTTPException(status_code=503, detail=f"ASR engine unavailable: {exc}") from exc
    audio = file.file.read()
    try:
        result = adapter.transcribe(audio, language=language)
    except Exception as exc:
        logger.error("ASR worker transcription failed: %s", exc)
        status = getattr(exc, "status_code", 503)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return {"text": str(result.get("text") or "")}


@app.on_event("startup")
def _startup_preload() -> None:
    if os.getenv("AIIH_ASR_PRELOAD", "").strip().lower() in ("1", "true", "yes"):
        threading.Thread(target=get_adapter, daemon=True, name="asr-worker-preload").start()
```

- [ ] **Step 4: Implement — `node/node_agent.py`**

模組層（`app = FastAPI(...)` 之後、`class NodeAgent` 之前）加：

```python
def _probe_asr_worker() -> dict[str, Any]:
    port = int(os.getenv("AIIH_ASR_PORT", "9450"))
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if response.ok:
            data = response.json()
            return {"port": port, "healthy": True, "engine": data.get("engine"), "model": data.get("model")}
    except (requests.RequestException, ValueError):
        pass
    return {"port": port, "healthy": False, "engine": None, "model": None}
```

`NodeAgent.payload()` 的 `metadata` dict（L166-172）加一鍵：

```python
            "metadata": {
                "rpc_port": settings.worker_rpc_port,
                "node_port": settings.node_agent_port,
                "worker_runtime": worker_runtime,
                "detail_refresh_s": self._detail_refresh_s,
                "detail_cached_at": self._last_detail_refresh,
                "asr_worker": _probe_asr_worker(),
            },
```

- [ ] **Step 5: Implement — `runtime/launcher/launcher.py`**

`SERVICE_DEFS` 的 `worker_agent` 條目（L98-105）之後插入：

```python
    {
        "name": "asr_worker",
        "module": "node.asr_worker:app",
        "host": "0.0.0.0",
        "port_env": "AIIH_ASR_PORT",
        "port_default": 9450,
        "desc": "ASR transcription worker (OpenAI-compatible /v1/audio/transcriptions)",
    },
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_asr_worker.py -v`
Expected: 9 passed

- [ ] **Step 7: 既有 node_agent / launcher 測試回歸**

Run: `pytest tests/ -k "node_agent or launcher or cluster" -q`
Expected: 無新增失敗（`payload()` 加鍵不影響既有斷言；若既有 payload 測試對 metadata 做**精確 dict 相等**斷言，更新該斷言加入 `asr_worker` 鍵）

- [ ] **Step 8: Commit**

```bash
git add node/asr_worker.py node/node_agent.py runtime/launcher/launcher.py tests/test_asr_worker.py
git commit -m "feat(node): asr_worker service with node_agent probe and launcher entry"
```

---

### Task 10: Dashboard — `/api/asr/config` proxy + ASR Settings 切換表單

**Files:**
- Modify: `dashboard/dashboard_server.py`（`api` router 區塊，`audio_status` L2272 之後加 proxy 端點與 `_router_api_key()`）
- Modify: `dashboard/static/admin_audio.js`（L86-106 `loadAsrSettings` 整段替換為表單版 + 新增 `applyAsrConfig`/`pollAsrState`/helper）
- Test: `tests/test_dashboard_asr.py`（新檔）

**Interfaces:**
- Consumes: Task 8 的 router 端 `GET/POST /api/asr/config`（400/404/409/503 原樣轉傳）；既有 `api = APIRouter(prefix="/api")`（L90）、`_require_admin(request)`（L216）、`ROUTER_BASE_URL`（L39）、`httpx`。
- Produces:
  - dashboard `GET /api/asr/config`、`POST /api/asr/config`（admin cookie 保護；x-api-key 由 `_router_api_key()` 伺服器端注入，**金鑰不進瀏覽器**；router 回應 JSON 原樣 + 原狀態碼；連不上 router → `503`）。
  - `admin_audio.js`：ASR Settings 卡片渲染狀態 pill + active 資訊 + Target/Provider/Model 下拉 + Apply 按鈕 + 2s 輪詢（上限 120s）。函式名 `loadAsrSettings()` 不變（既有呼叫點不動，見檔案底部 init 區）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_asr.py
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dashboard.dashboard_server as dashboard_server


@pytest.mark.asyncio
async def test_asr_config_proxy_get_passthrough(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            return MagicMock(status_code=200, json=lambda: {"state": "ready", "version": 1}, text="")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("AIIH_API_KEY", "key-123")
    response = await dashboard_server._asr_config_proxy("GET")
    assert response.status_code == 200
    assert json.loads(response.body) == {"state": "ready", "version": 1}
    assert captured["url"] == f"{dashboard_server.ROUTER_BASE_URL}/api/asr/config"
    assert captured["headers"]["x-api-key"] == "key-123"


@pytest.mark.asyncio
async def test_asr_config_proxy_post_forwards_body_and_status(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, content=None, headers=None):
            captured["url"] = url
            captured["content"] = content
            captured["headers"] = headers
            return MagicMock(status_code=400, json=lambda: {"detail": {"message": "bad"}}, text="")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    response = await dashboard_server._asr_config_proxy("POST", body=b'{"target_id": "x"}')
    assert response.status_code == 400
    assert json.loads(response.body) == {"detail": {"message": "bad"}}
    assert captured["content"] == b'{"target_id": "x"}'
    assert captured["headers"]["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_asr_config_proxy_router_down_maps_503(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            raise dashboard_server.httpx.ConnectError("refused")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    with pytest.raises(dashboard_server.HTTPException) as excinfo:
        await dashboard_server._asr_config_proxy("GET")
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_asr_config_proxy_non_json_response_passthrough_text(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return MagicMock(status_code=500, json=MagicMock(side_effect=ValueError("not json")), text="server exploded")

    monkeypatch.setattr(dashboard_server.httpx, "AsyncClient", FakeClient)
    response = await dashboard_server._asr_config_proxy("GET")
    assert response.status_code == 500
    assert json.loads(response.body) == {"message": "server exploded"}


def test_asr_config_get_requires_admin():
    app = FastAPI()
    app.include_router(dashboard_server.api)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/asr/config")
    assert resp.status_code in (401, 403)


def test_asr_config_post_requires_admin():
    app = FastAPI()
    app.include_router(dashboard_server.api)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/asr/config", json={"target_id": "local:cpu"})
    assert resp.status_code in (401, 403)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_asr.py -v`
Expected: FAIL（dashboard_server 無 `_asr_config_proxy` attribute error；`/api/asr/config` 404）

- [ ] **Step 3: Implement — `dashboard/dashboard_server.py` proxy**

在 `audio_status`（L2272 `return {"tts": ..., "asr": ...}` 之後、`@api.get("/audio/voices")` 之前）插入：

```python
def _router_api_key() -> str:
    env_key = os.environ.get("AIIH_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip().split(",")[0].strip()
    if hasattr(settings, "api_key") and settings.api_key:
        return str(settings.api_key)
    return ""


async def _asr_config_proxy(method: str, body: bytes | None = None):
    headers = {"x-api-key": _router_api_key()}
    if body is not None:
        headers["content-type"] = "application/json"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if method == "GET":
                resp = await client.get(f"{ROUTER_BASE_URL}/api/asr/config", headers=headers)
            else:
                resp = await client.post(f"{ROUTER_BASE_URL}/api/asr/config", content=body or b"", headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"ASR config proxy error: {exc}")
    try:
        payload = resp.json()
    except ValueError:
        payload = {"message": resp.text[:200]}
    return JSONResponse(status_code=resp.status_code, content=payload)


@api.get("/asr/config")
async def asr_config_get(request: Request):
    _require_admin(request)
    return await _asr_config_proxy("GET")


@api.post("/asr/config")
async def asr_config_post(request: Request):
    _require_admin(request)
    body = await request.body()
    return await _asr_config_proxy("POST", body=body)
```

- [ ] **Step 4: Implement — `dashboard/static/admin_audio.js`**

L86-106 的 `async function loadAsrSettings() {...}` 整段替換為以下內容（`escapeHtml` 沿用既有全域；函式其餘呼叫點不動）：

```javascript
  var ASR_STATE_PILL = { ready: 'ok', loading: 'warn', failed: 'bad', disabled: 'disabled', idle: 'disabled' };

  async function loadAsrSettings() {
    var panel = document.getElementById('asr-settings-panel');
    if (!panel) return;
    try {
      var resp = await fetch('/api/asr/config');
      if (resp.status === 401 || resp.status === 403) { redirectLogin(); return; }
      if (resp.status === 503) { panel.innerHTML = '<span class="pill warn">Router unreachable</span>'; return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load</span>'; return; }
      renderAsrSettings(panel, await resp.json());
    } catch(e) {
      panel.innerHTML = '<span class="pill warn">Error</span>';
    }
  }

  function asrTargetLabel(t) {
    if (t.memory_free_mb != null) {
      return t.id + ' — ' + (t.name || '') + ' (free ' + (t.memory_free_mb / 1024).toFixed(1) + ' GB)';
    }
    return t.id + (t.name ? ' — ' + t.name : '');
  }

  function updateAsrModelOptions(engines, provider, selected) {
    var modelSel = document.getElementById('asr-model');
    if (!modelSel) return;
    var models = engines[provider] || [];
    modelSel.innerHTML = models.map(function(m) {
      return '<option value="' + escapeHtml(m) + '"' + (m === selected ? ' selected' : '') + '>' + escapeHtml(m) + '</option>';
    }).join('');
  }

  function renderAsrSettings(panel, cfg) {
    var state = cfg.state || 'idle';
    var pillCls = ASR_STATE_PILL[state] || 'disabled';
    var active = cfg.active || {};
    var requested = cfg.requested || {};
    var html = '<div class="settings-grid">' +
      '<div class="label">Status</div><div class="value"><span class="pill ' + pillCls + '">' + escapeHtml(state) + '</span>' +
      (state === 'failed' && cfg.error ? ' <span style="color:var(--bad)">' + escapeHtml(cfg.error) + '</span>' : '') + '</div><div></div>' +
      '<div class="label">Active</div><div class="value"><code>' + escapeHtml(active.provider || '—') + ' / ' + escapeHtml(active.model || '—') + ' @ ' + escapeHtml(active.target_id || '—') + '</code></div><div></div>' +
      '</div>';
    if (state === 'disabled') {
      panel.innerHTML = html + '<div class="hint">ASR is disabled (AIIH_ASR_ENABLED=false). Enable it in .env and restart the router.</div>';
      return;
    }
    var localTargets = (cfg.targets && cfg.targets.local) || [];
    var nodeTargets = (cfg.targets && cfg.targets.nodes) || [];
    var options = '';
    localTargets.forEach(function(t) {
      options += '<option value="' + escapeHtml(t.id) + '">' + escapeHtml(asrTargetLabel(t)) + '</option>';
    });
    nodeTargets.forEach(function(t) {
      var label = t.id + (t.engine ? ' — ' + t.engine + '/' + t.model : ' — asr worker');
      options += '<option value="' + escapeHtml(t.id) + '"' + (t.reachable ? '' : ' disabled') + '>' + escapeHtml(label) + (t.reachable ? '' : ' (unreachable)') + '</option>';
    });
    var engineOptions = '';
    Object.keys(cfg.engines || {}).forEach(function(p) {
      engineOptions += '<option value="' + escapeHtml(p) + '">' + escapeHtml(p) + '</option>';
    });
    var selectedTarget = requested.target_id || active.target_id || '';
    var selectedProvider = requested.provider || active.provider || '';
    var selectedModel = requested.model || active.model || '';
    html += '<div class="upload-row" style="margin-top:10px">' +
      '<select id="asr-target">' + options + '</select>' +
      '<select id="asr-provider">' + engineOptions + '</select>' +
      '<select id="asr-model"></select>' +
      '<button class="btn btn-sm" id="btn-asr-apply" onclick="window.applyAsrConfig()">Apply</button>' +
      '<span id="asr-apply-msg" style="margin-left:8px;font-size:0.82rem;color:var(--muted)"></span>' +
      '</div>' +
      '<div class="hint">Switching loads the new engine in the background; the current one keeps serving until the swap completes. First load can take 30-60s.</div>' +
      '<div class="hint">Use <code>POST /v1/audio/transcriptions</code> or <code>/v1/audio/translations</code> to transcribe audio files.</div>';
    panel.innerHTML = html;
    var targetSel = document.getElementById('asr-target');
    var providerSel = document.getElementById('asr-provider');
    var modelSel = document.getElementById('asr-model');
    if (selectedTarget) targetSel.value = selectedTarget;
    if (selectedProvider) providerSel.value = selectedProvider;
    updateAsrModelOptions(cfg.engines || {}, providerSel.value, selectedModel);
    var syncEngineDisabled = function() {
      var isNode = targetSel.value.indexOf('node:') === 0;
      providerSel.disabled = isNode;
      modelSel.disabled = isNode;
    };
    providerSel.addEventListener('change', function() { updateAsrModelOptions(cfg.engines || {}, providerSel.value, ''); });
    targetSel.addEventListener('change', syncEngineDisabled);
    syncEngineDisabled();
  }

  window.applyAsrConfig = async function() {
    var btn = document.getElementById('btn-asr-apply');
    var msg = document.getElementById('asr-apply-msg');
    var target = document.getElementById('asr-target').value;
    var isNode = target.indexOf('node:') === 0;
    var body = { target_id: target };
    if (!isNode) {
      body.provider = document.getElementById('asr-provider').value;
      body.model = document.getElementById('asr-model').value;
    }
    if (btn) { btn.disabled = true; btn.textContent = 'Applying…'; }
    if (msg) { msg.textContent = 'Loading new engine… (first load 30-60s)'; msg.style.color = 'var(--warn)'; }
    try {
      var resp = await fetch('/api/asr/config', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (resp.status === 401 || resp.status === 403) { redirectLogin(); return; }
      if (!resp.ok) {
        var err = await resp.json().catch(function() { return {}; });
        var detail = err.detail || {};
        var text = detail.message || ('HTTP ' + resp.status);
        if (detail.valid_gpu_ids) text += ' (valid GPUs: ' + detail.valid_gpu_ids.join(', ') + ')';
        if (detail.valid_models) text += ' (valid models: ' + detail.valid_models.join(', ') + ')';
        if (detail.valid_providers) text += ' (valid providers: ' + detail.valid_providers.join(', ') + ')';
        if (msg) { msg.textContent = text; msg.style.color = 'var(--bad)'; }
        if (btn) { btn.disabled = false; btn.textContent = 'Apply'; }
        return;
      }
      await pollAsrState();
    } catch(e) {
      if (msg) { msg.textContent = 'Error: ' + e; msg.style.color = 'var(--bad)'; }
      if (btn) { btn.disabled = false; btn.textContent = 'Apply'; }
    }
  };

  async function pollAsrState() {
    var deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
      await new Promise(function(resolve) { setTimeout(resolve, 2000); });
      try {
        var resp = await fetch('/api/asr/config');
        if (!resp.ok) continue;
        var cfg = await resp.json();
        var panel = document.getElementById('asr-settings-panel');
        if (panel) renderAsrSettings(panel, cfg);
        if (cfg.state !== 'loading') {
          if (cfg.state === 'failed') {
            var msg = document.getElementById('asr-apply-msg');
            if (msg) { msg.textContent = cfg.error || 'load failed'; msg.style.color = 'var(--bad)'; }
          }
          return;
        }
      } catch(e) { continue; }
    }
    var msg = document.getElementById('asr-apply-msg');
    if (msg) { msg.textContent = 'Still loading after 120s — check the router log.'; msg.style.color = 'var(--warn)'; }
  }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_asr.py -v`
Expected: 6 passed

- [ ] **Step 6: Dashboard 全測試回歸**

Run: `pytest tests/ -k "dashboard" -q`
Expected: 無新增失敗（既有 `test_dashboard_auth.py` 的環境相關失敗數不變）

- [ ] **Step 7: Commit**

```bash
git add dashboard/dashboard_server.py dashboard/static/admin_audio.js tests/test_dashboard_asr.py
git commit -m "feat(dashboard): ASR provider switching form with live state polling"
```

---

### Task 11: 全套件驗證 + 真實引擎安裝（選配）+ 手動驗收

**Files:**
- No source changes（本 task 只驗證；若回歸發現問題，回到所屬 task 修復後重跑）

**前置基線：** 1042 passed + 16 個 pre-existing 環境失敗（`test_custom_providers` / `test_dashboard_auth` / `test_image_gen` / `test_security` 各檔內，與本計畫無關）+ realtime 45/45。

- [ ] **Step 1: 新測試全數通過**

Run: `pytest tests/test_settings.py tests/test_gpu_discovery.py tests/test_qwen_asr_adapter.py tests/test_remote_asr_adapter.py tests/test_asr_manager.py tests/test_audio_router.py tests/test_realtime_router.py tests/test_asr_admin_router.py tests/test_asr_worker.py tests/test_dashboard_asr.py -q`
Expected: 全部 passed，0 failed（新檔 6+5+14+9+19+2+1+7+9+6 = 78 個新測試，含改寫；`test_language_code_mapping` 參數化 ×4 已計入）

- [ ] **Step 2: 全套件回歸與基線比對**

Run: `pytest tests/ -q`
Expected: passed 數 ≥ 基線（1042 + 75 ≈ 1117）；failed 恰為已知 16 個 pre-existing 環境失敗，**無任何新增失敗**。任何新失敗先歸因（本計畫 commit 逐一套 `git bisect` 或逐檔重跑），修復後才進 Step 3。

- [ ] **Step 3: 選配 — 安裝真實 Qwen3-ASR 套件（若此機器要跑 qwen_asr 引擎）**

Run: `pip install -U qwen-asr`
Expected: 安裝成功（會一併帶入 torch/transformers；Windows 需 CUDA 版 torch，詳 spec §7）。若此機器僅跑 faster_whisper / 純 remote 客戶端，可跳過——所有單元測試不依賴此套件。

- [ ] **Step 4: 手動驗收（spec §7，需真實 GPU / 叢集環境）**

啟動 `python -m runtime.launcher.launcher` 後依序確認：

1. `GET http://127.0.0.1:8001/api/asr/config`（帶 API key）→ `state=ready`、`targets.local` 列出 GPU 0（RTX 5090）與 GPU 1（RTX 4070 Ti SUPER）及各自 `memory_free_mb`、`engines` 含兩 provider。
2. Dashboard `/admin/audio` ASR Settings 卡片顯示狀態 pill 與 active 引擎。
3. 選 `local:cuda:1` + `qwen_asr` + `Qwen/Qwen3-ASR-1.7B` → Apply → pill 轉 `loading`、舊引擎持續服務（期間打 `/v1/audio/transcriptions` 應成功）→ 30-60s 後轉 `ready`，active 顯示 qwen_asr。
4. 切換後 `nvidia-smi` 確認舊引擎 VRAM 已釋放（無雙佔）。
5. 重啟 router → active 仍為 qwen_asr（`config/asr_config.json` 持久化生效）。
6. 故意選超大模型或拔掉 GPU 觸發失敗 → pill 轉 `failed` 並顯示錯誤、舊引擎續用、version 不變。
7. （有遠端節點時）節點跑 `asr_worker` → `GET /cluster/nodes` 該節點 `metadata.asr_worker.healthy=true` → dashboard 下拉出現 `node:<id>` → 切換後轉錄走遠端。
8. Realtime WebSocket 會話進行中套用切換 → 下一幀起用新引擎，連線不中斷。

- [ ] **Step 5: 收尾 commit（僅當 Step 2 有修正）**

```bash
git add -A
git commit -m "fix: address regressions found during full-suite verification"
```
