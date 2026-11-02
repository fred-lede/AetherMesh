# ASR Provider 熱切換設計 — faster-whisper / Qwen3-ASR / 叢集遠端節點

- 日期：2026-11-03
- 狀態：待用戶審閱
- 前置脈絡：`docs/superpowers/specs/2026-10-25-realtime-subtitle-translation-design.md`（realtime 字幕翻譯契約）、2026-11-02 realtime 修正（ASR 啟動預載、transcript `text`/`seg`/`is_final` 欄位、斷線日誌）

## 1. 背景與目標

目前 ASR 唯一支援的引擎是本機 faster-whisper（`large-v3`，CUDA），且 provider 寫死在 `provider_router._get_asr_adapter()` 的 lazy singleton。2026-10-28 起多次因 GPU VRAM 不足（`CUDA failed with error out of memory`）導致 ASR 完全不可用，realtime 字幕翻譯隨之失效。

本設計提供：

1. **雙引擎並存**：faster-whisper 與 Qwen3-ASR（`Qwen/Qwen3-ASR-1.7B`，必要時 `0.6B`）皆可作為 ASR 引擎。
2. **Dashboard 熱切換**：在 `/admin/audio` 頁的 ASR Settings 卡片切換引擎/模型/執行位置，**不需重啟 router**；切換期間舊引擎持續服務。
3. **多 GPU 選擇**：偵測到兩張以上 GPU 時，使用者可指定 ASR 用哪張卡（含即時 VRAM 餘量顯示）；單卡系統照常顯示、無特殊分支。
4. **叢集遠端節點**：遠端機器跑 AetherMesh node 服務組（node_agent + 新增的 asr_worker），主機可把 ASR 指派到叢集任一節點的 GPU，本機 VRAM 歸零佔用。

### 成功標準

- Dashboard 切換後，realtime 字幕翻譯與 `POST /v1/audio/transcriptions` 均使用新引擎/位置；切換過程中進行中的 session 不中斷（舊引擎服務到替換完成）。
- 切換失敗（OOM、套件缺失、遠端不可達）時：舊引擎繼續服務、UI 顯示明確錯誤、設定檔保留新選擇（重啟重試）。
- 設定重啟後保留（`config/asr_config.json`）。
- 新引擎載入完成後，舊引擎的 VRAM 全數釋放（nvidia-smi 可驗證）。

## 2. 現況程式碼地圖（工程師需知的檔案）

| 檔案 | 現況 | 本設計的變更 |
|---|---|---|
| `runtime/orchestration/provider_router.py`（L242-269 附近） | `_get_asr_adapter()` lazy singleton + 失敗 cooldown | 改為委派給 `asr_manager`；對外介面不變 |
| `providers/faster_whisper_adapter.py` | `FasterWhisperAdapter`，`__init__` 載入 `WhisperModel`；`.transcribe(audio_bytes)` 檔案式介面；`._model` 為 `WhisperModel`（`transcribe(samples, **kwargs) -> (segments, info)` 串流契約） | 不改；實例生命週期移交 manager |
| `providers/streaming_asr.py`（L116-120） | `StreamingASR` 呼叫 `self._model.transcribe(samples_16k_float32, **kwargs)`，取 `seg.text`/`seg.language`/`info.language` | 不改（契約不變） |
| `providers/asr_base.py` | `ASRProviderAdapter`、`ASRProviderError` 基類 | 不改；新 adapter 繼承 |
| `router/realtime_router.py` | 2026-11-02 加入 `_asr_ready`/`_preloaded_asr_model`/`preload_asr()`；`process_audio` 首選預載模型 | 改為向 manager 查版本號；預載改呼 `manager.preload()` |
| `config/settings.py` | `asr_enabled/asr_model/asr_device/asr_device_index/asr_compute_type/asr_models_dir`；`save_*` JSON 原子寫入模式（L416-451） | 新增 `asr_provider` 欄位（`AIIH_ASR_PROVIDER`，預設 `faster_whisper`）；新增 `load_asr_config()`/`save_asr_config()` |
| `cluster/gpu_discovery.py` | `discover_gpus()`：nvidia-smi 列舉（id/name/memory=總量/utilization），**無 free memory**；無驅動安全回空 | 改：nvidia-smi 查詢加入 `memory.used`，回傳新增 `memory_used_mb`/`memory_free_mb`（NVIDIA 路徑；MPS 路徑此二欄為 `null`），既有欄位不動 |
| `control_plane/cluster_manager.py` | `GET /cluster/nodes`（節點登記簿：node_id/ip/gpus/workers/metadata）、`GET /cluster/gpu` | 不改；router 端讀取 |
| `node/node_agent.py` | 心跳 payload：`{node_id, ip, gpus, workers, metadata}`；worker 探索走 `/api/tags`（ollama 相容） | 新增 asr_worker 探測（見 4.6） |
| `dashboard/dashboard_server.py` | `/admin/audio` 頁 + `proxy_audio_*`（x-api-key 轉發 router 模式） | 新增 `/api/asr/config` GET/POST proxy |
| `dashboard/static/admin_audio.js` | ASR Settings 卡片唯讀渲染（`/api/audio/status`） | 改為可切換表單 + 狀態輪詢 |
| `runtime/launcher/launcher.py` | 9 服務啟動表 | 新增 `asr_worker` 服務列（本機也可選裝） |

## 3. 架構總覽

**新增/修改元件**：

| 元件 | 層 | 職責 |
|---|---|---|
| `runtime/asr/asr_manager.py`（新） | runtime | 唯一事實來源：目標解析、adapter 生命週期、狀態機、版本號、`reload()` 熱切換 |
| `providers/qwen_asr_adapter.py`（新） | providers | Qwen3-ASR 引擎 adapter（transformers backend） |
| `providers/remote_asr_adapter.py`（新） | providers | 遠端節點 ASR 的 HTTP 客戶端 adapter |
| `node/asr_worker.py`（新） | node | 遠端機器上的 OpenAI 相容轉錄服務（複用 providers adapters） |
| `router/asr_admin_router.py`（新） | router | 純協定：`GET/POST /api/asr/config` |
| `runtime/orchestration/provider_router.py`（改） | runtime | `_get_asr_adapter()` 委派 manager |
| `router/realtime_router.py`（改） | router | 預載/快取跟隨 manager 版本號 |
| `dashboard/dashboard_server.py` + `static/admin_audio.js`（改） | dashboard | proxy + 切換表單 UI |
| `config/settings.py`（改） | config | `asr_provider` 欄位 + `load_asr_config()`/`save_asr_config()` |

**資料流**：

```
UI 下拉選單 → POST dashboard /api/asr/config (cookie auth)
           → POST router /api/asr/config (x-api-key)
           → asr_manager.reload(target, provider, model)
              ├─ 驗證 target（provider 已知、local device_index 在 GPU 範圍內）
              ├─ 寫 config/asr_config.json（接受即持久化）
              ├─ 背景執行緒：建構新 adapter（舊 adapter 繼續服務）
              ├─ 成功 → 原子替換 + version += 1 → 卸載舊 adapter（釋放 VRAM）
              └─ 失敗 → state=failed、error 記錄、舊 adapter 續用、version 不變
UI 每 2s 輪詢 GET /api/asr/config 直到 state ≠ loading
router 啟動：讀 config/asr_config.json（存在則覆蓋 .env 預設）→ manager.preload()
```

## 4. 詳細設計

### 4.1 `runtime/asr/asr_manager.py` — 狀態機與生命週期

```python
@dataclass
class ASRTarget:
    target_id: str            # "local:cuda:0" | "local:cpu" | "node:<node_id>"
    kind: str                 # "local_cuda" | "local_cpu" | "node"
    node_id: str | None       # kind=node 時有值
    base_url: str | None      # kind=node 時，由叢集登記簿解析 http://<ip>:<asr_port>
    device_index: int | None  # kind=local_cuda 時有值
```

`ASRManager` 單例，執行緒安全（`threading.Lock` 保護狀態交換）：

- **狀態**：`state ∈ {disabled, idle, loading, ready, failed}`；`disabled` = `settings.asr_enabled=false`。對外快照：`{state, active, requested, error, version}`；`active`/`requested` 各含 `{target_id, provider, model, loaded_at}`。
- **`get_adapter()`**：回傳目前生效的 adapter 物件（含 `._model` 串流契約與 `.transcribe(audio_bytes)` 檔案契約）。`loading` 期間回傳**舊** adapter；從未載入成功過則回 `None`（呼叫端走既有 "ASR adapter unavailable" 錯誤路徑）。
- **`reload(target, provider, model)`**：
  1. 驗證：provider ∈ {faster_whisper, qwen_asr}（僅 local 目標需要）；local_cuda 的 `device_index` 必須 < `len(discover_gpus())`，否則回 400 與有效清單。
  2. `requested` 更新 + `state=loading` + 寫 `config/asr_config.json`（接受即持久化，見 4.10）。
  3. daemon thread（沿用 `realtime_router.preload_asr` 的 thread 模式）建構新 adapter；同一時間僅允許一個進行中的 reload（進行中收到新請求 → 409）。
  4. 成功：lock 內原子交換 `_active_adapter`，`version += 1`，隨後在 lock 外卸載舊 adapter；`state=ready`、`loaded_at=now`、`error=None`。
  5. 失敗：`state=failed`、`error=原因`；`_active_adapter` 與 `version` 不變（記憶體回滾）。
- **VRAM 雙重駐留**：替換完成前新舊模型短暫並存（large-v3 ~3GB + Qwen1.7B ~4.6GB ≈ 7.6GB）。同卡餘量不足 → 載入失敗走回滾，error 含 CUDA OOM 訊息。不做事前 VRAM 估算（事後失敗比事前猜測可靠）。
- **卸載**：faster-whisper → `del adapter` + `gc.collect()`（CTranslate2 於 gc 時釋放 CUDA）；qwen → 另加 `torch.cuda.empty_cache()`；remote → 無本地資源，僅丟棄參照。
- **`list_targets()`**：組合三種來源：
  - `local:cuda:N`：`discover_gpus()`（先依第 2 節 code map 擴充 `memory.used` 查詢），附 `name`/`memory_free_mb`（MPS 節點此欄為 `null`）（5s TTL 快取，避免 UI 輪詢打爆 nvidia-smi）。
  - `local:cpu`：固定存在。
  - `node:<node_id>`：向 control-plane `GET /cluster/nodes`（`settings.control_plane_url`）取節點 `node_id`/`ip`；節點 heartbeat metadata 含 asr_worker 探測結果（見 4.6）者，附 `reachable`/`engine`/`model`。control-plane 不可達 → 節點區塊標 `unreachable`，local 目標不受影響。
- **`preload()`**：啟動時依持久化設定建構 adapter（同 reload 的 thread 路徑，但無舊 adapter 可回滾；失敗 → `state=failed`）。

### 4.2 執行位置與引擎的語意

- `provider`/`model`（引擎選擇）**僅對 local 目標生效**：`faster_whisper` → `{large-v3, medium, small}`；`qwen_asr` → `{Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B}`。
- `node` 目標：引擎與模型由**遠端節點的 asr_worker 自己的設定決定**；主機端 UI 對 node 目標顯示 worker 回報的 `engine`/`model`（唯讀）。`reload(node_target)` 時忽略 provider/model。
- `device`（執行位置）對兩種 local 引擎的映射：
  - faster-whisper：`device="cuda", device_index=N`；cpu → `device="cpu", compute_type="int8"`。
  - qwen（transformers）：`torch.device(f"cuda:{N}")`；cpu → `torch.device("cpu")`。

### 4.3 `router/asr_admin_router.py` — 管理端點（純協定）

- `GET /api/asr/config` → manager 快照 + `list_targets()` + 引擎/模型目錄：
  ```json
  {
    "state": "ready",
    "active": {"target_id": "local:cuda:1", "provider": "faster_whisper", "model": "large-v3", "loaded_at": "...", "version": 3},
    "requested": {"...同上..."},
    "error": null,
    "targets": {
      "local": [{"id": "local:cuda:0", "name": "NVIDIA GeForce RTX 5090", "memory_free_mb": 31756},
                 {"id": "local:cpu", "name": "CPU"}],
      "nodes": [{"id": "node:m4-strata", "reachable": true, "engine": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}]
    },
    "engines": {"faster_whisper": ["large-v3", "medium", "small"],
                 "qwen_asr": ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"]}
  }
  ```
- `POST /api/asr/config` body：`{"target_id": "local:cuda:1", "provider": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B"}`（node 目標可省略 provider/model）。回應：`202 {"state": "loading", ...}`；驗證失敗 400（附有效清單）、reload 進行中 409。
- 認證：沿用既有 `runtime.security.middleware`（`/api/*` 需 API key；dashboard 以 x-api-key 伺服器端轉發，金鑰不進瀏覽器）。
- 註冊進 `router/openai_router.py` 的 app。

### 4.4 `providers/qwen_asr_adapter.py` — Qwen3-ASR 引擎

```python
class QwenASRAdapter(ASRProviderAdapter):
    provider_name = "qwen_asr"

    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B", device="cuda",
                 device_index=0, download_dir=None): ...
    # _load_model(): 建構子內 import（套件缺失 → ASRProviderError 含安裝提示）
    #   使用 qwen-asr 套件 transformers backend；權重快取至 download_dir
    #   ._model 為內部 _QwenTranscriber：.transcribe(samples, language="") -> (segments, info)
    #   segments[0].text / info.language —— 與 faster-whisper 串流契約一致

    def transcribe(self, audio: bytes, task="transcribe", language="",
                   prompt="", temperature=0.0, response_format="json") -> dict:
    # 檔案式介面（/v1/audio/transcriptions 用）：bytes → 記憶體解碼 16k → ._model
    # 回傳 {"text": ...}；異常包裝為 ASRProviderError(503)

    def health_check(self) -> dict: ...  # provider/model_loaded/device/device_index
```

- **依賴**：`qwen-asr`（PyPI，會帶 transformers；torch 已在 venv）。**僅用 transformers backend**（vLLM 不支援 Windows，不採用）。
- **輸入**：16kHz（與 `StreamingASR` 一致）；`language=""` 由模型自動偵測。
- **效能預期**：1.7B 於 2.5s 窗口單次推論應遠低於 1s（5090/4070）；實測若過慢，驗收時保留退回 faster-whisper 的能力（engine 切換本身即是本功能）。
- **BF16 VRAM**：1.7B ≈ 4.6GB、0.6B ≈ 1.6GB。

### 4.5 `providers/remote_asr_adapter.py` — 叢集節點客戶端

```python
class RemoteASRAdapter(ASRProviderAdapter):
    provider_name = "remote_asr"

    def __init__(self, node_id, base_url, timeout_s=10.0): ...
    # transcribe(audio: bytes, language=""):
    #   audio(PCM16 16k) → 記憶體 WAV 封裝 → multipart POST {base_url}/v1/audio/transcriptions
    #   → 解析 {"text": ...}；._model 為內部 _RemoteTranscriber
    #   （samples-based transcribe：float32 → int16 → WAV → 同上，串流契約一致）
    #   逾時/連線失敗 → ASRProviderError(503, 含 node_id 與 base_url)
    # health_check(): GET {base_url}/health → {"reachable": bool, "engine": ..., "model": ...}
```

- 節點 IP 由 manager 從 `GET /cluster/nodes` 解析；node 無心跳 → 建構時即標 `unreachable` 並在 reload 結果回報錯誤。
- 認證：asr_worker 預設 LAN 信任（與現有 ollama worker 一致）；選配 `AIIH_ASR_WORKER_TOKEN`，設定時以 `Authorization: Bearer` 附帶。

### 4.6 `node/asr_worker.py` — 遠端節點上的轉錄服務

- FastAPI 小服務（與 `node/node_agent.py` 同模式）：
  - `GET /health` → `{"status":"ok","service":"asr-worker","engine":"qwen_asr","model":"...","model_loaded":true}`
  - `POST /v1/audio/transcriptions`（OpenAI 相容：multipart `file`/`language`）→ 內部呼叫 `providers/qwen_asr_adapter` 或 `providers/faster_whisper_adapter`（依**遠端機器自己的** `AIIH_ASR_PROVIDER`/`AIIH_ASR_MODEL`）→ `{"text": ...}`
- 預設 port **9450**（`AIIH_ASR_PORT`）；lazy 載入模型（首次請求）+ 選配啟動預載（遠端 `.env` 的 `AIIH_ASR_PRELOAD=true`）。
- **node_agent 整合**：node_agent 心跳前探測 `http://127.0.0.1:{AIIH_ASR_PORT}/health`（2s timeout），結果放進 payload `metadata.asr_worker = {"port": 9450, "healthy": bool, "engine": ..., "model": ...}`；control-plane 照既有機制儲存，router 端由此得知節點的 asr 能力與埠號。探測失敗 → 該節點在 ASR 目標清單標 `unreachable`，不影響節點本身註冊。
- launcher 服務表新增 `asr_worker` 列（`module: node.asr_worker:app`，port 9450），本機/遠端皆可選裝。

### 4.7 `provider_router.py` 整合

`_get_asr_adapter()`（與其 cooldown 邏輯）改為：

```python
def _get_asr_adapter():
    return asr_manager.get_adapter()   # manager 內含失敗語意
```

- `asr_enabled=false` → manager `state=disabled`，`get_adapter()` 回 `None`（既有 "ASR is not enabled" 路徑不變）。
- 移除舊 module-level `_asr_adapter`/`_asr_adapter_failed_at`（職責移入 manager；manager 的 reload 失敗已涵蓋 cooldown 的目的）。

### 4.8 `realtime_router.py` 整合（版本失效）

- 移除 module-level `_asr_ready`/`_preloaded_asr_model`/`_preload_thread`，改：
  - `preload_asr()` → `asr_manager.preload()`
  - `process_audio` 開頭記錄 `version_before = asr_manager.version()`；每次 `queue.get()` 後若 `asr_manager.version() != version_before` 或 `audio_pipeline is None` → 重建 pipeline（`asr_model = asr_manager.get_adapter()._model`）。
- 語意：管理員切換後，**新音訊幀**用新引擎；切換瞬間已緩衝的 interim 音訊隨 pipeline 丟棄（明確管理動作，可接受）。進行中的翻譯請求不受影響。

### 4.9 Dashboard

**`dashboard_server.py`**（proxy，同 `proxy_audio_speech` 模式）：

- `GET /api/asr/config` → 轉發 router（x-api-key）。
- `POST /api/asr/config` → 轉發 router；router 4xx/5xx 原樣轉傳。
- 兩者皆走 cookie session 認證（現行 dashboard 認證）。

**`static/admin_audio.js`** — ASR Settings 卡片重寫：

- 狀態列：`state` pill（ready/loading/failed/disabled）+ `active`（provider/model/target）+ `error` 紅字（failed 時）。
- 表單：Target 下拉（`local:cuda:N — <GPU 名>（餘 X.X GB）`、`node:<id> — engine/model`（unreachable 灰顯）、`local:cpu`）+ Provider 下拉（僅 local 目標顯示）+ Model 下拉（隨 provider 連動，目錄來自 `engines`）。
- Apply → POST → 禁用按鈕 → 每 2s 輪詢 GET 直到 `state ≠ loading`（上限 120s，超時提示查看 router log）。
- 保留現有「Use POST /v1/audio/transcriptions…」提示。

### 4.10 持久化 `config/asr_config.json`

```json
{
  "target_id": "local:cuda:1",
  "provider": "qwen_asr",
  "model": "Qwen/Qwen3-ASR-1.7B",
  "updated_at": "2026-11-03T12:00:00"
}
```

- 由 router 在 POST 接受時原子寫入（`settings.save_asr_config()`，tmp+replace，同 `save_cloud_credentials` 模式）；dashboard 不直接寫檔。
- router 啟動時 `settings.load_asr_config()`：存在 → 覆蓋 `.env` 的 asr 預設；不存在/損毀 → 用 env 預設並 log 警告。
- 失敗政策：載入失敗**不**回寫舊設定（重啟會重試新選擇）；`error` 與 `active` 的差異即「請求 vs 實際」。

## 5. 錯誤處理政策（彙總）

| 情境 | 行為 |
|---|---|
| 新引擎載入失敗（OOM/套件缺失/權重下載失敗） | 記憶體回滾：舊 adapter 續用、version 不變；`state=failed`、error 給 UI；設定檔保留新選擇 |
| reload 進行中再收到切換請求 | 409，UI 提示等待 |
| POST 帶未知 provider / 越界 device_index | 400 + 有效清單 |
| node 目標：control-plane 不可達 | 該目標 `unreachable`；local 目標不受影響 |
| node 目標：asr_worker 不可達 | reload 失敗回滾（同第一列）；UI 顯示 node_id + base_url |
| realtime session 中途切換 | 新音訊幀用新引擎；緩衝 interim 音訊丟棄；翻譯中請求不受影響 |
| `asr_enabled=false` | `state=disabled`，UI 顯示停用；表單隱藏 |

## 6. 測試計畫

| 測試檔 | 覆蓋 |
|---|---|
| `tests/test_asr_manager.py`（新） | 狀態機轉換；reload 成功 → version+1 + 舊 adapter 卸載（mock teardown 計數）；失敗 → 回滾（active/version 不變、error 記錄）；loading 期間 `get_adapter()` 回舊 adapter；config 持久化與啟動載入；`list_targets()` 組合（mock discover_gpus/control-plane）；device_index 越界拒絕；單一進行中 reload（第二次 → 409） |
| `tests/test_qwen_asr_adapter.py`（新） | 契約一致性（sys.modules 預先 mock `qwen_asr`，CI 無 GPU/無套件）；`_model.transcribe` 回 (segments, info) 形狀；language 傳遞；bytes `.transcribe()` 回 `{"text"}`；套件缺失 → ASRProviderError 含安裝提示；device 映射 |
| `tests/test_remote_asr_adapter.py`（新） | httpx MockTransport：WAV 封裝、`{"text"}` 解析、逾時 → ASRProviderError(503) 含 node_id；`_model` 串流契約（float32 → WAV → text） |
| `tests/test_asr_admin_router.py`（新） | GET 形狀（state/active/targets/engines）；POST 成功 202；壊 provider 400；越界 device 400；reload 中 409；disabled 狀態 |
| `tests/test_asr_worker.py`（新） | FastAPI TestClient：`/health`；POST wav → mock adapter → `{"text"}`；未載入時 503 |
| `tests/test_realtime_router.py`（擴充） | manager version 變更 → pipeline 重建、新幀使用新 model（mock manager）；version 不變 → pipeline 複用 |
| `tests/test_audio_router.py`（既有，改寫） | 現有 ASR cooldown 測試（直接操作 `pr._asr_adapter`/`_asr_adapter_failed_at` module globals）改寫為 manager 委派語意：`_get_asr_adapter()` 回 manager 結果；disabled → None |
| dashboard 測試（擴充 `tests/test_dashboard_*.py` 模式） | proxy GET/POST 轉發與 401 路徑 |

全數不依賴 GPU/網路（adapter 層 mock；manager 的 thread 邏輯以注入的假建構器測試）。

## 7. 驗收標準（部署後手動）

1. `nvidia-smi` 顯示 ASR 模型載入於指定卡；dashboard 切換到另一卡後 VRAM 於舊卡歸零。
2. Dashboard 切換 faster_whisper ↔ qwen_asr：realtime session 保持連線，新幀 transcript 由新引擎產生（log 顯示新 engine/model）。
3. 故意選 Qwen1.7B 於餘量不足的卡 → UI 顯示 failed + OOM 原因，realtime 仍以舊引擎產生 transcript。
4. 重啟 router：設定保留、log 出現 preload 結果（成功載入秒數或失敗原因）。
5. （有遠端節點時）切換到 `node:<id>`：本機 ASR VRAM 歸零，遠端節點 log 有請求；拔線後 UI 標 unreachable。

## 8. 不在範圍內（Out of Scope)

- 遠端節點的自動化部署腳本/映像（node 服務組的安裝文件另行提供）。
- Qwen3-ASR 的 vLLM backend / 真 VAD 串流（維持現行窗口式管線）。
- ASR 負載平衡或多節點同時服務（單一目標生效制）。
- `/v1/models` 對 ASR engine 目錄的動態揭露（維持現狀 `whisper-large-v3 (asr)` 註冊；後續如需再議）。
- realtime 的 OpenAI 官方事件名相容性（另案）。
