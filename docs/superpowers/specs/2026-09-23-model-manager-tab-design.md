# Dashboard Model Manager Tab — 設計規格

- 日期：2026-09-23
- 狀態：Draft（待審）
- 關聯：`config/models.yaml`、`dashboard/`、`runtime/orchestration/`、`providers/registry.py`

## 1. 目標與範圍

### 目標
在 Dashboard 新增一個 **Models** Tab，讓使用者（含遠端）以 UI 完成 `models.yaml` 的新增 / 編輯 / 刪除，取代用文字編輯器手改檔案。

### 成功標準
- 透過 UI 存檔後，**不需重啟任何服務**，新模型即出現在 `/v1/models` 並可被路由。
- 能區分**本地模型**與**雲端模型**。
- 能力以**複選**方式設定；可設定**上下文窗口（context_length）**。
- 本地模型可編輯 `worker_bindings`（node_id + port）。
- 改名時能偵測並（可選）更新內部參照，避免孤立設定。

### 非目標（第二階段）
- 為 `image_gen` / `video` 接上路由評分（本階段僅標註 + UI）。
- 編輯 provider 憑證（沿用既有 Cloud Credentials / custom_providers 流程）。
- 在 Models Tab 直接編輯 `routing_rules.yaml` 的 alias（改名掃描時僅「一併更新既有參照」）。
- 編輯本地 Ollama 實際已安裝但未列於 `models.yaml` 的模型（探索/匯入留待後續）。

## 2. 已定決策

| # | 決策 | 內容 |
|---|---|---|
| D1 | 持久化 + 熱載入 | **Approach A**：直接原子寫入 `config/models.yaml`；跨行程靠檔案 mtime 自動熱載（無 IPC） |
| D2 | Capability 擴充 | 新增 `image_gen`、`video`，拆分既有 `image → vision` alias；本階段**僅標註 + UI** |
| D3 | worker_bindings | 支援本地模型 `worker_bindings` 編輯（node_id + port 可增刪） |
| D4 | 改名 | **可改名**，掃描 `routing_rules.yaml`（alias/fallback/rules）參照並提供「一併更新 / 取消」；集群 runtime assignments 僅警示；外部客戶端以警示呈現 |

## 3. 現況與熱載入阻礙

`config/settings.py:212 model_registry()` 每次呼叫都重新讀檔（**無快取**），因此磁碟讀取本身可即時反映變更。問題在於**建構時快照**的 consumer：

| 元件 | 位置 | 現狀 | 需要 |
|---|---|---|---|
| `RouterService.registry` | `runtime/orchestration/openai_handler.py:85` | 建構時快照（`router/openai_router.py:48` 模組單例） | 入口檢查 mtime → 重讀 |
| `AnthropicRouter.registry` | `runtime/orchestration/anthropic_converter.py:30` | 同上（`router/anthropic_router.py:44`） | 同上 |
| routing engine provider tables | `runtime/orchestration/routing_engine.py:128`（`_load_config()`） | 啟動時載入 | mtime 變更時重跑 |
| `settings.model_registry()` | `config/settings.py:212` | 已即時 | 無需改 |
| `provider_router._CUSTOM_PROVIDERS` | `runtime/orchestration/provider_router.py:52` | provider 層 | 模型異動不需 |

既有可沿用模式：
- 熱載入前例：`reload_custom_providers()`（`provider_router.py:116`）。
- 原子寫入 + Windows 重試：`routing_engine.py:163-175`。
- mtime 熱重載前例：`runtime/alerting/alert_manager.py:54-64`。

## 4. UI 設計

### 4.1 Tab
新增 `data-tab="models"` 的 tab 與對應 `tab-content`（現有 tabs：overview / cluster / providers / charts / traces / system）。沿用 `switchTab()`（`index.html:961`）機制。

### 4.2 工具列
- `＋ 新增模型`
- 搜尋框（依 `name` 過濾）
- `Provider` 下拉篩選
- 類別分段鈕：`全部 / 本地 / 雲端`
- `重新載入`（呼叫 `POST /api/models/reload`）
- 摘要 chips：總數 / 本地 / 雲端

### 4.3 清單表
欄位：`名稱 | 類別 | Provider | 能力(徽章) | Context | Workers | 操作`
- 類別：本地（有 `worker_bindings`）/ 雲端（`provider` 屬雲端集）。
- Workers：本地顯示 `node-01:11434`，多筆以 `+N` 收合；雲端顯示 `—`。
- 操作：`編輯 / 複製 / 刪除`；點列開啟 Drawer。

### 4.4 編輯 Drawer（右側滑出）
- 標題：`新增模型` / `編輯：<name>`
- `name`：模型 ID，必填、唯一、格式 `^[A-Za-z0-9._:/+-]+$`。
- `provider`：依類別過濾
  - 本地：`ollama`、`xtts`…
  - 雲端：`openai`、`gemini`、`nvidia_nim`、`ollama_cloud` + 自訂 provider（讀 `custom_providers.json`）；旁顯示憑證狀態。
- 本地專屬：
  - `worker_bindings`：可增刪列，`node_id` 下拉（來源 `cluster.yaml` `node_hosts`）、`port`（1–65535）。
  - `estimated_vram_mb`：選填整數 ≥ 0。
- 雲端專屬：`worker_ports`（唯讀、預設空）。
- **Capabilities**：checkbox 方格（複選），分組如下：

| 分組 | UI 標籤 | canonical 值 |
|---|---|---|
| 核心 | Chat | `chat` |
| 核心 | Responses | `responses` |
| 核心 | Streaming | `streaming` |
| 推理/工具 | Reasoning | `thinking` |
| 推理/工具 | Tool | `tools` |
| 推理/工具 | MCP | `mcp` |
| 推理/工具 | Web Search | `web_search` |
| 模態 | Vision（影像輸入） | `vision` |
| 模態 | Image（生圖） | `image_gen` |
| 模態 | Audio | `audio` |
| 模態 | Video | `video` |
| 模態 | Documents | `documents` |
| 其他 | Embedding | `embeddings` |
| 其他 | Reranker | `rerank` |

- **Context Window**：數字輸入（token）+ 快捷鈕 `32K / 131K / 200K / 1M` + `自動抓取`（呼叫 `POST /api/models/{name}/fetch-context`）。
- 頁尾：`儲存 / 取消`；編輯時多 `刪除`。驗證錯誤行內顯示。

## 5. 後端 API（`dashboard/dashboard_server.py`，`api = APIRouter(prefix="/api")`）

| Method | Path | 權限 | 說明 |
|---|---|---|---|
| GET | `/api/models` | 已認證 | 清單（附 category / workers / 憑證狀態） |
| GET | `/api/models/capabilities` | 已認證 | 能力詞彙（value + label + group） |
| GET | `/api/models/providers` | 已認證 | 依類別列出可用 provider + 憑證狀態 |
| GET | `/api/models/nodes` | 已認證 | 節點清單（`cluster.yaml`），供 worker_bindings 下拉 |
| POST | `/api/models` | **admin** | 新增 |
| PUT | `/api/models/{name}` | **admin** | 更新（可含改名） |
| DELETE | `/api/models/{name}` | **admin** | 刪除 |
| POST | `/api/models/{name}/fetch-context` | **admin** | 自動抓取 context_length |
| POST | `/api/models/reload` | **admin** | 強制重載 |

- 權限：讀取走既有認證 middleware（`dashboard_server.py:218`）；寫入呼叫 `_require_admin()`（`:212`）。
- 錯誤碼：`400` 驗證失敗（含明細）、`404` 模型不存在、`409` 名稱重複、`423` 檔案鎖定（重試後仍失敗）。
- 所有寫入成功後呼叫 `reload_models()`。

### `PUT` 改名流程
1. 驗證新 `name` 唯一且格式合法。
2. 掃描內部參照（見 §7）；若無參照 → 直接改。
3. 若有參照 → 回應 `200` 並附 `references: [...]` 清單，前端以對話框詢問「一併更新 / 僅改模型 / 取消」：
   - **一併更新**：`PUT` 帶 `update_references=true`，同步改寫 `routing_rules.yaml`（集群 runtime assignments 不持久化，僅警示）。
   - **僅改模型**：只改 `models.yaml`。
   - **取消**：不動作。

## 6. 資料層與熱載入

### 6.1 新模組 `runtime/orchestration/model_registry_store.py`
```python
def models_path() -> Path
def load_models() -> list[dict]                 # 讀 models.yaml，保留其他頂層 key
def save_models(models: list[dict]) -> None     # 原子寫 + .bak
def get_models() -> list[dict]                  # mtime 快取：未變回快取，變了重讀
def reload_models() -> None                     # 重讀 + 刷新 consumer 快照
def models_mtime() -> float

def normalize_model(entry: dict) -> dict        # 正規化欄位與 capability 值
def validate_model(entry, existing_names, providers, capabilities) -> tuple[dict, list[str]]
```

- `save_models`：`models.yaml.tmp` → `os.replace()`，`PermissionError` 重試 ×5（backoff `0.1s*(attempt+1)`），並保留 `models.yaml.bak`。
- 寫入保留頂層其他 key（目前僅 `models:`，但以防萬一）。
- 正規化欄位順序：`name, provider, worker_bindings|worker_ports, capabilities, estimated_vram_mb, context_length`。

### 6.2 驗證規則（`validate_model`）
| 欄位 | 規則 |
|---|---|
| `name` | 非空、符合 `^[A-Za-z0-9._:/+-]+$`、在 `models.yaml` 內唯一 |
| `provider` | 在允許集合內（依類別） |
| `worker_bindings`（本地） | 至少 1 筆；`node_id` 存在於 `cluster.yaml`；`port` ∈ 1–65535 |
| `worker_ports`（雲端） | 允許空陣列 |
| `capabilities` | 子集於 canonical 詞彙；寫入前正規化（去重、排序） |
| `context_length` | 正整數或 `null` |
| `estimated_vram_mb` | 整數 ≥ 0 或 `null` |

### 6.3 consumer mtime 刷新
- `RouterService`：新增 `_ensure_registry()`，於 `handle_chat` / `handle_streaming_chat` / `handle_responses` / `handle_streaming_responses` / `list_models` 入口比對 `models_mtime()`，變更則 `self.registry = settings.model_registry()`。
- `AnthropicRouter`：同上。
- `routing_engine.route()`：進入時比對 mtime，變更則重跑 `_load_config()`。
- 跨行程：讀寫同檔，各行程以 mtime 自我刷新；`POST /api/models/reload` 為手動強制選項。

## 7. 改名參考掃描（D4 / C）

`scan_model_references(old_name, new_name)` 掃描：
- `config/routing_rules.yaml`：alias 對映（alias → model）、fallback 鏈、明確路由規則中的模型名。
- 集群 runtime `model_assignments` / worker registry：**非持久化設定**，不自動改寫，僅以 `source: "runtime"` 提示。

回傳 `[{source, path, value}]`。`update_model_references(old, new)` 以同樣的原子寫入模式改寫 `routing_rules.yaml`。

- 外部客戶端（opencode.jsonc、Telegram、GUI、腳本）無法自動更新 → UI 明確警示使用者需自行更換。
- 歷史資料（token 用量、episodic memory）不改寫，僅斷連續性。

## 8. Capability 詞彙擴充

- `providers/registry.py`：`Capability` 新增 `IMAGE_GEN = "image_gen"`、`VIDEO = "video"`；`CAPABILITY_ALIASES` 將 `image` 改指向 `IMAGE_GEN`（不再是 `VISION`）。
- 注意：`providers/registry.py` 的 registry **未被執行路徑使用**（僅測試）；路由直接比對字串。因此本階段 `image_gen` / `video` 僅**標註 + UI**，不影響路由。
- 第二階段若需參與路由，須同步改：`routing_engine.CAPABILITY_PROVIDER_SCORES`（`:22-30`）、`runtime/intelligence/provider_scoring.py` `ProviderCapabilities`、必要時 `runtime/orchestration/capabilities.py`。
- **遷移**：掃描既有 `models.yaml` 的 `capabilities:` 是否含 `image`（舊義為 vision）；若有，於上線時改為 `image_gen` 或 `vision`（依模型性質人工判定），並在 UI 顯示提示。

## 9. 測試計畫

| 檔案 | 覆蓋 |
|---|---|
| `tests/test_model_registry_store.py` | load/save 原子性、`.bak`、PermissionError 重試、mtime 快取失效、保留頂層 key、壞 yaml 處理、normalize/validate 規則 |
| `tests/test_models_api.py` | CRUD、admin 403、驗證 400、重複 409、不存在 404、reload、providers/nodes/capabilities 端點 |
| `tests/test_model_references.py` | `scan_model_references` 於 alias/fallback/rules 的偵測、`update_model_references` 原子改寫且保留其他內容 |
| 熱載入 | `RouterService._ensure_registry()` 與 routing engine mtime 重載吃到新檔 |
| capability | `image_gen` / `video` parse、`image` alias 拆分回歸 |

手動 E2E：Dashboard 存檔 → `/v1/models` 無重啟即出現；改名掃描 → 更新參照後 alias 仍指向有效模型。

## 10. 交付與部署

- 後端：`dashboard_server.py` 新增端點；需**重啟 dashboard(9001)** 載入。
- 前端：`dashboard/templates/index.html` + `dashboard/static/dashboard.js` 為磁碟靜態檔，重整頁面即生效（沿用 cache-buster）。
- 文件：README（新 Tab + 端點）、PROGRESS.md（append）、TASK.md、必要時 HANDOFF.md。

## 11. 風險與緩解

| 風險 | 緩解 |
|---|---|
| 多行程同時寫檔（Windows 檔案鎖） | 原子寫 + `PermissionError` 重試；dashboard 為唯一寫入者 |
| `yaml.safe_dump` 重排格式、遺失註解 | models.yaml 目前無註解；UI 成為唯一編輯來源 |
| consumer 未刷新導致仍用舊模型 | 入口 mtime 檢查覆蓋三個 consumer |
| 改名孤立外部參照 | 掃描 + 一併更新 + UI 警示 |
| 刪除使用中模型 | 允許但路由 fallback；UI 警告 |
| 舊 `image` 能力語意變更 | 上線前掃描 + 人工判定遷移 |
