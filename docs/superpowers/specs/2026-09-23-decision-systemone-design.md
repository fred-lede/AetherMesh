# Decision Models (`/v1/systemone`) — 設計規格

- 日期：2026-09-23
- 狀態：Draft（待審）
- 關聯：`router/`、`providers/ollama_adapter.py`、`runtime/orchestration/provider_router.py`、`runtime/orchestration/routing_engine.py`、`config/models.yaml`、Dashboard Models capability 詞彙

## 1. 目標與範圍

### 目標
讓 AetherMesh 支援 Ollama 的 **decision 模型**與其 `POST /v1/systemone` API：用戶以 OpenAI router（8001）呼叫 `/v1/systemone`，AetherMesh 依 `model` 路由到對應 Ollama worker 並回傳結果。

### 成功標準
- `POST http://<aethermesh>:8001/v1/systemone` 帶合法 payload，能路由到裝有該模型的 Ollama worker 並回傳 Ollama 的原始結果。
- `tev1:0.8b`、`tev1:4b`、`nimble:9b` 可透過 `models.yaml` 手動登錄並被子模型表/Dashboard 顯示。
- 能力詞彙新增 `decision`，可在 Dashboard Models 勾選、且驗證接受。

### 非目標（第二階段）
- 自動探索 Ollama 已安裝的 decision 模型（本階段**手動**登錄）。
- 將 `decision` 做成一等 routing 能力與評分（本階段比照 image_gen 第一階段：只路由 + 端點）。
- 串流（觀察到 `/v1/systemone` 為單次 JSON）。
- 雲端 decision provider（目前不存在）。

## 2. 已實測的 API contract（Ollama）

以本機 Ollama（`http://127.0.0.1:11434`）實測：

- `GET /v1/systemone` → **405**（端點存在，POST-only）。
- `POST {}` → `{"error":"model is required"}`
- 缺 `state` → `{"error":"state: must be a string, object, or array"}`
- `questions` 非物件/為空 → `{"error":"questions must contain 1–64 fields"}`
- `questions.<key>` 值必須是 `decision.Question` 物件：`{"type","instructions","criteria"}`
  - 缺 `instructions` → `instructions must be a nonempty string, object, or array`
  - `type` ∈ `choice | noul | score`
  - `choice`：`criteria` = 選項→描述（或 `null`）的 map
  - `score`：`criteria` = 描述陣列

**合法請求**：
```json
{
  "model": "nimble:9b",
  "state": "Book a flight to Tokyo",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What does the user want to do?",
      "criteria": {"book": "book travel", "cancel": "cancel something", "other": null}
    }
  }
}
```

**回應**：
```json
{
  "model": "nimble:9b",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "book",
      "probabilities": {"book": 0.99, "cancel": 0.001, "other": 0.008},
      "confidence": 0.948
    }
  },
  "usage": {"input_tokens": 171, "output_tokens": 1}
}
```
`score` 型別回應範例：`{"type":"score","score":0.636,"legend":{"0":"...","1":"..."},"probabilities":{...},"confidence":0.26}`。

已安裝模型（Ollama `/api/tags` 的 `capabilities` 含 `decision`）：`tev1:0.8b`（752M, ctx 262144）、`tev1:4b`（4.2B, ctx 262144）、`nimble:9b`（9.0B, ctx 262144）。

## 3. 架構與流程

### 元件
1. **Capability 詞彙**（`providers/registry.py`）：新增 `Capability.DECISION = "decision"` 與 aliases（`"decision"`）。
2. **Dashboard 能力清單**（`dashboard/dashboard_server.py` `_CAPABILITY_GROUPS`）：新增 `{"value":"decision","label":"Decision","group":"other"}`。
3. **models.yaml**：手動登錄三個模型（見 §4）。
4. **Adapter**（`providers/ollama_adapter.py`）：新增 `systemone(payload) -> dict`，`POST {base_url}/v1/systemone`。
5. **Route**（`router/decision_router.py`，新檔，比照 `image_router.py`）：`POST /v1/systemone`；於 `router/openai_router.py` 掛載。
6. **Provider 解析**：`provider_router.resolve_provider(model, {"models": model_registry_store.get_models()})` → `(provider, worker)`。

### 請求流程
1. `POST /v1/systemone` 收到 payload（標準 dashboard/auth middleware）。
2. 取 `model`（必要）；無 → 400（AetherMesh 訊息），其餘欄位原樣轉發。
3. `resolve_provider(model, registry)`：
   - `provider == "ollama"` → `provider_router.adapter("ollama", worker)` 取得帶 worker `base_url` 的 `OllamaAdapter`。
   - 其他 provider：
     - 若 adapter 有 `systemone` → 呼叫之（目前無）。
     - 否則 → **501** `provider '<p>' does not support decision (/v1/systemone) yet`。（不再默默打本地。）
4. 以 `asyncio.to_thread(adapter.systemone, payload)` 呼叫。
5. 回傳 Ollama 的 JSON 原樣；`usage` 可選擇記錄到 token tracker（若可對應 user/model）。
6. 上游錯誤（4xx/5xx）：轉成對應 HTTPException，附上游訊息。

### 模型名稱處理
- 沿用既有 `settings.strip_model_route_prefix()`（若有 route 前綴，如 `ollama/`）後再送 worker；worker 端 `/v1/systemone` 需要真實模型名（如 `nimble:9b`）。

## 4. models.yaml 登錄（手動）

```yaml
- name: tev1:0.8b
  provider: ollama
  worker_bindings:
  - node_id: node-01
    port: 11434
  capabilities: [decision, tools, thinking, chat]
  estimated_vram_mb: null
  context_length: 262144
- name: tev1:4b
  provider: ollama
  worker_bindings:
  - node_id: node-01
    port: 11434
  capabilities: [decision, tools, thinking, chat]
  estimated_vram_mb: null
  context_length: 262144
- name: nimble:9b
  provider: ollama
  worker_bindings:
  - node_id: node-01
    port: 11434
  capabilities: [decision, tools, thinking, chat]
  estimated_vram_mb: null
  context_length: 262144
```

## 5. 驗證與錯誤處理

- `model` 缺失 → AetherMesh 回 400（`model is required`）。
- `payload` 其他格式錯誤由 Ollama 回 400，AetherMesh 透傳其 `error` 訊息（附 provider 標記）。
- 找不到模型的 worker → 依既有 fallback（`local_ollama_fallback`）或 400/404，訊息清楚。
- 401/403/429 等上游狀態碼保留。

## 6. 測試計畫

| 檔案 | 覆蓋 |
|---|---|
| `tests/test_ollama_systemone.py` | `OllamaAdapter.systemone()` → `POST {base}/v1/systemone`、回傳 JSON、HTTP 錯誤轉 `ProviderError` |
| `tests/test_decision_router.py` | 合法請求經路由回傳 provider 結果；`model` 缺失 400；不支援 provider（如 custom）501；上游錯誤透傳 |

工具：mock `OllamaAdapter.systemone` / `provider_router.adapter`，`fastapi.testclient`。capability 驗證測試加在 `tests/test_model_registry_store.py`（`decision` 被接受）。

## 7. 文件

- README：新增 `### Decision Models (/v1/systemone)` 章節（contract、curl 範例、模型登錄）。
- PROGRESS.md append、TASK.md 記錄。
- `.env.example`：無新變數（沿用 `AIIH_RERANK_BASE_URL` 無關；decision 走一般 Ollama worker）。
- `docs/providers/`：可選一份 `decision-models.md`。

## 8. 風險與緩解

| 風險 | 緩解 |
|---|---|
| Ollama `/v1/systemone` contract 為非標準/可能變動 | 本 spec 以實測記錄；AetherMesh 主要做透傳，不重新實作決策邏輯 |
| 大 payload（state 為大型 object/array） | 透傳，不解析；timeout 沿用 `settings.request_timeout_s`（決策可能較慢，必要時提高） |
| 誤把 decision 模型當 chat 用 | capability 標註；models.yaml 標 `decision` |
| 未來多 worker | 依 `resolve_provider` 選有該模型的 worker；未知模型走既有 fallback |

## 9. 第二階段（不在本次）

- Ollama `/api/tags` capabilities 自動探索並匯入 models.yaml。
- `decision` 納入 `CAPABILITY_PROVIDER_SCORES` 與 `ProviderCapabilities` 評分。
- 串流 / 批次 decision。
