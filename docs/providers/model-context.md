# 模型上下文長度 (Context Length)

AetherMesh 用**模型層**的上下文長度來判斷某個模型能否容納請求的 token 量。上下文長度是**模型**屬性，不是 provider 屬性，因此可在 `config/models.yaml` 的每個模型上獨立設定。

## 解析順序

`runtime/orchestration/model_context.py::resolve_effective_max_context(provider, model)` 依序解析：

1. **models.yaml 該模型的 `context_length`**（用戶手填，最高優先）
2. **自動抓取快取**（透過 CLI `refresh` 取得）
3. **provider 層 fallback**（`provider_scoring.py` 內建值：gemini 200K、openai 128K、nvidia_nim 128K、ollama 32K、ollama_cloud 32K）

## 手動設定

在 `config/models.yaml` 的模型下加 `context_length`：

```yaml
- name: gemma4:e2b
  provider: ollama
  worker_bindings:
    - node_id: node-01
      port: 11434
  context_length: 32768
```

## 自動抓取 (CLI)

不想手填時，可用 CLI 自動抓取並（可選）回寫 models.yaml。

```bash
# 抓單一模型
.venv\Scripts\python.exe -m runtime.orchestration.model_context_cli fetch gemma4:e2b

# 指定 provider / base_url（雲端通常需要 api_key）
.venv\Scripts\python.exe -m runtime.orchestration.model_context_cli fetch gpt-4.1-mini \
  --provider openai --base-url https://api.openai.com/v1 --api-key sk-...
```

抓取來源：

- **Ollama**：`POST {base_url}/api/show`。依序嘗試頂層 `context_length` → `parameters.num_ctx` → `model_info.{architecture}.context_length`（例 `gemma4.context_length`）。
- **雲端 (OpenAI 相容)**：`GET {base_url}/models` → 依序嘗試欄位 `context_window` / `context_length` / `max_context_window` / `max_context_length` / `max_model_len` / `max_sequence_length`。

> **多 worker / 遠端節點**：Ollama 模型若 `worker_bindings` 有多個 binding（含遠端節點），`refresh` 會依序嘗試每個 binding（透過 `cluster.yaml` 的 `node_hosts` 解析 IP），直到抓到 ctx 為止。離線節點會優雅略過，不中斷。

掃描 models.yaml 中所有**未設定** `context_length` 的模型並自動抓取：

```bash
# 只印出結果（不寫入）
.venv\Scripts\python.exe -m runtime.orchestration.model_context_cli refresh

# 抓取後回寫 models.yaml
.venv\Scripts\python.exe -m runtime.orchestration.model_context_cli refresh --write
```

## `/v1/models` 暴露

`GET /v1/models`（OpenAI + Anthropic 路徑）每次請求都從 `config/models.yaml` 讀檔，並在每個模型的 metadata 帶上 `context_length`（與 AI 平台的 `context_window` 對映）。任何 OpenAI 相容客戶端（GUI、dashboard、自訂腳本）可直接讀取真實視窗，不需內建 models DB。

> 注意：opencode（及其 Telegram bot）的 context 顯示**不讀** AetherMesh `/v1/models`，而是用 opencode.jsonc 中該模型的 `limit.context`（缺省 200000）。要在 opencode 側修正顯示，請在 `~/.config/opencode/opencode.jsonc` 的 provider 模型定義補 `"limit": {"context": <ctx>, "output": <out>}`。

## 影響

`resolve_effective_max_context` 用於兩處評分：

- `runtime/intelligence/provider_scoring.py` — `_score_provider` 的 context penalty
- `runtime/intelligence/execution_selector.py` — `_context_penalty`

當 `estimated_input_tokens` 超過模型的 `context_length` 時，該 provider 會被扣分（選取時降權）。此為選取參考，不會實際截斷請求。