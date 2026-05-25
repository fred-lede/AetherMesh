# Phase 21 — Responses API 完整多輪工具循環實作計畫

> 目標：讓 OpenAI Responses API 對接所有雲端與本地供應商，支援自動 tool call 執行。

---

## 問題分析

目前 `RouterService.handle_responses()` 只發送一次請求到模型就回傳。完整的 Responses API 規格要求：

```
Client: POST /v1/responses { tools: [...], input: "..." }
  │
  ▼
Model: 回傳 tool_calls (status: requires_action, type: function_call)
  │
  ▼
Server: 自動執行 tool call
  │
  ▼
Server: 把 tool_result 送回 model
  │
  ▼
Model: 再次回傳 (可能又有 tool_calls，或最終 text)
  │
  ▼
Server: status: completed → 回 Client
```

**需要實作的是這個 loop。**

---

## 現有可用基礎

| 元件 | 檔案 | 狀態 |
|------|------|------|
| Response Runtime (store) | `runtime/responses/response_runtime.py` | ✅ 完整 |
| Response Models | `runtime/responses/response_models.py` | ✅ 完整 |
| Input Converter (Responses→Chat) | `runtime/responses/input_converter.py` | ✅ 完整 |
| Output Converter (Chat→Responses) | `runtime/responses/output_converter.py` | ✅ 完整 |
| Stream Encoder (SSE) | `runtime/responses/response_stream.py` | ✅ 完整 |
| Tool Registry | `runtime/tools/tool_registry.py` | ✅ 完整 |
| Tool Executor | `runtime/tools/tool_executor.py` | ✅ 完整 |
| Tool Runtime | `runtime/tools/tool_runtime.py` | ✅ 完整 |
| Tool Normalizer | `runtime/tools/tool_normalizer.py` | ✅ 完整 |
| Provider Adapters | `providers/*.py` | ✅ 都有 chat()/responses() |
| MCP Session Manager | `runtime/mcp/mcp_session_manager.py` | ✅ 完整 |
| MCP Tool Bridge | `runtime/mcp/mcp_tool_bridge.py` | ✅ 完整 |
| RouterService | `runtime/orchestration/openai_handler.py` | ⚠️ 只有單次呼叫 |

---

## 設計決策

### 關鍵決定

1. **不改寫現有 `handle_responses()`** — 保留現有行為作為 `tool_choice: "none"` 的快速路徑
2. **新增 `execute_responses_tool_loop()`** — 獨立方法，處理多輪工具循環
3. **不改 `ProviderAdapter` 接口** — 現有 `chat()` + `responses()` 已足夠
4. **同步 + 串流兩套 loop** — `execute_responses_tool_loop()` (sync) + `execute_streaming_tool_loop()` (async generator)
5. **Tool 來源分離** — client 提供的 function tools vs AetherMesh builtin tools vs MCP tools

### 供應商差異處理

| 供應商 | 處理方式 |
|--------|---------|
| OpenAI | 原生 Responses API，tool loop 由 OpenAI 服务端處理，AetherMesh 只做 passthrough |
| Ollama (local) | AetherMesh 側執行 tool loop，model ↔ chat() ↔ 執行 tool ↔ 加回 messages ↔ chat() |
| Gemini | 同上 |
| Nvidia NIM | 同上 |
| Ollama Cloud | 同上 |

---

## 實作階段

### 階段 1：Responses Tool Loop 核心（必要）

新增 `runtime/responses/tool_loop.py`：

```python
class ResponsesToolLoop:
    """OpenAI Responses API 多輪工具調用循環"""

    def __init__(
        self,
        *,
        max_turns: int = 16,           # OpenAI 預設 16
        tool_timeout_s: int = 30,      # 工具執行超時
        parallel_tool_calls: bool = True,  # 是否並行執行 tool
    ):
        ...

    def run(
        self,
        *,
        provider: str,
        worker: dict | None,
        adapter: ProviderAdapter,
        payload: dict,                 # 已轉換的 chat payload
        tools: list[dict],             # OpenAI function tools
        responses_instructions: str,   # original instructions
        response_id: str,
        model: str,
        previous_response_id: str,
        metadata: dict | None,
    ) -> ResponseObject:
        """執行多輪工具循環，回傳最終 ResponseObject"""
        ...

    async def run_streaming(
        self,
        ...
    ) -> Iterable[str]:
        """串流版本，產生 SSE events"""
        ...
```

**核心流程**:
```
messages = responses_input_to_messages(input, instructions)
turn = 0

while turn < max_turns:
    turn += 1
    completion = adapter.chat({messages, tools, model, ...})
    
    tool_calls = completion.choices[0].message.tool_calls
    
    if not tool_calls:
        # Model 完成，只有 text
        response = chat_completion_to_response(completion)
        response.status = COMPLETED
        return response
    
    # 執行 tool calls
    tool_results = execute_tools(tool_calls)
    
    # 把 assistant message + tool calls 加入 messages
    messages.append(completion.choices[0].message)
    
    # 把 tool results 加入 messages
    for tr in tool_results:
        messages.append({
            role: "tool",
            tool_call_id: tr.id,
            content: tr.output
        })
```

需要修改的檔案：
- **[新增]** `runtime/responses/tool_loop.py` — 核心循環邏輯 (~200 lines)
- **[修改]** `runtime/responses/response_models.py` — 新增 `ResponseStatus.REQUIRES_ACTION` 和 `ResponseOutputStatus` enum
- **[修改]** `runtime/orchestration/openai_handler.py` — `handle_responses()` 和 `handle_streaming_responses()` 加入 tool loop 條件分支

### 階段 2：Tool 來源整合

目標：讓 tool loop 能執行三種類型的工具：

#### 2a. Client 自定義 Function Tools

Client 透過 `POST /v1/responses` 發送 `tools: [{type: "function", function: {...}}]`。
這些工具的執行邏輯已存在於 `runtime/tools/tool_registry.py` + `tool_executor.py`。

需要實作：
- **[修改]** `runtime/responses/tool_loop.py` — 將 OpenAI function tool 動態註冊到 ToolRegistry，執行後移除

#### 2b. AetherMesh Builtin Tools

現有 builtin tools：shell, filesystem, python, http_request, web_search。

- **[新增]** `runtime/responses/builtin_injection.py` — 自動將 builtin tools 注入 tool loop

#### 2c. MCP Tools

MCP 工具已經有 bridge (`runtime/mcp/mcp_tool_bridge.py`) 和 session 管理。

- **[修改]** `runtime/responses/tool_loop.py` — 偵測 MCP tool call，路由到 MCP bridge 執行

### 階段 3：Responses API 格式完整支援

| 功能 | 規格 | 實作 |
|------|------|------|
| `parallel_tool_calls` | 控制是否並行執行 tool call | 在 tool_loop 中加 flag |
| `tool_choice: "auto"\|"required"\|"none"` | 控制 model 行為 | 已處理（透傳到 provider） |
| `truncation: "auto"\|"disabled"` | 長對話截斷 | 加到 input converter |
| `max_output_tokens` | Token 上限 | 加到 payload 傳遞 |
| `metadata` | 自定義 JSON 欄位 | ✅ 現已有 |
| `previous_response_id` | 多輪對話上下文 | 部分支持，需加強 |
| `user` | 使用者識別 | 透傳到 provider |
| `temperature` / `top_p` | 生成參數 | 已處理 |
| `include[]` | 回傳 reasoning/images | **[新增]** output converter 支援 |
| `stream: true` | SSE 串流 | **[修改]** wrap_streaming_chunks 加入 tool loop events |

### 階段 4：非 OpenAI 供應商的 Responses API 串流

目前 `handle_streaming_responses()` 的串流只做了 chat→responses 事件轉換，**沒有 tool loop**。需要：

- **[修改]** `runtime/responses/tool_loop.py` — async streaming version，產生完整的 Responses API SSE events：
  - `response.created`
  - `response.in_progress`
  - `response.function_call.queue`
  - `response.function_call.call`
  - `response.function_call.output`
  - `response.output_item.added`
  - `response.content_part.added`
  - `response.text.delta`
  - `response.completed`

### 階段 5：測試

- **[新增]** `tests/test_responses_tool_loop.py` — 單元測試 tool loop 邏輯
- **[新增]** `tests/test_responses_integration.py` — 端對端測試 POST /v1/responses with tools
- **[新增]** `tests/test_responses_streaming.py` — 串流測試

---

## 檔案清單

### 新增檔案
| 檔案 | 功能 | 估算行数 |
|------|------|---------|
| `runtime/responses/tool_loop.py` | 核心多輪工具循環 | ~300 |
| `runtime/responses/response_status.py` | 完整 Responses API 狀態枚舉 | ~50 |
| `tests/test_responses_tool_loop.py` | Tool loop 單元測試 | ~150 |
| `tests/test_responses_streaming_tool_loop.py` | 串流 tool loop 測試 | ~150 |

### 修改檔案
| 檔案 | 修改內容 |
|------|---------|
| `runtime/responses/response_models.py` | 加 `REQUIRES_ACTION` status, `FunctionCallStatus` |
| `runtime/responses/input_converter.py` | 加 `truncation`, `max_output_tokens` 支援 |
| `runtime/responses/output_converter.py` | 加 `include[]` 支援 (reasoning, images) |
| `runtime/responses/response_stream.py` | 加 tool loop SSE events |
| `runtime/responses/__init__.py` | 更新出口 |
| `runtime/orchestration/openai_handler.py` | `handle_responses()` + `handle_streaming_responses()` 加入 tool loop 邏輯 |
| `runtime/tools/tool_registry.py` | 加臨時 tool 註冊/移除功能 |

### 不需修改的檔案
- `providers/*.py` — 現有接口已足夠
- `router/openai/responses_adapter.py` — 路由層不需改
- `runtime/mcp/*.py` — MCP bridge 不需改

---

## 執行順序

1. ✅ 階段 1: Tool Loop 核心
2. → 階段 2a: Client function tools
3. → 階段 3: Responses API 格式完整化
4. → 階段 4: 串流 tool loop
5. → 階段 5: 測試
6. → 階段 2b/c: Builtin + MCP tools (可延後)

## 風險評估

| 風險 | 影響 | 緩解 |
|------|------|------|
| Tool loop 無限循環 | 高 | max_turns=16, 可配置 |
| Tool 執行超時 | 中 | tool_timeout_s + async |
| Gemini 的 tool_call 格式不同 | 中 | tool_normalizer 已處理 |
| 大型模型的 token 限制 | 低 | truncation 支援 |
| 串流 tool loop 事件順序 | 高 | 嚴格遵循 OpenAI SSE 規格 |

---

## 預估工時

| 階段 | 時間 |
|------|------|
| 階段 1: 核心 loop | 1-2h |
| 階段 2a: Client tools | 30min |
| 階段 3: 格式完整化 | 1h |
| 階段 4: 串流 loop | 2h |
| 階段 5: 測試 | 1h |
| **Total** | **~5-6h** |
