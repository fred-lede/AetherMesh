# 白龍馬 Vision 整合指南

日期: 2025-07-06
狀態: AetherMesh 端已完成, 白龍馬待接入

## 背景

白龍馬在 vision 設計文件中問了以下問題：

1. **可以用哪個 endpoint？** → `/v1/chat/completions`，不需要新 endpoint
2. **需不需要新增 provider？** → 不用，Ollama/Gemini adapter 已支援 `image_url`
3. **VLM 跑在哪裡？** → GPU 1 (RTX 4070 Ti, port 11435)，隔離於 text models (GPU 0, 5090)
4. **model name 怎麼選？** → 隨便填 (如 `gpt-4o`)，routing engine 會自動 reroute 到 VLM
5. **本地 VLM 沒有時怎麼辦？** → `AIIH_VISION_FALLBACK=any` 會自動 fallback 到 Gemini/OpenAI

## 我們最終的做法 (AetherMesh 已實現)

### 核心 routing 邏輯

```
白龍馬 POST /v1/chat/completions with image_url
  → openai_handler._resolve_provider_and_worker()
    → capabilities.py: 偵測到 image_url → requires ["chat", "vision"]
    → find_registry_model(): 使用者填的 model 不支援 vision
    → local_ollama_fallback(): 找 AIIH_VISION_MODEL (qwen2.5-vl:7b) on GPU 1 (port 11435)
    → 找不到 → cloud fallback: Gemini (vision score 98) or OpenAI (95)
    → ollama_adapter: image_url base64 → Ollama images[] 格式
    → VLM 處理圖片 → 回傳文字
```

### 已修改的檔案

| 檔案 | 修改內容 |
|---|---|
| `config/settings.py` | 新增 `vision_model`, `vision_fallback` 欄位 |
| `.env.example` | 新增 `AIIH_VISION_MODEL`, `AIIH_VISION_FALLBACK` |
| `runtime/orchestration/provider_router.py` | `local_ollama_fallback()` 針對 vision request 回傳 GPU 1 worker |
| `runtime/orchestration/routing_engine.py` | `_local_model_fallback()` 加入 VLM 動態查找, 新增 `_resolve_fallback_worker()` |
| `runtime/orchestration/openai_handler.py` | vision request 在無本地模型時 fallback 到 Gemini/OpenAI |
| `config/cluster.yaml` | GPU 1 role 改為 `embeddings+vision` |
| `README.md` | 新增 VLM setup guide + env vars 表格 |
| `tests/test_vision.py` | 9 個測試 (routing fallback, capability detection, adapter) |

## 白龍馬需要做的事

### 1. 發送 image_url

在 audio chat 流程中，當使用者說「看這個」、「這是什麼」等觸發詞時，白龍馬應：

- 從 camera 或 screenshot 取得圖片
- 轉為 base64 data URI (`data:image/jpeg;base64,...`)
- 將 `image_url` block 放入 messages content array

```python
# 白龍馬內部的發送範例
import base64

def image_to_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    # 或用 image_path.endswith() 判斷格式
    return f"data:image/jpeg;base64,{data}"

# 原有的 audio chat 流程，加上 image block
payload = {
    "model": "gpt-4o",  # 任何 model name 都可以
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in Chinese"},
                {"type": "image_url",
                 "image_url": {"url": image_to_data_uri("photo.jpg")}}
            ]
        }
    ],
    "stream": True  # 支援 streaming
}
```

### 2. Response 處理

回傳格式與一般 `/v1/chat/completions` 完全相同：

```python
# 白龍馬解析 streaming response
for chunk in response.iter_lines():
    # standard SSE parsing (data: {"choices":[{"delta":{"content":"..."}}]})
    text = extract_content(chunk)
    if text:
        yield text  # 餵給 TTS 或顯示
```

### 3. 建議的觸發邏輯

- **被動**: 使用者說「看一下」、「這個是什麼」等語音指令 → 白龍馬截圖 + 發 vision request
- **主動**: (未來) 白龍馬可選定時截圖分析環境，但建議先只做被動觸發

### 4. 建議的 model 設定 (client 端可寫死)

白龍馬發 request 時建議用以下 model，不需要依賴 AetherMesh 環境變數：

```python
VISION_MODEL = "gpt-4o"  # 只是一個 routing token，不會真的送到 OpenAI
```

### 5. 同一輪支援 multi-image

```python
content = [
    {"type": "text", "text": "Compare these two images"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,img1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,img2"}},
]
```

## 注意事項

1. **Streaming 完整支援** — `stream: True` 可以正常運作，SSE 事件與純文字相同
2. **Cloud fallback 自動** — 如果本地 Ollama VLM worker 不在線，會自動走 Gemini/OpenAI (前提是 `AIIH_VISION_FALLBACK=any`)，白龍馬不需要做任何事
3. **封包大小** — base64 圖片可能很大 (一張 1080p JPEG ~500KB → base64 ~700KB)，確保 HTTP client timeout 設定合理 (>30s)
4. **錯誤處理** — 如果 VLM 服務不可用且 cloud fallback 也無 API key，AetherMesh 會回傳 HTTP 400 `unsupported_capabilities`。白龍馬應該 catch 這個錯誤並降級為純文字回覆

## 測試

```bash
# AetherMesh 端驗證 vision routing (9 tests)
cd C:\ai\AetherMesh
& .venv\Scripts\python.exe -m pytest tests/test_vision.py -x -v

# 實際 curl 測試 (需先啟動 AetherMesh + Ollama on port 11435)
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in Chinese"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
      ]
    }]
  }'
```
