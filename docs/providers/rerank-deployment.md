# Reranker 部署

AetherMesh 使用 **llama.cpp 的 `llama-server --rerank`** 獨立進程提供 rerank 能力，而非標準 Ollama。

## 為什麼不用 Ollama 的 `/api/rerank`？

AetherMesh 隨附的標準 Ollama binary **沒有暴露 `/api/rerank`**。在支援的 Ollama 版本上直接呼叫 `/api/rerank` 會得到 `404 page not found`。因此 rerank 模型由一個獨立的 `llama-server --rerank` 進程提供，端點為 `/rerank`。

## 架構

```
Client ──> POST /v1/rerank (AetherMesh Router :8001)
              └─> handle_rerank
                   └─> RerankAdapter (providers/rerank_adapter.py)
                        └─> POST {base_url}/rerank   (llama-server --rerank)
```

- Router 端點：`POST /v1/rerank`，OpenAI 相容格式
- Reranker server：llama.cpp 原生 `/rerank`（`--rerank` 旗標）
- `AIIH_RERANK_BASE_URL`（預設 `http://127.0.0.1:11436`）指定 rerank server 位址

## 平台 GPU 支援

reranker 模型很小（~600MB），單一 GPU 即可。llama.cpp 的 GPU backend 因平台而異：

| 平台 | Backend | 說明 |
|------|---------|------|
| **Windows / Linux (NVIDIA)** | CUDA | 需下載 llama.cpp **CUDA 版** binary（Ollama 附帶的是 CPU-only）|
| **macOS (Apple Silicon)** | Metal | Ollama 附帶的 macOS `llama-server` 即 Metal 版，原生支援 GPU，**不需額外下載** |
| Linux (AMD) | ROCm / Vulkan | 需對應 build |

> **注意**：llama.cpp **沒有 MPS backend**（MPS 是 PyTorch 的 API）。macOS 用的是 **Metal**。

Ollama 隨附的 `llama-server` 在 NVIDIA 平台是 **CPU-only 建置**，`-mg N` 無效。要啟用 GPU，把 `AIIH_LLAMA_SERVER` 指向 llama.cpp **CUDA 版** binary。

#### Windows (NVIDIA)

1. 從 https://github.com/ggml-org/llama.cpp 的 nightly release 下載 `*-bin-win-cuda-12.4-x64.zip`
2. 解壓後設定路徑。**注意**：`set` 只對當前 cmd session 有效，`sc create`/工作排程器啟動 `.bat` 是獨立 session，`set` 不會帶過去，會落回 Ollama CPU-only 版。要持久化 GPU 路徑，改用 `setx` 寫入使用者環境變數（重開 cmd 生效）：
   ```bat
   setx AIIH_LLAMA_SERVER "C:\ai\tools\llama-cpp\b10964\llama-server.exe"
   ```
   （臨時測試才用 `set`；`.bat` 會依 `AIIH_LLAMA_SERVER` 是否已設定決定是否回退 CPU-only，見 `scripts/start-rerank-server.bat`。）
3. 用 `llama-server.exe --list-devices` 驗證偵測到 GPU。

#### Linux (NVIDIA)

1. 從 https://github.com/ggml-org/llama.cpp 的 release 下載對應的 Linux CUDA build，或安裝 llama.cpp 的 CUDA 版至 `/usr/local/bin/llama-server`：
   ```bash
   # 以 Ubuntu x64 CUDA 12.8 為例（版本號視 release 而定）
   wget https://github.com/ggml-org/llama.cpp/releases/download/b11057/llama-b11057-bin-ubuntu-cuda-12.8-x64.tar.gz
   sudo tar -xzf llama-b11057-bin-ubuntu-cuda-12.8-x64.tar.gz -C /usr/local
   ```
   > **版本無關**：llama.cpp 是獨立 binary，**不依賴 torch**。其 CUDA build 的版本只需 **NVIDIA 驅動**支持，與環境中的 torch `+cu128` 無關。cu12.4 build 僅需驅動 ≥ 525（新 GPU 建議 ≥ 550）；能跑 torch cu128 的機器驅動必然足夠，直接使用即可。
2. 設定 binary 路徑（`export` 只對當前 shell 有效，重開終端即失效）：
   ```bash
   export AIIH_LLAMA_SERVER=/usr/local/bin/llama-server
   ```
   **持久化**：啟動腳本在 `AIIH_LLAMA_SERVER` 未設定時回退到 `/usr/local/bin/llama-server`（Linux 預設路徑），所以只要 binary 放在該路徑即可；systemd service 已用 `Environment=AIIH_LLAMA_SERVER=...` 寫死此值（見 `systemd/aiih-rerank.service`），重啟服務仍有效，無需額外設定。
3. 用 `llama-server --list-devices` 驗證偵測到 GPU。

#### 選 GPU（Windows 與 Linux 通用）

用 `AIIH_RERANK_DEVICE`（CUDA ordinal，傳 `-mg N`）選 GPU：

- `AIIH_RERANK_DEVICE=0`：第一張離散 GPU
- `AIIH_RERANK_DEVICE=1`：第二張離散 GPU（預設）

在 bash 設定：`export AIIH_RERANK_DEVICE=1`；在 cmd 設定：`set AIIH_RERANK_DEVICE=1`。亦可寫入 `scripts/start-rerank-server.sh`（Linux/macOS）或 `.bat`（Windows）。

### macOS (Metal)

Ollama 附帶的 macOS `llama-server` 已是 Metal 版，直接用即可，無需 `AIIH_LLAMA_SERVER`。Apple Silicon 只有單一 GPU，腳本自動用 `-mg 0`。

```bash
# 直接使用 Ollama 附帶的 Metal 版
./scripts/start-rerank-server.sh
```

## 模型註冊

`config/models.yaml` 已內建一個 rerank 模型（指向獨立 server）：

```yaml
- name: rerank/bge-reranker-v2-m3
  provider: rerank
  worker_bindings:
    - node_id: node-01
      port: 11436
  capabilities: [rerank]
```

呼叫時 model 名用 `rerank/bge-reranker-v2-m3`。

> 既有 `bge-reranker-v2-m3:latest`、`pdurugyan/qwen3-reranker-0.6b-q8_0:latest` 是透過 Ollama `/api/rerank` 的路徑，在本環境不可用。

## 調用方式

端點：`POST /v1/rerank`（OpenAI 相容格式），需 `AIIH_API_KEY`。model 名用 `rerank/bge-reranker-v2-m3`。

```bash
curl -s -X POST http://127.0.0.1:8001/v1/rerank \
  -H "Authorization: Bearer $AIIH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank/bge-reranker-v2-m3",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "Apples grow on trees."],
    "top_n": 1
  }'
```

OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="<AIIH_API_KEY>")

resp = client.rerank.create(
    model="rerank/bge-reranker-v2-m3",
    query="What is the capital of France?",
    documents=["Paris is the capital of France.", "Apples grow on trees."],
    top_n=1,
)
print(resp.data[0].relevance_score)
```

> 若你的 SDK 沒有內建 `rerank`，可改用 `client.post("/rerank", json=payload)`。

## 前置需求

1. 取得有 GPU offload 的 `llama-server` binary：
   - **macOS**：用 Ollama 附帶的 Metal 版 `~/Library/Application Support/Ollama/bin/llama-server`（原生 GPU）
   - **Windows**：Ollama 附帶的 `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe` 是 **CPU-only**，需下載 llama.cpp CUDA 版
   - **Linux (NVIDIA)**：Ollama 附帶的是 CPU-only，需下載 llama.cpp CUDA 版（或安裝 llama.cpp 的 CUDA 版至 `/usr/local/bin/llama-server`）
   - 亦可 `AIIH_LLAMA_SERVER=<path>` 指定。
2. GGUF blob 位於 Ollama model store：
   - **Windows**：`%USERPROFILE%\.ollama\models\blobs`
   - **macOS/Linux**：`$HOME/.ollama/models/blobs`（可用 `OLLAMA_MODELS` 覆寫）
   - BGE：`sha256-a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3`
   - Qwen3：`sha256-22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48`

## 手動啟動（所有平台）

由 `scripts/` 提供啟動腳本，自動尋找模型 blob 與 llama-server：

```bash
# macOS / Linux
./scripts/start-rerank-server.sh                 # bge-reranker-v2-m3, port 11436
./scripts/start-rerank-server.sh bge-reranker-v2-m3 11436
./scripts/start-rerank-server.sh qwen3-reranker-0.6b 11437
```

```bat
:: Windows
start-rerank-server.bat
start-rerank-server.bat bge-reranker-v2-m3 11436
start-rerank-server.bat qwen3-reranker-0.6b 11437
```

也可直接用 `llama-server` 啟動：

```bash
llama-server --model <path-to-bge-blob> --rerank --host 127.0.0.1 --port 11436 --ctx-size 8192
```

## 系統服務（自動啟動）

### Linux (systemd)

```bash
sudo cp systemd/aiih-rerank.service /etc/systemd/system/
# 編輯 User=/Group= 與 ExecStart 中的專案路徑
sudo systemctl daemon-reload
sudo systemctl enable --now aiih-rerank
```

### macOS (launchd)

```bash
cp launchd/com.aiih.rerank.plist.example ~/Library/LaunchAgents/com.aiih.rerank.plist
# 將檔案內的 __ROOT_DIR__ 取代為 AetherMesh 實際路徑
launchctl load ~/Library/LaunchAgents/com.aiih.rerank.plist
```

### Windows

用工作排程器或 `sc create` 指向 `start-rerank-server.bat`。簡單作法：

```bat
:: 先持久化 GPU binary 路徑（見上方「Windows (NVIDIA)」），否則會用 CPU-only 版
setx AIIH_LLAMA_SERVER "C:\ai\tools\llama-cpp\b10964\llama-server.exe"
sc create aiih-rerank binPath= "cmd /c C:\ai\AetherMesh\scripts\start-rerank-server.bat" start= auto
```

## 驗證

```bash
# 1. rerank server 存活
curl -s http://127.0.0.1:11436/health

# 2. 透過 AetherMesh 端點（需 AIIH_API_KEY）
curl -s -X POST http://127.0.0.1:8001/v1/rerank \
  -H "Authorization: Bearer $AIIH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank/bge-reranker-v2-m3",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "Apples grow on trees."],
    "top_n": 1
  }'
```

預期回應（OpenAI 相容格式）：

```json
{
  "object": "list",
  "data": [
    {"index": 0, "relevance_score": 4.668, "document": "Paris is the capital of France."}
  ],
  "model": "rerank/bge-reranker-v2-m3",
  "usage": {}
}
```

## 已知限制

- 一次只能載入一個模型（單一 `llama-server` 進程綁定單一 GGUF）。
- 多模型需啟動多個 server（不同 port），並在 `models.yaml` 各別註冊。
- 原生 OpenAI API 沒有 rerank endpoint，故 `providers/openai_adapter.py` 的 rerank 回報「未實作」，屬預期行為。