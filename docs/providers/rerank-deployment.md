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
| **macOS (Apple Silicon)** | Metal | `brew install llama.cpp` 的 `llama-server` 即 Metal 版；官方 Ollama app 也附帶 `llama-server`（Homebrew 的 `ollama` formula 不附帶） |
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

1. 從 https://github.com/ggml-org/llama.cpp 的 release 下載對應的 Linux CUDA build，解壓後把 binary 放進 `/usr/local/bin`：
   ```bash
   # 以 Ubuntu x64 CUDA 12.8 為例（版本號視 release 而定）
   wget https://github.com/ggml-org/llama.cpp/releases/download/b11057/llama-b11057-bin-ubuntu-cuda-12.8-x64.tar.gz
   sudo tar -xzf llama-b11057-bin-ubuntu-cuda-12.8-x64.tar.gz -C /usr/local
   # 解壓會產生目錄 /usr/local/llama-b11057/，llama-server 不在 /usr/local/bin。
   # 建立 symlink 讓系統預設路徑 /usr/local/bin/llama-server 可用：
   sudo ln -sf /usr/local/llama-b11057/llama-server /usr/local/bin/llama-server
   ```
   > **版本無關**：llama.cpp 是獨立 binary，**不依賴 torch**。其 CUDA build 的版本只需 **NVIDIA 驅動**支持，與環境中的 torch `+cu128` 無關。此 cu12.8 build 僅需驅動 ≥ 560；能跑 torch cu128 的機器驅動必然足夠，直接使用即可。
2. 設定 binary 路徑（`export` 只對當前 shell 有效，重開終端即失效）：
   ```bash
   export AIIH_LLAMA_SERVER=/usr/local/bin/llama-server
   ```
   **持久化**：啟動腳本在 `AIIH_LLAMA_SERVER` 未設定時回退到 `/usr/local/bin/llama-server`（Linux 預設路徑），所以只要 binary 放在該路徑即可；systemd service 已用 `Environment=AIIH_LLAMA_SERVER=...` 寫死此值（見 `systemd/aiih-rerank.service`），重啟服務仍有效，無需額外設定。
3. 用 `llama-server --list-devices` 驗證偵測到 GPU。

#### 選 GPU（Windows 與 Linux 通用）

reranker 跑在**實體 CUDA ordinal** 上：

- `0`：第一張離散 GPU（例如 RTX 5090）
- `1`：第二張離散 GPU（例如 RTX 4070 Ti SUPER，**預設**）

**解析優先序**（由上到下，先命中者勝出）：

1. `AIIH_RERANK_CUDA_DEVICES` 環境變數（主設定）
2. `AIIH_RERANK_DEVICE`（**已棄用**的別名，為相容舊設定保留）
3. `config/cluster.yaml` 中 `local_workers[]` 對應 port 的 `gpu_id`
4. 預設 `1`

在 bash 設定：`export AIIH_RERANK_CUDA_DEVICES=1`；在 cmd 設定：`set AIIH_RERANK_CUDA_DEVICES=1`。
若要持久化到 Windows，請用 `setx AIIH_RERANK_CUDA_DEVICES 1`（重開 cmd 生效）。

> **注意 `.env` 的影響**：launcher 會用 `load_dotenv(..., override=True)` 載入 `.env`，
> 因此**若 `.env` 內含 `AIIH_RERANK_CUDA_DEVICES`，它會覆蓋 shell 中的同名變數**。
> 想讓 shell / 系統層級的環境變數優先，請勿把該變數寫進 `.env`（只需寫在
> `config/cluster.yaml`，由第 3 級規則接手）。

`config/cluster.yaml` 的對應設定：

```yaml
local_workers:
  - port: 11436
    gpu_id: 1        # 實體 CUDA ordinal；1 = RTX 4070 Ti SUPER
    role: rerank
```

> **重要**：腳本把解析結果寫入 `CUDA_VISIBLE_DEVICES`，因此**只有被選中的那張卡對進程可見**，其他 GPU 完全不會被初始化。
> 僅靠 llama-server 的 `-mg N` **無法**隔離 GPU —— llama.cpp 仍會在所有可見裝置上初始化 CUDA，在多卡機器上可能導致載入失敗或佔用其他卡的 VRAM。
> 由於設定 `CUDA_VISIBLE_DEVICES` 後進程只看得到一張卡（內部編號為 `0`），腳本固定使用 `-mg 0`。

手動啟動時也可直接用第三個位置參數指定：

```bash
./scripts/start-rerank-server.sh bge-reranker-v2-m3 11436 0   # 指定第一張卡
cmd /c scripts\start-rerank-server.bat bge-reranker-v2-m3 11436 0
```

環境變數仍優先於位置參數。

### macOS (Metal)

Apple Silicon 只有單一 GPU，`CUDA_VISIBLE_DEVICES` 對 Metal 無效，腳本仍使用 `-mg 0`。

啟動腳本依序搜尋 `llama-server`：`AIIH_LLAMA_SERVER` → PATH 中的 `llama-server` → `/opt/homebrew/bin` → `/usr/local/bin` → `/Applications/Ollama.app/Contents/Resources/llama-server` → `~/Library/Application Support/Ollama/bin/llama-server`。找不到就報 `ERROR: llama-server binary not found.`

Homebrew 的 `ollama` formula **不附帶** `llama-server`（只有 `ollama` 主程式），所以 Mac 上最可靠的做法是：

```bash
brew install llama.cpp          # Metal 版 llama-server
./scripts/start-rerank-server.sh
```

官方 Ollama macOS app 則附帶 `llama-server`，腳本會自動找到。

開機自動啟動用 launchd（**plist 不讀 `.env`**，環境變數只能寫在 plist 的 `EnvironmentVariables`）：

```bash
REPO=/path/to/AetherMesh
sed "s|__ROOT_DIR__|$REPO|g" launchd/com.aiih.rerank.plist.example \
  > ~/Library/LaunchAgents/com.aiih.rerank.plist
plutil -lint ~/Library/LaunchAgents/com.aiih.rerank.plist
launchctl load ~/Library/LaunchAgents/com.aiih.rerank.plist
```

若 Ollama 的模型目錄不是預設位置，必須在 plist 中設定 `OLLAMA_MODELS`（例如 `/Users/Shared/ollama_models`），否則 launchd 環境裡找不到 GGUF blob。也可以改用 `scripts/start-mac-node.sh`：它偵測到 `llama-server` 時會把 `rerank` 加入 launcher 服務清單（此路徑會讀 `.env`）。

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

## 多機 rerank 池（自動 failover）

`RerankAdapter` 不只連一台，它維護一個**端點池**：第一個健康端點優先使用，連不上時自動換下一台，失敗的端點會進入 cooldown（30 秒起，每次失敗加倍，上限 120 秒），之後自動重试。全部端點都失效才報錯（`provider_unreachable`）。

池的来源（优先级由左到右）：

1. `AIIH_RERANK_POOL` 環境變數（逗號分隔，順序即優先序）
   ```bash
   AIIH_RERANK_POOL=http://127.0.0.1:11436,http://192.168.1.100:11436,http://192.168.1.200:11436
   ```
2. **自動發現**：`node_agent` 每次 heartbeat 會 probe 本機 `cluster.yaml` 的 `local_workers` 中 `role: rerank` 的端口（`GET http://127.0.0.1:<port>/health`），健康者以 `metadata.rerank_workers` 回報，控制平面把它注册成 provider `llama-server`、`metadata.kind = "rerank"` 的 worker。adapter 讀 `/cluster/workers`（30 秒 TTL 快取）拿到端点列表：本機 node 映射為 loopback，远端 node 用註冊的 `base_url`，排序为「本機優先、VRAM 大的优先」。
3. 若 registry 沒有 rerank worker，讀 `config/models.yaml` 中 `rerank/bge-reranker-v2-m3` 的 `worker_bindings`（本機 node 自動映射為 loopback，远端 node 用 `cluster.yaml` 的 `node_hosts` IP）。

這是**冗余（redundancy）**不是 load balancing：请求永远先打第一台，只有它挂了才 failover，所以多台机器各自加载模型不会互相抢负载。

## 在 Dashboard 检视 rerank 池

Dashboard → **Cluster** 页（需 admin 登入）：

- **Cluster Nodes** 表格新增 `Rerank` 与 `ASR` 两栏：该节点是否有可用的 rerank 端点（含端口）与 ASR worker（engine + model）。
- **Rerank Pool** 表格列出池内每个端点的优先序、所属节点与 GPU、健康状态（`online` / `cooldown` / `down`）、连续失败次数、剩余冷却时间、最后一次被使用的时间。
- 数据来自 `providers/rerank_adapter.RerankAdapter.pool_status()`（`GET <endpoint>/health` + 内部 `_POOL_STATUS` 冷却状态），端点顺序与 adapter 实际使用的顺序一致，因此面板可以直接判断 failover 会走向哪一台。

要让其他机器成为 rerank 端点：

```bash
# 该机器上的 rerank server 必须对外绑定（预设只绑 127.0.0.1）
AIIH_RERANK_HOST=0.0.0.0 AIIH_RERANK_DEVICE=<gpu index> \
  bash scripts/start-rerank-server.sh bge-reranker-v2-m3 11436 <gpu>
```

- macOS：`launchd/com.aiih.rerank.plist.example` 内加入 `AIIH_RERANK_HOST=0.0.0.0`
- Windows：`scripts/start-worker.bat` 加 `rerank`，并在防火墙放行 11436
  ```bat
  netsh advfirewall firewall add rule name="AetherMesh rerank 11436" dir=in action=allow protocol=TCP localport=11436
  ```
  開機排定任务（`scripts/register-worker-task.bat`）用 `/RU SYSTEM` 执行，此时 `%USERPROFILE%` 是 `C:\WINDOWS\system32\config\systemprofile`，沒有 Ollama 模型库。`scripts/start-rerank-server.bat` 会自动扫描 `C:\Users\*\ .ollama\models`；若仍找不到 GGUF，在 `.env` 明确指定：
  ```bat
  OLLAMA_MODELS=C:\Users\<user>\.ollama\models
  ```
- 每台 rerank 端点需要 ~1.2 GB VRAM 空闲；如果某台 GPU 已被其他模型占满，就把它从池里移除（或不要绑它）。

> `workers` 字段仍是 `list[int]`（Ollama 端口），rerank worker 走 `metadata.rerank_workers`，所以舊的 node agent 不受影響。控制平面 dashboard 可以看到 rerank worker 的 GPU/狀態；`/cluster/workers` 中它的 provider 是 `llama-server`。

## 前置需求

1. 取得有 GPU offload 的 `llama-server` binary：
   - **macOS**：`brew install llama.cpp`（Metal 版）；官方 Ollama app 的 `llama-server` 也可，Homebrew `ollama` formula 不附帶 `llama-server`
   - **Windows**：Ollama 附帶的 `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe` 是 **CPU-only**，需下載 llama.cpp CUDA 版
   - **Linux (NVIDIA)**：Ollama 附帶的是 CPU-only，需下載 llama.cpp CUDA 版（或安裝 llama.cpp 的 CUDA 版至 `/usr/local/bin/llama-server`）
   - 亦可 `AIIH_LLAMA_SERVER=<path>` 指定。
2. GGUF blob 位於 Ollama model store：
   - **Windows**：`%USERPROFILE%\.ollama\models\blobs`（或 `%LOCALAPPDATA%\Ollama\models`）
   - **macOS/Linux**：`$HOME/.ollama/models/blobs`（可用 `OLLAMA_MODELS` 覆寫）
   - BGE：`sha256-a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3`
   - Qwen3：`sha256-22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48`
   - 上面是**pinned fallback**：`ollama pull` 會改變 blob sha，重新 pull 後 pinned 值就會過期。啟動腳本會先從 Ollama manifest（`manifests/*/…/<model>/latest`）解析目前真正的 GGUF digest（`.sh` 用 python3、`.bat` 用 `scripts/resolve-rerank-blob.ps1`），解析不到才退回 pinned 值；因此重新 pull 後不需要手動改腳本。
   - **`rerank/bge-reranker-v2-m3` 不是 registry 模型**（`ollama pull rerank/...` 會報 `pull model manifest: file does not exist`）。它是在本機用 `ollama create` 從 GGUF 建立的名字。新機器上取得模型的方式：
     ```bash
     ollama pull qllama/bge-reranker-v2-m3:f16      # 1.2 GB, bert F16
     # 或直接把 GGUF 路徑當作第一個參數 / AIIH_RERANK_MODEL_PATH
     ./scripts/start-rerank-server.sh /path/to/bge-reranker-v2-m3.gguf 11436 0
     ```

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

rerank 已納入 launcher 的 `SERVICE_DEFS`，**若你的 AetherMesh 用 launcher 啟動（`start_all.bat` / `start_supervisor.bat`），rerank 會自動一併啟動並由 watchdog 監督，無需下面各平台的獨立服務**。獨立服務僅供不使用 launcher 的部署。

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

rerank 已納入 launcher 的 `SERVICE_DEFS`（`runtime/launcher/launcher.py`），launcher 會自動啟動並由 watchdog 監督（掛了自動重啟）。因此**只需排程啟動 `scripts/start_supervisor.bat`**，rerank 會隨 launcher 一併拉起，不需額外 `sc create`。

前置：先持久化 GPU binary 路徑（見上方「Windows (NVIDIA)」），否則會用 CPU-only 版：

```bat
setx AIIH_LLAMA_SERVER "C:\ai\tools\llama-cpp\b10964\llama-server.exe"
```

launcher 內建的 rerank 命令等同執行：

```bat
cmd /c scripts\start-rerank-server.bat bge-reranker-v2-m3 11436
```

若想用其他模型/port，改 `_rerank_command()`（launcher.py）中的模型名與 port。

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