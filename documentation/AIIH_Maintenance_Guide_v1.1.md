# ✨ AI Inference Hub 維護與操作指南 v1.1 ✨

**審核人/升級者:** Hermes AI Agent
**審核日期:** [系統日期]

## 🎯 系統總覽 (Overview)

AI Inference Hub (AIIH) 是一個用於本地部署的、高度可擴展的 AI 資料中心控制平面。它旨在提供一個統一、穩定的介面 (OpenAI-compatible API 格式) 來管理和調用多個異構的 LLM 後端 (Provider)，包括 Ollama, OpenAI, Gemini 等。

**核心角色:**
1. **API 閘道 (Gateway):** 統一接收來自外部的 API 請求。
2. **路由與分派 (Routing & Dispatch):** 根據模型名稱和請求類型，判斷將任務分派給哪個底層 Worker/Adapter。
3. **狀態管理 (State Management):** 實時管理數百個 GPU/Worker 的健康狀態和任務佇列。

---

## 🚀 環境建置 (Setup) (📌 必須遵循的 SOP)

### 🏆 第一優先級：Docker Compose 部署 (最佳選擇)

這是目前最穩定、可複製性最強的部署方式。它將所有依賴和服務運行環境完全隔離。

**範例指令:**
\`\`\`bash
docker compose up -d --force-recreate
\`\`\`

**停止指令:**
\`\`\`bash
docker compose down
\`\`\`

### 💔 第二優先級：原生依賴啟動 (不推薦)
*   **警告:** 這種方式依賴全局環境狀態，極易因端口衝突或進程管理不當而崩潰。請盡量使用 Docker Compose。

---

## 🧩 核心組件深度解析 (Deep Dive)

### A. Adapter Layer (`providers/`) - [穩定性重點]

**職責:** 這是與所有外部模型 API 互動的單一抽象層。必須將底層 API 轉換為統一的語義層。

**✨ 穩定性核心：指數退避重試機制 (Exponential Backoff Retry)**
**📌 關鍵改動點:** 這是本次升級的最大貢獻。所有外部 I/O 呼叫都必須採用此機制，以確保在網路和 API 層級的瞬時故障下，系統能自我恢復，這是系統穩定的最高保障。

### 🔶 路由層 (`router/`)
**關注點:** 必須實現 **[故障轉移 (Failure Fallbacks)]**。如果首選的 Adapter 失敗，應自動嘗試報告錯誤並調用到備用 Adapter，而不是直接讓請求進入錯誤。

---

## 🆘 故障診斷與維護 (Troubleshooting & Debugging)

| 錯誤類型 | 服務組件 | 原因分析 | 應對策略 |
| :--- | :--- | :--- | :--- |
| **API 呼叫失敗 (非預期)** | Adapter 層 | 網路瞬斷、服務過載 (50x)、速率限制 (429)。 | **檢查 `robust-api-retry-wrapper` 技能是否生效。** 確認系統有正確捕捉並重試這些狀態碼。 |
| **端口衝突** | 啟動腳本 | 服務端口已被其他進程佔用。 | 優先使用 **Docker Compose**。如果無法使用，需檢查所有服務的端口綁定是否在 `.env` 文件中統一管理。 |
| **心跳失聯** | `Worker/Node Agent` | Worker/Node Agent 進程異常退出或網絡隔離。 | 檢查 `Heartbeat` 的過期邏輯，確認判定狀態的時機是否過於激進。 |

---
**結語:** 本指南應作為 AIIH 的官方 SOP，所有新開發功能都必須向其穩定性標準進行對標檢查。