@echo off
REM Start a standalone llama.cpp rerank server for AetherMesh on Windows.
REM AetherMesh ships with the standard Ollama binary which does NOT expose
REM /api/rerank. Rerank is served by a dedicated llama-server --rerank process.
REM
REM To use a GPU you must point AIIH_LLAMA_SERVER at a CUDA build of
REM llama.cpp (Ollama's bundled llama-server.exe is CPU-only). For example:
REM   set AIIH_LLAMA_SERVER=C:\ai\tools\llama-cpp\b10964\llama-server.exe
REM
REM Usage: start-rerank-server.bat [model] [port] [gpu]
REM   start-rerank-server.bat                bge-reranker-v2-m3, port 11436
REM   start-rerank-server.bat bge-reranker-v2-m3 11436
REM   start-rerank-server.bat qwen3-reranker-0.6b 11437
REM
REM [gpu] is the physical CUDA ordinal to run on (default 1). It is exported
REM as CUDA_VISIBLE_DEVICES so every other GPU stays hidden from the process.
REM AIIH_RERANK_CUDA_DEVICES overrides it; AIIH_RERANK_DEVICE is a deprecated
REM alias. Because only one device is visible, llama-server's -mg must be 0.
setlocal

REM Priority: explicit AIIH_LLAMA_SERVER, else Ollama's bundled binary.
if defined AIIH_LLAMA_SERVER (
  set "LLAMA_SERVER=%AIIH_LLAMA_SERVER%"
) else (
  set "LLAMA_SERVER=%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe"
)
if not exist "%LLAMA_SERVER%" (
  echo ERROR: llama-server not found at "%LLAMA_SERVER%". Set AIIH_LLAMA_SERVER. 1>&2
  exit /b 1
)

set "BLOB_DIR=%USERPROFILE%\.ollama\models\blobs"
if defined OLLAMA_MODELS set "BLOB_DIR=%OLLAMA_MODELS%\blobs"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=bge-reranker-v2-m3"

if /i "%MODEL%"=="bge-reranker-v2-m3" (
  set "BLOB=sha256-a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3"
) else if /i "%MODEL%"=="qwen3-reranker-0.6b" (
  set "BLOB=sha256-22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"
) else (
  echo ERROR: unknown model "%MODEL%". Known: bge-reranker-v2-m3, qwen3-reranker-0.6b 1>&2
  exit /b 1
)

if not exist "%BLOB_DIR%\%BLOB%" (
  echo ERROR: GGUF blob not found at "%BLOB_DIR%\%BLOB%". 1>&2
  exit /b 1
)

set "PORT=%~2"
if "%PORT%"=="" set "PORT=11436"
if not defined AIIH_RERANK_HOST set "AIIH_RERANK_HOST=127.0.0.1"
if not defined AIIH_RERANK_CTX_SIZE set "AIIH_RERANK_CTX_SIZE=8192"

REM Resolve the GPU ordinal: env override, deprecated alias, then argument.
if not defined AIIH_RERANK_CUDA_DEVICES if defined AIIH_RERANK_DEVICE set "AIIH_RERANK_CUDA_DEVICES=%AIIH_RERANK_DEVICE%"
if not defined AIIH_RERANK_CUDA_DEVICES set "AIIH_RERANK_CUDA_DEVICES=%~3"
if not defined AIIH_RERANK_CUDA_DEVICES set "AIIH_RERANK_CUDA_DEVICES=1"
set "GPU=%AIIH_RERANK_CUDA_DEVICES%"
set "CUDA_VISIBLE_DEVICES=%GPU%"

echo Starting reranker on %AIIH_RERANK_HOST%:%PORT% ^(ctx=%AIIH_RERANK_CTX_SIZE%, gpu=%GPU%^, CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%^)...
"%LLAMA_SERVER%" --model "%BLOB_DIR%\%BLOB%" --rerank --host %AIIH_RERANK_HOST% --port %PORT% --ctx-size %AIIH_RERANK_CTX_SIZE% -mg 0 -ngl 99