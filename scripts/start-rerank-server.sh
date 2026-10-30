#!/usr/bin/env bash
# Start a standalone llama.cpp rerank server for AetherMesh.
#
# AetherMesh ships with the standard Ollama binary, which does NOT expose
# /api/rerank. Rerank models are therefore served by a dedicated
# `llama-server --rerank` process. This script launches it.
#
# Usage:
#   ./scripts/start-rerank-server.sh [model_name] [port] [gpu]
#
# Examples:
#   ./scripts/start-rerank-server.sh                        # bge-reranker-v2-m3, port 11436
#   ./scripts/start-rerank-server.sh bge-reranker-v2-m3 11436
#   ./scripts/start-rerank-server.sh qwen3-reranker-0.6b 11437
#
# The model name selects a GGUF blob from the Ollama model store.
# [gpu] is the physical CUDA ordinal (default 1); it is exported as
# CUDA_VISIBLE_DEVICES so the other GPUs stay hidden from the process.
# AIIH_RERANK_CUDA_DEVICES overrides it; AIIH_RERANK_DEVICE is deprecated.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the bundled llama-server shipped with Ollama. Override via
# AIIH_LLAMA_SERVER if you installed llama.cpp elsewhere.
LLAMA_SERVER="${AIIH_LLAMA_SERVER:-}"
if [[ -z "$LLAMA_SERVER" ]]; then
  case "$(uname -s)" in
    Darwin)
      LLAMA_SERVER="$HOME/Library/Application Support/Ollama/bin/llama-server"
      ;;
    Linux)
      LLAMA_SERVER="/usr/local/bin/llama-server"
      ;;
  esac
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: llama-server not found at '$LLAMA_SERVER'." >&2
  echo "Set AIIH_LLAMA_SERVER to the llama-server binary path." >&2
  exit 1
fi

# Map model name -> GGUF blob sha256 (inside the Ollama model store).
BLOB_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs"

case "${1:-bge-reranker-v2-m3}" in
  bge-reranker-v2-m3)
    BLOB="sha256-a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3"
    ;;
  qwen3-reranker-0.6b)
    BLOB="sha256-22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"
    ;;
  *)
    echo "ERROR: unknown model '${1}'. Known: bge-reranker-v2-m3, qwen3-reranker-0.6b" >&2
    exit 1
    ;;
esac

if [[ ! -f "$BLOB_DIR/$BLOB" ]]; then
  echo "ERROR: GGUF blob not found at '$BLOB_DIR/$BLOB'." >&2
  exit 1
fi

PORT="${2:-11436}"
HOST="${AIIH_RERANK_HOST:-127.0.0.1}"
CTX_SIZE="${AIIH_RERANK_CTX_SIZE:-8192}"

# GPU selection: physical CUDA ordinal to run on.
#   AIIH_RERANK_CUDA_DEVICES  (primary)
#   AIIH_RERANK_DEVICE        (deprecated alias)
#   third CLI argument        (resolved by the AetherMesh launcher)
#   default                   1
GPU="${AIIH_RERANK_CUDA_DEVICES:-${AIIH_RERANK_DEVICE:-${3:-1}}}"

# Only the selected GPU is visible to the process. llama-server's -mg must
# therefore be 0: with a single visible device that is the only valid index.
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Starting reranker on $HOST:$PORT (ctx=$CTX_SIZE, gpu=$GPU, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
exec "$LLAMA_SERVER" \
  --model "$BLOB_DIR/$BLOB" \
  --rerank \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  -ngl 99 \
  -mg 0