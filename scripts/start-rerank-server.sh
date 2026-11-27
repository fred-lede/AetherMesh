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
  candidates=()
  if command -v llama-server >/dev/null 2>&1; then
    candidates+=("$(command -v llama-server)")
  fi
  candidates+=("/opt/homebrew/bin/llama-server" "/usr/local/bin/llama-server")
  if [[ "$(uname -s)" == "Darwin" ]]; then
    candidates+=(
      "/Applications/Ollama.app/Contents/Resources/llama-server"
      "$HOME/Library/Application Support/Ollama/bin/llama-server"
    )
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      LLAMA_SERVER="$candidate"
      break
    fi
  done
fi

if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: llama-server binary not found." >&2
  echo "Install llama.cpp (macOS: brew install llama.cpp) or set AIIH_LLAMA_SERVER to the binary path." >&2
  exit 1
fi

# Map model name -> pinned GGUF blob sha256 (last-resort fallback; the
# preferred path resolves the digest from the Ollama manifest below, because
# re-pulling a model changes its sha256).
BLOB_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs"

resolve_blob() {
  local model="$1" fallback="$2"
  local store manifest digest blob
  local -a stores=()
  if [[ -n "${OLLAMA_MODELS:-}" ]]; then
    stores+=("$OLLAMA_MODELS")
  fi
  stores+=("$HOME/.ollama/models" "/usr/share/ollama/.ollama/models")
  for store in "${stores[@]}"; do
    [[ -d "$store/manifests" || -d "$store/manifests-v2" ]] || continue
    while IFS= read -r manifest; do
      digest="$(python3 - "$manifest" 2>/dev/null <<'PY'
import json, sys
try:
    manifest = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
for layer in manifest.get("layers", []):
    if str(layer.get("mediaType", "")).endswith("image.model"):
        print(layer.get("digest", ""))
        break
PY
)" || digest=""
      [[ "$digest" == sha256:* ]] || continue
      blob="sha256-${digest#sha256:}"
      if [[ -f "$store/blobs/$blob" ]]; then
        printf '%s\n' "$store/blobs/$blob"
        return 0
      fi
    done < <(find "$store/manifests" "$store/manifests-v2" -type f -path "*/$model/*" 2>/dev/null | sort)
  done
  printf '%s\n' "$BLOB_DIR/$fallback"
}

MODEL="${1:-bge-reranker-v2-m3}"

# A direct GGUF path is accepted too: either the first argument is a file, or
# AIIH_RERANK_MODEL_PATH points at one. Useful when the model is not in the
# Ollama store (e.g. a GGUF downloaded from Hugging Face).
if [[ -f "$MODEL" ]]; then
  BLOB_PATH="$MODEL"
elif [[ -n "${AIIH_RERANK_MODEL_PATH:-}" && -f "${AIIH_RERANK_MODEL_PATH}" ]]; then
  BLOB_PATH="${AIIH_RERANK_MODEL_PATH}"
else
  case "$MODEL" in
    bge-reranker-v2-m3)
      FALLBACK_BLOB="sha256-a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3"
      ;;
    qwen3-reranker-0.6b)
      FALLBACK_BLOB="sha256-22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"
      ;;
    *)
      echo "ERROR: unknown model '${MODEL}'. Known: bge-reranker-v2-m3, qwen3-reranker-0.6b" >&2
      exit 1
      ;;
  esac
  BLOB_PATH="$(resolve_blob "$MODEL" "$FALLBACK_BLOB")"
fi

if [[ ! -f "$BLOB_PATH" ]]; then
  echo "ERROR: GGUF blob not found at '$BLOB_PATH'." >&2
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
  --model "$BLOB_PATH" \
  --rerank \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  -ngl 99 \
  -mg 0