#!/usr/bin/env bash
set -euo pipefail

# AetherMesh macOS Node Starter
# Starts node_agent + worker_agent + asr_worker and registers with the control plane

AETHERMESH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$AETHERMESH_DIR/.env"

# launchd starts the daemon with a minimal PATH that has no Homebrew, and decoding
# audio files needs ffmpeg, which Homebrew installs on Apple Silicon.
if [ -d /opt/homebrew/bin ]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

# Read one KEY=value pair from .env (first match, ignores CR and surrounding quotes).
env_value() {
  if [ ! -f "$ENV_FILE" ]; then
    return
  fi
  awk -v key="$1" '
    substr($0, 1, length(key) + 1) == key "=" {
      value = substr($0, length(key) + 2)
      sub(/\r$/, "", value)
      gsub(/^["'"'"']|["'"'"']$/, "", value)
      print value
      exit
    }' "$ENV_FILE"
}

# Control plane target: explicit env, then .env, then the legacy AIIH_CONTROL_IP override.
CONTROL_URL="${AIIH_CONTROL_URL:-}"
CONTROL_URL_SOURCE="environment"
if [ -z "$CONTROL_URL" ]; then
  CONTROL_URL="$(env_value AIIH_CONTROL_URL)"
  CONTROL_URL_SOURCE=".env"
fi
if [ -z "$CONTROL_URL" ] && [ -n "${AIIH_CONTROL_IP:-}" ]; then
  CONTROL_PORT="${AIIH_CONTROL_PORT:-$(env_value AIIH_CONTROL_PORT)}"
  CONTROL_URL="http://${AIIH_CONTROL_IP}:${CONTROL_PORT:-9200}"
  CONTROL_URL_SOURCE="AIIH_CONTROL_IP"
fi

cd "$AETHERMESH_DIR"

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found at $AETHERMESH_DIR/.venv"
  exit 1
fi

if [ -z "$CONTROL_URL" ]; then
  echo "ERROR: no control plane target found. Set AIIH_CONTROL_URL in $ENV_FILE (or AIIH_CONTROL_IP)." >&2
  exit 1
fi

export AIIH_CONTROL_URL="$CONTROL_URL"

# Rerank needs a llama.cpp binary: Ollama itself has no /api/rerank endpoint.
# Only add the rerank service when a usable llama-server is actually present.
find_llama_server() {
  if [[ -n "${AIIH_LLAMA_SERVER:-}" && -x "${AIIH_LLAMA_SERVER}" ]]; then
    return 0
  fi
  for candidate in /opt/homebrew/bin/llama-server /usr/local/bin/llama-server \
    "/Applications/Ollama.app/Contents/Resources/llama-server"; do
    if [[ -x "$candidate" ]]; then
      export AIIH_LLAMA_SERVER="$candidate"
      return 0
    fi
  done
  return 1
}

SERVICES="node_agent worker_agent asr_worker"
if find_llama_server; then
  # Bind on all interfaces so other nodes' rerank pools can reach this machine.
  export AIIH_RERANK_HOST="${AIIH_RERANK_HOST:-0.0.0.0}"
  SERVICES="$SERVICES rerank"
else
  echo "NOTE: no llama-server found - rerank service not started (brew install llama.cpp)" >&2
fi

echo "Starting macOS node agents..."
echo "  control plane: $CONTROL_URL  (from $CONTROL_URL_SOURCE)"
echo "  services:      $SERVICES"
python -m runtime.launcher start $SERVICES
