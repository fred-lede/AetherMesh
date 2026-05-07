#!/usr/bin/env bash
set -euo pipefail

PROJECT="${AIIH_PROJECT_DIR:-}"
VOLUME="${AIIH_VOLUME_DIR:-}"
LOG="${AIIH_LAUNCHER_LOG:-}"

if [ -z "$PROJECT" ]; then
  if [ -n "$VOLUME" ]; then
    PROJECT="$VOLUME/ai/ai_inference_hub"
  else
    PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

if [ -z "$LOG" ]; then
  LOG="$PROJECT/logs/aiih_launcher.log"
fi

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "==== $(date) AI Launcher Start ===="

# Wait for mounted volume/project path when the project lives on external storage.
for _ in {1..30}; do
  [ -d "$PROJECT" ] && break
  echo "Waiting for project path: $PROJECT"
  sleep 2
done

if [ ! -d "$PROJECT" ]; then
  echo "[ERROR] Project path not available: $PROJECT"
  exit 1
fi

cd "$PROJECT"

echo "Starting AI cluster..."
/bin/bash scripts/start_cluster_macos.sh

echo "Launcher finished"
