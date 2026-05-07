#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

stop_pattern() {
  local pattern="$1"
  if pgrep -f "$pattern" > /dev/null; then
    pkill -f "$pattern"
    echo "Stopped: $pattern"
  else
    echo "Not running: $pattern"
  fi
}

stop_pattern "uvicorn node.worker_agent:app"
stop_pattern "uvicorn node.node_agent:app"
