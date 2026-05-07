#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

stop_by_pid_file() {
  local name="$1"
  local pid_file="logs/${name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "${name} not running (no pid file)"
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped ${name} (PID ${pid})"
  else
    echo "${name} not running (stale pid file)"
  fi

  rm -f "$pid_file"
}

stop_by_pid_file "aiih-worker-agent"
stop_by_pid_file "aiih-node-agent"

pkill -f "uvicorn node.worker_agent:app" 2>/dev/null || true
pkill -f "uvicorn node.node_agent:app" 2>/dev/null || true
