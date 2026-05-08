#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p logs

echo "Starting AetherMesh Ubuntu node services..."
echo "Assumption: Ollama workers are already running on ports 11434/11435 (or your configured ports)."

nohup python -m uvicorn node.worker_agent:app --host 0.0.0.0 --port "${AIIH_WORKER_RPC_PORT:-9300}" > logs/worker_agent.log 2>&1 &
WORKER_PID=$!

nohup python -m uvicorn node.node_agent:app --host 0.0.0.0 --port "${AIIH_NODE_PORT:-9400}" > logs/node_agent.log 2>&1 &
NODE_PID=$!

echo "Worker Agent PID: ${WORKER_PID}"
echo "Node Agent PID: ${NODE_PID}"
echo "Logs:"
echo "  logs/worker_agent.log"
echo "  logs/node_agent.log"
