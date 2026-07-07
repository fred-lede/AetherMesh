#!/usr/bin/env bash
set -euo pipefail

AETHERMESH_DIR="/Volumes/Ai-2TB/ai/AetherMesh"

# Wait for external drive to mount (up to 60 seconds)
for i in $(seq 1 60); do
  if [ -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
    break
  fi
  sleep 1
done

# If still not mounted after timeout, exit (launchd will retry via KeepAlive)
if [ ! -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
  exit 1
fi

cd "$AETHERMESH_DIR"

# Activate virtual environment
source .venv/bin/activate

# Fix control_plane_url
CONTROL_PLANE_IP="${AIIH_CONTROL_IP:-192.168.1.200}"
sed -i '' "s|control_plane_url: http://127.0.0.1:9200|control_plane_url: http://${CONTROL_PLANE_IP}:9200|" config/cluster.yaml 2>/dev/null || true

# Start agents
exec .venv/bin/python -m runtime.launcher start node_agent worker_agent
