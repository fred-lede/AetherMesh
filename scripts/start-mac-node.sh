#!/usr/bin/env bash
set -euo pipefail

# AetherMesh macOS Node Starter
# Starts node_agent + worker_agent and registers with Windows control plane

AETHERMESH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER_YAML="$AETHERMESH_DIR/config/cluster.yaml"
CONTROL_PLANE_IP="${AIIH_CONTROL_IP:-192.168.1.200}"

cd "$AETHERMESH_DIR"

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found at $AETHERMESH_DIR/.venv"
  exit 1
fi

# Ensure control_plane_url points to Windows host
if grep -q 'control_plane_url: http://127.0.0.1:9200' "$CLUSTER_YAML"; then
  sed -i '' "s|control_plane_url: http://127.0.0.1:9200|control_plane_url: http://${CONTROL_PLANE_IP}:9200|" "$CLUSTER_YAML"
  echo "Updated control_plane_url -> http://${CONTROL_PLANE_IP}:9200"
fi

echo "Starting macOS node agents..."
python -m runtime.launcher start node_agent worker_agent
