@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"
if not defined AIIH_HOST set "AIIH_HOST=0.0.0.0"
if not defined AIIH_CONTROL_PORT set "AIIH_CONTROL_PORT=9200"
if not defined AIIH_WORKER_RPC_PORT set "AIIH_WORKER_RPC_PORT=9300"
if not defined AIIH_NODE_PORT set "AIIH_NODE_PORT=9400"

echo Starting AetherMesh cluster services...
start "AIIH Control Plane" cmd /k python -m uvicorn control_plane.cluster_manager:app --host %AIIH_HOST% --port %AIIH_CONTROL_PORT%
start "AIIH Worker Agent" cmd /k python -m uvicorn node.worker_agent:app --host %AIIH_HOST% --port %AIIH_WORKER_RPC_PORT%
start "AIIH Node Agent" cmd /k python -m uvicorn node.node_agent:app --host %AIIH_HOST% --port %AIIH_NODE_PORT%
start "AIIH Task Worker" cmd /k python -m ai_queue.task_worker

echo Control plane stack launched.
