@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"

REM Check for admin privileges (some cleanup requires elevation)
>nul 2>&1 net session || (
  echo WARNING: Not running as administrator — some cleanup steps may fail.
)

echo Stopping existing AIIH windows and processes...
taskkill /F /T /FI "WINDOWTITLE eq AIIH*" >nul 2>&1

REM Also stop known python module processes in case window-title match misses them.
powershell -NoProfile -Command ^
  "$mods=@('control_plane.cluster_manager:app','node.worker_agent:app','node.node_agent:app','ai_queue.task_worker','router.openai_router:app','router.anthropic_router:app','metrics.prometheus_exporter:app','dashboard.dashboard_server:app');" ^
  "$ps=Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python(.exe)?$' -and $_.CommandLine };" ^
  "foreach($p in $ps){ foreach($m in $mods){ if($p.CommandLine -like ('*'+$m+'*')){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{}; break } } }" >nul 2>&1

REM Free key ports used by AIIH in case orphan listeners still exist.
powershell -NoProfile -Command ^
  "$ports=@(8001,8002,9001,9100,9200,9300,9400);" ^
  "foreach($port in $ports){ try{ $procs=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach($pid in $procs){ if($pid){ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } } }catch{} }" >nul 2>&1

timeout /t 1 /nobreak >nul

echo ==========================================
echo AetherMesh startup sequence
echo 1. Ensure Ollama workers are already running
echo 2. Ensure Redis is already running
echo 3. Launching control plane
echo 4. Launching router API
echo 5. Launching metrics exporter
echo 6. Launching dashboard
echo ==========================================

call scripts\start_cluster.bat
if errorlevel 1 (
  echo ERROR: start_cluster.bat failed with exit code %ERRORLEVEL%.
  endlocal
  exit /b %ERRORLEVEL%
)

start "AIIH Router" cmd /k scripts\start_router.bat
start "AIIH Anthropic Router" cmd /k scripts\start_anthropic_router.bat
start "AIIH Metrics" cmd /k scripts\start_metrics.bat
start "AIIH Dashboard" cmd /k scripts\start_dashboard.bat

echo.
REM Waiting for service health (up to 90s)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0wait_services.ps1"

endlocal