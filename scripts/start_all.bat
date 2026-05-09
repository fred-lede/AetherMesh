@echo off
setlocal
cd /d "%~dp0.."

echo ==========================================
echo AetherMesh Startup
echo ==========================================
echo.
echo Stopping existing AIIH processes...
taskkill /F /T /FI "WINDOWTITLE eq AIIH*" >nul 2>&1
powershell -NoProfile -Command ^
  "$mods=@('control_plane.cluster_manager:app','node.worker_agent:app','node.node_agent:app','ai_queue.task_worker','router.openai_router:app','router.anthropic_router:app','metrics.prometheus_exporter:app','dashboard.dashboard_server:app');" ^
  "$ps=Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python(.exe)?$' -and $_.CommandLine };" ^
  "foreach($p in $ps){ foreach($m in $mods){ if($p.CommandLine -like ('*'+$m+'*')){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{}; break } } }" >nul 2>&1
powershell -NoProfile -Command ^
  "$ports=@(8001,8002,9001,9100,9200,9300,9400);" ^
  "foreach($port in $ports){ try{ $procs=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach($pid in $procs){ if($pid){ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } } }catch{} }" >nul 2>&1
timeout /t 1 /nobreak >nul

echo Starting all services with runtime.launcher...
echo.
echo   Ctrl+C to stop all services
echo.
call .venv\Scripts\python.exe -m runtime.launcher

endlocal
