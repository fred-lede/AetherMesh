@echo off
setlocal
cd /d "%~dp0.."

:: asr_worker needs ffmpeg to decode audio; SYSTEM account cannot resolve the user's LOCALAPPDATA.
set "FFMPEG_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build-shared\bin"
if not exist "%FFMPEG_BIN%" set "FFMPEG_BIN=C:\Users\fred\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build-shared\bin"
if exist "%FFMPEG_BIN%" set "PATH=%FFMPEG_BIN%;%PATH%"

echo ==========================================
echo AetherMesh Worker Node
echo ==========================================
echo.
echo Stopping stale worker services...
powershell -NoProfile -Command ^
  "$mods=@('node.worker_agent:app','node.node_agent:app','node.asr_worker:app');" ^
  "$ps=Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python(.exe)?$' -and $_.CommandLine };" ^
  "foreach($p in $ps){ foreach($m in $mods){ if($p.CommandLine -like ('*'+$m+'*')){ try{ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }catch{}; break } } }" >nul 2>&1
powershell -NoProfile -Command ^
  "$ports=@(9300,9400,9450);" ^
  "foreach($port in $ports){ try{ $procs=Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach($pid in $procs){ if($pid){ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } } }catch{} }" >nul 2>&1
timeout /t 1 /nobreak >nul

echo Starting node_agent + worker_agent + asr_worker...
echo.
echo   Ctrl+C to stop all services
echo.
call .venv\Scripts\python.exe -m runtime.launcher start node_agent worker_agent asr_worker

endlocal
