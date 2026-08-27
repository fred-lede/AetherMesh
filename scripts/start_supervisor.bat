@echo off
setlocal
cd /d "%~dp0.."

:: Add FFmpeg shared DLLs to PATH (needed by torchcodec for TTS)
set "FFMPEG_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build-shared\bin"
if exist "%FFMPEG_BIN%" set "PATH=%FFMPEG_BIN%;%PATH%"

echo ==========================================
echo AetherMesh Launcher Supervisor
echo ==========================================
echo.
echo Ensuring launcher + services are up, then supervising...
echo.

:: Start the supervisor (it boot-starts the stack if nothing is running,
:: then keeps restarting the launcher whenever the whole stack dies).
call .venv\Scripts\python.exe -m runtime.launcher supervise --check-interval 30

endlocal
