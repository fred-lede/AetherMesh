@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"
if not defined AIIH_HOST set "AIIH_HOST=0.0.0.0"
if not defined AIIH_DASHBOARD_PORT set "AIIH_DASHBOARD_PORT=8003"

echo Starting Dashboard on port %AIIH_DASHBOARD_PORT%...
python -m uvicorn dashboard.dashboard_server:app --host %AIIH_HOST% --port %AIIH_DASHBOARD_PORT%
