@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"
if not defined AIIH_HOST set "AIIH_HOST=0.0.0.0"
if not defined AIIH_ROUTER_PORT set "AIIH_ROUTER_PORT=8001"

echo Starting Router API on port %AIIH_ROUTER_PORT%...
python -m uvicorn router.openai_router:app --host %AIIH_HOST% --port %AIIH_ROUTER_PORT%
