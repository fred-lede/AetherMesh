@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"
if not defined AIIH_HOST set "AIIH_HOST=0.0.0.0"
if not defined AIIH_ANTHROPIC_PORT set "AIIH_ANTHROPIC_PORT=8002"

echo Starting Anthropic Router API on port %AIIH_ANTHROPIC_PORT%...
python -m uvicorn router.anthropic_router:app --host %AIIH_HOST% --port %AIIH_ANTHROPIC_PORT%
