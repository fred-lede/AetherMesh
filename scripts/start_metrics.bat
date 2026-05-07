@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"
if not defined AIIH_HOST set "AIIH_HOST=0.0.0.0"
if not defined AIIH_METRICS_PORT set "AIIH_METRICS_PORT=8002"

echo Starting Metrics Exporter on port %AIIH_METRICS_PORT%...
python -m uvicorn metrics.prometheus_exporter:app --host %AIIH_HOST% --port %AIIH_METRICS_PORT%
