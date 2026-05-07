@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"

set WORKER_ID=%AIIH_TASK_WORKER_ID%
if "%WORKER_ID%"=="" set WORKER_ID=worker-1

title AIIH Task Worker %WORKER_ID%
echo Starting Task Worker %WORKER_ID%...

python -m ai_queue.task_worker --worker-id %WORKER_ID%