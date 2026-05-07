@echo off
setlocal
cd /d "%~dp0.."
call scripts\load_env.bat "%cd%"

set BASE_URL=http://127.0.0.1:8001
set MODEL=qwen3.5:27b
set TOTAL=20
set CONCURRENCY=8
set API_KEY=local-dev-key
if not "%AIIH_API_KEY%"=="" set API_KEY=%AIIH_API_KEY%

python scripts\stress_429_check.py --base-url %BASE_URL% --model %MODEL% --total %TOTAL% --concurrency %CONCURRENCY% --api-key %API_KEY% %*
exit /b %errorlevel%
