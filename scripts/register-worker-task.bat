@echo off
setlocal

net session >nul 2>&1
if errorlevel 1 (
  echo ERROR: run this script as Administrator.
  exit /b 1
)

set "TASK_HOST=AetherMesh"
set "TASK_WORKER=AetherMesh-Worker"
set "START_BAT=%~dp0start-worker.bat"

schtasks /Query /TN "%TASK_HOST%" >nul 2>&1
if not errorlevel 1 (
  echo Disabling old host task "%TASK_HOST%"...
  schtasks /Change /TN "%TASK_HOST%" /DISABLE
)

echo Creating worker task "%TASK_WORKER%"...
schtasks /Create /SC ONSTART /TN "%TASK_WORKER%" /TR "\"%START_BAT%\"" /RU SYSTEM /RL HIGHEST /F

echo.
echo Verify with: schtasks /Query /TN "%TASK_WORKER%"
endlocal
