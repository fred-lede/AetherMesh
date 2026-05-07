@echo off
setlocal
set "PATH=C:\Users\fred\miniconda3\envs\ai_inference_hub;C:\Users\fred\miniconda3\envs\ai_inference_hub\Scripts;%PATH%"
cd /d D:\Ai\ai_inference_hub
call scripts\start_all.bat >> D:\Ai\ai_inference_hub\logs\AIIH-Platform.log 2>&1
endlocal
