@echo off
setlocal EnableDelayedExpansion

set "ROOT_DIR=%~1"
if "%ROOT_DIR%"=="" set "ROOT_DIR=%cd%"
set "ENV_FILE=%ROOT_DIR%\.env"

if not exist "%ENV_FILE%" (
    endlocal
    exit /b 0
)

for /f "usebackq tokens=* delims=" %%L in ("%ENV_FILE%") do (
    set "line=%%L"
    if not "!line!"=="" if not "!line:~0,1!"=="#" (
        for /f "tokens=1* delims==" %%A in ("!line!") do (
            endlocal
            set "%%~A=%%~B"
            setlocal EnableDelayedExpansion
        )
    )
)

endlocal
exit /b 0
