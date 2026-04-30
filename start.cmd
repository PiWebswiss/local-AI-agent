@echo off
REM One-command launcher for the local AI agent. Asks the user once whether
REM to enable voice input (WhisperX), persists the choice to .env, and then
REM runs `docker compose up -d --build` with the right profile so WhisperX
REM is only built/installed when the user actually wants it.

setlocal enabledelayedexpansion

set "ENV_FILE=%~dp0.env"
set "VOICE_PREF="

if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B /C:"ENABLE_VOICE=" "%ENV_FILE%" 2^>nul`) do (
        set "VOICE_PREF=%%B"
    )
)

if "!VOICE_PREF!"=="" (
    echo.
    echo This project includes optional voice input via WhisperX.
    echo Enabling it adds ~1.5 GB to the Docker build and downloads model weights on first start.
    echo You can change this later by re-running this script ^(after deleting the
    echo ENABLE_VOICE line from .env^) or from the web UI Settings tab.
    echo.
    set /p "ANSWER=Enable voice input? [y/N] "
    if /I "!ANSWER!"=="y" (
        set "VOICE_PREF=true"
    ) else if /I "!ANSWER!"=="yes" (
        set "VOICE_PREF=true"
    ) else (
        set "VOICE_PREF=false"
    )

    REM Persist preference to .env so future runs don't ask again.
    if not exist "%ENV_FILE%" (
        type nul > "%ENV_FILE%"
    )
    echo.>> "%ENV_FILE%"
    echo ENABLE_VOICE=!VOICE_PREF!>> "%ENV_FILE%"
    echo Saved preference: ENABLE_VOICE=!VOICE_PREF!
)

if /I "!VOICE_PREF!"=="true" (
    set "COMPOSE_PROFILES=voice"
    echo Starting with voice input enabled...
) else (
    set "COMPOSE_PROFILES="
    echo Starting without voice input...
)

docker compose up -d --build
endlocal
