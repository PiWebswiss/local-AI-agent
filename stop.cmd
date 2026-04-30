@echo off
REM Stops and uninstalls the local AI agent: removes containers, named
REM volumes (Ollama models + WhisperX cache + RAG indexes inside the volume),
REM networks, and the locally-built images for this project. Pulled images
REM (e.g. ollama/ollama:latest) are kept. Asks for confirmation because it's
REM destructive — restart with start.cmd to set up again from scratch.

setlocal enabledelayedexpansion

set /p "CONFIRM=Stop and uninstall everything (containers, Ollama models, WhisperX cache, RAG indexes, built images)? [y/N] "
if /I not "!CONFIRM!"=="y" if /I not "!CONFIRM!"=="yes" (
    echo Aborted.
    endlocal
    exit /b 0
)

REM --profile voice ensures the whisperx service is included if it was created.
docker compose --profile voice down -v --rmi local --remove-orphans

REM Clear voice preference so the next start.cmd run re-asks the user.
if exist "%~dp0.env" (
    type "%~dp0.env" | findstr /V "^ENABLE_VOICE=" > "%~dp0.env.tmp"
    move /Y "%~dp0.env.tmp" "%~dp0.env" > nul
)

echo.
echo Uninstalled. Run start.cmd to set up again from scratch.
endlocal
