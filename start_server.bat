@echo off
setlocal

cd /d %~dp0

echo [voice-input] Starting WebSocket server...
echo.

REM Ollama settings
set OLLAMA_URL=http://127.0.0.1:11434
set LLM_MODEL=gemma3:12b-it-qat
set VISION_MODEL=qwen3-vl:8b-instruct

REM Whisper model (adjust if you want higher accuracy)
REM Recommended: large-v3-turbo (best speed/quality ratio)
REM If your machine is low on RAM, change this to: small
set WHISPER_MODEL=large-v3-turbo

REM Download/cache Whisper models inside this repo (portable)
set "WHISPER_DOWNLOAD_ROOT=%~dp0models\whisper"

REM Optional: force CPU or CUDA
REM set WHISPER_DEVICE=cpu
REM set WHISPER_DEVICE=cuda

if not exist ".venv\Scripts\python.exe" (
  echo [voice-input] .venv not found. Bootstrapping with uv...
  echo.
  where uv >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] uv not found on PATH.
    echo Install uv: https://astral.sh/uv
    pause
    exit /b 1
  )
  uv venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv with uv.
    pause
    exit /b 1
  )
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
  )
)

echo Ollama:   %OLLAMA_URL%
echo LLM:      %LLM_MODEL%
echo Vision:   %VISION_MODEL%
echo Whisper:  %WHISPER_MODEL%
echo Cache:    %WHISPER_DOWNLOAD_ROOT%
echo.

.venv\Scripts\python.exe ws_server.py --host 127.0.0.1 --port 8991

pause
