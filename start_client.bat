@echo off
setlocal

cd /d %~dp0

echo [voice-input] Starting Windows client...
echo.

set SERVER=ws://127.0.0.1:8991
set MODEL=gemma3:4b
set HOTKEY=f8

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
    echo [ERROR] Failed to install server requirements.
    pause
    exit /b 1
  )
  uv pip install --python .venv\Scripts\python.exe sounddevice numpy websockets pynput pyperclip pillow
  if errorlevel 1 (
    echo [ERROR] Failed to install Windows client requirements.
    pause
    exit /b 1
  )
)

echo Server: %SERVER%
echo Model:  %MODEL%
echo Hotkey: %HOTKEY%
echo.

REM Tips:
REM - Hold %HOTKEY% to record, release to paste.
REM - Hold Ctrl while releasing %HOTKEY% to paste without Enter.
REM - Disable screenshot context: add --no-screenshot

.venv\Scripts\python.exe win_client.py --server %SERVER% --model %MODEL% --hotkey %HOTKEY% --language zh

pause
