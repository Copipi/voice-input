@echo off
setlocal

cd /d %~dp0

echo [voice-input] Starting Windows client...
echo.

set SERVER=ws://127.0.0.1:8991
set MODEL=gemma3:12b-it-qat
set HOTKEY=f9
set NO_ENTER_HOTKEY=f8
set TOGGLE_HOTKEY=f10
set PAUSE_HOTKEY=f11

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
  uv pip install --python .venv\Scripts\python.exe sounddevice numpy websockets pynput pyperclip pillow keyboard
  if errorlevel 1 (
    echo [ERROR] Failed to install Windows client requirements.
    pause
    exit /b 1
  )
)

REM Ensure `keyboard` is installed even if .venv already existed.
.venv\Scripts\python.exe -m pip show keyboard >nul 2>nul
if errorlevel 1 (
  echo [voice-input] Installing missing dependency: keyboard
  .venv\Scripts\python.exe -m pip install keyboard
  if errorlevel 1 (
    echo [ERROR] Failed to install keyboard.
    pause
    exit /b 1
  )
)

echo Server: %SERVER%
echo Model:  %MODEL%
echo Hotkey: %HOTKEY%
echo No-Enter Hotkey: %NO_ENTER_HOTKEY%
echo Toggle Hotkey: %TOGGLE_HOTKEY%
echo Pause Hotkey: %PAUSE_HOTKEY%
echo.

REM Tips:
REM - Hold %HOTKEY% to record, release to paste.
REM - Hold %NO_ENTER_HOTKEY% to record, release to paste without Enter.
REM - Press %TOGGLE_HOTKEY% once to start continuous dictation,
REM   auto-send by silence every ~10-20s, press again to stop.
REM - While continuous dictation is ON, press %PAUSE_HOTKEY% to pause/resume audio capture.
REM - Disable screenshot context: add --no-screenshot

  .venv\Scripts\python.exe win_client.py --server %SERVER% --model %MODEL% --hotkey %HOTKEY% --no-enter-hotkey %NO_ENTER_HOTKEY% --toggle-hotkey %TOGGLE_HOTKEY% --pause-hotkey %PAUSE_HOTKEY% --language zh --output-language zh

pause
