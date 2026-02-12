@echo off
setlocal

cd /d %~dp0

echo [voice-input] Restarting WebSocket server (free :8991 then start)...
echo.

call "%~dp0kill_8991.bat" --no-pause

echo.
echo [voice-input] Starting server...
echo.

call "%~dp0start_server.bat"
