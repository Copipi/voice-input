@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d %~dp0

echo [voice-input] Freeing TCP port 8991...
echo.

REM Some environments set a broken FINDSTR default options variable.
REM Clear it to avoid: "FINDSTR: invalid command line".
set "FINDSTR="

set "FOUND="

REM Allow non-interactive usage: kill_8991.bat --no-pause
set "NO_PAUSE="
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C":8991"') do (
  if not defined KILLED_%%P (
    set "KILLED_%%P=1"
    set "FOUND=1"
    echo - Killing PID %%P uses :8991
    taskkill /PID %%P /F >nul 2>nul
  )
)

if not defined FOUND (
  echo - No process found using :8991
) else (
  echo.
  echo [OK] Port 8991 should be free now.
)

echo.
if not defined NO_PAUSE pause
