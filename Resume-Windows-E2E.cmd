@echo off
setlocal EnableExtensions
title wLLM Windows E2E - Resume
chcp 65001 >nul 2>nul
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo.
  echo Node.js 20 LTS is not installed or is not on PATH.
  echo Install Node.js 20 LTS, reopen this window, and run this file again.
  echo.
  pause
  exit /b 1
)

node -e "process.exit(Number(process.versions.node.split('.')[0]) >= 20 ? 0 : 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Node.js 20 LTS or newer is required. Current version:
  node --version
  echo Install Node.js 20 LTS, reopen this window, and run this file again.
  echo.
  pause
  exit /b 1
)

node "%~dp0evaluation\windows\run-windows-e2e.mjs" --resume %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Windows E2E resume finished.
) else (
  echo Windows E2E resume stopped with exit code %EXIT_CODE%.
  echo Inspect the newest folder under impl\handoff\windows.
)
echo.
pause
exit /b %EXIT_CODE%
