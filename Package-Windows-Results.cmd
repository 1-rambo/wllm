@echo off
setlocal EnableExtensions
title wLLM Windows E2E - Package Results
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

node "%~dp0evaluation\windows\run-windows-e2e.mjs" --package-only %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Result package refreshed.
) else (
  echo Result packaging failed with exit code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
