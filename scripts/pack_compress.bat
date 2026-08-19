@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "ROOT=%CD%"

if exist "%ROOT%\vendor\python\python.exe" (
  "%ROOT%\vendor\python\python.exe" "%ROOT%\scripts\pack_compress.py" --platform windows %*
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\pack_compress.py" --platform windows %*
) else (
  echo ERROR: no Python found. Run install.bat first.
  exit /b 1
)
exit /b %ERRORLEVEL%
