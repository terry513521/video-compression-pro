@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "ROOT=%CD%"

if exist "%ROOT%\vendor\python\python.exe" (
  "%ROOT%\vendor\python\python.exe" "%ROOT%\scripts\pack_production.py" %*
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\pack_production.py" %*
) else (
  python "%ROOT%\scripts\pack_production.py" %*
)
exit /b %ERRORLEVEL%
