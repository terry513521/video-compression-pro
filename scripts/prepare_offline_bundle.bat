@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "ROOT=%CD%"

if exist "%ROOT%\vendor\python\python.exe" (
  "%ROOT%\vendor\python\python.exe" "%ROOT%\scripts\prepare_offline_bundle.py" %*
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\prepare_offline_bundle.py" %*
) else (
  python "%ROOT%\scripts\prepare_offline_bundle.py" %*
)
exit /b %ERRORLEVEL%
