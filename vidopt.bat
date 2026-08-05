@echo off
setlocal EnableExtensions

rem Relocatable launcher — works after moving/extracting this folder anywhere.
rem Uses bundled Python + ffmpeg only. No OS install required.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VIDOPT_ROOT=%ROOT%"
set "VIDOPT_FFMPEG_DIR=%ROOT%\vendor\ffmpeg\bin"
set "PATH=%VIDOPT_FFMPEG_DIR%;%ROOT%\vendor\python;%ROOT%\vendor\python\Scripts;%PATH%"

if not exist "%ROOT%\vendor\python\python.exe" (
  echo ERROR: bundled Python missing: %ROOT%\vendor\python\python.exe
  echo This package is incomplete. Re-pack after install.bat on the build machine.
  exit /b 1
)
if not exist "%VIDOPT_FFMPEG_DIR%\ffmpeg.exe" (
  echo ERROR: bundled ffmpeg missing: %VIDOPT_FFMPEG_DIR%\ffmpeg.exe
  exit /b 1
)

"%ROOT%\vendor\python\python.exe" -m vidopt %*
exit /b %ERRORLEVEL%
