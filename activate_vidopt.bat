@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VIDOPT_ROOT=%ROOT%"
set "VIDOPT_FFMPEG_DIR=%ROOT%\vendor\ffmpeg\bin"
set "PATH=%VIDOPT_FFMPEG_DIR%;%ROOT%\vendor\python;%ROOT%\vendor\python\Scripts;%PATH%"
echo vidopt ready (bundled Python + ffmpeg). Examples:
echo   vidopt.bat doctor
echo   vidopt.bat dev video\corpus --config cpu --set jobs.cpu_workers=3
echo   vidopt.bat compress in.mp4 -o out\out.mp4 --target 89 --verify
echo.
cmd /k
