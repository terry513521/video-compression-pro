@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================================
rem  vidopt — offline install / REPAIR (Windows)
rem
rem  Works with NO network. Rebuilds the runtime from installables already
rem  inside this package:
rem
rem      vendor\installers\python-3.11.9-embed-amd64.zip   (preferred rebuild)
rem      vendor\installers\python-3.11.9-amd64.exe         (full installer fallback)
rem      vendor\installers\get-pip.py
rem      vendor\wheelhouse\*.whl                           (all Python libraries)
rem      vendor\ffmpeg\bin\ffmpeg.exe + ffprobe.exe        (libvmaf build)
rem      src\vidopt\                                       (application source)
rem
rem  Normal use: extract the zip and run vidopt.bat (already installed).
rem  When the environment is damaged: run this script.
rem
rem      install.bat
rem ============================================================================

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENDOR=%ROOT%\vendor"
set "PY_HOME=%VENDOR%\python"
set "PY=%PY_HOME%\python.exe"
set "FFMPEG_BIN=%VENDOR%\ffmpeg\bin"
set "WHEELHOUSE=%VENDOR%\wheelhouse"
set "INSTALLERS=%VENDOR%\installers"
set "EMBED_ZIP=%INSTALLERS%\python-3.11.9-embed-amd64.zip"
set "PY_SETUP=%INSTALLERS%\python-3.11.9-amd64.exe"
set "GET_PIP=%INSTALLERS%\get-pip.py"
if not exist "%GET_PIP%" if exist "%PY_HOME%\get-pip.py" set "GET_PIP=%PY_HOME%\get-pip.py"

echo.
echo ============================================================
echo  vidopt offline install / repair
echo  root: %ROOT%
echo ============================================================
echo.

rem ---- wheelhouse (libraries) ----------------------------------------------
if not exist "%WHEELHOUSE%\" goto :missing_wheelhouse
dir /b "%WHEELHOUSE%\*.whl" >nul 2>&1
if errorlevel 1 goto :missing_wheelhouse
echo Using wheelhouse: %WHEELHOUSE%

rem ---- ffmpeg --------------------------------------------------------------
if not exist "%FFMPEG_BIN%\ffmpeg.exe" goto :missing_ffmpeg
if not exist "%FFMPEG_BIN%\ffprobe.exe" goto :missing_ffmpeg
echo Using ffmpeg: %FFMPEG_BIN%\ffmpeg.exe

rem ---- Python: reuse or rebuild from installers ----------------------------
set "NEED_PY=0"
if not exist "%PY%" set "NEED_PY=1"
if "%NEED_PY%"=="0" (
  "%PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
  if errorlevel 1 set "NEED_PY=1"
)

if "%NEED_PY%"=="1" (
  echo.
  echo Bundled Python missing or broken — rebuilding from vendor\installers ...
  call :rebuild_python
  if errorlevel 1 exit /b 1
) else (
  echo Using bundled Python: %PY%
)

"%PY%" -c "import sys; print('  ', sys.version.split()[0])"

rem ---- ensure pip ----------------------------------------------------------
echo.
echo [1/3] Ensuring pip ^(offline from wheelhouse^) ...
"%PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
  if not exist "%GET_PIP%" (
    echo ERROR: get-pip.py not found at vendor\installers\get-pip.py
    echo The package is incomplete.
    exit /b 1
  )
  echo       bootstrapping pip via get-pip.py
  "%PY%" "%GET_PIP%" --no-index --find-links "%WHEELHOUSE%" --no-warn-script-location
  if errorlevel 1 (
    echo ERROR: get-pip failed
    exit /b 1
  )
)

"%PY%" -m pip install --no-index --find-links "%WHEELHOUSE%" --upgrade --no-warn-script-location pip setuptools wheel
if errorlevel 1 (
  echo ERROR: could not install pip/setuptools/wheel from wheelhouse
  exit /b 1
)

rem ---- libraries + vidopt --------------------------------------------------
echo.
echo [2/3] Installing libraries + vidopt from wheelhouse + src ^(offline^) ...
"%PY%" -m pip install --no-index --find-links "%WHEELHOUSE%" --no-warn-script-location ^
  numpy==2.1.3 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2 ^
  opencv-python-headless==4.10.0.84 scenedetect==0.6.5 PyYAML==6.0.2
if errorlevel 1 (
  echo ERROR: pip install of dependencies failed
  exit /b 1
)

if not exist "%ROOT%\src\vidopt\__init__.py" (
  echo ERROR: missing src\vidopt — cannot repair application code
  exit /b 1
)
set "SITE_PKG=%PY_HOME%\Lib\site-packages\vidopt"
if exist "%SITE_PKG%" rmdir /s /q "%SITE_PKG%"
xcopy /E /I /Y /Q "%ROOT%\src\vidopt" "%SITE_PKG%" >nul
if errorlevel 1 (
  echo ERROR: failed to copy src\vidopt into bundled Python
  exit /b 1
)

rem ---- verify --------------------------------------------------------------
echo.
echo [3/3] Verifying ...
set "VIDOPT_FFMPEG_DIR=%FFMPEG_BIN%"
set "PATH=%FFMPEG_BIN%;%PY_HOME%;%PY_HOME%\Scripts;%PATH%"
"%PY%" -m vidopt doctor
if errorlevel 1 (
  echo.
  echo WARNING: vidopt doctor reported problems. See REPAIR.txt
  exit /b 1
)

echo.
echo ============================================================
echo  Install / repair complete — still fully offline.
echo ============================================================
echo.
echo    vidopt.bat doctor
echo    vidopt.bat dev video\corpus --encoder libx265 --cpu-workers 4
echo    vidopt.bat compress in.mp4 -o out\out.mp4 --target 89 --encoder libx265 --verify
echo.
echo  Repair again anytime with:  install.bat
echo  Checklist: REPAIR.txt
echo.
endlocal
exit /b 0

rem ========================================================================
:rebuild_python
if not exist "%INSTALLERS%" mkdir "%INSTALLERS%"

if exist "%EMBED_ZIP%" (
  echo       using embed zip: %EMBED_ZIP%
  if exist "%PY_HOME%" rmdir /s /q "%PY_HOME%"
  mkdir "%PY_HOME%"
  powershell -NoProfile -Command "Expand-Archive -Path '%EMBED_ZIP%' -DestinationPath '%PY_HOME%' -Force"
  if errorlevel 1 (
    echo ERROR: failed to extract embeddable Python
    exit /b 1
  )
  rem enable site-packages
  powershell -NoProfile -Command ^
    "$p = Get-ChildItem '%PY_HOME%' -Filter 'python*._pth' | Select-Object -First 1; if (-not $p) { exit 1 }; " ^
    "$c = Get-Content $p.FullName; $o = @(); foreach ($l in $c) { if ($l -match '^\s*#\s*import site') { $o += 'Lib\site-packages'; $o += 'import site' } elseif ($l -match '^\s*import site') { if ($o -notcontains 'Lib\site-packages') { $o += 'Lib\site-packages' }; $o += 'import site' } else { $o += $l } }; " ^
    "if ($o -notcontains 'import site') { $o += 'Lib\site-packages'; $o += 'import site' }; $o | Set-Content $p.FullName -Encoding ASCII"
  if not exist "%PY%" (
    echo ERROR: python.exe missing after extract
    exit /b 1
  )
  if exist "%GET_PIP%" copy /Y "%GET_PIP%" "%PY_HOME%\get-pip.py" >nul
  exit /b 0
)

if exist "%PY_SETUP%" (
  echo       using official installer: %PY_SETUP%
  echo       TargetDir=%PY_HOME%
  if exist "%PY_HOME%" rmdir /s /q "%PY_HOME%"
  "%PY_SETUP%" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_doc=0 Include_launcher=0 InstallLauncherAllUsers=0 SimpleInstall=1 TargetDir="%PY_HOME%"
  if errorlevel 1 (
    echo ERROR: Python installer failed
    exit /b 1
  )
  if not exist "%PY%" (
    echo ERROR: python.exe not found after installer — TargetDir may differ
    exit /b 1
  )
  exit /b 0
)

echo ERROR: cannot rebuild Python — missing installers:
echo   %EMBED_ZIP%
echo   %PY_SETUP%
echo The offline package is incomplete. Re-download the production zip.
exit /b 1

:missing_wheelhouse
echo ERROR: vendor\wheelhouse is missing or empty.
echo Offline repair needs the .whl libraries that shipped in the package.
echo Re-extract / re-download the production zip.
exit /b 1

:missing_ffmpeg
echo ERROR: vendor\ffmpeg\bin\ffmpeg.exe ^(and ffprobe.exe^) required.
echo Restore those two files from a good copy of the package, then re-run install.bat.
exit /b 1
