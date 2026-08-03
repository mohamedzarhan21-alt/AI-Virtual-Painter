@echo off
rem ===========================================================================
rem  AI Virtual Painter - Windows launcher
rem
rem  Finds a suitable Python, checks the dependencies, and starts the app from
rem  this folder so it can find its icon. Keeps the console open on failure so
rem  the error stays readable.
rem ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
title AI Virtual Painter

rem Run from the folder this script lives in, whatever the caller's directory is.
pushd "%~dp0"

echo.
echo   AI VIRTUAL PAINTER
echo   air canvas engine
echo.

rem --- locate Python 3.11+ ---------------------------------------------------
set "PYTHON="
for %%I in ("py -3.11" "py -3" "python") do (
    if not defined PYTHON (
        %%~I -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
        if !errorlevel! equ 0 set "PYTHON=%%~I"
    )
)

if not defined PYTHON (
    echo   [X] Python 3.11 or newer was not found.
    echo       Install it from https://www.python.org/downloads/
    echo       and make sure "Add python.exe to PATH" is ticked.
    echo.
    pause
    popd
    endlocal
    exit /b 1
)

for /f "delims=" %%V in ('%PYTHON% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%V"
echo   Python !PYVER! via "!PYTHON!"

rem --- check the dependencies ------------------------------------------------
%PYTHON% -c "import cv2, mediapipe, numpy, PIL, customtkinter" >nul 2>&1
if errorlevel 1 (
    echo   [X] Some dependencies are missing.
    echo.
    set "INSTALL="
    set /p "INSTALL=      Install them now with pip? [Y/n] "
    if /i "!INSTALL!"=="n" (
        popd
        endlocal
        exit /b 1
    )
    echo.
    %PYTHON% -m pip install --upgrade "opencv-python>=4.9" "mediapipe>=0.10.30" "numpy>=1.26" "Pillow>=10" "customtkinter>=5.2"
    if errorlevel 1 (
        echo.
        echo   [X] Dependency installation failed. See the pip output above.
        echo.
        pause
        popd
        endlocal
        exit /b 1
    )
)

rem --- launch ----------------------------------------------------------------
echo   Starting... ^(the hand model is downloaded once on first run^)
echo.

%PYTHON% "ai_virtual_painter.py"
set "EXITCODE=!errorlevel!"

if not "!EXITCODE!"=="0" (
    echo.
    echo   [X] AI Virtual Painter exited with code !EXITCODE!.
    echo.
    pause
)

popd
endlocal & exit /b %EXITCODE%
