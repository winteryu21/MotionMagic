@echo off
chcp 65001 > nul
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "PIP=%ROOT%.venv\Scripts\pip.exe"
set "REQ=%ROOT%requirements.txt"
set "MODEL=%ROOT%models\hand_landmarker.task"
set "CAMERA_ID=0"
set "DEBUG_CAMERA=0"

:parse_args
if "%~1"=="" goto after_args
if /I "%~1"=="debug" (
    set "DEBUG_CAMERA=1"
) else (
    set "CAMERA_ID=%~1"
)
shift
goto parse_args

:after_args
echo =============================================
echo  MotionMagic game launcher
echo =============================================

if not exist "%PYTHON%" (
    echo [INFO] .venv was not found. Setting up the environment first...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install_python.ps1"
    if errorlevel 1 goto setup_failed
)

"%PYTHON%" -c "import pygame, cv2, mediapipe" > nul 2> nul
if errorlevel 1 (
    echo [INFO] Required packages are missing in .venv. Installing requirements...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed
    "%PYTHON%" -m pip install -r "%REQ%"
    if errorlevel 1 goto setup_failed
)

if not exist "%MODEL%" (
    echo [INFO] MediaPipe hand landmarker model is missing. Downloading...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "New-Item -ItemType Directory -Force '%ROOT%models' | Out-Null; Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' -OutFile '%MODEL%'"
    if errorlevel 1 goto setup_failed
)

set "MOTIONMAGIC_CAMERA_ID=%CAMERA_ID%"
set "MOTIONMAGIC_DEBUG_CAMERA=%DEBUG_CAMERA%"

echo [INFO] Python: %PYTHON%
echo [INFO] Camera ID: %MOTIONMAGIC_CAMERA_ID%
if "%MOTIONMAGIC_DEBUG_CAMERA%"=="1" echo [INFO] Debug camera overlay: ON
echo.
echo If the webcam does not open, try: run_game.bat 1
echo Toggle debug overlay in game: F3
echo.

"%PYTHON%" -m src.game.app
exit /b %ERRORLEVEL%

:setup_failed
echo.
echo [ERROR] Environment setup failed.
echo Try running setup_env.bat, then run_game.bat again.
pause
exit /b 1
