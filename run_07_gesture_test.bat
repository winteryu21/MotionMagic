@echo off
chcp 65001 > nul
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "REQ=%ROOT%requirements.txt"
set "MODEL=%ROOT%models\hand_landmarker.task"
set "CAMERA_ID=0"

:parse_args
if "%~1"=="" goto after_args
set "CAMERA_ID=%~1"
shift
goto parse_args

:after_args
echo =============================================
echo  MotionMagic 07 gesture test launcher
echo =============================================

if not exist "%PYTHON%" (
    echo [INFO] .venv was not found. Setting up the environment first...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\install_python.ps1"
    if errorlevel 1 goto setup_failed
)

"%PYTHON%" -c "import cv2, mediapipe, numpy" > nul 2> nul
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

echo [INFO] Python: %PYTHON%
echo [INFO] Camera ID: %CAMERA_ID%
echo.
echo If the webcam does not open, try: run_07_gesture_test.bat 1
echo.

"%PYTHON%" "%ROOT%scripts\07_gesture_mode_test.py" --camera "%CAMERA_ID%"
exit /b %ERRORLEVEL%

:setup_failed
echo.
echo [ERROR] Environment setup failed.
echo Try running setup_env.bat, then run_07_gesture_test.bat again.
pause
exit /b 1
