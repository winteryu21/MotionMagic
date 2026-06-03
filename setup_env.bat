@echo off
:: MotionMagic Python & Environment Setup Launcher
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo =============================================
echo  MotionMagic 환경 구성 도구를 시작합니다.
echo =============================================

:: PowerShell 실행 정책을 임시로 Bypass하여 스크립트 실행
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_python.ps1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [오류] 환경 구성 중 에러가 발생했습니다.
) else (
    echo.
    echo [성공] 모든 환경 설정이 완료되었습니다!
    echo.
    echo 게임 실행: run_game.bat
    echo 07 진단 실행: run_07_gesture_test.bat
    echo 웹캠이 안 열리면: run_game.bat 1
)

echo.
pause
