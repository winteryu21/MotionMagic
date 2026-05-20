<#
.SYNOPSIS
    MotionMagic 프로젝트 협업을 위한 Python 3.12.10 자동 설치 및 가상환경 구성 스크립트.
.DESCRIPTION
    이 스크립트는 시스템에 Python 3.12.10이 설치되어 있는지 확인하고,
    없다면 공식 파이썬 웹사이트에서 인스톨러를 다운로드하여 Silent 모드로 자동 설치합니다.
    설치 완료 후 프로젝트 가상환경(.venv)을 구축하고 의존성 패키지를 설치합니다.
#>

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$PythonVersionTarget = "3.12.10"
$InstallerUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
$InstallerPath = "$env:TEMP\python-3.12.10-amd64.exe"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " MotionMagic 환경 구축 및 Python 설치 도구" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. 파이썬 3.12.10 설치 여부 확인
Write-Host "1. Python $PythonVersionTarget 설치 여부를 확인하는 중..." -ForegroundColor Yellow

$pythonInstalled = $false
$installedVersion = ""

# py 런처가 있는지 확인
$pyExecutable = Get-Command py -ErrorAction SilentlyContinue
if ($pyExecutable) {
    $installedVersion = & py -3.12 --version 2>&1
    if ($installedVersion -like "*3.12.10*") {
        $pythonInstalled = $true
    }
} else {
    # python 명령어로 확인
    $pythonExecutable = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonExecutable) {
        $installedVersion = & python --version 2>&1
        if ($installedVersion -like "*3.12.10*") {
            $pythonInstalled = $true
        }
    }
}

if ($pythonInstalled) {
    Write-Host "[OK] 이미 Python $PythonVersionTarget 버전이 설치되어 있습니다. ($installedVersion)" -ForegroundColor Green
} else {
    Write-Host "[INFO] Python $PythonVersionTarget 버전을 찾을 수 없습니다. 설치를 진행합니다." -ForegroundColor Cyan
    
    # 인스톨러 다운로드
    Write-Host "2. 파이썬 인스톨러 다운로드 중..." -ForegroundColor Yellow
    Write-Host "URL: $InstallerUrl" -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $InstallerUrl -OutFile $InstallerPath -UseBasicParsing
        Write-Host "[OK] 다운로드 완료: $InstallerPath" -ForegroundColor Green
    } catch {
        Write-Error "파이썬 인스톨러 다운로드 중 오류 발생: $_"
        Exit 1
    }

    # Silent 설치 실행 (InstallAllUsers=0으로 관리자 권한 팝업 없이 사용자 디렉토리에 설치)
    Write-Host "3. Python $PythonVersionTarget 자동 설치 중 (잠시만 기다려주세요)..." -ForegroundColor Yellow
    $installArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 AssociateFiles=1"
    
    $process = Start-Process -FilePath $InstallerPath -ArgumentList $installArgs -Wait -PassThru
    if ($process.ExitCode -eq 0) {
        Write-Host "[OK] Python $PythonVersionTarget 설치 성공!" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] 파이썬 설치에 실패했습니다. 종료 코드: $($process.ExitCode)" -ForegroundColor Red
        Exit 1
    }

    # 다운로드 파일 삭제
    Remove-Item $InstallerPath -Force -ErrorAction SilentlyContinue

    # PATH 환경변수 강제 갱신
    Write-Host "4. 시스템 PATH 환경 변수 갱신 중..." -ForegroundColor Yellow
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 3. 가상환경(.venv) 설정 및 의존성 설치
Write-Host "5. 프로젝트 가상환경(.venv) 설정 중..." -ForegroundColor Yellow

# 프로젝트 루트 경로 확보
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."

# 가상환경이 위치할 경로
$VenvPath = Join-Path $ProjectRoot ".venv"

if (Test-Path $VenvPath) {
    Write-Host "[INFO] 이미 가상환경(.venv) 폴더가 존재합니다. 기존 가상환경을 업데이트합니다." -ForegroundColor Cyan
} else {
    Write-Host "가상환경 생성 중..." -ForegroundColor Yellow
    # Python Launcher(py)가 있으면 py -3.12를 사용하고, 없으면 python 사용
    if ($pyExecutable) {
        & py -3.12 -m venv $VenvPath
    } else {
        # 설치 직후의 환경변수 미반영을 감안하여 유력한 설치 경로 확인
        $localPythonPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
        if (Test-Path $localPythonPath) {
            & $localPythonPath -m venv $VenvPath
        } else {
            & python -m venv $VenvPath
        }
    }
}

# 가상환경 내 pip 및 requirements.txt 설치
$PipPath = Join-Path $VenvPath "Scripts\pip.exe"
$ReqPath = Join-Path $ProjectRoot "requirements.txt"

if (Test-Path $PipPath) {
    Write-Host "가상환경 의존성 라이브러리 설치 중 (pip, requirements.txt)..." -ForegroundColor Yellow
    & $PipPath install --upgrade pip
    if (Test-Path $ReqPath) {
        & $PipPath install -r $ReqPath
        Write-Host "[OK] 모든 패키지가 가상환경(.venv)에 성공적으로 설치되었습니다!" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] requirements.txt 파일을 찾을 수 없습니다. 라이브러리 설치를 생략합니다." -ForegroundColor Yellow
    }
} else {
    Write-Host "[ERROR] 가상환경 생성 실패 또는 pip 경로를 찾을 수 없습니다." -ForegroundColor Red
    Exit 1
}

Write-Host "=============================================" -ForegroundColor Green
Write-Host " 환경 구축 완료! 프로젝트 루트의 .venv 가상환경을 사용하세요." -ForegroundColor Green
Write-Host " Pygame 게임 실행 방법: .venv\Scripts\python -m src.game.app" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
