# Clean APO - PowerShell Setup & Launch Script
# Automates virtual environment creation, dependency installation, and application execution.

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor White
Write-Host "       Clean APO for Equalizer APO - Setup & Launch       " -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor White

# 1. Verify Python installation
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python is not found in your PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from python.org and check 'Add Python to PATH'." -ForegroundColor Yellow
    Exit 1
}

$pyVersion = & python --version
Write-Host "Detected: $pyVersion" -ForegroundColor Green

# 2. Setup Virtual Environment inside $PSScriptRoot\.venv (independent of caller CWD)
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "`nCreating virtual environment in: $venvPath..." -ForegroundColor Yellow
    & python -m venv $venvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
}

# 3. Install / Verify Dependencies directly with the venv Python (avoids ExecutionPolicy activation issues)
Write-Host "`nInstalling / verifying required packages..." -ForegroundColor Yellow
$reqPath = Join-Path $PSScriptRoot "requirements.txt"
& $venvPython -m pip install -r $reqPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "Package installation encountered an error (exit code $LASTEXITCODE)." -ForegroundColor Red
    Exit 1
}

# 4. Launch Application directly with the venv Python
Write-Host "`nLaunching Clean APO..." -ForegroundColor White
& $venvPython (Join-Path $PSScriptRoot "main.py")
Exit $LASTEXITCODE
