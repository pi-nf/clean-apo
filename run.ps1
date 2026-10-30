# Clean APO - PowerShell Setup & Launch Script
# Automates virtual environment creation, dependency installation, and application execution.

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

$pyVersion = python --version
Write-Host "Detected: $pyVersion" -ForegroundColor Green

# 2. Setup Virtual Environment
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "`nCreating virtual environment in: $venvPath..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
}

# 3. Activate Virtual Environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    & $activateScript
} else {
    Write-Host "Virtual environment activation script not found. Using system python." -ForegroundColor Yellow
}

# 4. Install / Verify Dependencies
Write-Host "`nInstalling / verifying required packages..." -ForegroundColor Yellow
pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Package installation encountered an error." -ForegroundColor Red
}

# 5. Launch Application
Write-Host "`nLaunching Clean APO..." -ForegroundColor White
python (Join-Path $PSScriptRoot "main.py")
