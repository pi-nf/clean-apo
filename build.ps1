# Clean APO - Build Executable Script
# Compiles Clean APO into a standalone, portable Windows executable (.exe).

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor White
Write-Host "           Building Clean APO Standalone Executable       " -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor White

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

Write-Host "Using Python: $venvPython" -ForegroundColor Gray

Push-Location $PSScriptRoot
try {
    # 1. Ensure PyInstaller is installed (avoid forced network upgrade if already available)
    Write-Host "`nChecking PyInstaller..." -ForegroundColor Yellow
    & $venvPython -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller not found. Installing..." -ForegroundColor Yellow
        & $venvPython -m pip install "pyinstaller>=6.0.0"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`nFailed to install PyInstaller (exit code $LASTEXITCODE)." -ForegroundColor Red
            Exit 1
        }
    }

    # 2. Run PyInstaller with clean_apo.spec from repo root
    $specPath = Join-Path $PSScriptRoot "clean_apo.spec"
    Write-Host "`nCompiling Clean APO executable..." -ForegroundColor Yellow
    & $venvPython -m PyInstaller --clean -y $specPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nBuild failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Exit 1
    }

    $exePath = Join-Path $PSScriptRoot "dist\CleanAPO.exe"
    if (-not (Test-Path $exePath)) {
        Write-Host "`nBuild completed but output executable is missing: $exePath" -ForegroundColor Red
        Exit 1
    }

    $sizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
    Write-Host "`nBuild succeeded!" -ForegroundColor Green
    Write-Host "Output: $exePath ($sizeMB MB)" -ForegroundColor White
    Write-Host "You can now distribute this single .exe file to anyone!" -ForegroundColor Green
} finally {
    Pop-Location
}
