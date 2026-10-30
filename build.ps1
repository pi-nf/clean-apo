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

# 1. Ensure pyinstaller is installed
Write-Host "`nChecking PyInstaller..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pyinstaller

# 2. Run PyInstaller with clean_apo.spec
Write-Host "`nCompiling Clean APO executable..." -ForegroundColor Yellow
& $venvPython -m PyInstaller --clean -y clean_apo.spec

if ($LASTEXITCODE -eq 0) {
    $exePath = Join-Path $PSScriptRoot "dist\CleanAPO.exe"
    if (Test-Path $exePath) {
        $sizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
        Write-Host "`nBuild succeeded!" -ForegroundColor Green
        Write-Host "Output: $exePath ($sizeMB MB)" -ForegroundColor White
        Write-Host "You can now distribute this single .exe file to anyone!" -ForegroundColor Green
    }
} else {
    Write-Host "`nBuild failed with exit code $LASTEXITCODE." -ForegroundColor Red
    Exit 1
}
