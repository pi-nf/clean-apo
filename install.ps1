# Clean APO - Windows Application Installer Script
# Installs Clean APO into the standard Windows user applications directory:
# %LOCALAPPDATA%\Programs\Clean APO
# Creates Desktop & Start Menu shortcuts with the official application icon.

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor White
Write-Host "             Clean APO - Windows App Installer            " -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor White

$repoDir = $PSScriptRoot
$distExe = Join-Path $repoDir "dist\CleanAPO.exe"
$iconSrc = Join-Path $repoDir "assets\clean_apo.ico"

# If dist\CleanAPO.exe does not exist, offer to build it
if (-not (Test-Path $distExe)) {
    Write-Host "`nCleanAPO.exe not found in dist\. Building executable first..." -ForegroundColor Yellow
    $buildScript = Join-Path $repoDir "build.ps1"
    if (Test-Path $buildScript) {
        & $buildScript
    } else {
        Write-Error "Could not find build.ps1 to build Clean APO executable."
        Exit 1
    }
}

if (-not (Test-Path $distExe)) {
    Write-Error "Build failed or CleanAPO.exe missing: $distExe"
    Exit 1
}

$installDir = Join-Path $env:LOCALAPPDATA "Programs\Clean APO"
Write-Host "`nInstalling Clean APO to: $installDir" -ForegroundColor Cyan

if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

$targetExe = Join-Path $installDir "CleanAPO.exe"
$targetIcon = Join-Path $installDir "clean_apo.ico"

Copy-Item $distExe $targetExe -Force
if (Test-Path $iconSrc) {
    Copy-Item $iconSrc $targetIcon -Force
}

# Create Shortcuts
$WshShell = New-Object -ComObject WScript.Shell

# Desktop shortcut
$desktopPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Clean APO.lnk"
$shortcut = $WshShell.CreateShortcut($desktopPath)
$shortcut.TargetPath = $targetExe
$shortcut.WorkingDirectory = $installDir
if (Test-Path $targetIcon) {
    $shortcut.IconLocation = "$targetIcon,0"
}
$shortcut.Description = "Clean APO - High-Performance Audio Equalizer & AutoEQ Studio"
$shortcut.Save()

# Start Menu shortcut
$startMenuPath = Join-Path ([Environment]::GetFolderPath("Programs")) "Clean APO.lnk"
$shortcut = $WshShell.CreateShortcut($startMenuPath)
$shortcut.TargetPath = $targetExe
$shortcut.WorkingDirectory = $installDir
if (Test-Path $targetIcon) {
    $shortcut.IconLocation = "$targetIcon,0"
}
$shortcut.Description = "Clean APO - High-Performance Audio Equalizer & AutoEQ Studio"
$shortcut.Save()

# Windows App Paths registry (enables Win+R 'cleanapo')
try {
    $appPathKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\App Paths\CleanAPO.exe"
    if (-not (Test-Path $appPathKey)) {
        New-Item -Path $appPathKey -Force | Out-Null
    }
    Set-ItemProperty -Path $appPathKey -Name "(Default)" -Value $targetExe
    Set-ItemProperty -Path $appPathKey -Name "Path" -Value $installDir
} catch {
    Write-Host "Notice: App Paths registry entry skipped." -ForegroundColor Gray
}

# Windows Installed Apps registry entry
try {
    $uninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CleanAPO"
    if (-not (Test-Path $uninstallKey)) {
        New-Item -Path $uninstallKey -Force | Out-Null
    }
    Set-ItemProperty -Path $uninstallKey -Name "DisplayName" -Value "Clean APO"
    Set-ItemProperty -Path $uninstallKey -Name "DisplayVersion" -Value "1.1.0"
    if (Test-Path $targetIcon) {
        Set-ItemProperty -Path $uninstallKey -Name "DisplayIcon" -Value "$targetIcon,0"
    }
    Set-ItemProperty -Path $uninstallKey -Name "Publisher" -Value "pi-nf"
    Set-ItemProperty -Path $uninstallKey -Name "InstallLocation" -Value $installDir
} catch {
    Write-Host "Notice: Uninstall registry entry skipped." -ForegroundColor Gray
}

Write-Host "`nInstallation Succeeded!" -ForegroundColor Green
Write-Host "  • Desktop Shortcut:    $desktopPath" -ForegroundColor White
Write-Host "  • Start Menu Shortcut: $startMenuPath" -ForegroundColor White
Write-Host "`nYou can now search 'Clean APO' in your Start menu or launch it directly from your Desktop!" -ForegroundColor Green
