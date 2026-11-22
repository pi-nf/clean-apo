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

# If dist\CleanAPO.exe does not exist, build it first
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

# Stop any running CleanAPO instance so CleanAPO.exe is not locked during overwrite
$runningProcs = Get-Process -Name "CleanAPO" -ErrorAction SilentlyContinue
if ($runningProcs) {
    Write-Host "Stopping running Clean APO instance(s)..." -ForegroundColor Yellow
    foreach ($proc in $runningProcs) {
        try {
            $null = $proc.CloseMainWindow()
        } catch {
            # Ignore if process has no main window
        }
    }
    Start-Sleep -Milliseconds 600
    $stillRunning = Get-Process -Name "CleanAPO" -ErrorAction SilentlyContinue
    if ($stillRunning) {
        try {
            $stillRunning | Stop-Process -Force -ErrorAction Stop
            Start-Sleep -Milliseconds 300
        } catch {
            Write-Error "Clean APO is currently running and could not be closed automatically. Please close Clean APO and re-run install.ps1."
            Exit 1
        }
    }
}

if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

$targetExe = Join-Path $installDir "CleanAPO.exe"
$targetIcon = Join-Path $installDir "clean_apo.ico"
$uninstallScript = Join-Path $installDir "uninstall.ps1"

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

# Generate Clean APO uninstaller script (preserves %APPDATA%\CleanAPO user presets)
$uninstallContent = @'
$ErrorActionPreference = "SilentlyContinue"
$installDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Get-Process -Name "CleanAPO" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 300

$desktopLnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Clean APO.lnk"
$startMenuLnk = Join-Path ([Environment]::GetFolderPath("Programs")) "Clean APO.lnk"
if (Test-Path $desktopLnk) { Remove-Item $desktopLnk -Force }
if (Test-Path $startMenuLnk) { Remove-Item $startMenuLnk -Force }

Remove-Item -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\App Paths\CleanAPO.exe" -Recurse -Force
Remove-Item -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CleanAPO" -Recurse -Force

$exeFile = Join-Path $installDir "CleanAPO.exe"
$icoFile = Join-Path $installDir "clean_apo.ico"
if (Test-Path $exeFile) { Remove-Item $exeFile -Force }
if (Test-Path $icoFile) { Remove-Item $icoFile -Force }

# Schedule removal of uninstall.ps1 and empty install folder after this script exits
$escapedDir = $installDir.Replace('"', '""')
Start-Process -FilePath "cmd.exe" -ArgumentList "/c ping 127.0.0.1 -n 2 >nul & rmdir /s /q `"$escapedDir`"" -WindowStyle Hidden
'@
Set-Content -Path $uninstallScript -Value $uninstallContent -Encoding UTF8

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
    $uninstallCmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$uninstallScript`""
    Set-ItemProperty -Path $uninstallKey -Name "DisplayName" -Value "Clean APO"
    Set-ItemProperty -Path $uninstallKey -Name "DisplayVersion" -Value "1.4.2"
    if (Test-Path $targetIcon) {
        Set-ItemProperty -Path $uninstallKey -Name "DisplayIcon" -Value "$targetIcon,0"
    }
    Set-ItemProperty -Path $uninstallKey -Name "Publisher" -Value "pi-nf"
    Set-ItemProperty -Path $uninstallKey -Name "InstallLocation" -Value $installDir
    Set-ItemProperty -Path $uninstallKey -Name "UninstallString" -Value $uninstallCmd
    Set-ItemProperty -Path $uninstallKey -Name "QuietUninstallString" -Value $uninstallCmd
    Set-ItemProperty -Path $uninstallKey -Name "NoModify" -Type DWord -Value 1
    Set-ItemProperty -Path $uninstallKey -Name "NoRepair" -Type DWord -Value 1
} catch {
    Write-Host "Notice: Uninstall registry entry skipped." -ForegroundColor Gray
}

Write-Host "`nInstallation Succeeded!" -ForegroundColor Green
Write-Host "  - Desktop Shortcut:    $desktopPath" -ForegroundColor White
Write-Host "  - Start Menu Shortcut: $startMenuPath" -ForegroundColor White
Write-Host "`nYou can now search 'Clean APO' in your Start menu or launch it directly from your Desktop!" -ForegroundColor Green
