# RVU ID Card Detector - runs after RVU-ID-Detector-Setup.exe has copied the project here.
# It runs the normal installer from this folder and puts a shortcut on the Desktop.
$ErrorActionPreference = "Stop"
$dir = Split-Path $PSScriptRoot -Parent
Set-Location $dir

try {
    & (Join-Path $dir "install-windows.ps1") @args

    $desktop = [Environment]::GetFolderPath("Desktop")
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path $desktop "RVU ID Card Detector.lnk"))
    $link.TargetPath = Join-Path $dir "start-windows.bat"
    $link.WorkingDirectory = $dir
    $link.Save()
    Write-Host ""
    Write-Host "A shortcut called 'RVU ID Card Detector' is now on the Desktop." -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check the internet connection and run the setup again."
    exit 1
}
