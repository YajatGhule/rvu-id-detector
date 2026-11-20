# RVU ID Card Detector - Windows installer
#
# What it does:
#   1. installs Git and uv if they are missing (uses winget, which comes with Windows 10 and 11)
#   2. gets the project (clones it if you only downloaded this script)
#   3. downloads Python 3.12 and every package with "uv sync"
#   4. checks that the YOLO model file is there and runs "idcard doctor"
#
# How to run (PowerShell, no administrator needed):
#   powershell -ExecutionPolicy Bypass -File .\install-windows.ps1
#
# Intel Arc graphics (optional, slower to set up, not tested on this project):
#   powershell -ExecutionPolicy Bypass -File .\install-windows.ps1 -Openvino
#
# After it finishes, double click start-windows.bat to run the website.

param(
    [switch]$Openvino,
    [string]$Repo = "https://github.com/YajatGhule/rvu-id-detector",
    [string]$Branch = "web-refactor"
)

$ErrorActionPreference = "Stop"

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Have($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
# run a program and stop if it fails (PowerShell does not do this by itself)
function Run {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" }
}
function RefreshPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user;$HOME\.local\bin"
}

# ---- 1. Git and uv ----------------------------------------------------------------------
Step "Checking Git and uv"
if (-not (Have "git")) {
    if (-not (Have "winget")) { throw "Git is missing and winget is not available. Install Git from https://git-scm.com and run this script again." }
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    RefreshPath
}
if (-not (Have "uv")) {
    if (Have "winget") {
        winget install --id astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements
    } else {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    }
    RefreshPath
}
if (-not (Have "uv")) { throw "uv did not install. Close this window, open a new PowerShell and run the script again." }
Write-Host "git: $(git --version)"
Write-Host "uv:  $(uv --version)"

# ---- 2. The project folder ---------------------------------------------------------------
Step "Finding the project"
if (Test-Path (Join-Path $PSScriptRoot "pyproject.toml")) {
    $dir = $PSScriptRoot
} else {
    $dir = Join-Path $HOME "rvu-id-detector"
    if (-not (Test-Path $dir)) {
        git clone --branch $Branch $Repo $dir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Branch $Branch not found on GitHub, using the main branch." -ForegroundColor Yellow
            Run git clone $Repo $dir
        }
    }
}
Set-Location $dir
Write-Host "project folder: $dir"

# ---- 3. Python and packages --------------------------------------------------------------
Step "Installing Python 3.12 and the packages (a few minutes the first time)"
Run uv sync

# ---- 4. The model -----------------------------------------------------------------------
Step "Checking the model file"
$model = Join-Path $dir "models\yolo11n-pose.onnx"
if (-not (Test-Path $model)) { throw "models\yolo11n-pose.onnx is missing. Run: git checkout $Branch" }
Write-Host "model found: $model ($([math]::Round((Get-Item $model).Length / 1MB, 1)) MB)"

# ---- 5. Optional Intel Arc support ---------------------------------------------------------
if ($Openvino) {
    Step "Adding OpenVINO for Intel graphics"
    try {
        Run uv pip uninstall onnxruntime-directml
        Run uv pip install onnxruntime-openvino
        Run uv run --no-sync idcard doctor
    } catch {
        Write-Host "OpenVINO did not work, going back to the normal CPU version." -ForegroundColor Yellow
        Run uv sync --reinstall-package onnxruntime-directml
    }
}

# ---- 6. Check that it runs ---------------------------------------------------------------
Step "Testing the model on this laptop"
Run uv run --no-sync idcard doctor

Write-Host ""
Write-Host "Done. Double click start-windows.bat to open the website." -ForegroundColor Green
Write-Host "Everything runs on this laptop. Nothing is sent to the internet."
