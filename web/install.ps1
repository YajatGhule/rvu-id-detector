# RVU ID Card Detector - one line Windows installer (Node.js version)
#
# Paste this into PowerShell (no administrator needed):
#   irm https://yajatghule.github.io/rvu-id-detector/install.ps1 | iex
#
# What it does:
#   1. uses Node.js if it is installed, otherwise downloads the official portable Node.js
#      (signed by the OpenJS Foundation) into your user folder
#   2. downloads this project from GitHub into %USERPROFILE%\rvu-id-detector
#   3. puts a "RVU ID Card Detector" shortcut on the Desktop
#   4. starts the local server and opens http://localhost:3000 in the browser
# Nothing is installed for other users and no settings are changed. To remove it, delete the
# rvu-id-detector folder, the node folder inside it and the Desktop shortcut.

& {
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"   # makes downloads much faster in Windows PowerShell
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    $dir = Join-Path $HOME "rvu-id-detector"
    $zipUrl = "https://github.com/YajatGhule/rvu-id-detector/archive/refs/heads/main.zip"

    function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

    try {
        # ---- 1. Node.js ---------------------------------------------------------------------
        Step "Checking Node.js"
        $node = $null
        $found = Get-Command node -ErrorAction SilentlyContinue
        if ($found) {
            $node = $found.Source
        } else {
            $portable = Get-ChildItem (Join-Path $dir "node") -Filter node.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($portable) { $node = $portable.FullName }
        }
        if (-not $node) {
            Write-Host "Node.js not found, downloading the portable LTS version from nodejs.org"
            $releases = Invoke-RestMethod "https://nodejs.org/dist/index.json"
            $lts = $releases | Where-Object { $_.lts -and ($_.files -contains "win-x64-zip") } | Select-Object -First 1
            $name = "node-$($lts.version)-win-x64"
            $nodeZip = Join-Path $env:TEMP "$name.zip"
            Invoke-WebRequest "https://nodejs.org/dist/$($lts.version)/$name.zip" -OutFile $nodeZip
            New-Item -ItemType Directory -Force (Join-Path $dir "node") | Out-Null
            Expand-Archive $nodeZip -DestinationPath (Join-Path $dir "node") -Force
            Remove-Item $nodeZip
            $node = Join-Path $dir "node\$name\node.exe"
        }
        Write-Host "node: $node ($(& $node --version))"

        # ---- 2. The project -----------------------------------------------------------------
        Step "Downloading the project from GitHub"
        $zip = Join-Path $env:TEMP "rvu-id-detector-main.zip"
        $unpack = Join-Path $env:TEMP "rvu-id-detector-unpack"
        Invoke-WebRequest $zipUrl -OutFile $zip
        if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
        Expand-Archive $zip -DestinationPath $unpack -Force
        $src = Join-Path $unpack "rvu-id-detector-main"
        New-Item -ItemType Directory -Force $dir | Out-Null
        # copy only what the website needs; a node folder from step 1 is left alone
        foreach ($part in @("web", "node-server", "README.md", "LICENSE")) {
            $target = Join-Path $dir $part
            if (Test-Path $target) { Remove-Item $target -Recurse -Force }
            Copy-Item (Join-Path $src $part) $target -Recurse
        }
        Remove-Item $zip
        Remove-Item $unpack -Recurse -Force
        Write-Host "project folder: $dir"

        # ---- 3. Desktop shortcut ------------------------------------------------------------
        Step "Making the Desktop shortcut"
        $server = Join-Path $dir "node-server\server.js"
        $desktop = [Environment]::GetFolderPath("Desktop")
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut((Join-Path $desktop "RVU ID Card Detector.lnk"))
        $link.TargetPath = $node
        $link.Arguments = "`"$server`""
        $link.WorkingDirectory = $dir
        $link.Description = "Starts the RVU ID Card Detector on http://localhost:3000"
        $link.Save()
        Write-Host "Shortcut 'RVU ID Card Detector' is on the Desktop."

        # ---- 4. Start it ----------------------------------------------------------------------
        Step "Starting the detector at http://localhost:3000"
        Start-Process -FilePath $node -ArgumentList "`"$server`"" -WorkingDirectory $dir
        Write-Host ""
        Write-Host "Done. In the browser press 'Use my camera' and allow the camera." -ForegroundColor Green
        Write-Host "Next time just open the Desktop shortcut. Close the black server window to stop it."
    } catch {
        Write-Host ""
        Write-Host "Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Check the internet connection and run the command again."
    }
}
