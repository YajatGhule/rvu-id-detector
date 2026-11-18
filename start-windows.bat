@echo off
rem Starts the RVU ID Card Detector and opens the website in the browser.
rem Close this window to stop the detector.
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed. Run install-windows.ps1 first.
  pause
  exit /b 1
)
rem open the browser after 8 seconds, once the server is up
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8000"
uv run --no-sync idcard serve
pause
