@echo off
echo ==========================================
echo   Rack Inventory Scanner
echo ==========================================
echo.

cd /d "%~dp0"

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Check arguments
if "%1"=="scan" (
    echo Scanning for cameras...
    python run_rack_inventory.py --scan
) else if "%1"=="setup" (
    echo Running setup wizard...
    python run_rack_inventory.py --setup
) else (
    echo Starting rack inventory viewer...
    python run_rack_inventory.py %*
)

pause
