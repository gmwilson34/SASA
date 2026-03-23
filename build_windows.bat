@echo off
REM ──────────────────────────────────────────────────────────────
REM SASA — Windows Build Script
REM Builds a standalone .exe using PyInstaller
REM
REM Usage:
REM   build_windows.bat
REM
REM Prerequisites:
REM   pip install -r requirements.txt pyinstaller
REM ──────────────────────────────────────────────────────────────

echo ======================================================
echo    SASA — Windows Build
echo    Ridgeback Defense
echo ======================================================
echo.

cd /d "%~dp0"

REM ── 1. Check Python ──
echo [1/5] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo.

REM ── 2. Create/activate venv ──
echo [2/5] Setting up virtual environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo.

REM ── 3. Install dependencies ──
echo [3/5] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

REM Install Node.js dependencies for UI
if exist "ui\package.json" (
    echo   Installing UI dependencies...
    where npm >nul 2>nul
    if not errorlevel 1 (
        cd ui
        call npm install --production
        cd ..
    ) else (
        echo   WARNING: npm not found. Install Node.js for full UI support.
    )
)
echo.

REM ── 4. Build with PyInstaller ──
echo [4/5] Building Windows executable...
pyinstaller sasa.spec --noconfirm --clean
echo.

REM ── 5. Report ──
echo [5/5] Build complete!
echo.

if exist "dist\SASA.exe" (
    echo   Executable: dist\SASA.exe
    echo.
    echo   To run: dist\SASA.exe
) else (
    echo   Standalone: dist\SASA\
    echo   To run:     dist\SASA\SASA.exe
)

echo.
echo Done.
pause
