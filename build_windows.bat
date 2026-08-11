@echo off
REM ──────────────────────────────────────────────────────────────
REM SASA — Windows Build Script
REM Builds a standalone .exe using PyInstaller
REM
REM Usage:
REM   build_windows.bat
REM
REM Prerequisites:
REM   pip install ".[video,build]"
REM
REM NOTE: the resulting SASA.exe is NOT Authenticode-signed. SmartScreen will
REM show "Windows protected your PC" on other machines until it is signed with
REM a code-signing certificate (signtool sign /fd SHA256 /tr <timestamp-url>
REM /td SHA256 /f cert.pfx /p <password> dist\SASA.exe). This script does not
REM attempt to sign anything.
REM ──────────────────────────────────────────────────────────────

echo ======================================================
echo    SASA - Windows Build
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
REM The [video] extra (moviepy + imageio-ffmpeg) is needed at BUILD time:
REM sasa.spec only bundles ffmpeg and the moviepy submodules if they are
REM importable while PyInstaller runs. Building without them produces an exe
REM that silently refuses video input.
echo [3/5] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install ".[video,build]"
if errorlevel 1 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
)
echo.

REM The frozen app embeds a pure-Python HTTP server (app.py) and does not use
REM ui\server.js, so Node.js is NOT required for the build. Run `npm install`
REM in ui\ only if you want the Node development server.

REM ── 4. Build with PyInstaller ──
echo [4/5] Building Windows executable...
pyinstaller sasa.spec --noconfirm --clean
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)
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
echo   WARNING: this build is unsigned. SmartScreen will warn users on first
echo            run until an Authenticode signature is applied.
echo.
echo Done.
pause
