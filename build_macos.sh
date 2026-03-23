#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SASA — macOS Build Script
# Builds a standalone .app bundle using PyInstaller
#
# Usage:
#   chmod +x build_macos.sh
#   ./build_macos.sh
#
# Prerequisites:
#   pip install -r requirements.txt pyinstaller
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════╗"
echo "║   SASA — macOS Build                             ║"
echo "║   Ridgeback Defense                              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ──
PYTHON="${PYTHON:-python3}"
echo "[1/5] Checking Python..."
$PYTHON --version
echo ""

# ── 2. Create/activate venv if not already in one ──
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[2/5] Setting up virtual environment..."
    if [ ! -d ".venv" ]; then
        $PYTHON -m venv .venv
    fi
    source .venv/bin/activate
else
    echo "[2/5] Already in virtual environment: $VIRTUAL_ENV"
fi

# ── 3. Install dependencies ──
echo "[3/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Install Node.js dependencies for the UI
if [ -d "ui" ] && [ -f "ui/package.json" ]; then
    echo "  Installing UI dependencies..."
    if command -v npm &> /dev/null; then
        cd ui && npm install --production && cd ..
    else
        echo "  WARNING: npm not found. UI server dependencies not installed."
        echo "  Install Node.js from https://nodejs.org/ for full UI support."
    fi
fi

# ── 4. Build with PyInstaller ──
echo "[4/5] Building macOS app..."
pyinstaller sasa.spec --noconfirm --clean

# ── 5. Report ──
echo ""
echo "[5/5] Build complete!"
echo ""

if [ -d "dist/SASA.app" ]; then
    APP_SIZE=$(du -sh "dist/SASA.app" | cut -f1)
    echo "  App:  dist/SASA.app ($APP_SIZE)"
    echo ""
    echo "  To run:  open dist/SASA.app"
    echo "  To distribute: zip -r SASA-macOS.zip dist/SASA.app"
else
    echo "  Standalone: dist/SASA/"
    echo "  To run:     ./dist/SASA/SASA"
fi

echo ""
echo "Done."
