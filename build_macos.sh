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
#   pip install '.[video,build]'
#
# Signing:
#   The build signs, notarizes and staples automatically when a
#   "Developer ID Application" certificate and notarization credentials are
#   present, via scripts/sign_macos.sh. When they are not, it says which piece
#   is missing and leaves an unsigned build — it never ships something that
#   claims to be signed and is not.
#
#   SASA_SKIP_SIGN=1   build only, do not attempt to sign.
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
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "[2/5] Already in virtual environment: $VIRTUAL_ENV"
fi

# ── 3. Install dependencies ──
# The [video] extra (moviepy + imageio-ffmpeg) matters at BUILD time, not just
# at run time: sasa.spec only bundles ffmpeg and the moviepy submodules if they
# are importable while PyInstaller runs. Building without them produces an app
# that silently refuses video input.
echo "[3/5] Installing dependencies..."
pip install --upgrade pip
pip install '.[video,build]'

# The frozen app embeds a pure-Python HTTP server (app.py) and does not use
# ui/server.js, so Node.js is not required for the build. `npm install` is only
# needed if you want to run the Node development server (`node ui/server.js`).

# ── 4. Build with PyInstaller ──
echo "[4/6] Building macOS app..."
pyinstaller sasa.spec --noconfirm --clean

# ── 5. Report ──
echo ""
echo "[5/6] Build complete!"
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

# ── 6. Sign, notarize and staple ──
#
# One description of how a release is signed, shared with the CI workflow, so
# the two cannot drift. The script explains what is missing when it cannot
# finish, and exits 0 so an unsigned local build is still usable.
if [ "${SASA_SKIP_SIGN:-0}" = "1" ]; then
    echo ""
    echo "  Signing skipped (SASA_SKIP_SIGN=1)."
else
    echo ""
    echo "[6/6] Signing..."
    "$SCRIPT_DIR/scripts/sign_macos.sh"
fi

echo "Done."
