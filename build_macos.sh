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
# ⚠️  THE OUTPUT IS UNSIGNED AND UN-NOTARIZED.
#     It runs on the machine that built it and nowhere else without manual
#     intervention. See the "CODE SIGNING" section printed at the end of this
#     script for the exact commands. This script deliberately does not attempt
#     to sign anything — that needs a Developer ID certificate and credentials.
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

cat <<'SIGNING_NOTICE'

╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  UNSIGNED / UN-NOTARIZED BUILD                                    ║
╚══════════════════════════════════════════════════════════════════════╝

This build is NOT code-signed and NOT notarized.

  • It will launch on THIS machine.
  • On any other Mac, Gatekeeper will block it. Because the .zip carries the
    com.apple.quarantine attribute, the user sees
    "SASA.app is damaged and can't be opened. You should move it to the Trash."
    That message means "unsigned", not "corrupt".
  • Right-click > Open does NOT work around this for an app that was never
    signed at all — the ad-hoc workaround is:
        xattr -dr com.apple.quarantine /Applications/SASA.app

To ship this properly you need an Apple Developer Program membership and a
"Developer ID Application" certificate in your login keychain. Then:

  # 1. Sign every nested binary, then the bundle, with a hardened runtime.
  #    PyInstaller bundles dylibs/.so files that must each be signed.
  codesign --force --deep --options runtime --timestamp \
      --sign "Developer ID Application: Ridgeback Defense (TEAMID)" \
      dist/SASA.app

  # 2. Verify before submitting.
  codesign --verify --deep --strict --verbose=2 dist/SASA.app

  # 3. Notarization requires a ZIP (or DMG/PKG), not a bare .app.
  ditto -c -k --keepParent dist/SASA.app SASA-macOS.zip

  # 4. Store credentials once (app-specific password from appleid.apple.com):
  xcrun notarytool store-credentials "SASA-NOTARY" \
      --apple-id "you@ridgebackdefense.com" \
      --team-id  "TEAMID" \
      --password "abcd-efgh-ijkl-mnop"

  # 5. Submit and wait for the result.
  xcrun notarytool submit SASA-macOS.zip --keychain-profile "SASA-NOTARY" --wait

  # 6. Staple the ticket to the .app, then re-zip THAT for distribution.
  xcrun stapler staple dist/SASA.app
  xcrun stapler validate dist/SASA.app
  ditto -c -k --keepParent dist/SASA.app SASA-macOS.zip

  # 7. Final check - this is what Gatekeeper will do on the user's machine.
  spctl --assess --type execute --verbose=4 dist/SASA.app

Replace TEAMID with your 10-character Apple Developer Team ID
(`xcrun notarytool store-credentials` will not accept a placeholder).

SIGNING_NOTICE

echo "Done."
