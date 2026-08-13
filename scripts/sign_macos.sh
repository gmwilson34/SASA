#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SASA — sign, notarize and staple the macOS app, then produce the shipping zip.
#
# Called by build_macos.sh and by the CI workflow, so there is one description
# of how a release is signed rather than two that can drift.
#
# Nothing here is conditional on being "the release build". It signs whenever
# the pieces are present and says exactly which piece is missing when they are
# not, because an unsigned build that SAYS it is unsigned is recoverable, and
# one that quietly ships is what puts "SASA.app is damaged" in front of a
# customer.
#
# What Gatekeeper actually requires, on macOS 10.15 and later:
#   1. a "Developer ID Application" certificate — NOT "Apple Development" and
#      NOT "Apple Distribution", neither of which Gatekeeper accepts for an app
#      distributed outside the App Store;
#   2. the hardened runtime (--options runtime) and a secure timestamp;
#   3. notarization by Apple, and the resulting ticket stapled to the bundle so
#      a machine with no network still passes.
# Missing any one of the three and the download is blocked.
#
# Environment:
#   SASA_SIGN_IDENTITY     Full identity name. Default: the first
#                          "Developer ID Application" identity in the keychain.
#   SASA_NOTARY_PROFILE    notarytool keychain profile. Default: SASA-NOTARY.
#   SASA_NOTARY_APPLE_ID   \
#   SASA_NOTARY_PASSWORD    > used instead of the profile, for CI.
#   SASA_NOTARY_TEAM_ID    /
#   SASA_APP               Bundle to sign. Default: dist/SASA.app
#   SASA_ZIP               Zip to produce. Default: SASA-macOS.zip
#   SASA_ENTITLEMENTS      Entitlements plist. Default: build/sasa.entitlements
#   SASA_REQUIRE_SIGNED    "1" makes a missing certificate a hard failure
#                          instead of a warning. Use it on release tags.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP="${SASA_APP:-dist/SASA.app}"
ZIP="${SASA_ZIP:-SASA-macOS.zip}"
PROFILE="${SASA_NOTARY_PROFILE:-SASA-NOTARY}"
ENTITLEMENTS="${SASA_ENTITLEMENTS:-build/sasa.entitlements}"
REQUIRE="${SASA_REQUIRE_SIGNED:-0}"

say()  { printf '  %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }

fail_or_warn() {
    if [ "$REQUIRE" = "1" ]; then
        warn "$1"
        warn "SASA_REQUIRE_SIGNED=1, so this is a failure rather than a warning."
        exit 1
    fi
    warn "$1"
}

# ── The bundle ────────────────────────────────────────────────────────────────

if [ ! -d "$APP" ]; then
    # A one-file build produces dist/SASA rather than a bundle. There is nothing
    # here that can be notarized, and saying so is more use than a stack trace.
    warn "No app bundle at $APP — nothing to sign."
    warn "Only a .app (or a DMG/PKG containing one) can be notarized."
    exit 0
fi

# ── The certificate ───────────────────────────────────────────────────────────

IDENTITY="${SASA_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
    IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
        | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
fi

if [ -z "$IDENTITY" ]; then
    fail_or_warn "No \"Developer ID Application\" certificate in the keychain; the app cannot be signed."
    cat <<'NO_CERT'

  Identities that WILL NOT work for a downloaded app, and why:

    Apple Development   — for running on your own registered devices.
    Apple Distribution  — for App Store submission and enterprise/ad-hoc.

  Gatekeeper accepts neither for direct distribution. You need a separate
  "Developer ID Application" certificate:

    1. https://developer.apple.com/account/resources/certificates/add
    2. Choose "Developer ID Application".
       (Only the team's Account Holder may create one — an Admin cannot.)
    3. Download it and double-click to install into the login keychain.

  Then re-run this script. It will sign, notarize and staple without any
  further changes.

NO_CERT
    exit 0
fi

say "Signing identity: $IDENTITY"

# ── Entitlements ──────────────────────────────────────────────────────────────
#
# Deliberately minimal: the hardened runtime with no exceptions. Every
# entitlement here is a security check switched off, so none is added on
# suspicion.
#
# If the SIGNED app refuses to launch with a code-signing or library-loading
# error, the entitlement that class of failure usually needs is
# com.apple.security.cs.disable-library-validation — PyInstaller relocates and
# dlopens dylibs from inside the bundle. Add it only if you actually see that
# failure, and note that this script re-signs every nested Mach-O with the same
# identity, which is what normally makes it unnecessary.
if [ ! -f "$ENTITLEMENTS" ]; then
    mkdir -p "$(dirname "$ENTITLEMENTS")"
    cat > "$ENTITLEMENTS" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
</dict>
</plist>
PLIST
fi

# ── Sign ──────────────────────────────────────────────────────────────────────
#
# Every nested Mach-O first, the bundle last. NOT --deep: Apple documents that
# as unsuitable here because it applies one set of entitlements to everything it
# finds and skips things it does not recognise as code.

# Extended attributes first. codesign refuses any file carrying a resource
# fork or Finder information — "resource fork, Finder information, or similar
# detritus not allowed" — and this repository lives inside a Google Drive
# folder, which attaches them to everything it syncs. Without this the very
# first binary fails and the cause is not obvious from the message.
say "Clearing extended attributes..."
xattr -cr "$APP"

say "Signing nested binaries..."
signed=0
while IFS= read -r -d '' target; do
    if file -b "$target" | grep -q 'Mach-O'; then
        # Errors are shown, not swallowed. A signing failure that reports only
        # the filename leaves you guessing at which of a dozen causes it was.
        if ! codesign --force --options runtime --timestamp \
                      --entitlements "$ENTITLEMENTS" \
                      --sign "$IDENTITY" "$target"; then
            warn "could not sign $target — see the codesign error above"
            exit 1
        fi
        signed=$((signed + 1))
    fi
done < <(find "$APP/Contents" -type f -print0)
say "Signed $signed nested binaries."

say "Signing the bundle..."
codesign --force --options runtime --timestamp \
         --entitlements "$ENTITLEMENTS" \
         --sign "$IDENTITY" "$APP"

say "Verifying..."
codesign --verify --deep --strict --verbose=2 "$APP"

# ── Package ───────────────────────────────────────────────────────────────────
#
# ditto, not zip: notarization needs the bundle's symlinks and extended
# attributes intact, and this is the archiver Apple's own instructions use.

say "Packaging $ZIP..."
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

# ── Notarize ──────────────────────────────────────────────────────────────────

notary_args=()
if [ -n "${SASA_NOTARY_APPLE_ID:-}" ] && [ -n "${SASA_NOTARY_PASSWORD:-}" ] \
   && [ -n "${SASA_NOTARY_TEAM_ID:-}" ]; then
    notary_args=(--apple-id "$SASA_NOTARY_APPLE_ID"
                 --password "$SASA_NOTARY_PASSWORD"
                 --team-id  "$SASA_NOTARY_TEAM_ID")
elif xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    notary_args=(--keychain-profile "$PROFILE")
fi

if [ ${#notary_args[@]} -eq 0 ]; then
    fail_or_warn "Signed, but NOT notarized — no notarization credentials."
    cat <<NO_NOTARY

  A signed but un-notarized app is still blocked on first launch, though the
  message becomes the accurate "cannot be opened because Apple cannot check it
  for malicious software" rather than "is damaged".

  Store credentials once, with an app-specific password from
  https://appleid.apple.com (Sign-In and Security > App-Specific Passwords):

    xcrun notarytool store-credentials "$PROFILE" \\
        --apple-id "you@ridgebackdefense.com" \\
        --team-id  "YOURTEAMID" \\
        --password "abcd-efgh-ijkl-mnop"

  Then re-run this script.

NO_NOTARY
    exit 0
fi

say "Submitting for notarization (this usually takes a few minutes)..."
xcrun notarytool submit "$ZIP" "${notary_args[@]}" --wait

say "Stapling the ticket..."
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"

# The zip submitted for notarization does not contain the ticket, so the one
# that ships has to be made again from the stapled bundle. Skipping this is the
# classic way to notarize successfully and still ship something that fails on a
# machine with no network.
say "Re-packaging the stapled bundle..."
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

# ── The check that matters ────────────────────────────────────────────────────
#
# This is what Gatekeeper itself will do on the customer's machine.

say "Assessing as Gatekeeper will..."
spctl --assess --type execute --verbose=4 "$APP"

say "Signed, notarized and stapled: $ZIP"
