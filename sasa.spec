# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SASA — Shot Acoustic Spectral Analysis
Builds both macOS (.app) and Windows (.exe) standalone applications.

Entry point: app.py — a pure-Python HTTP + WebSocket server that serves the
web UI and bridges to the analysis backend. No Node.js required.

Usage:
    macOS:   pyinstaller sasa.spec
    Windows: pyinstaller sasa.spec

Requires PyInstaller >= 6.0. The bytecode-encryption ("cipher"/block_cipher)
scaffolding that used to live here was removed in PyInstaller 6 and has been
dropped; a.zipfiles / a.zipped_data are likewise always empty since 6.0.

NOTE ON SIGNING: nothing here signs or notarizes the output. See build_macos.sh
for the codesign/notarytool commands required before distributing SASA.app.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

PROJECT_ROOT = os.path.abspath(SPECPATH)


def _sasa_version() -> str:
    """
    The version the bundle reports to Finder, read from provenance.py.

    It was hardcoded to "1.0.0" and stayed there through every release, so a
    signed 2.4.1 build told Get Info, the installer and any management tool
    that it was 1.0.0 — and two different releases were indistinguishable to
    anything that reads the bundle rather than the app's own About panel.
    Imported textually rather than by importing the module, because a spec file
    is executed by PyInstaller before the package is importable.
    """
    source = Path(PROJECT_ROOT, 'provenance.py').read_text(encoding='utf-8')
    for line in source.splitlines():
        if line.startswith('__version__'):
            return line.split('=', 1)[1].strip().strip('\'"')
    raise SystemExit('sasa.spec: provenance.py has no __version__ to read')


SASA_VERSION = _sasa_version()

# Icon paths. NOTE: assets/ is *build-time only* — the icons are compiled into
# the bundle by BUNDLE/EXE below and nothing under assets/ is read at runtime,
# so the directory is deliberately not added to `datas`.
ICON_ICNS = os.path.join(PROJECT_ROOT, 'assets', 'sasa.icns')
ICON_ICO = os.path.join(PROJECT_ROOT, 'assets', 'sasa.ico')

# ── Data files ───────────────────────────────────────────────────────────────
# app.py serves the static UI from ui/renderer (see _find_renderer_dir()), and
# resolves the Python sources via _find_source_dir(); both look in _MEIPASS and,
# on macOS, in Contents/Resources. ui/server.js and ui/bridge are the Node
# development server and are NOT needed by the frozen app.
# The analysis modules are shipped as SOURCE next to the app because app.py
# resolves and runs them from _find_source_dir() rather than importing them all
# directly. That means PyInstaller's import analysis does NOT see them, so this
# list is the only thing putting them in the bundle -- and a module missing from
# it fails at runtime, not at build time.
#
# It is therefore derived from the filesystem rather than hand-maintained. The
# previous hand-written list had gone stale: report.py and provenance.py were
# both absent, and app.py imports report, so the frozen app would have raised
# ModuleNotFoundError on first use.
_MODULE_DIR = Path(SPECPATH)
_EXCLUDED_MODULES = {'sasa.spec'}  # nothing generated or build-only

_ANALYSIS_MODULES = sorted(
    p.name for p in _MODULE_DIR.glob('*.py')
    if p.name not in _EXCLUDED_MODULES and not p.name.startswith('_')
)

datas = [
    ('ui/renderer', 'ui/renderer'),
    ('LICENSE', '.'),
]
datas += [(name, '.') for name in _ANALYSIS_MODULES]

# Fail the BUILD rather than the running app if something the entry point needs
# is not being shipped.
for _required in ('main.py', 'app.py', 'report.py', 'provenance.py',
                  'metrics.py', 'calibration.py', 'weighting.py', 'bands.py',
                  'shot_detect.py', 'STFT.py', 'plots.py'):
    if _required not in _ANALYSIS_MODULES:
        raise SystemExit(f'sasa.spec: {_required} is missing from the bundle')

# plots.py calls fig.write_html() with the default include_plotlyjs=True, which
# inlines plotly/package_data/plotly.min.js. Without this data the interactive
# *_full.html outputs are produced but render blank. pyinstaller-hooks-contrib
# ships hook-plotly.py which does the same; collected explicitly so the build
# does not silently depend on the hook being present.
datas += collect_data_files('plotly', includes=['package_data/**/*.*'])

# Bundle the ffmpeg binary from imageio-ffmpeg for video support
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if _ffmpeg_exe and os.path.isfile(_ffmpeg_exe):
        # app.py looks for this in _MEIPASS/imageio_ffmpeg_bin (and, on macOS,
        # in Contents/Resources/imageio_ffmpeg_bin).
        datas.append((_ffmpeg_exe, 'imageio_ffmpeg_bin'))
except ImportError:
    print('sasa.spec: WARNING - imageio-ffmpeg not installed; the bundled app '
          'will not be able to read video files. Install with: pip install ".[video]"')

# ── Hidden imports ───────────────────────────────────────────────────────────
hiddenimports = [
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.signal.windows',   # STFT.get_window fallback
    'scipy.fft',
    'scipy.special',
    'soundfile',              # libsndfile itself is collected by hook-soundfile.py
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.colors',
    'matplotlib.figure',
    'matplotlib.backends.backend_agg',
    # plots.py calls matplotlib.use('Agg'), so PyInstaller's matplotlib hook
    # auto-collects Agg and nothing else. `--formats pdf,svg` makes savefig()
    # lazily import these two, which would then fail in the frozen app.
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_svg',
    'plotly',
    'plotly.graph_objects',
    'plotly.io',
    'narwhals',               # plotly >= 6 dataframe shim, imported lazily
    'imageio_ffmpeg',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'json',
    'csv',
    'dataclasses',
    'pathlib',
    'hashlib',
    'struct',
    'webbrowser',
    'http.server',
    'threading',
    'signal',
    'subprocess',
]

# moviepy is imported by ExtractAudio.py (`from moviepy import VideoFileClip`).
# main.py wraps that import in try/except ImportError and silently sets
# _VIDEO_SUPPORT = False, so a missing moviepy does not fail the build - it just
# makes video input quietly unavailable in the shipped app. Collect it eagerly.
try:
    import moviepy  # noqa: F401
    hiddenimports += collect_submodules('moviepy')
except ImportError:
    print('sasa.spec: WARNING - moviepy not installed; video input will be '
          'DISABLED in the bundled app. Install with: pip install ".[video]"')

# Optional ISO 532-1 loudness backend. Not imported by any module today; picked
# up only so a future import does not require a spec change.
try:
    import mosqito  # noqa: F401
    hiddenimports.append('mosqito')
except ImportError:
    pass

a = Analysis(
    ['app.py'],  # Entry point: the Python HTTP server
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

if IS_MACOS:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='SASA',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        # UPX is not used on macOS: it corrupts arm64 Mach-O binaries and
        # invalidates any code signature applied afterwards.
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        # Left None deliberately - signing is a separate, credentialed step.
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='SASA',
    )
    app = BUNDLE(
        coll,
        name='SASA.app',
        icon=ICON_ICNS if os.path.exists(ICON_ICNS) else None,
        bundle_identifier='com.ridgebackdefense.sasa',
        info_plist={
            'CFBundleDisplayName': 'SASA',
            'CFBundleShortVersionString': SASA_VERSION,
            'CFBundleVersion': SASA_VERSION,
            'NSHighResolutionCapable': True,
            'LSBackgroundOnly': False,
            'LSUIElement': True,  # App runs as agent (no dock icon bouncing)
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'WAV Audio',
                    'CFBundleTypeExtensions': ['wav'],
                    'CFBundleTypeRole': 'Viewer',
                },
            ],
        },
    )
else:
    # Windows — single-file exe
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='SASA',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=ICON_ICO if os.path.exists(ICON_ICO) else None,
    )
