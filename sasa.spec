# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SASA — Shot Acoustic Spectral Analysis
Builds both macOS (.app) and Windows (.exe) standalone applications.

Entry point: app.py — a pure-Python HTTP + WebSocket server that serves the
web UI and bridges to the analysis backend. No Node.js required.

Usage:
    macOS:   pyinstaller sasa.spec
    Windows: pyinstaller sasa.spec
"""

import sys
import os

block_cipher = None

IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

PROJECT_ROOT = os.path.abspath(SPECPATH)

# Icon paths
ICON_ICNS = os.path.join(PROJECT_ROOT, 'assets', 'sasa.icns')
ICON_ICO = os.path.join(PROJECT_ROOT, 'assets', 'sasa.ico')

# Data files to bundle — UI static assets + all Python source modules
# (Python sources are bundled as data so app.py can spawn `python main.py`)
datas = [
    ('ui/renderer', 'ui/renderer'),
    # Bundle all analysis Python modules alongside the app
    ('main.py', '.'),
    ('calibration.py', '.'),
    ('weighting.py', '.'),
    ('shot_detect.py', '.'),
    ('bands.py', '.'),
    ('metrics.py', '.'),
    ('plots.py', '.'),
    ('STFT.py', '.'),
    ('WavLoader.py', '.'),
    ('WaveformPlot.py', '.'),
    ('SignalGenerator.py', '.'),
    ('ExtractAudio.py', '.'),
    ('FileSelector.py', '.'),
]

# Bundle the ffmpeg binary from imageio-ffmpeg for video support
try:
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if _ffmpeg_exe and os.path.isfile(_ffmpeg_exe):
        # Bundle ffmpeg binary into a 'ffmpeg' directory
        datas.append((_ffmpeg_exe, 'imageio_ffmpeg_bin'))
except ImportError:
    pass

hiddenimports = [
    'numpy',
    'numpy.core',
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'soundfile',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.colors',
    'matplotlib.figure',
    'matplotlib.backends.backend_agg',
    'plotly',
    'plotly.graph_objects',
    'plotly.io',
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

try:
    import mosqito
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
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
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
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
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
        a.zipfiles,
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
