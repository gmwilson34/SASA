#!/usr/bin/env python3
"""
SASA Application Server — Pure Python

Self-contained HTTP + WebSocket server. This is the entry point for the packaged
macOS/Windows app.

  - Serves the web UI (static HTML/CSS/JS from ui/renderer/)
  - Streams uploads to disk, lists analyses, serves result files
  - Runs analysis IN-PROCESS via analyze_file() with WebSocket progress streaming
  - Opens the default browser on startup and stays alive until closed

Measurement-integrity rules enforced here (they are not cosmetic):

  * The GUI and the CLI must measure the SAME THING. Audio extracted from a video
    container keeps the source sample rate, bit depth and channel count; nothing is
    resampled or down-mixed on the way in, because an impulse destroyed at import
    cannot be recovered by any later stage.
  * Calibration never silently defaults. If the operator asked for a calibrated
    measurement and did not supply every number it needs, the run is REFUSED with a
    message naming the missing field. A wrong Pa/FS produces confident, wrong dB SPL.
  * Every served path is confined to an allow-list of roots, resolved through
    symlinks. The local server is not a file-read oracle.

No external dependencies beyond the Python standard library + the SASA modules.
"""

from __future__ import annotations

import atexit
import base64
import dataclasses
import hashlib
import http.server
import inspect
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


# ═══════════════════════════════════════════════════════════
#  Safe I/O — handle sys.stdout/stderr being None
#  (PyInstaller console=False sets them to None on macOS/Windows)
# ═══════════════════════════════════════════════════════════

_LOG_FILE = None


def _setup_logging():
    """Set up a log file and fix None stdout/stderr for the packaged app."""
    global _LOG_FILE
    log_dir = Path.home() / '.sasa' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'sasa.log'
    _LOG_FILE = open(log_path, 'a', encoding='utf-8', buffering=1)

    # If stdout/stderr are None (PyInstaller console=False), redirect to the log file
    if sys.stdout is None or not hasattr(sys.stdout, 'write'):
        sys.stdout = _LOG_FILE
    if sys.stderr is None or not hasattr(sys.stderr, 'write'):
        sys.stderr = _LOG_FILE


def _log(msg: str):
    """Write to the log file (always works, even if stdout is None)."""
    if _LOG_FILE:
        try:
            _LOG_FILE.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
            _LOG_FILE.flush()
        except Exception:
            pass
    try:
        out = _real_stdout()
        if out is not None:
            out.write(str(msg) + '\n')
            out.flush()
    except Exception:
        pass


# ── Resolve paths relative to this file (works inside a PyInstaller bundle) ──

def _get_base_dir() -> Path:
    """Get the project root, whether running from source or a PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _find_renderer_dir() -> Path:
    """Find the ui/renderer directory — it may be in _MEIPASS or in the macOS
    Resources directory depending on how PyInstaller packages data files."""
    if getattr(sys, 'frozen', False):
        candidates = [
            Path(sys._MEIPASS) / 'ui' / 'renderer',  # type: ignore[attr-defined]
        ]
        # macOS .app bundle: data files go to Contents/Resources/
        if sys.platform == 'darwin':
            frameworks = Path(sys._MEIPASS)  # type: ignore[attr-defined]
            resources = frameworks.parent / 'Resources'
            candidates.insert(0, resources / 'ui' / 'renderer')
        for c in candidates:
            if c.is_dir() and (c / 'index.html').is_file():
                return c
        return Path(sys._MEIPASS) / 'ui' / 'renderer'  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / 'ui' / 'renderer'


def _find_source_dir() -> Path:
    """Find the directory containing the Python analysis modules (main.py, ...)."""
    if getattr(sys, 'frozen', False):
        candidates = [Path(sys._MEIPASS)]  # type: ignore[attr-defined]
        if sys.platform == 'darwin':
            frameworks = Path(sys._MEIPASS)  # type: ignore[attr-defined]
            candidates.append(frameworks.parent / 'Resources')
        for c in candidates:
            if (c / 'main.py').is_file():
                return c
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


BASE_DIR = _get_base_dir()
RENDERER_DIR = _find_renderer_dir()
SOURCE_DIR = _find_source_dir()

# Writable data directory for uploads and analysis output
if getattr(sys, 'frozen', False):
    DATA_DIR = Path.home() / '.sasa'
else:
    DATA_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = DATA_DIR / 'Audio' / 'uploads'
ANALYSIS_DIR = DATA_DIR / 'Audio' / 'analysis'
# Audio extracted from video containers is kept apart from the operator's uploads
# so a cache entry can never be mistaken for a source recording.
EXTRACT_DIR = DATA_DIR / 'Audio' / 'extracted'

PORT = int(os.environ.get('SASA_PORT', '3847'))

# ── MIME map. Also the allow-list of file types /api/image will serve. ──
MIME_MAP = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.pdf': 'application/pdf',
    '.csv': 'text/csv',
    '.txt': 'text/plain',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
}

# ═══════════════════════════════════════════════════════════
#  Accepted media — ONE definition, used everywhere
# ═══════════════════════════════════════════════════════════
#  Defect: .mts/.mxf used to be accepted for upload but were missing from the
#  video list, so a camcorder transport stream was handed straight to the WAV
#  loader. There is now exactly one source of truth for each question:
#  "may this be uploaded?" and "does this need audio extraction?".

AUDIO_EXTS = {
    '.wav', '.wave', '.bwf', '.rf64', '.w64',
    '.flac', '.aif', '.aiff', '.aifc', '.caf',
    '.ogg', '.oga', '.opus', '.mp3', '.m4a', '.aac', '.wma', '.alac',
}

VIDEO_EXTS = {
    '.mp4', '.m4v', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm',
    '.mpeg', '.mpg', '.mts', '.m2ts', '.ts', '.mxf', '.3gp', '.asf', '.vob',
}

MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

# Upload limits
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024      # 8 GB — a long 192 kHz multitrack take
MAX_FIELD_BYTES = 64 * 1024                    # non-file multipart fields
UPLOAD_CHUNK = 1 << 20                         # 1 MiB streaming chunks

# How long a finished run stays retrievable so a reconnecting client can collect it
RUN_RETENTION_S = 24 * 60 * 60
MAX_RUN_LOG_LINES = 5000


# ═══════════════════════════════════════════════════════════
#  Per-run stdout capture (thread-routed)
# ═══════════════════════════════════════════════════════════
#  Defect: analysis used to assign the process-global sys.stdout to a per-request
#  capture object and restore it in a finally block. Two overlapping runs
#  cross-wired their logs, and any failure between the swap and the restore left
#  stdout permanently pointing at a dead socket.
#
#  Instead, ONE proxy is installed for the lifetime of the process. It routes each
#  write by the identity of the calling thread: a thread running an analysis writes
#  to that run's sink, and every other thread writes to the real stream unchanged.
#  Nothing is ever swapped back, so there is no window in which stdout is broken.

class _ThreadRoutedStream:
    """A stdout/stderr stand-in that dispatches writes per thread."""

    def __init__(self, base):
        self._base = base
        self._sinks: Dict[int, Callable[[str], None]] = {}
        self._lock = threading.Lock()

    # -- registration --------------------------------------------------
    def bind(self, sink: Callable[[str], None]) -> None:
        with self._lock:
            self._sinks[threading.get_ident()] = sink

    def unbind(self) -> None:
        with self._lock:
            self._sinks.pop(threading.get_ident(), None)

    def _sink_for_current_thread(self) -> Optional[Callable[[str], None]]:
        with self._lock:
            return self._sinks.get(threading.get_ident())

    # -- file protocol -------------------------------------------------
    def write(self, text: str) -> int:
        # The real stream is written FIRST: whatever the per-run sink does, the
        # console and log file still receive the line.
        written = len(text)
        base = self._base
        if base is not None:
            try:
                written = base.write(text)
            except Exception:
                pass

        sink = self._sink_for_current_thread()
        if sink is not None:
            try:
                sink(text)
            except Exception:
                # A broken sink must never break stdout for the whole process.
                # Control-flow signals (cancellation) derive from BaseException
                # and deliberately pass straight through this guard.
                pass
        return written

    def flush(self) -> None:
        base = self._base
        if base is not None:
            try:
                base.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        base = self._base
        try:
            return bool(base is not None and base.isatty())
        except Exception:
            return False

    def fileno(self):
        base = self._base
        if base is None:
            raise OSError('no file descriptor')
        return base.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._base, 'encoding', 'utf-8')

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def close(self) -> None:  # never close the underlying stream on our behalf
        self.flush()


_stdout_router: Optional[_ThreadRoutedStream] = None
_stderr_router: Optional[_ThreadRoutedStream] = None
_router_lock = threading.Lock()


def _install_stream_routers() -> None:
    """Install the thread-routing proxies once. Idempotent."""
    global _stdout_router, _stderr_router
    with _router_lock:
        if _stdout_router is None:
            _stdout_router = _ThreadRoutedStream(sys.stdout)
            sys.stdout = _stdout_router
        if _stderr_router is None:
            _stderr_router = _ThreadRoutedStream(sys.stderr)
            sys.stderr = _stderr_router


def _real_stdout():
    """The genuine stdout, bypassing any per-run capture."""
    if _stdout_router is not None:
        return _stdout_router._base
    return sys.stdout


class _LineAssembler:
    """Turns a stream of writes into complete lines for one run."""

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._buf = ''
        self._lock = threading.Lock()

    def feed(self, text: str) -> None:
        with self._lock:
            self._buf += text
            lines: List[str] = []
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                lines.append(line.rstrip('\r'))
            # A single runaway line must not grow without bound.
            if len(self._buf) > 1 << 20:
                lines.append(self._buf)
                self._buf = ''
        for line in lines:
            self._emit(line)

    def close(self) -> None:
        with self._lock:
            tail, self._buf = self._buf, ''
        if tail.strip():
            self._emit(tail.rstrip('\r'))


# ═══════════════════════════════════════════════════════════
#  Path containment
# ═══════════════════════════════════════════════════════════
#  Defect: /api/image and /api/results took an arbitrary absolute "dir" from the
#  query string. Anything the user account could read was readable through the
#  browser, from any page able to reach localhost. Every path that reaches the
#  filesystem now goes through resolve_within_roots().

def _real(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        return Path(os.path.abspath(str(path)))


def _is_inside(root: Path, target: Path) -> bool:
    """True when `target` is `root` itself or lives beneath it."""
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_within_roots(
    target: str | Path,
    roots: List[Path],
    *,
    must_exist: bool = True,
) -> Optional[Path]:
    """
    Resolve `target` and prove it lives inside one of `roots`.

    Containment is decided on REAL (symlink-resolved) paths, so neither '../..'
    segments nor a symlink pointing outside a root can escape. Returns None when
    the path is not allowed — callers must treat None as "not found", never as
    "fall back to the raw path".
    """
    if target is None:
        return None
    raw = str(target)
    if not raw or '\x00' in raw:
        return None

    resolved = _real(Path(raw))
    real_roots = [_real(r) for r in roots]

    if not any(_is_inside(root, resolved) for root in real_roots):
        return None

    if not resolved.exists():
        if must_exist:
            return None
        # Not created yet (e.g. an output directory): check the nearest existing
        # ancestor once symlinks are resolved, so a symlinked parent cannot move
        # the eventual path outside a root.
        anchor = resolved
        while not anchor.exists():
            parent = anchor.parent
            if parent == anchor:
                return None
            anchor = parent
        real_anchor = _real(anchor)
        tail = resolved.relative_to(anchor) if _is_inside(anchor, resolved) else None
        if tail is None:
            return None
        real_target = _real(real_anchor / tail)
        if not any(_is_inside(root, real_target) for root in real_roots):
            return None
        return resolved

    return resolved


# Characters that survive into a filename. Everything else is replaced, so a
# crafted upload name cannot contain separators, control characters or NULs.
_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9._-]+')
_WINDOWS_RESERVED = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
}


def safe_upload_name(raw: str) -> Optional[str]:
    """
    Reduce a client-supplied filename to a leaf name safe to join onto a directory.

    Defect: the uploaded name was used verbatim, so '../../.ssh/authorized_keys'
    or a Windows 'C:\\...' spelling wrote outside the upload directory. Returns
    None when nothing usable remains.
    """
    if not raw or not isinstance(raw, str):
        return None
    if '\x00' in raw:
        return None

    # Take the leaf under BOTH separator conventions — a Windows client sends
    # backslashes that POSIX Path() would treat as ordinary characters.
    leaf = raw.replace('\\', '/').split('/')[-1].strip()
    # Drop any drive letter that survived ("C:evil.wav")
    if len(leaf) > 1 and leaf[1] == ':':
        leaf = leaf[2:]

    leaf = _SAFE_NAME_RE.sub('_', leaf)
    leaf = leaf.lstrip('.')                    # no dotfiles, no '..'
    leaf = leaf.strip('_')
    if not leaf:
        return None

    stem, dot, ext = leaf.rpartition('.')
    if not dot:
        stem, ext = leaf, ''
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f'_{stem}'
    stem = stem[:120] or 'upload'
    ext = ext[:16]
    return f'{stem}.{ext}' if ext else stem


def unique_destination(directory: Path, name: str) -> Path:
    """A non-colliding path for `name` inside `directory`."""
    dest = directory / name
    if not dest.exists():
        return dest
    stem, dot, ext = name.rpartition('.')
    if not dot:
        stem, ext = name, ''
    suffix = f'.{ext}' if ext else ''
    for _ in range(1000):
        candidate = directory / f'{stem}_{int(time.time() * 1000) % 10_000_000}{suffix}'
        if not candidate.exists():
            return candidate
    return directory / f'{stem}_{uuid.uuid4().hex[:8]}{suffix}'


# ═══════════════════════════════════════════════════════════
#  ffmpeg — audio extraction that preserves the measurement
# ═══════════════════════════════════════════════════════════

def _find_ffmpeg() -> Optional[str]:
    """Find an ffmpeg binary — bundled, imageio-ffmpeg, then system PATH."""
    if getattr(sys, 'frozen', False):
        bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        search = [bundle_dir / 'imageio_ffmpeg_bin', bundle_dir]
        if sys.platform == 'darwin':
            search.append(bundle_dir.parent / 'Resources' / 'imageio_ffmpeg_bin')
        for candidate_dir in search:
            if candidate_dir.is_dir():
                for name in ('ffmpeg', 'ffmpeg.exe'):
                    exe = candidate_dir / name
                    if exe.is_file():
                        if sys.platform != 'win32':
                            try:
                                exe.chmod(0o755)
                            except OSError:
                                pass
                        return str(exe)

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(exe)
    except Exception:
        pass

    return shutil.which('ffmpeg')


def _find_ffprobe(ffmpeg: Optional[str]) -> Optional[str]:
    """ffprobe sitting next to ffmpeg, or on PATH."""
    if ffmpeg:
        here = Path(ffmpeg).parent
        for name in ('ffprobe', 'ffprobe.exe'):
            candidate = here / name
            if candidate.is_file():
                return str(candidate)
    return shutil.which('ffprobe')


# ffmpeg sample formats → the smallest PCM encoding that loses nothing.
_SAMPLE_FMT_TO_PCM = {
    'u8': 'pcm_s16le', 'u8p': 'pcm_s16le',     # 8-bit up into 16-bit: lossless
    's16': 'pcm_s16le', 's16p': 'pcm_s16le',
    's32': 'pcm_s32le', 's32p': 'pcm_s32le',   # narrowed to s24 when the source is 24-bit
    's64': 'pcm_s32le', 's64p': 'pcm_s32le',
    'flt': 'pcm_f32le', 'fltp': 'pcm_f32le',
    'dbl': 'pcm_f64le', 'dblp': 'pcm_f64le',
}


@dataclasses.dataclass
class AudioStreamInfo:
    """What the source container actually holds."""
    codec_name: str = ''
    sample_fmt: str = ''
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bits_per_raw_sample: Optional[int] = None
    # True: an audio stream was found. False: the container was read and has none.
    # None: the container could not be inspected, so nothing may be concluded.
    found: Optional[bool] = None

    def pcm_codec(self) -> str:
        """
        The PCM encoding to extract into.

        Chosen to be at least as wide as the source in every dimension. A lossy
        source (AAC, Vorbis) decodes to float, so pcm_f32le is exact for it too.
        When nothing is known, 32-bit float is the safe default: it cannot clip
        and cannot quantise below any realistic source.
        """
        fmt = (self.sample_fmt or '').lower()
        codec = _SAMPLE_FMT_TO_PCM.get(fmt)
        if codec == 'pcm_s32le' and self.bits_per_raw_sample == 24:
            return 'pcm_s24le'
        if codec:
            return codec
        # Unknown sample format: infer from the codec name where it is explicit.
        name = (self.codec_name or '').lower()
        if name.startswith('pcm_'):
            if '24' in name:
                return 'pcm_s24le'
            if '32' in name and 'f' in name:
                return 'pcm_f32le'
            if '32' in name:
                return 'pcm_s32le'
            if '16' in name:
                return 'pcm_s16le'
        return 'pcm_f32le'

    def describe(self) -> str:
        parts = []
        if self.codec_name:
            parts.append(self.codec_name)
        if self.sample_rate:
            parts.append(f'{self.sample_rate} Hz')
        if self.channels:
            parts.append(f'{self.channels} ch')
        if self.sample_fmt:
            bits = f'/{self.bits_per_raw_sample}-bit' if self.bits_per_raw_sample else ''
            parts.append(f'{self.sample_fmt}{bits}')
        return ', '.join(parts) if parts else 'unknown'


_FFMPEG_STREAM_RE = re.compile(
    r'Stream #\d+:\d+.*?:\s*Audio:\s*([A-Za-z0-9_]+).*?,\s*(\d+)\s*Hz,\s*([^,]+),\s*([A-Za-z0-9]+)'
)
_FFMPEG_BITS_RE = re.compile(r'\((\d+)\s*bit\)')
_CHANNEL_LAYOUTS = {'mono': 1, 'stereo': 2, '2.1': 3, 'quad': 4, '5.0': 5, '5.1': 6, '7.1': 8}


def probe_audio_stream(media_path: Path, ffmpeg: Optional[str] = None) -> AudioStreamInfo:
    """
    Read the source audio stream's real parameters.

    Prefers ffprobe; falls back to parsing `ffmpeg -i`, because imageio-ffmpeg
    ships ffmpeg without ffprobe and the packaged app may only have the former.
    Returns an empty AudioStreamInfo when nothing could be determined — callers
    must then preserve the source implicitly rather than inventing values.
    """
    ffmpeg = ffmpeg or _find_ffmpeg()
    ffprobe = _find_ffprobe(ffmpeg)

    if ffprobe:
        try:
            proc = subprocess.run(
                [ffprobe, '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries',
                 'stream=codec_name,sample_fmt,sample_rate,channels,bits_per_raw_sample',
                 '-of', 'json', str(media_path)],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                streams = (json.loads(proc.stdout) or {}).get('streams') or []
                if not streams:
                    # ffprobe read the container and found no audio stream at all.
                    return AudioStreamInfo(found=False)
                if streams:
                    s = streams[0]

                    def _int(value):
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return None

                    return AudioStreamInfo(
                        found=True,
                        codec_name=str(s.get('codec_name') or ''),
                        sample_fmt=str(s.get('sample_fmt') or ''),
                        sample_rate=_int(s.get('sample_rate')),
                        channels=_int(s.get('channels')),
                        bits_per_raw_sample=_int(s.get('bits_per_raw_sample')),
                    )
        except Exception:
            pass

    if ffmpeg:
        try:
            proc = subprocess.run(
                [ffmpeg, '-hide_banner', '-i', str(media_path)],
                capture_output=True, text=True, timeout=60,
            )
            text = (proc.stderr or '') + (proc.stdout or '')
            match = _FFMPEG_STREAM_RE.search(text)
            if not match and re.search(r'Stream #\d+:\d+', text) \
                    and not re.search(r':\s*Audio:', text):
                # Streams were listed and none of them is audio.
                return AudioStreamInfo(found=False)
            if match:
                codec, rate, layout, fmt = match.groups()
                bits = None
                bits_match = _FFMPEG_BITS_RE.search(text[match.start():match.end() + 40])
                if bits_match:
                    bits = int(bits_match.group(1))
                layout = layout.strip().lower()
                channels = _CHANNEL_LAYOUTS.get(layout)
                if channels is None:
                    ch_match = re.match(r'(\d+)\s*channels?', layout)
                    channels = int(ch_match.group(1)) if ch_match else None
                return AudioStreamInfo(
                    found=True,
                    codec_name=codec, sample_fmt=fmt.strip(),
                    sample_rate=int(rate), channels=channels,
                    bits_per_raw_sample=bits,
                )
        except Exception:
            pass

    return AudioStreamInfo()


def _ffmpeg_error_summary(stderr: Optional[str], limit: int = 400) -> str:
    """
    The lines of ffmpeg's output that state the failure.

    ffmpeg prints its whole stream inventory before the error, so echoing the tail
    verbatim buries the cause in metadata the operator cannot act on.
    """
    text = (stderr or '').strip()
    if not text:
        return 'ffmpeg produced no diagnostic output.'
    interesting = [line.strip() for line in text.splitlines()
                   if re.search(r'error|invalid|unable|failed|no such|not found|denied',
                                line, re.IGNORECASE)]
    summary = ' '.join(interesting) if interesting else text.splitlines()[-1].strip()
    return summary[:limit]


def _extraction_fingerprint(video_path: Path, info: AudioStreamInfo, codec: str) -> str:
    """
    Identify an extraction by its source AND its settings.

    A cached WAV produced by an older, lossier extraction must never be reused, so
    the encoding parameters are part of the cache key rather than just the name.
    """
    try:
        stat = video_path.stat()
        stamp = f'{stat.st_size}:{stat.st_mtime_ns}'
    except OSError:
        stamp = 'nostat'
    key = f'{video_path.name}|{stamp}|{codec}|{info.sample_rate}|{info.channels}|{info.sample_fmt}'
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]


def extract_audio_from_video(
    video_path: Path,
    output_dir: Path,
    *,
    log: Optional[Callable[[str], None]] = None,
    timeout_s: float = 3600.0,
) -> Path:
    """
    Extract audio from a video container WITHOUT changing the measurement.

    Defect: this used to run `-acodec pcm_s16le -ar 44100 -ac 1`, which
      * resampled 48/96/192 kHz down to 44.1 kHz, low-passing the blast at 22 kHz
        and destroying the rise time the analysis is built on,
      * quantised 24-bit and float sources to 16 bits, raising the noise floor,
      * mixed multi-channel down to mono, averaging away a channel that may have
        been the only unclipped one.
    The GUI therefore disagreed with the CLI on the very same file.

    The source sample rate, bit depth and channel count are now preserved, and the
    PCM encoding is the narrowest one that is lossless for the source.
    """
    say = log or (lambda _m: None)

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            'Cannot extract audio from video: ffmpeg not found.\n'
            'Install ffmpeg:\n'
            '  macOS:   brew install ffmpeg\n'
            '  Windows: winget install ffmpeg\n'
            '  Or:      pip install imageio-ffmpeg'
        )

    info = probe_audio_stream(video_path, ffmpeg)
    if info.found is False:
        raise RuntimeError(
            f'{video_path.name} contains no audio stream, so there is nothing to '
            f'measure. Check that the camera or recorder was actually capturing audio.'
        )
    codec = info.pcm_codec()
    say(f'  Source audio: {info.describe()}')
    say(f'  Extracting as {codec} at the source rate and channel count (no resampling, no down-mix)')

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _extraction_fingerprint(video_path, info, codec)
    stem = safe_upload_name(video_path.stem) or 'extracted'
    wav_path = output_dir / f'{stem}.{fingerprint}.wav'

    if wav_path.is_file() and wav_path.stat().st_size > 0:
        say(f'  Reusing previously extracted audio: {wav_path.name}')
        return wav_path

    tmp_path = wav_path.with_suffix('.wav.partial')
    cmd = [ffmpeg, '-hide_banner', '-nostdin', '-i', str(video_path), '-vn', '-sn', '-dn',
           '-map', '0:a:0', '-acodec', codec]
    # Passing the probed values explicitly is a no-op that documents intent in the
    # log; omitting them entirely (unknown probe) also preserves the source.
    if info.sample_rate:
        cmd += ['-ar', str(info.sample_rate)]
    if info.channels:
        cmd += ['-ac', str(info.channels)]
    # A long high-rate take can exceed the 4 GB WAV limit; RF64 handles it.
    # -f wav is explicit because the temp file's extension is '.partial', which
    # ffmpeg cannot map to a muxer on its own.
    cmd += ['-rf64', 'auto', '-f', 'wav', '-y', str(tmp_path)]

    say('  ffmpeg ' + ' '.join(cmd[1:]))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f'ffmpeg timed out after {timeout_s:.0f} s extracting {video_path.name}')

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f'Audio extraction failed for {video_path.name}: '
                           f'{_ffmpeg_error_summary(result.stderr)}')
    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg produced no output')

    tmp_path.replace(wav_path)

    # Verify what we actually got. Silent divergence here would be indistinguishable
    # from a correct extraction in every later stage.
    out_info = probe_audio_stream(wav_path, ffmpeg)
    if info.sample_rate and out_info.sample_rate and out_info.sample_rate != info.sample_rate:
        raise RuntimeError(
            f'Extraction changed the sample rate ({info.sample_rate} Hz -> '
            f'{out_info.sample_rate} Hz). Refusing to measure a resampled impulse.'
        )
    if info.channels and out_info.channels and out_info.channels != info.channels:
        say(f'  WARNING: channel count changed ({info.channels} -> {out_info.channels})')
    say(f'  Extracted: {wav_path.name} ({out_info.describe()})')
    return wav_path


# Back-compat alias for existing callers.
_extract_audio_from_video = extract_audio_from_video


# ═══════════════════════════════════════════════════════════
#  Lazy import of the analysis modules
# ═══════════════════════════════════════════════════════════

_analysis_modules: Dict[str, Any] = {}
_import_lock = threading.Lock()


def _ensure_analysis_imports():
    """Import analysis modules on first use. Raises ImportError on failure."""
    with _import_lock:
        if _analysis_modules:
            return
        src = str(SOURCE_DIR)
        if src not in sys.path:
            sys.path.insert(0, src)
        _log(f'Analysis import: SOURCE_DIR={SOURCE_DIR}')

        from main import analyze_file, AnalysisConfig
        from calibration import Calibration

        _analysis_modules['analyze_file'] = analyze_file
        _analysis_modules['AnalysisConfig'] = AnalysisConfig
        _analysis_modules['Calibration'] = Calibration
        _log('Analysis modules imported successfully')


# ═══════════════════════════════════════════════════════════
#  Configuration validation
# ═══════════════════════════════════════════════════════════
#  Defect: every field was read as `if config.get('x'):`. A user who typed 0 —
#  a legitimate value for a threshold in dB, a pre-roll in ms or a preamp gain —
#  got the built-in default instead, silently. Presence is now tested explicitly
#  and 0 survives.

class ConfigError(Exception):
    """A configuration the operator must fix. Carries per-field messages."""

    def __init__(self, message: str, fields: Optional[List[Dict[str, str]]] = None):
        super().__init__(message)
        self.fields = fields or []


def _present(config: dict, key: str) -> bool:
    """True when the client actually supplied a value — 0 and False included."""
    if key not in config:
        return False
    value = config[key]
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == '':
        return False
    return True


def _get_number(config: dict, key: str, *, minimum=None, maximum=None,
                exclusive_min: bool = False) -> Optional[float]:
    """Validated float, or None when not supplied. Never coerces a bad value."""
    if not _present(config, key):
        return None
    raw = config[key]
    if isinstance(raw, bool):
        raise ConfigError(f'{key} must be a number.', [{'field': key, 'message': 'Must be a number.'}])
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ConfigError(f'{key} must be a number, got {raw!r}.',
                          [{'field': key, 'message': 'Must be a number.'}])
    if value != value or value in (float('inf'), float('-inf')):
        raise ConfigError(f'{key} must be finite.', [{'field': key, 'message': 'Must be finite.'}])
    if minimum is not None:
        if (value <= minimum) if exclusive_min else (value < minimum):
            word = 'greater than' if exclusive_min else 'at least'
            raise ConfigError(f'{key} must be {word} {minimum}.',
                              [{'field': key, 'message': f'Must be {word} {minimum}.'}])
    if maximum is not None and value > maximum:
        raise ConfigError(f'{key} must be at most {maximum}.',
                          [{'field': key, 'message': f'Must be at most {maximum}.'}])
    return value


def _get_int(config: dict, key: str, *, minimum=None, maximum=None) -> Optional[int]:
    value = _get_number(config, key, minimum=minimum, maximum=maximum)
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    raise ConfigError(f'{key} must be a whole number.',
                      [{'field': key, 'message': 'Must be a whole number.'}])


def _get_bool(config: dict, key: str) -> Optional[bool]:
    if key not in config or config[key] is None:
        return None
    raw = config[key]
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ('true', '1', 'yes', 'on'):
            return True
        if low in ('false', '0', 'no', 'off'):
            return False
    raise ConfigError(f'{key} must be true or false.',
                      [{'field': key, 'message': 'Must be true or false.'}])


def _get_str(config: dict, key: str, *, max_length: int = 500) -> Optional[str]:
    if not _present(config, key):
        return None
    raw = config[key]
    if not isinstance(raw, str):
        raise ConfigError(f'{key} must be text.', [{'field': key, 'message': 'Must be text.'}])
    if len(raw) > max_length:
        raise ConfigError(f'{key} must be at most {max_length} characters.',
                          [{'field': key, 'message': f'Must be at most {max_length} characters.'}])
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', raw):
        raise ConfigError(f'{key} must not contain control characters.',
                          [{'field': key, 'message': 'Must not contain control characters.'}])
    return raw.strip()


ALLOWED_FORMATS = ('png', 'pdf', 'svg', 'html')


def _calibration_from_tone(config: dict, Calibration,
                           description: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Resolve a calibration from a recorded calibrator tone.

    The tone is read here rather than handed to the engine as a filename,
    because this server's contract is that the calibration is DECIDED before
    the run starts -- the backend never gets the chance to fall back to a
    default on our behalf. The reading itself is WavLoader's, the same code the
    engine would have used, and the channel policy matches main._read_calibrator:
    the tone is read on the channel the test will be read on.

    The level is required. A calibrator whose level is assumed is a calibration
    that was invented, which is the failure this whole module is built around.
    """
    raw = _get_str(config, 'calibratorTone', max_length=4096)
    if not raw:
        raise ConfigError(
            'Tone calibration needs the calibrator recording.',
            [{'field': 'calibratorTone', 'message': 'Required for tone calibration.'}])

    tone_path = resolve_within_roots(raw, [UPLOAD_DIR, EXTRACT_DIR, DATA_DIR / 'Audio'])
    if tone_path is None or not tone_path.is_file():
        raise ConfigError(
            'The calibrator recording is not an accessible file in this workspace.',
            [{'field': 'calibratorTone', 'message': 'Not an accessible uploaded recording.'}])

    level_dB = _get_number(config, 'calibratorLevelDb', minimum=0, maximum=200)
    if level_dB is None:
        raise ConfigError(
            'Tone calibration needs the level printed on the calibrator. There is no '
            'default: an assumed calibrator level puts every dB SPL in the report out '
            'by the difference between the assumption and the instrument.',
            [{'field': 'calibratorLevelDb', 'message': 'Required for tone calibration.'}])

    frequency_Hz = _get_number(config, 'calibratorFreqHz', minimum=1, maximum=200000)
    channel = _get_int(config, 'channel', minimum=0, maximum=1024) or 0

    try:
        from WavLoader import get_wav_info, load_wav  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(f'The calibrator recording could not be read: {exc}',
                          [{'field': 'calibratorTone', 'message': str(exc)}]) from exc

    try:
        frames, sample_rate, _duration, channels = get_wav_info(tone_path)
        if frames <= 0:
            raise ValueError('the recording is empty')
        data = load_wav(tone_path, dtype='float64',
                        channel=channel if channel < channels else 0)
    except Exception as exc:  # noqa: BLE001 - surfaced as a configuration error
        raise ConfigError(
            f'The calibrator recording could not be read: {exc}',
            [{'field': 'calibratorTone', 'message': str(exc)}]) from exc

    kwargs: Dict[str, Any] = {
        'calibrator_level_dB': float(level_dB),
        'description': description or f'Calibrator tone at {float(level_dB):g} dB',
    }
    if frequency_Hz is not None:
        kwargs['tone_frequency_Hz'] = float(frequency_Hz)

    try:
        calibration = Calibration.from_calibrator_tone(
            data.samples, int(sample_rate), **kwargs)
    except ValueError as exc:
        raise ConfigError(
            f'The calibrator tone in {tone_path.name} is unusable: {exc}',
            [{'field': 'calibratorTone', 'message': str(exc)}]) from exc

    inputs: Dict[str, Any] = {'Pa_per_FS': float(calibration.Pa_per_FS)}
    return calibration, inputs


def build_calibration(config: dict, Calibration) -> Tuple[Any, Dict[str, Any]]:
    """
    Build the Calibration for this run — or refuse the run.

    Returns (calibration, native_inputs). native_inputs holds the operator's raw
    numbers so they can be mirrored onto the backend's own configuration fields,
    keeping the saved run config re-runnable rather than reduced to a bare factor.

    Defect: a half-filled microphone-sensitivity form (say, sensitivity typed but
    full-scale voltage left blank) silently reverted to the built-in 143.96 Pa/FS.
    Every level in the report was then wrong by however far the operator's rig
    differed from the one that constant came from, and nothing anywhere said so.

    There is no default here. The calibration mode is explicit, each mode names
    exactly what it needs, and a missing field raises instead of substituting.
    """
    mode = _get_str(config, 'calMode', max_length=32)
    if mode:
        mode = mode.strip().lower()

    has_direct = _present(config, 'paPerFS')
    has_sens = _present(config, 'sensitivityMv')
    # adcFullScaleV is what the interface calls the recorder's full-scale
    # voltage; vPerFS is this server's older name for the same number, and the
    # engine's own field carries both spellings too.
    has_vfs = _present(config, 'vPerFS') or _present(config, 'adcFullScaleV')
    has_tone = _present(config, 'calibratorTone')
    said_uncalibrated = bool(config.get('uncalibrated'))

    if not mode:
        # Infer the intent from what was sent, but never invent a calibration.
        #
        # Three of these four branches did not exist, and the interface uses
        # them: it sends `uncalibrated`, or a calibrator tone, or the chain
        # under the name adcFullScaleV. None of those were read here, so the
        # only calibration method that worked end to end in the packaged app
        # was a saved profile -- every other choice was refused with the
        # message below, which is true of the request but was not true of what
        # the operator had actually filled in.
        if said_uncalibrated:
            mode = 'uncalibrated'
        elif has_tone:
            mode = 'tone'
        elif has_direct and not (has_sens or has_vfs):
            mode = 'direct'
        elif has_sens or has_vfs:
            mode = 'sensitivity'
        else:
            raise ConfigError(
                'No calibration was supplied. Choose a calibration mode: '
                '"direct" (Pa per full scale), "sensitivity" (microphone sensitivity '
                'and recorder full-scale voltage), "tone" (a recorded calibrator), '
                'or "uncalibrated" (results are RELATIVE, in dB re FS, and must not '
                'be reported as dB SPL).',
                [{'field': 'calMode', 'message': 'Choose direct, sensitivity, tone or uncalibrated.'}],
            )

    description = _get_str(config, 'calDesc', max_length=200) or ''

    if mode in ('uncalibrated', 'relative', 'none'):
        return Calibration.uncalibrated(), {'uncalibrated': True}

    if mode in ('tone', 'calibrator', 'calibrator_tone'):
        return _calibration_from_tone(config, Calibration, description)

    if mode == 'direct':
        if not has_direct:
            raise ConfigError(
                'Direct calibration needs Pa per full scale. It was left blank, and '
                'there is no default: an assumed conversion factor would produce '
                'confident, wrong dB SPL.',
                [{'field': 'paPerFS', 'message': 'Required for direct calibration.'}],
            )
        pa_per_fs = _get_number(config, 'paPerFS', minimum=0, exclusive_min=True, maximum=1e9)
        return Calibration(
            Pa_per_FS=float(pa_per_fs),
            calibrated=True,
            method='direct',
            description=description or f'Direct: {pa_per_fs:.6g} Pa/FS',
        ), {'Pa_per_FS': float(pa_per_fs)}

    if mode in ('sensitivity', 'chain', 'recording_chain'):
        missing = []
        if not has_sens:
            missing.append({'field': 'sensitivityMv',
                            'message': 'Required: microphone sensitivity in mV/Pa.'})
        if not has_vfs:
            missing.append({'field': 'adcFullScaleV',
                            'message': 'Required: recorder full-scale voltage.'})
        if missing:
            names = ', '.join(m['field'] for m in missing)
            raise ConfigError(
                f'Microphone-sensitivity calibration is incomplete ({names}). '
                f'Both the sensitivity and the recorder full-scale voltage are needed '
                f'to convert full scale to Pascals. The run was refused rather than '
                f'falling back to a built-in factor measured on different hardware.',
                missing,
            )
        sensitivity = _get_number(config, 'sensitivityMv', minimum=0, exclusive_min=True, maximum=1e6)
        v_per_fs = _get_number(config, 'vPerFS', minimum=0, exclusive_min=True, maximum=1e6)
        if v_per_fs is None:
            v_per_fs = _get_number(config, 'adcFullScaleV', minimum=0, exclusive_min=True,
                                   maximum=1e6)
        gain_dB = _get_number(config, 'preampGainDb', minimum=-200, maximum=200)
        inputs = {
            'sensitivity_mV_per_Pa': float(sensitivity),
            'adc_full_scale_V': float(v_per_fs),
            'preamp_gain_dB': float(gain_dB) if gain_dB is not None else 0.0,
        }
        return Calibration.from_recording_chain(description=description, **inputs), inputs

    raise ConfigError(
        f'Unknown calibration mode {mode!r}. Use "direct", "sensitivity" or "uncalibrated".',
        [{'field': 'calMode', 'message': 'Unknown calibration mode.'}],
    )


# UI field  ->  candidate AnalysisConfig field names, most preferred first.
# main.py is being rewritten in parallel; matching by introspection means this
# server adapts to its field names instead of guessing one spelling and breaking.
_CONFIG_ALIASES: Dict[str, Tuple[str, ...]] = {
    'threshold_dB': ('detection_threshold_dB', 'threshold_dB'),
    'threshold_relative_dB': ('detection_threshold_relative_dB', 'threshold_relative_dB'),
    'refractory_ms': ('refractory_ms',),
    'pre_ms': ('pre_shot_ms', 'pre_ms'),
    'post_ms': ('post_shot_ms', 'post_ms'),
    'nperseg': ('nperseg',),
    'overlap_fraction': ('overlap_fraction',),
    'compute_bands': ('compute_bands',),
    'save_per_shot_plots': ('save_per_shot_plots',),
    'plot_formats': ('plot_formats',),
    'load_dtype': ('load_dtype', 'dtype'),
    'protection_NRR_dB': ('protection_NRR_dB',),
    'test_metadata': ('test_metadata',),
    'reference_dir': ('reference_dir', 'reference_analysis_dir'),
    'channel': ('channel',),
    'mono_mix': ('mono_mix', 'mono'),
}

_CALIBRATION_ALIASES: Dict[str, Tuple[str, ...]] = {
    'object': ('calibration', 'cal'),
    'Pa_per_FS': ('Pa_per_FS', 'pa_per_fs'),
    'sensitivity': ('sensitivity_mV_per_Pa',),
    'adc_full_scale_V': ('adc_full_scale_V', 'V_per_FS'),
    'preamp_gain_dB': ('preamp_gain_dB',),
    'uncalibrated': ('uncalibrated',),
    'calibrated': ('calibrated', 'is_calibrated'),
    'method': ('calibration_method',),
    'description': ('calibration_description', 'cal_description'),
}


def _apply_native_calibration(kwargs: dict, available: set, calibration,
                              inputs: Dict[str, Any]) -> Optional[str]:
    """
    Mirror the calibration onto the backend's own fields.

    main.py's resolve_calibration() refuses a config that names MORE THAN ONE
    calibration source, so exactly one is set here — the same one the operator
    chose. This keeps the config written into the run record re-runnable: loading
    it back reproduces this calibration and no other.

    Returns the source that was set, or None when the backend has no such fields.
    """
    if not calibration.calibrated:
        if _assign(kwargs, available, _CALIBRATION_ALIASES['uncalibrated'], True):
            return 'uncalibrated'
        if _assign(kwargs, available, _CALIBRATION_ALIASES['calibrated'], False):
            return 'uncalibrated'
        return None

    if calibration.method == 'recording_chain' and 'sensitivity_mV_per_Pa' in inputs and \
            set(_CALIBRATION_ALIASES['sensitivity']) & available:
        # The chain is one source; the full-scale voltage and gain are its operands,
        # not a second source.
        _assign(kwargs, available, _CALIBRATION_ALIASES['sensitivity'],
                inputs['sensitivity_mV_per_Pa'])
        _assign(kwargs, available, _CALIBRATION_ALIASES['adc_full_scale_V'],
                inputs['adc_full_scale_V'])
        _assign(kwargs, available, _CALIBRATION_ALIASES['preamp_gain_dB'],
                inputs['preamp_gain_dB'])
        _assign(kwargs, available, _CALIBRATION_ALIASES['description'],
                calibration.description)
        return 'recording_chain'

    if _assign(kwargs, available, _CALIBRATION_ALIASES['Pa_per_FS'], calibration.Pa_per_FS):
        _assign(kwargs, available, _CALIBRATION_ALIASES['calibrated'], True)
        _assign(kwargs, available, _CALIBRATION_ALIASES['description'],
                calibration.description)
        return 'Pa_per_FS'

    return None


def _config_field_names(AnalysisConfig) -> set:
    if dataclasses.is_dataclass(AnalysisConfig):
        return {f.name for f in dataclasses.fields(AnalysisConfig)}
    try:
        return set(inspect.signature(AnalysisConfig).parameters)
    except (TypeError, ValueError):
        return set()


def _assign(kwargs: dict, available: set, aliases: Tuple[str, ...], value) -> bool:
    for name in aliases:
        if name in available:
            kwargs[name] = value
            return True
    return False


def build_analysis_config(config: dict, AnalysisConfig, Calibration, analyze_fn=None):
    """
    Turn a validated UI config into an AnalysisConfig, or raise ConfigError.

    Returns (analysis_config, calibration, settings_used, call_kwargs), where
    call_kwargs are extra keyword arguments for analyze_file().
    """
    available = _config_field_names(AnalysisConfig)
    call_params: set = set()
    if analyze_fn is not None:
        try:
            call_params = set(inspect.signature(analyze_fn).parameters)
        except (TypeError, ValueError):
            call_params = set()

    kwargs: Dict[str, Any] = {}
    call_kwargs: Dict[str, Any] = {}
    settings: Dict[str, Any] = {}

    # ── Calibration ───────────────────────────────────────────────────
    # Resolved HERE, from the operator's input, and then handed to the backend
    # already decided. The backend never gets the chance to fall back to a
    # default on our behalf.
    calibration, native_inputs = build_calibration(config, Calibration)

    # Mirror the choice onto the backend's own fields so the config saved with the
    # run reproduces this calibration and no other.
    native = _apply_native_calibration(kwargs, available, calibration, native_inputs)

    # Transport the resolved object itself where the backend accepts one.
    if _assign(kwargs, available, _CALIBRATION_ALIASES['object'], calibration):
        transport = 'config field'
    elif 'calibration' in call_params:
        call_kwargs['calibration'] = calibration
        transport = 'analyze_file(calibration=...)'
    elif native:
        transport = f'config field ({native})'
    else:
        raise ConfigError(
            'The analysis backend exposes no calibration interface; refusing to run '
            'rather than measure with an unknown conversion factor.',
            [{'field': 'calMode', 'message': 'Backend calibration API not found.'}],
        )

    # Last line of defence: an uncalibrated run must be *expressible* end to end.
    # If it is not, relative levels would be published as dB SPL, which is the
    # exact failure this server exists to prevent.
    if not calibration.calibrated and transport.startswith('config field (') \
            and native != 'uncalibrated':
        raise ConfigError(
            'This analysis backend cannot represent an uncalibrated measurement, '
            'so relative levels would be reported as dB SPL. Supply a calibration '
            'or update the backend.',
            [{'field': 'calMode', 'message': 'Backend cannot express "uncalibrated".'}],
        )

    settings['calibration'] = calibration.to_dict()
    settings['calibration_transport'] = transport

    # ── Shot detection ──
    threshold = _get_number(config, 'thresholdDb', minimum=-200, maximum=300)
    if threshold is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['threshold_dB'], threshold)
        settings['detection_threshold_dB'] = threshold

    threshold_rel = _get_number(config, 'thresholdRelativeDb', minimum=0, maximum=200)
    if threshold_rel is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['threshold_relative_dB'], threshold_rel)
        settings['detection_threshold_relative_dB'] = threshold_rel

    for ui_key, alias_key, lo, hi in (
        ('refractoryMs', 'refractory_ms', 0, 600000),
        ('preMs', 'pre_ms', 0, 600000),
        ('postMs', 'post_ms', 0, 600000),
    ):
        value = _get_number(config, ui_key, minimum=lo, maximum=hi)
        if value is not None:
            _assign(kwargs, available, _CONFIG_ALIASES[alias_key], value)
            settings[alias_key] = value

    nperseg = _get_int(config, 'nperseg', minimum=64, maximum=1_048_576)
    if nperseg is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['nperseg'], nperseg)
        settings['nperseg'] = nperseg

    # The interface offers this in Settings and it was going nowhere: the STFT
    # ran at the engine's default overlap whatever the operator chose.
    overlap = _get_number(config, 'overlapFraction', minimum=0, maximum=0.99)
    if overlap is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['overlap_fraction'], overlap)
        settings['overlap_fraction'] = overlap

    dtype = _get_str(config, 'dtype', max_length=16)
    if dtype is not None:
        if dtype not in ('float32', 'float64'):
            raise ConfigError('dtype must be float32 or float64.',
                              [{'field': 'dtype', 'message': 'Must be float32 or float64.'}])
        _assign(kwargs, available, _CONFIG_ALIASES['load_dtype'], dtype)
        settings['load_dtype'] = dtype

    nrr = _get_number(config, 'protectionNrrDb', minimum=0, maximum=60)
    if nrr is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['protection_NRR_dB'], nrr)
        settings['protection_NRR_dB'] = nrr

    # ── Options. `noBands`/`noPerShot` are inverted flags: False must be honoured
    #    as "compute them", not treated as absent.
    no_bands = _get_bool(config, 'noBands')
    if no_bands is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['compute_bands'], not no_bands)
        settings['compute_bands'] = not no_bands

    no_per_shot = _get_bool(config, 'noPerShot')
    if no_per_shot is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['save_per_shot_plots'], not no_per_shot)
        settings['save_per_shot_plots'] = not no_per_shot

    # ── Plot formats ──
    if _present(config, 'formats'):
        raw = config['formats']
        items = raw if isinstance(raw, list) else str(raw).split(',')
        cleaned: List[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ConfigError('formats must be text.',
                                  [{'field': 'formats', 'message': 'Must be text.'}])
            fmt = item.strip().lower()
            if not fmt:
                continue
            if fmt not in ALLOWED_FORMATS:
                raise ConfigError(
                    f'Unsupported plot format {fmt!r}. Allowed: {", ".join(ALLOWED_FORMATS)}.',
                    [{'field': 'formats', 'message': f'Allowed: {", ".join(ALLOWED_FORMATS)}.'}])
            if fmt not in cleaned:
                cleaned.append(fmt)
        if not cleaned:
            raise ConfigError('At least one plot format is required.',
                              [{'field': 'formats', 'message': 'At least one format required.'}])
        _assign(kwargs, available, _CONFIG_ALIASES['plot_formats'], cleaned)
        settings['plot_formats'] = cleaned

    # ── Channel selection ──
    # Now that extraction preserves every channel, which one is measured is a real
    # choice rather than an accident of the import step.
    channel = _get_int(config, 'channel', minimum=0, maximum=1024)
    if channel is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['channel'], channel)
        settings['channel'] = channel
    mono_mix = _get_bool(config, 'monoMix')
    if mono_mix is not None:
        _assign(kwargs, available, _CONFIG_ALIASES['mono_mix'], mono_mix)
        settings['mono_mix'] = mono_mix

    # ── Test metadata (operator, weapon, mic position, atmosphere) ──
    # The interface sends this as `metadata`; `testMetadata` is this server's
    # older name for it. Reading only the older one dropped the entire test
    # record -- microphone distance and angle, temperature, humidity, pressure,
    # weapon, operator -- from every run started through the packaged app. The
    # atmospheric correction and the whole provenance block are built from it,
    # so the loss was silent and total.
    raw_meta = config.get('testMetadata')
    if not isinstance(raw_meta, dict) or not raw_meta:
        raw_meta = config.get('metadata')
    if isinstance(raw_meta, dict) and raw_meta:
        clean_meta = {k: v for k, v in raw_meta.items()
                      if isinstance(k, str) and not isinstance(v, (dict, list))}
        if not _assign(kwargs, available, _CONFIG_ALIASES['test_metadata'], clean_meta):
            if 'metadata' in call_params:
                try:
                    from provenance import TestMetadata
                    call_kwargs['metadata'] = TestMetadata.from_dict(clean_meta)
                except Exception as exc:  # noqa: BLE001
                    raise ConfigError(f'Invalid test metadata: {exc}',
                                      [{'field': 'testMetadata', 'message': str(exc)}])
        settings['test_metadata'] = clean_meta

    # ── Output directory ──
    # Confined to the results area, exactly as the development bridge confines
    # it: this is a local server, but a client-supplied path is still a path
    # the client chose, and nothing else in this file lets one out of the
    # workspace. Ignored before this, so the operator's choice in Settings had
    # no effect on where anything was written.
    if _present(config, 'outputDir'):
        raw_out = config['outputDir']
        if not isinstance(raw_out, str):
            raise ConfigError('The output directory must be text.',
                              [{'field': 'outputDir', 'message': 'Must be text.'}])
        out_dir = resolve_within_roots(raw_out, [ANALYSIS_DIR], must_exist=False)
        if out_dir is None:
            raise ConfigError(
                'That output directory is outside the results area.',
                [{'field': 'outputDir', 'message': 'Outside the permitted results area.'}])
        if 'output_dir' in call_params:
            call_kwargs['output_dir'] = out_dir
            settings['output_dir'] = str(out_dir)
        else:
            raise ConfigError(
                'This analysis backend does not accept an output directory.',
                [{'field': 'outputDir', 'message': 'Not supported by the backend.'}])

    # ── Reference (unsuppressed) analysis for insertion loss ──
    if _present(config, 'referenceDir'):
        ref = resolve_within_roots(str(config['referenceDir']), [ANALYSIS_DIR])
        if ref is None or not ref.is_dir():
            raise ConfigError(
                'The reference analysis directory is not a previous analysis in this workspace.',
                [{'field': 'referenceDir', 'message': 'Not an accessible analysis directory.'}])
        if not _assign(kwargs, available, _CONFIG_ALIASES['reference_dir'], str(ref)):
            if 'reference_dir' in call_params:
                call_kwargs['reference_dir'] = ref
            else:
                raise ConfigError(
                    'This analysis backend cannot compute insertion loss against a '
                    'reference measurement.',
                    [{'field': 'referenceDir', 'message': 'Not supported by the backend.'}])
        settings['reference_dir'] = str(ref)

    try:
        analysis_config = AnalysisConfig(**kwargs)
    except TypeError as exc:
        raise ConfigError(f'The analysis backend rejected the configuration: {exc}')
    except ValueError as exc:
        raise ConfigError(f'Invalid configuration: {exc}')

    return analysis_config, calibration, settings, call_kwargs


def validate_input_file(raw_path: Any) -> Path:
    """Resolve the requested input file inside the workspace and check its type."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ConfigError('An input file is required.',
                          [{'field': 'filePath', 'message': 'An input file is required.'}])

    roots = [UPLOAD_DIR, EXTRACT_DIR, ANALYSIS_DIR, DATA_DIR / 'Audio']
    resolved = resolve_within_roots(raw_path, roots)
    if resolved is None or not resolved.is_file():
        raise ConfigError(
            'The input file is not an accessible recording in this workspace.',
            [{'field': 'filePath', 'message': 'Not an accessible uploaded recording.'}])
    if resolved.suffix.lower() not in MEDIA_EXTS:
        raise ConfigError(
            f'{resolved.suffix or "This file type"} is not a supported recording format.',
            [{'field': 'filePath', 'message': 'Unsupported file type.'}])
    return resolved


# ═══════════════════════════════════════════════════════════
#  Run registry — cancellable, and recoverable after a reconnect
# ═══════════════════════════════════════════════════════════
#  Defect: a running analysis could not be stopped, and its completion was
#  delivered exactly once to the socket that started it. A browser refresh (or the
#  UI's own 2-second reconnect) lost the result entirely even though the analysis
#  had finished and the files were on disk.

class AnalysisCancelled(BaseException):
    """
    Raised inside the worker when the operator cancels.

    Derived from BaseException, not Exception, so that the broad `except Exception`
    handlers in the stdout router and in the analysis pipeline cannot swallow a
    cancellation and quietly carry on computing.
    """


class AnalysisRun:
    """One analysis: its state, its log, its subscribers, its cancel switch."""

    def __init__(self, run_id: str, file_path: Path, config: dict):
        self.id = run_id
        self.file_path = file_path
        self.config = config
        self.state = 'starting'            # starting|running|complete|error|cancelled
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.percent = 0.0
        self.message = ''
        self.output_dir: Optional[str] = None
        self.error: Optional[str] = None
        self.log: List[Tuple[int, str]] = []
        self._seq = 0
        self.cancel_event = threading.Event()
        self.delivered = False             # terminal state reached a live client
        self.subscribers: set = set()
        self.lock = threading.RLock()

    # -- state -------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self.state in ('starting', 'running')

    def summary(self) -> dict:
        with self.lock:
            return {
                'runId': self.id,
                'state': self.state,
                'pct': round(self.percent, 1),
                'message': self.message,
                'inputFile': str(self.file_path),
                'outputDir': self.output_dir,
                'error': self.error,
                'startedAt': self.started_at,
                'finishedAt': self.finished_at,
                'elapsedS': round((self.finished_at or time.time()) - self.started_at, 2),
                'lastSeq': self._seq,
            }

    # -- log ---------------------------------------------------------
    def append_log(self, line: str) -> Tuple[int, str]:
        with self.lock:
            self._seq += 1
            entry = (self._seq, line)
            self.log.append(entry)
            if len(self.log) > MAX_RUN_LOG_LINES:
                del self.log[:len(self.log) - MAX_RUN_LOG_LINES]
            return entry

    def log_since(self, seq: int) -> List[Tuple[int, str]]:
        with self.lock:
            return [entry for entry in self.log if entry[0] > seq]

    # -- fan-out -----------------------------------------------------
    def subscribe(self, conn) -> None:
        with self.lock:
            self.subscribers.add(conn)

    def unsubscribe(self, conn) -> None:
        with self.lock:
            self.subscribers.discard(conn)

    def broadcast(self, payload: dict) -> bool:
        payload = dict(payload, runId=self.id)
        with self.lock:
            targets = list(self.subscribers)
        reached = False
        for conn in targets:
            if conn.send(payload):
                reached = True
            else:
                self.unsubscribe(conn)
        return reached


class RunRegistry:
    """All runs, past and present. One analysis at a time."""

    def __init__(self):
        self._runs: Dict[str, AnalysisRun] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def active(self) -> Optional[AnalysisRun]:
        with self._lock:
            for run_id in reversed(self._order):
                run = self._runs[run_id]
                if run.is_active:
                    return run
        return None

    def latest(self) -> Optional[AnalysisRun]:
        with self._lock:
            return self._runs[self._order[-1]] if self._order else None

    def get(self, run_id: str) -> Optional[AnalysisRun]:
        with self._lock:
            return self._runs.get(run_id)

    def add(self, run: AnalysisRun) -> None:
        with self._lock:
            self._runs[run.id] = run
            self._order.append(run.id)
            self._prune_locked()

    def all(self) -> List[AnalysisRun]:
        with self._lock:
            return [self._runs[r] for r in self._order]

    def _prune_locked(self) -> None:
        cutoff = time.time() - RUN_RETENTION_S
        keep: List[str] = []
        for run_id in self._order:
            run = self._runs[run_id]
            if run.is_active or (run.finished_at or run.started_at) >= cutoff:
                keep.append(run_id)
            else:
                self._runs.pop(run_id, None)
        # Always keep the most recent handful regardless of age.
        if not keep and self._order:
            keep = self._order[-5:]
            self._runs = {r: self._runs[r] for r in keep if r in self._runs}
        self._order = keep


RUNS = RunRegistry()


# ═══════════════════════════════════════════════════════════
#  Minimal WebSocket Implementation (RFC 6455)
# ═══════════════════════════════════════════════════════════

def _ws_accept_key(key: str) -> str:
    magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    sha1 = hashlib.sha1((key + magic).encode()).digest()
    return base64.b64encode(sha1).decode()


def _ws_frame_size(data: bytes) -> int:
    """Total byte length of the first frame in `data`, or 0 when incomplete."""
    if len(data) < 2:
        return 0
    b1 = data[1]
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    offset = 2
    if length == 126:
        if len(data) < 4:
            return 0
        length = struct.unpack('!H', data[2:4])[0]
        offset = 4
    elif length == 127:
        if len(data) < 10:
            return 0
        length = struct.unpack('!Q', data[2:10])[0]
        offset = 10
    if masked:
        offset += 4
    total = offset + length
    return total if len(data) >= total else 0


def _ws_decode_frame(data: bytes):
    """Decode one frame. Returns (opcode, payload, bytes_consumed)."""
    size = _ws_frame_size(data)
    if size == 0:
        return None, None, 0

    b0, b1 = data[0], data[1]
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    offset = 2
    if length == 126:
        length = struct.unpack('!H', data[2:4])[0]
        offset = 4
    elif length == 127:
        length = struct.unpack('!Q', data[2:10])[0]
        offset = 10

    if masked:
        mask = data[offset:offset + 4]
        offset += 4
        payload = bytearray(data[offset:offset + length])
        for i in range(length):
            payload[i] ^= mask[i % 4]
        payload = bytes(payload)
    else:
        payload = data[offset:offset + length]

    return opcode, payload, size


def _ws_encode_frame(payload: bytes, opcode: int = 1) -> bytes:
    """Encode a frame (server→client, unmasked)."""
    frame = bytearray()
    frame.append(0x80 | opcode)
    length = len(payload)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack('!H', length))
    else:
        frame.append(127)
        frame.extend(struct.pack('!Q', length))
    frame.extend(payload)
    return bytes(frame)


class WSConnection:
    """A websocket with its own send lock, so concurrent senders cannot interleave frames."""

    def __init__(self, sock):
        self.sock = sock
        self._lock = threading.Lock()
        self.closed = False

    def send(self, payload: dict) -> bool:
        if self.closed:
            return False
        try:
            frame = _ws_encode_frame(json.dumps(payload).encode('utf-8'))
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self.closed:
                return False
            try:
                self.sock.sendall(frame)
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.closed = True
                return False

    def send_control(self, payload: bytes, opcode: int) -> bool:
        if self.closed:
            return False
        with self._lock:
            try:
                self.sock.sendall(_ws_encode_frame(payload, opcode=opcode))
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.closed = True
                return False

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.close()
        except Exception:
            pass


_ws_connections: set = set()
_ws_lock = threading.Lock()

# Track whether the browser has been opened this session
_browser_opened = False


# ═══════════════════════════════════════════════════════════
#  Streaming multipart parser
# ═══════════════════════════════════════════════════════════
#  Defect: the whole request body was read into memory with
#  self.rfile.read(content_length) and then copied again by str.split(). A
#  30-minute 192 kHz 24-bit stereo take is ~2 GB on the wire and needed roughly
#  three times that in RAM before a single byte reached disk. The body is now
#  consumed in 1 MiB chunks and the file part is written straight to its
#  destination.

class MultipartError(Exception):
    pass


class _BoundedReader:
    """Reads at most `limit` bytes from a stream."""

    def __init__(self, stream, limit: int):
        self._stream = stream
        self.remaining = max(0, int(limit))

    def read(self, size: int) -> bytes:
        if self.remaining <= 0:
            return b''
        chunk = self._stream.read(min(size, self.remaining))
        if not chunk:
            self.remaining = 0
            return b''
        self.remaining -= len(chunk)
        return chunk


def parse_multipart_stream(
    stream,
    content_length: int,
    content_type: str,
    *,
    file_dir: Path,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Parse multipart/form-data, streaming any file part to `file_dir`.

    Returns (fields, files) where files maps the field name to
    {'filename': original, 'path': Path (a .part temp file), 'size': int}.
    The caller renames or deletes the temp files.
    """
    match = re.search(r'boundary=("?)([^";]+)\1', content_type or '')
    if not match:
        raise MultipartError('Malformed multipart request: no boundary.')
    boundary = match.group(2).strip().encode('utf-8')
    if not boundary:
        raise MultipartError('Malformed multipart request: empty boundary.')

    if content_length <= 0:
        raise MultipartError('Empty upload.')
    if content_length > max_bytes:
        raise MultipartError(f'Upload exceeds the {max_bytes // (1024**3)} GB limit.')

    delimiter = b'--' + boundary
    reader = _BoundedReader(stream, content_length)
    buf = bytearray()
    fields: Dict[str, str] = {}
    files: Dict[str, Dict[str, Any]] = {}
    file_dir.mkdir(parents=True, exist_ok=True)

    def fill(minimum: int) -> bool:
        """Ensure at least `minimum` bytes are buffered. False at end of body."""
        while len(buf) < minimum:
            chunk = reader.read(UPLOAD_CHUNK)
            if not chunk:
                return False
            buf.extend(chunk)
        return True

    def cleanup_partials():
        for info in files.values():
            try:
                Path(info['path']).unlink(missing_ok=True)
            except OSError:
                pass

    try:
        # Preamble up to the first delimiter
        while True:
            index = buf.find(delimiter)
            if index >= 0:
                del buf[:index + len(delimiter)]
                break
            if len(buf) > 1 << 20:
                del buf[:len(buf) - len(delimiter)]
            if not fill(len(buf) + 1):
                raise MultipartError('Malformed multipart request: no opening boundary.')

        while True:
            # After a delimiter: '--' means the end, CRLF means another part.
            if not fill(2):
                break
            if buf[:2] == b'--':
                break
            if buf[:2] == b'\r\n':
                del buf[:2]
            else:  # tolerate a bare LF
                del buf[:1]

            # Headers
            while True:
                header_end = buf.find(b'\r\n\r\n')
                if header_end >= 0:
                    break
                if len(buf) > MAX_FIELD_BYTES:
                    raise MultipartError('Multipart headers too large.')
                if not fill(len(buf) + 1):
                    raise MultipartError('Malformed multipart request: truncated headers.')
            headers_raw = bytes(buf[:header_end]).decode('utf-8', errors='replace')
            del buf[:header_end + 4]

            name_match = re.search(r'name="([^"]*)"', headers_raw)
            filename_match = re.search(r'filename="([^"]*)"', headers_raw)
            field_name = name_match.group(1) if name_match else ''
            filename = filename_match.group(1) if filename_match else None
            is_file = filename is not None

            sink = None
            temp_path: Optional[Path] = None
            written = 0
            text_parts: List[bytes] = []
            if is_file:
                temp_path = file_dir / f'.upload-{uuid.uuid4().hex}.part'
                sink = open(temp_path, 'wb')

            # Body up to CRLF + delimiter. Hold back the tail that could still be
            # the start of the delimiter across a chunk boundary.
            terminator = b'\r\n' + delimiter
            try:
                while True:
                    index = buf.find(terminator)
                    if index >= 0:
                        payload = bytes(buf[:index])
                        del buf[:index + len(terminator)]
                        if sink:
                            sink.write(payload)
                            written += len(payload)
                        else:
                            text_parts.append(payload)
                        break

                    keep = len(terminator) - 1
                    if len(buf) > keep:
                        flushable = bytes(buf[:len(buf) - keep])
                        del buf[:len(flushable)]
                        if sink:
                            sink.write(flushable)
                            written += len(flushable)
                            if written > max_bytes:
                                raise MultipartError('Upload exceeds the size limit.')
                        else:
                            text_parts.append(flushable)
                            if sum(len(p) for p in text_parts) > MAX_FIELD_BYTES:
                                raise MultipartError(f'Field {field_name!r} is too large.')
                    if not fill(len(buf) + 1):
                        raise MultipartError('Malformed multipart request: truncated body.')
            finally:
                if sink:
                    sink.close()

            if is_file:
                files[field_name] = {
                    'filename': filename,
                    'path': temp_path,
                    'size': written,
                }
            else:
                fields[field_name] = b''.join(text_parts).decode('utf-8', errors='replace')

        return fields, files

    except Exception:
        cleanup_partials()
        raise


def parse_multipart(body: bytes, content_type: str) -> dict:
    """
    In-memory convenience wrapper, kept for small bodies and tests.

    Returns {field_name: (filename, data)}. Uploads do NOT go through this — see
    parse_multipart_stream().
    """
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fields, files = parse_multipart_stream(
            io.BytesIO(body), len(body), content_type, file_dir=Path(tmp))
        result: Dict[str, Tuple[Optional[str], bytes]] = {}
        for name, value in fields.items():
            result[name] = (None, value.encode('utf-8'))
        for name, info in files.items():
            data = Path(info['path']).read_bytes()
            Path(info['path']).unlink(missing_ok=True)
            result[name] = (info['filename'], data)
        return result


# ═══════════════════════════════════════════════════════════
#  Results helpers
# ═══════════════════════════════════════════════════════════

_SHOT_IMAGE_RE = re.compile(r'shot[_-]?(\d+)', re.IGNORECASE)


def collect_shot_images(output_dir: Path) -> Tuple[Dict[str, str], List[Optional[str]]]:
    """
    Map per-shot images to shot NUMBERS, not to sort order.

    Defect: images were matched to metrics by their position in a
    lexicographically sorted list. 'shot_100_summary.png' sorts before
    'shot_99_summary.png', so past 99 shots every image was attached to the wrong
    measurement — a report showing one shot's waveform beside another's numbers.

    Returns (by_shot_number, ordered) where `ordered` is indexed by shot number - 1
    with None for gaps, so the legacy positional consumer still lines up.
    """
    shots_dir = output_dir / 'shots'
    by_number: Dict[str, str] = {}
    unnumbered: List[str] = []
    if shots_dir.is_dir():
        for f in sorted(shots_dir.iterdir()):
            if f.suffix.lower() not in ('.png', '.svg', '.jpg', '.jpeg', '.pdf'):
                continue
            match = _SHOT_IMAGE_RE.search(f.stem)
            if match:
                by_number.setdefault(str(int(match.group(1))), f.name)
            else:
                unnumbered.append(f.name)

    ordered: List[Optional[str]] = []
    if by_number:
        highest = max(int(k) for k in by_number)
        ordered = [by_number.get(str(i)) for i in range(1, highest + 1)]
    ordered.extend(unnumbered)
    return by_number, ordered


def _analysis_timestamp(meta: dict, fallback: Path) -> float:
    """Sort key for history: when the analysis actually ran."""
    from datetime import datetime

    stamp = None
    if isinstance(meta, dict):
        analysis = meta.get('analysis')
        if isinstance(analysis, dict):
            stamp = analysis.get('timestamp')
        if not stamp:
            stamp = meta.get('timestamp')
    if isinstance(stamp, str) and stamp:
        try:
            return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
        except ValueError:
            # Legacy compact form "YYYYmmdd_HHMMSS"
            try:
                return datetime.strptime(stamp, '%Y%m%d_%H%M%S').timestamp()
            except ValueError:
                pass
    try:
        return fallback.stat().st_mtime
    except OSError:
        return 0.0


def _history_summary(meta: dict) -> dict:
    """A schema-tolerant digest for the history list (v1 and v2 metadata)."""
    def dig(*path, default=None):
        node: Any = meta
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node if node is not None else default

    aggregate = dig('aggregate', default={}) or {}
    stats = aggregate.get('statistics') or {}
    peak = None
    if isinstance(stats.get('Lpeak_Z'), dict):
        peak = stats['Lpeak_Z'].get('max')
    if peak is None:
        peak = aggregate.get('Lpeak_Z_max')

    return {
        'schema_version': meta.get('schema_version', '1.0'),
        'input_file': dig('analysis', 'input_file', default=meta.get('input_file', '')),
        'timestamp': dig('analysis', 'timestamp', default=meta.get('timestamp', '')),
        'sample_rate': dig('source', 'sample_rate', default=meta.get('sample_rate')),
        'duration_s': dig('source', 'duration_s', default=meta.get('duration_s')),
        'n_shots': aggregate.get('n_shots', meta.get('n_shots')),
        'Lpeak_Z_max': peak,
        'level_unit': dig('calibration', 'level_unit', default='dB'),
        'calibrated': dig('calibration', 'calibrated'),
        'is_valid': dig('quality', 'is_valid'),
        'test_id': dig('test_metadata', 'test_id', default=''),
    }


# ═══════════════════════════════════════════════════════════
#  HTTP + WebSocket Request Handler
# ═══════════════════════════════════════════════════════════

class SASAHandler(http.server.BaseHTTPRequestHandler):
    """Handles HTTP requests and WebSocket upgrades."""

    protocol_version = 'HTTP/1.1'
    _is_websocket = False

    def log_message(self, format, *args):  # noqa: A002
        pass  # Suppress default logging

    # ── Routing ──

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/ws':
            upgrade = self.headers.get('Upgrade', '').lower()
            key = self.headers.get('Sec-WebSocket-Key', '')
            if upgrade == 'websocket' and key:
                self._handle_ws_upgrade(key)
            else:
                self._send_text('Expected WebSocket upgrade', 400)
            return

        if path == '/api/analyses':
            return self._api_analyses()
        if path == '/api/results':
            return self._api_results(query)
        if path == '/api/image':
            return self._api_image(query)
        if path == '/api/runs':
            return self._send_json({'runs': [r.summary() for r in RUNS.all()]})
        if path == '/api/run':
            return self._api_run(query)
        if path == '/api/report':
            return self._api_report(query)

        self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/upload':
            return self._api_upload()
        if parsed.path == '/api/cancel':
            return self._api_cancel(parse_qs(parsed.query))
        self._send_json({'error': 'Not found'}, 404)

    # ── Static files ──

    def _serve_static(self, path: str):
        if path == '/':
            path = '/index.html'

        candidate = RENDERER_DIR / path.lstrip('/')
        file_path = resolve_within_roots(candidate, [RENDERER_DIR])
        if file_path is None or not file_path.is_file():
            self._send_text('Not Found', 404)
            return

        ext = file_path.suffix.lower()
        content_type = MIME_MAP.get(ext, 'application/octet-stream')
        try:
            data = file_path.read_bytes()
        except OSError:
            self._send_text('Not Found', 404)
            return

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    # ── API: Upload (streamed to disk) ──

    def _api_upload(self):
        content_type = self.headers.get('Content-Type', '')
        try:
            content_length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            return self._send_json({'error': 'Invalid Content-Length'}, 400)

        if content_length <= 0:
            return self._send_json({'error': 'Empty upload'}, 400)
        if content_length > MAX_UPLOAD_BYTES:
            return self._send_json(
                {'error': f'Upload exceeds the {MAX_UPLOAD_BYTES // (1024**3)} GB limit.'}, 413)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        try:
            _fields, files = parse_multipart_stream(
                self.rfile, content_length, content_type, file_dir=UPLOAD_DIR)
        except MultipartError as exc:
            return self._send_json({'error': str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            _log(f'Upload failed: {exc}')
            return self._send_json({'error': 'Upload failed'}, 500)

        part = files.get('file') or (next(iter(files.values())) if files else None)
        for name, info in files.items():
            if info is not part:
                Path(info['path']).unlink(missing_ok=True)

        if part is None:
            return self._send_json({'error': 'No file uploaded'}, 400)

        temp_path = Path(part['path'])
        safe_name = safe_upload_name(part['filename'] or '')
        if not safe_name:
            temp_path.unlink(missing_ok=True)
            return self._send_json({'error': 'No usable filename'}, 400)

        suffix = Path(safe_name).suffix.lower()
        if suffix not in MEDIA_EXTS:
            temp_path.unlink(missing_ok=True)
            return self._send_json(
                {'error': f'{suffix or "That file type"} is not a supported recording format.'}, 400)

        dest = unique_destination(UPLOAD_DIR, safe_name)
        # Belt and braces: prove the destination is inside the upload directory
        # even after every sanitising step above.
        confirmed = resolve_within_roots(dest, [UPLOAD_DIR], must_exist=False)
        if confirmed is None:
            temp_path.unlink(missing_ok=True)
            return self._send_json({'error': 'Rejected upload path'}, 400)

        try:
            temp_path.replace(confirmed)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            return self._send_json({'error': f'Could not store upload: {exc}'}, 500)

        self._send_json({
            'path': str(confirmed),
            'name': confirmed.name,
            'originalName': part['filename'],
            'size': part['size'],
            'needsExtraction': suffix in VIDEO_EXTS,
        })

    # ── API: List analyses (newest first, by analysis time) ──

    def _api_analyses(self):
        if not ANALYSIS_DIR.is_dir():
            return self._send_json([])

        entries = []
        for d in ANALYSIS_DIR.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / 'analysis_metadata.json'
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            entries.append({
                'name': d.name,
                'path': str(d),
                'meta': meta,
                'summary': _history_summary(meta),
                'sortKey': _analysis_timestamp(meta, meta_path),
            })

        # Defect: this used sorted(..., reverse=True) on Path objects, i.e.
        # reverse-alphabetical order. 'z_recording_20200101' outranked
        # 'a_recording_20260101'. Sort by when the analysis actually ran.
        entries.sort(key=lambda e: e['sortKey'], reverse=True)
        for entry in entries:
            entry.pop('sortKey', None)
        self._send_json(entries)

    # ── API: Load results ──

    def _api_results(self, query: dict):
        dir_param = query.get('dir', [''])[0]
        if not dir_param:
            return self._send_json({'error': 'Missing dir parameter'}, 400)

        output_dir = resolve_within_roots(dir_param, [ANALYSIS_DIR])
        if output_dir is None or not output_dir.is_dir():
            return self._send_json({'error': 'No such analysis'}, 404)

        meta_path = output_dir / 'analysis_metadata.json'
        if not meta_path.is_file():
            return self._send_json({'error': 'No analysis_metadata.json found'}, 404)

        try:
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            return self._send_json({'error': str(exc)}, 500)

        images: Dict[str, Dict[str, str]] = {}
        for f in output_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ('.png', '.html', '.svg', '.pdf'):
                key = f.stem
                kind = 'html' if f.suffix.lower() == '.html' else f.suffix.lower().lstrip('.')
                images.setdefault(key, {})[kind] = f.name

        by_shot, ordered = collect_shot_images(output_dir)

        csv_data = None
        csv_path = output_dir / 'metrics_summary.csv'
        if csv_path.is_file():
            try:
                csv_data = csv_path.read_text(encoding='utf-8')
            except OSError:
                csv_data = None

        self._send_json({
            'metadata': metadata,
            'summary': _history_summary(metadata),
            'images': images,
            'shotImages': [name for name in ordered if name],
            'shotImagesByNumber': by_shot,
            'shotImagesOrdered': ordered,
            'csv': csv_data,
            'hasReport': (output_dir / 'report.html').is_file(),
            'outputDir': str(output_dir),
        })

    # ── API: Serve a file from an analysis output ──

    def _api_image(self, query: dict):
        dir_param = query.get('dir', [''])[0]
        file_name = query.get('file', [''])[0]
        sub = query.get('sub', [''])[0]

        if not dir_param or not file_name:
            return self._send_text('Missing dir or file', 400)

        base = resolve_within_roots(dir_param, [ANALYSIS_DIR])
        if base is None or not base.is_dir():
            return self._send_text('Not found', 404)

        # Leaf names only — no traversal through the sub/file parameters.
        safe_name = Path(file_name.replace('\\', '/')).name
        safe_sub = Path(sub.replace('\\', '/')).name if sub else None
        if not safe_name or safe_name in ('.', '..'):
            return self._send_text('Not found', 404)

        candidate = base / safe_sub / safe_name if safe_sub else base / safe_name
        file_path = resolve_within_roots(candidate, [ANALYSIS_DIR])
        if file_path is None or not file_path.is_file():
            return self._send_text('Not found', 404)

        ext = file_path.suffix.lower()
        if ext not in MIME_MAP:
            return self._send_text('Unsupported file type', 415)

        try:
            data = file_path.read_bytes()
        except OSError:
            return self._send_text('Not found', 404)

        self.send_response(200)
        self.send_header('Content-Type', MIME_MAP[ext])
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    # ── API: Run status (recovery without a websocket) ──

    def _api_run(self, query: dict):
        run_id = query.get('id', [''])[0]
        run = RUNS.get(run_id) if run_id else RUNS.latest()
        if run is None:
            return self._send_json({'error': 'No such run'}, 404)
        try:
            since = int(query.get('since', ['0'])[0])
        except (TypeError, ValueError):
            since = 0
        payload = run.summary()
        payload['log'] = [{'seq': seq, 'line': line} for seq, line in run.log_since(since)]
        self._send_json(payload)

    def _api_cancel(self, query: dict):
        run_id = query.get('id', [''])[0]
        run = RUNS.get(run_id) if run_id else RUNS.active()
        if run is None:
            return self._send_json({'error': 'No such run'}, 404)
        if not run.is_active:
            return self._send_json({'runId': run.id, 'state': run.state, 'cancelled': False})
        run.cancel_event.set()
        return self._send_json({'runId': run.id, 'state': 'cancelling', 'cancelled': True})

    # ── API: Generate a customer report from an analysis directory ──

    def _api_report(self, query: dict):
        dir_param = query.get('dir', [''])[0]
        if not dir_param:
            return self._send_text('Missing dir', 400)
        output_dir = resolve_within_roots(dir_param, [ANALYSIS_DIR])
        if output_dir is None or not output_dir.is_dir():
            return self._send_text('Not found', 404)

        try:
            src = str(SOURCE_DIR)
            if src not in sys.path:
                sys.path.insert(0, src)
            from report import build_report_from_directory
            report_path = build_report_from_directory(output_dir)
            data = Path(report_path).read_bytes()
        except Exception as exc:  # noqa: BLE001
            _log(f'Report generation failed: {exc}\n{traceback.format_exc()}')
            return self._send_text(f'Report generation failed: {exc}', 500)

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Content-Disposition', 'inline; filename="report.html"')
        self.end_headers()
        self.wfile.write(data)

    # ═══════════════════════════════════════════════════
    #  WebSocket
    # ═══════════════════════════════════════════════════

    def _handle_ws_upgrade(self, key: str):
        """Perform the handshake and enter the frame loop."""
        accept = _ws_accept_key(key)
        response = (
            'HTTP/1.1 101 Switching Protocols\r\n'
            'Upgrade: websocket\r\n'
            'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Accept: {accept}\r\n'
            '\r\n'
        ).encode()
        self.wfile.write(response)
        self.wfile.flush()

        self._is_websocket = True
        self.close_connection = True

        conn = WSConnection(self.connection)
        with _ws_lock:
            _ws_connections.add(conn)

        try:
            self._ws_on_open(conn)
            self._ws_frame_loop(conn)
        finally:
            conn.closed = True
            for run in RUNS.all():
                run.unsubscribe(conn)
            with _ws_lock:
                _ws_connections.discard(conn)

    def _ws_on_open(self, conn: WSConnection):
        """
        Bring a newly connected client up to date.

        The UI reconnects automatically two seconds after any drop, so this is the
        path that makes a completed analysis recoverable instead of lost.
        """
        run = RUNS.active() or RUNS.latest()
        if run is None:
            conn.send({'type': 'status', 'run': None})
            return

        run.subscribe(conn)
        conn.send({'type': 'status', 'run': run.summary()})
        for seq, line in run.log_since(0):
            conn.send({'type': 'log', 'line': line, 'seq': seq, 'runId': run.id})

        if run.is_active:
            conn.send({'type': 'progress', 'pct': run.percent,
                       'message': run.message, 'runId': run.id})
        elif not run.delivered:
            # A terminal state that no live client ever saw.
            self._send_terminal(conn, run)

    @staticmethod
    def _send_terminal(conn: WSConnection, run: AnalysisRun):
        if run.state == 'complete':
            ok = conn.send({'type': 'complete', 'outputDir': run.output_dir,
                            'runId': run.id, 'elapsedS': run.summary()['elapsedS']})
        elif run.state == 'cancelled':
            ok = conn.send({'type': 'cancelled', 'runId': run.id,
                            'message': 'Analysis cancelled.'})
        else:
            ok = conn.send({'type': 'error', 'message': run.error or 'Analysis failed.',
                            'runId': run.id})
        if ok:
            run.delivered = True

    def _ws_frame_loop(self, conn: WSConnection):
        """Read frames and dispatch messages."""
        sock = conn.sock
        try:
            buffered = self.rfile.peek()
        except Exception:
            buffered = b''
        if buffered:
            buf = bytes(buffered)
            self.rfile.read(len(buffered))
        else:
            buf = b''

        sock.settimeout(30.0)

        while not conn.closed:
            while buf:
                opcode, payload, consumed = _ws_decode_frame(buf)
                if consumed == 0:
                    break
                buf = buf[consumed:]

                if payload is None:
                    continue
                if opcode == 0x8:  # Close
                    conn.send_control(b'', 0x8)
                    return
                if opcode == 0x9:  # Ping → Pong
                    if not conn.send_control(payload, 0xA):
                        return
                elif opcode == 0x1:  # Text
                    self._ws_on_message(conn, payload.decode('utf-8', errors='replace'))

            try:
                chunk = sock.recv(65536)
                if not chunk:
                    return
                buf += chunk
            except socket.timeout:
                if not conn.send_control(b'ping', 0x9):
                    return
            except (ConnectionResetError, BrokenPipeError, OSError):
                return

    def _ws_on_message(self, conn: WSConnection, text: str):
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return

        kind = msg.get('type')

        if kind == 'run-analysis':
            self._start_analysis(conn, msg.get('config') or {})
            return

        if kind in ('cancel-analysis', 'cancel'):
            run_id = msg.get('runId')
            run = RUNS.get(run_id) if run_id else RUNS.active()
            if run is None or not run.is_active:
                conn.send({'type': 'error', 'message': 'No analysis is running.',
                           'runId': run_id})
                return
            run.cancel_event.set()
            run.broadcast({'type': 'cancelling', 'message': 'Cancelling…'})
            conn.send({'type': 'cancelling', 'runId': run.id})
            return

        if kind == 'subscribe':
            run_id = msg.get('runId')
            run = RUNS.get(run_id) if run_id else (RUNS.active() or RUNS.latest())
            if run is None:
                conn.send({'type': 'status', 'run': None})
                return
            run.subscribe(conn)
            try:
                since = int(msg.get('sinceSeq') or 0)
            except (TypeError, ValueError):
                since = 0
            conn.send({'type': 'status', 'run': run.summary()})
            for seq, line in run.log_since(since):
                conn.send({'type': 'log', 'line': line, 'seq': seq, 'runId': run.id})
            if not run.is_active:
                self._send_terminal(conn, run)
            return

        if kind == 'status':
            run = RUNS.active() or RUNS.latest()
            conn.send({'type': 'status', 'run': run.summary() if run else None,
                       'runs': [r.summary() for r in RUNS.all()]})
            return

        if kind == 'ping':
            conn.send({'type': 'pong'})

    # ═══════════════════════════════════════════════════
    #  Analysis — runs IN-PROCESS (no subprocess)
    # ═══════════════════════════════════════════════════

    def _start_analysis(self, conn: WSConnection, config: dict):
        """Validate, then hand off to a worker thread."""
        if not isinstance(config, dict):
            conn.send({'type': 'error', 'message': 'Configuration must be an object.'})
            return

        active = RUNS.active()
        if active is not None:
            conn.send({
                'type': 'error',
                'message': 'An analysis is already running. Cancel it or wait for it to finish.',
                'runId': active.id,
                'busy': True,
            })
            active.subscribe(conn)
            return

        try:
            _ensure_analysis_imports()
        except ImportError as exc:
            conn.send({'type': 'error', 'message': f'Failed to import analysis modules: {exc}'})
            return

        AnalysisConfig = _analysis_modules['AnalysisConfig']
        Calibration = _analysis_modules['Calibration']
        analyze_file = _analysis_modules['analyze_file']

        try:
            input_file = validate_input_file(config.get('filePath'))
            analysis_config, calibration, settings, call_kwargs = build_analysis_config(
                config, AnalysisConfig, Calibration, analyze_fn=analyze_file)
        except ConfigError as exc:
            conn.send({'type': 'error', 'message': str(exc), 'fields': exc.fields})
            return
        except Exception as exc:  # noqa: BLE001
            conn.send({'type': 'error', 'message': f'Invalid configuration: {exc}'})
            return

        run = AnalysisRun(uuid.uuid4().hex[:16], input_file, settings)
        RUNS.add(run)
        run.subscribe(conn)
        conn.send({'type': 'started', 'runId': run.id, 'inputFile': str(input_file),
                   'calibration': calibration.to_dict()})

        threading.Thread(
            target=self._run_analysis_worker,
            args=(run, analysis_config, calibration, call_kwargs),
            name=f'sasa-analysis-{run.id}',
            daemon=True,
        ).start()

    @staticmethod
    def _run_analysis_worker(run: AnalysisRun, analysis_config, calibration,
                             call_kwargs: Optional[Dict[str, Any]] = None):
        """
        Execute one analysis with its own captured stdout.

        stdout is captured per THREAD (see _ThreadRoutedStream), so two runs — or a
        run and the server's own logging — can never appear in each other's logs,
        and no failure path can leave the process without a stdout.
        """
        progress_re = re.compile(r'^\[SASA-PROGRESS\]\s+(-?\d+(?:\.\d+)?)\s*(.*)$')
        output_re = re.compile(r'^\[SASA-OUTPUT\]\s+(.+)$')
        legacy_output_re = re.compile(
            r'^(?:Output directory|Results saved to|Saving results to)\s*:?\s*(.+)$', re.I)
        legacy_progress_re = re.compile(r'\[(\d+)/(\d+)\]')

        def emit_progress(pct: float, message: str = ''):
            pct = max(0.0, min(100.0, float(pct)))
            run.percent = pct
            if message:
                run.message = message
            run.broadcast({'type': 'progress', 'pct': round(pct, 1), 'message': message})

        def emit_line(line: str):
            # Cancellation is cooperative, and this is the only place the worker
            # regains control while the backend is running. It is checked FIRST,
            # before the control-line early returns below: during plotting the
            # pipeline emits nothing but progress markers, so a check placed after
            # them would not notice a cancellation until the plots had finished.
            if cancel_armed and run.cancel_event.is_set():
                raise AnalysisCancelled()

            # Control lines drive the UI; they are not shown as log noise.
            match = progress_re.match(line)
            if match:
                emit_progress(float(match.group(1)), match.group(2).strip())
                return
            match = output_re.match(line)
            if match:
                run.output_dir = match.group(1).strip()
                return
            match = legacy_output_re.match(line.strip())
            if match:
                run.output_dir = match.group(1).strip()
            match = legacy_progress_re.search(line)
            if match:
                num, den = int(match.group(1)), int(match.group(2))
                if den:
                    emit_progress(min(99.0, num / den * 100.0))
            seq, text = run.append_log(line)
            run.broadcast({'type': 'log', 'line': text, 'seq': seq})

        # Cancellation is raised from emit_line while the analysis is running. Once
        # we are tearing down, flushing the last partial line must NOT raise again —
        # doing so would abort the handler that marks the run cancelled and tells the
        # operator, leaving the UI waiting forever on a run that had already stopped.
        cancel_armed = True

        assembler = _LineAssembler(emit_line)

        def finish_capture():
            """Flush buffered output without re-raising a pending cancellation."""
            nonlocal cancel_armed
            cancel_armed = False
            try:
                assembler.close()
            except AnalysisCancelled:
                pass

        def say(message: str):
            emit_line(message)

        _install_stream_routers()
        if _stdout_router:
            _stdout_router.bind(assembler.feed)
        if _stderr_router:
            _stderr_router.bind(assembler.feed)

        run.state = 'running'
        try:
            say(f'Starting analysis: {run.file_path}')
            say(f'Calibration: {calibration.description} '
                f'({"calibrated" if calibration.calibrated else "UNCALIBRATED — relative levels"})')
            emit_progress(1, 'Preparing')

            if run.cancel_event.is_set():
                raise AnalysisCancelled()

            wav_path = run.file_path
            if wav_path.suffix.lower() in VIDEO_EXTS:
                say(f'[Video detected] Extracting audio from {wav_path.name}...')
                emit_progress(2, 'Extracting audio')
                wav_path = extract_audio_from_video(wav_path, EXTRACT_DIR, log=say)

            if run.cancel_event.is_set():
                raise AnalysisCancelled()

            emit_progress(5, 'Analysing')
            analyze_file = _analysis_modules['analyze_file']
            result = SASAHandler._call_analyze(
                analyze_file, wav_path, analysis_config, run, call_kwargs or {}, say,
                original_path=run.file_path)

            finish_capture()
            output_dir = getattr(result, 'output_dir', None) or run.output_dir
            run.output_dir = str(output_dir) if output_dir else run.output_dir
            run.state = 'complete'
            run.finished_at = time.time()
            emit_progress(100, 'Complete')
            reached = run.broadcast({'type': 'complete', 'outputDir': run.output_dir,
                                     'elapsedS': run.summary()['elapsedS']})
            run.delivered = reached
            _log(f'Run {run.id} complete: {run.output_dir}')

        except AnalysisCancelled:
            finish_capture()
            run.state = 'cancelled'
            run.finished_at = time.time()
            run.error = 'Analysis cancelled by the operator.'
            run.delivered = run.broadcast({'type': 'cancelled', 'message': run.error})
            _log(f'Run {run.id} cancelled')

        except Exception as exc:  # noqa: BLE001
            finish_capture()
            tb = traceback.format_exc()
            run.state = 'error'
            run.finished_at = time.time()
            run.error = str(exc)
            run.append_log(f'Error: {tb}')
            run.delivered = run.broadcast({'type': 'error', 'message': str(exc)})
            _log(f'Run {run.id} failed: {exc}\n{tb}')

        finally:
            if _stdout_router:
                _stdout_router.unbind()
            if _stderr_router:
                _stderr_router.unbind()

    @staticmethod
    def _call_analyze(analyze_file, wav_path: Path, analysis_config, run: AnalysisRun,
                      call_kwargs: Dict[str, Any], say: Callable[[str], None],
                      *, original_path: Optional[Path] = None):
        """
        Call analyze_file(), passing progress/cancel hooks when it accepts them.

        main.py is being rewritten in parallel. If it grows a cancel hook, the run
        stops promptly; if it does not, cancellation still works through the log
        callback, which fires at every stage boundary.
        """
        kwargs: Dict[str, Any] = dict(call_kwargs)
        try:
            params = inspect.signature(analyze_file).parameters
        except (TypeError, ValueError):
            params = {}

        # The operator's own file, before any extraction, belongs in the record.
        if 'original_path' in params and original_path is not None:
            kwargs['original_path'] = original_path

        # Collect the backend's non-fatal warnings so they reach the operator's log
        # rather than only the run record.
        backend_warnings: List[str] = []
        if 'warnings' in params:
            kwargs['warnings'] = backend_warnings

        for name in ('cancel_event', 'cancel', 'should_cancel'):
            if name in params:
                kwargs[name] = run.cancel_event
                break
        for name in ('progress_callback', 'on_progress', 'progress'):
            if name in params:
                def _progress(pct, message=''):
                    run.percent = max(0.0, min(100.0, float(pct)))
                    run.broadcast({'type': 'progress', 'pct': round(run.percent, 1),
                                   'message': str(message)})
                kwargs[name] = _progress
                break

        try:
            if 'output_base' in params:
                kwargs.setdefault('output_base', ANALYSIS_DIR)
                return analyze_file(wav_path, analysis_config, **kwargs)
            return analyze_file(wav_path, analysis_config, ANALYSIS_DIR, **kwargs)
        finally:
            for warning in backend_warnings:
                say(f'  WARNING: {warning}')

    # ── HTTP Helpers ──

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)


# ═══════════════════════════════════════════════════════════
#  Threaded HTTP Server
# ═══════════════════════════════════════════════════════════

class ThreadedHTTPServer(http.server.HTTPServer):
    """Handle each request in a separate thread (needed for WebSocket)."""
    allow_reuse_address = True
    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self._process, args=(request, client_address))
        t.daemon = True
        t.start()

    def _process(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            try:
                self.shutdown_request(request)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════
#  Port Management
# ═══════════════════════════════════════════════════════════

def _kill_existing_server(port: int):
    """Kill any existing SASA server on the given port so we can reuse it."""
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'],
                                    capture_output=True, text=True, timeout=5)
            pids = result.stdout.strip().split('\n')
            my_pid = str(os.getpid())
            for pid in pids:
                pid = pid.strip()
                if pid and pid != my_pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
            if any(p.strip() and p.strip() != my_pid for p in pids):
                time.sleep(0.5)
        except Exception:
            pass
    elif sys.platform == 'win32':
        try:
            result = subprocess.run(['netstat', '-ano'],
                                    capture_output=True, text=True, timeout=5)
            my_pid = str(os.getpid())
            for line in result.stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    pid = line.split()[-1].strip()
                    if pid and pid not in (my_pid, '0'):
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', pid],
                                           capture_output=True, timeout=5)
                        except Exception:
                            pass
            time.sleep(0.5)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    global _browser_opened

    _setup_logging()
    _install_stream_routers()
    _log(f'SASA starting — frozen={getattr(sys, "frozen", False)}, platform={sys.platform}')
    _log(f'BASE_DIR={BASE_DIR}')
    _log(f'RENDERER_DIR={RENDERER_DIR}')
    _log(f'SOURCE_DIR={SOURCE_DIR}')
    _log(f'DATA_DIR={DATA_DIR}')

    for d in (UPLOAD_DIR, ANALYSIS_DIR, EXTRACT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not RENDERER_DIR.is_dir() or not (RENDERER_DIR / 'index.html').is_file():
        _log(f'ERROR: UI renderer not found at {RENDERER_DIR}')
        return 1

    _kill_existing_server(PORT)

    port = PORT
    server = None
    for _ in range(10):
        try:
            server = ThreadedHTTPServer(('127.0.0.1', port), SASAHandler)
            break
        except OSError as e:
            _log(f'Port {port} in use ({e}), trying next...')
            port += 1

    if server is None:
        _log(f'ERROR: Could not bind to any port ({PORT}-{port})')
        return 1

    url = f'http://localhost:{port}'
    _log(f'Server bound to port {port}')
    _log(f"""
  ╔══════════════════════════════════════════╗
  ║   SASA — Shot Acoustic Spectral Analysis ║
  ║   Ridgeback Defense                      ║
  ╠══════════════════════════════════════════╣
  ║   UI running at: {url}          ║
  ╚══════════════════════════════════════════╝

  The UI will open in your default browser.
  Close this window or press Ctrl+C to stop.
""")

    if not _browser_opened:
        def open_browser():
            time.sleep(0.5)
            try:
                webbrowser.open(url)
                _log(f'Browser opened: {url}')
            except Exception as e:
                _log(f'Failed to open browser: {e}')

        threading.Thread(target=open_browser, daemon=True).start()
        _browser_opened = True

    _cleanup_done = False

    def cleanup():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        _log('Shutting down SASA server...')
        for run in RUNS.all():
            if run.is_active:
                run.cancel_event.set()
        with _ws_lock:
            for conn in list(_ws_connections):
                conn.close()
            _ws_connections.clear()
        try:
            server.server_close()
        except Exception:
            pass
        if _LOG_FILE:
            try:
                _LOG_FILE.close()
            except Exception:
                pass

    atexit.register(cleanup)

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _log(f'Server error: {e}\n{traceback.format_exc()}')
    finally:
        cleanup()

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        try:
            crash_log = Path.home() / '.sasa' / 'logs' / 'crash.log'
            crash_log.parent.mkdir(parents=True, exist_ok=True)
            with open(crash_log, 'a') as f:
                f.write(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} CRASH: {e}\n')
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise
