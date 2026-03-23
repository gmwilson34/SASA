#!/usr/bin/env python3
"""
SASA Application Server — Pure Python

Replaces the Node.js server.js with a self-contained Python HTTP + WebSocket
server. This is the entry point for the packaged macOS/Windows app.

- Serves the web UI (static HTML/CSS/JS from ui/renderer/)
- Handles file upload, analysis results, and image serving via REST API
- Runs analysis IN-PROCESS via analyze_file() with WebSocket progress streaming
- Opens the default browser on startup
- Stays alive in the background until the user closes it
- Cleans up all resources on exit

No external dependencies beyond Python stdlib + the SASA analysis modules.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import http.server
import json
import os
import re
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse


# ═══════════════════════════════════════════════════════════
#  Safe I/O — handle sys.stdout/stderr being None
#  (PyInstaller console=False sets them to None on macOS/Windows)
# ═══════════════════════════════════════════════════════════

_LOG_FILE = None

def _setup_logging():
    """Set up a log file and fix None stdout/stderr for packaged app."""
    global _LOG_FILE
    log_dir = Path.home() / '.sasa' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'sasa.log'
    _LOG_FILE = open(log_path, 'a', encoding='utf-8', buffering=1)

    # If stdout/stderr are None (PyInstaller console=False), redirect to log file
    if sys.stdout is None or not hasattr(sys.stdout, 'write'):
        sys.stdout = _LOG_FILE
    if sys.stderr is None or not hasattr(sys.stderr, 'write'):
        sys.stderr = _LOG_FILE

def _log(msg: str):
    """Write to log file (always works, even if stdout is None)."""
    if _LOG_FILE:
        try:
            _LOG_FILE.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
            _LOG_FILE.flush()
        except Exception:
            pass
    try:
        if sys.stdout and hasattr(sys.stdout, 'write'):
            print(msg)
    except Exception:
        pass


# ── Resolve paths relative to this file (works inside PyInstaller bundle) ──

def _get_base_dir() -> Path:
    """Get the project root, whether running from source or PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _find_renderer_dir() -> Path:
    """Find the ui/renderer directory — it may be in _MEIPASS or in the macOS
    Resources directory depending on how PyInstaller packages data files."""
    if getattr(sys, 'frozen', False):
        candidates = [
            Path(sys._MEIPASS) / 'ui' / 'renderer',
        ]
        # macOS .app bundle: data files go to Contents/Resources/
        if sys.platform == 'darwin':
            # _MEIPASS is .../Contents/Frameworks — Resources is a sibling
            frameworks = Path(sys._MEIPASS)  # type: ignore[attr-defined]
            resources = frameworks.parent / 'Resources'
            candidates.insert(0, resources / 'ui' / 'renderer')
        for c in candidates:
            if c.is_dir() and (c / 'index.html').is_file():
                return c
        # Last resort: return the standard path (will fail with a clear error)
        return Path(sys._MEIPASS)  # type: ignore[attr-defined] / 'ui' / 'renderer'
    return Path(__file__).resolve().parent / 'ui' / 'renderer'


def _find_source_dir() -> Path:
    """Find directory containing Python source modules (main.py, etc.)."""
    if getattr(sys, 'frozen', False):
        candidates = [
            Path(sys._MEIPASS),
        ]
        if sys.platform == 'darwin':
            frameworks = Path(sys._MEIPASS)  # type: ignore[attr-defined]
            resources = frameworks.parent / 'Resources'
            candidates.append(resources)
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

PORT = int(os.environ.get('SASA_PORT', '3847'))

# ── MIME map ──
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
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
}

# Video file extensions that need audio extraction before analysis
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}


def _find_ffmpeg() -> str | None:
    """Find an ffmpeg binary — try bundled, imageio-ffmpeg, then system PATH."""
    # 1. Try bundled ffmpeg (PyInstaller packages it into imageio_ffmpeg_bin/)
    if getattr(sys, 'frozen', False):
        bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        for candidate_dir in [bundle_dir / 'imageio_ffmpeg_bin', bundle_dir]:
            if candidate_dir.is_dir():
                for name in ['ffmpeg', 'ffmpeg.exe']:
                    exe = candidate_dir / name
                    if exe.is_file():
                        # Ensure it's executable on macOS/Linux
                        if sys.platform != 'win32':
                            exe.chmod(0o755)
                        return str(exe)
        # Also check Resources/ on macOS
        if sys.platform == 'darwin':
            resources = bundle_dir.parent / 'Resources' / 'imageio_ffmpeg_bin'
            if resources.is_dir():
                for name in ['ffmpeg', 'ffmpeg.exe']:
                    exe = resources / name
                    if exe.is_file():
                        exe.chmod(0o755)
                        return str(exe)

    # 2. Try imageio-ffmpeg (pip-installed)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(exe)
    except ImportError:
        pass

    # 3. Try system ffmpeg
    import shutil
    exe = shutil.which('ffmpeg')
    if exe:
        return exe

    return None


def _extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:
    """Extract audio from a video file to WAV using ffmpeg.

    Returns the path to the extracted WAV file.
    Raises RuntimeError if ffmpeg is not available or extraction fails.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            'Cannot extract audio from video: ffmpeg not found.\n'
            'Install ffmpeg:\n'
            '  macOS:   brew install ffmpeg\n'
            '  Windows: winget install ffmpeg\n'
            '  Or:      pip install imageio-ffmpeg'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / (video_path.stem + '.wav')

    # Skip extraction if we already have this file
    if wav_path.is_file():
        return wav_path

    import subprocess as _sp
    result = _sp.run(
        [ffmpeg, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', '-y', str(wav_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed: {result.stderr[:500]}')

    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError('ffmpeg produced no output')

    return wav_path


# ── Track active WebSocket connections for cleanup ──
_ws_sockets: set = set()
_ws_lock = threading.Lock()

# Track whether browser has been opened this session
_browser_opened = False


# ═══════════════════════════════════════════════════════════
#  Lazy import of analysis modules
# ═══════════════════════════════════════════════════════════
#  These are imported on first use so the server starts fast
#  and we can report import errors to the UI.

_analysis_modules = {}

def _ensure_analysis_imports():
    """Import analysis modules on first use. Raises ImportError on failure."""
    if _analysis_modules:
        return
    # Make sure SOURCE_DIR is on sys.path so main.py and its imports are found
    src = str(SOURCE_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    _log(f'Analysis import: SOURCE_DIR={SOURCE_DIR}, sys.path[0]={sys.path[0]}')

    from main import analyze_file, AnalysisConfig
    _analysis_modules['analyze_file'] = analyze_file
    _analysis_modules['AnalysisConfig'] = AnalysisConfig
    _log('Analysis modules imported successfully')


# ═══════════════════════════════════════════════════════════
#  Stdout Capture — streams print() output over WebSocket
# ═══════════════════════════════════════════════════════════

class WebSocketStdoutCapture:
    """Redirects stdout so that print() output is sent over WebSocket."""

    def __init__(self, ws_send_fn):
        self.ws_send = ws_send_fn
        self.original = sys.stdout
        self._buf = ''
        self._lock = threading.Lock()

    def write(self, text: str):
        # Always write to original stdout too (for debugging)
        if self.original:
            try:
                self.original.write(text)
            except Exception:
                pass

        with self._lock:
            self._buf += text
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                if line:
                    self.ws_send({'type': 'log', 'line': line})
                    # Parse progress markers
                    pct_match = re.search(r'\[(\d+)/(\d+)\]|\[(\d+)%\]', line)
                    if pct_match:
                        if pct_match.group(3):
                            pct = int(pct_match.group(3))
                        else:
                            num, den = int(pct_match.group(1)), int(pct_match.group(2))
                            pct = int(num / den * 100) if den else 0
                        self.ws_send({'type': 'progress', 'pct': min(100, pct)})

    def flush(self):
        with self._lock:
            if self._buf.strip():
                self.ws_send({'type': 'log', 'line': self._buf.rstrip()})
                self._buf = ''
        if self.original:
            try:
                self.original.flush()
            except Exception:
                pass

    # Forward other attributes to original stdout
    def __getattr__(self, name):
        return getattr(self.original, name)


# ═══════════════════════════════════════════════════════════
#  Minimal WebSocket Implementation (RFC 6455)
# ═══════════════════════════════════════════════════════════

def _ws_accept_key(key: str) -> str:
    magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    sha1 = hashlib.sha1((key + magic).encode()).digest()
    return base64.b64encode(sha1).decode()


def _ws_frame_size(data: bytes) -> int:
    """Calculate total byte length of the first WebSocket frame in data.
    Returns 0 if there isn't enough data for a complete frame."""
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
    """Decode one WebSocket frame. Returns (opcode, payload, bytes_consumed)."""
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
    """Encode a WebSocket frame (server→client, no masking)."""
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


# ═══════════════════════════════════════════════════════════
#  Multipart Form Parser
# ═══════════════════════════════════════════════════════════

def parse_multipart(body: bytes, content_type: str) -> dict:
    """Parse multipart/form-data body. Returns {field_name: (filename, data)}."""
    match = re.search(r'boundary=([^\s;]+)', content_type)
    if not match:
        return {}
    boundary = match.group(1).strip().encode()
    if boundary.startswith(b'"') and boundary.endswith(b'"'):
        boundary = boundary[1:-1]

    parts = body.split(b'--' + boundary)
    result = {}
    for part in parts:
        part = part.strip()
        if not part or part == b'--':
            continue
        header_end = part.find(b'\r\n\r\n')
        if header_end < 0:
            continue
        headers_raw = part[:header_end].decode('utf-8', errors='replace')
        file_data = part[header_end + 4:]
        if file_data.endswith(b'\r\n'):
            file_data = file_data[:-2]

        name_match = re.search(r'name="([^"]+)"', headers_raw)
        filename_match = re.search(r'filename="([^"]+)"', headers_raw)
        if name_match:
            field_name = name_match.group(1)
            filename = filename_match.group(1) if filename_match else None
            result[field_name] = (filename, file_data)
    return result


# ═══════════════════════════════════════════════════════════
#  HTTP + WebSocket Request Handler
# ═══════════════════════════════════════════════════════════

class SASAHandler(http.server.BaseHTTPRequestHandler):
    """Handles HTTP requests and WebSocket upgrades."""

    # Flag set when connection has been upgraded to WebSocket
    _is_websocket = False

    def log_message(self, format, *args):  # noqa: A002
        pass  # Suppress default logging

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # ── WebSocket upgrade ──
        if path == '/ws':
            upgrade = self.headers.get('Upgrade', '').lower()
            key = self.headers.get('Sec-WebSocket-Key', '')
            if upgrade == 'websocket' and key:
                self._handle_ws_upgrade(key)
            else:
                self._send_text('Expected WebSocket upgrade', 400)
            return

        # ── API endpoints ──
        if path == '/api/analyses':
            return self._api_analyses()
        if path == '/api/results':
            return self._api_results(query)
        if path == '/api/image':
            return self._api_image(query)

        # ── Static file serving ──
        self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/upload':
            return self._api_upload()
        self._send_json({'error': 'Not found'}, 404)

    # ── Static files ──

    def _serve_static(self, path: str):
        if path == '/':
            path = '/index.html'

        safe_path = Path(path.lstrip('/')).parts
        file_path = RENDERER_DIR
        for part in safe_path:
            if part in ('..', '.'):
                continue
            file_path = file_path / part

        if not file_path.is_file():
            self._send_text('Not Found', 404)
            return

        ext = file_path.suffix.lower()
        content_type = MIME_MAP.get(ext, 'application/octet-stream')
        data = file_path.read_bytes()

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    # ── API: Upload ──

    def _api_upload(self):
        content_type = self.headers.get('Content-Type', '')
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        parts = parse_multipart(body, content_type)
        if 'file' not in parts:
            return self._send_json({'error': 'No file uploaded'}, 400)

        filename, file_data = parts['file']
        if not filename:
            return self._send_json({'error': 'No filename'}, 400)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        dest = UPLOAD_DIR / filename
        if dest.exists():
            stem = dest.stem
            ext = dest.suffix
            dest = UPLOAD_DIR / f'{stem}_{int(time.time())}{ext}'

        dest.write_bytes(file_data)
        self._send_json({'path': str(dest), 'name': filename})

    # ── API: List analyses ──

    def _api_analyses(self):
        if not ANALYSIS_DIR.is_dir():
            return self._send_json([])

        entries = []
        for d in sorted(ANALYSIS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / 'analysis_metadata.json'
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            entries.append({'name': d.name, 'path': str(d), 'meta': meta})
        self._send_json(entries)

    # ── API: Load results ──

    def _api_results(self, query: dict):
        dir_path = query.get('dir', [''])[0]
        if not dir_path:
            return self._send_json({'error': 'Missing dir parameter'}, 400)

        output_dir = Path(dir_path)
        meta_path = output_dir / 'analysis_metadata.json'
        if not meta_path.is_file():
            return self._send_json({'error': 'No analysis_metadata.json found'}, 404)

        try:
            metadata = json.loads(meta_path.read_text())
        except Exception as e:
            return self._send_json({'error': str(e)}, 500)

        images = {}
        for f in output_dir.iterdir():
            if f.suffix in ('.png', '.html'):
                key = f.stem
                if key not in images:
                    images[key] = {}
                images[key]['html' if f.suffix == '.html' else 'png'] = f.name

        shot_images = []
        shots_dir = output_dir / 'shots'
        if shots_dir.is_dir():
            shot_images = sorted(f.name for f in shots_dir.iterdir() if f.suffix == '.png')

        csv_data = None
        csv_path = output_dir / 'metrics_summary.csv'
        if csv_path.is_file():
            csv_data = csv_path.read_text()

        self._send_json({
            'metadata': metadata,
            'images': images,
            'shotImages': shot_images,
            'csv': csv_data,
            'outputDir': str(output_dir),
        })

    # ── API: Serve image / file from analysis output ──

    def _api_image(self, query: dict):
        dir_path = query.get('dir', [''])[0]
        file_name = query.get('file', [''])[0]
        sub = query.get('sub', [''])[0]

        if not dir_path or not file_name:
            return self._send_text('Missing dir or file', 400)

        safe_name = Path(file_name).name
        safe_sub = Path(sub).name if sub else None

        if safe_sub:
            file_path = Path(dir_path) / safe_sub / safe_name
        else:
            file_path = Path(dir_path) / safe_name

        if not file_path.is_file():
            return self._send_text('Not found', 404)

        ext = file_path.suffix.lower()
        content_type = MIME_MAP.get(ext, 'application/octet-stream')
        data = file_path.read_bytes()

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ═══════════════════════════════════════════════════
    #  WebSocket
    # ═══════════════════════════════════════════════════

    def _handle_ws_upgrade(self, key: str):
        """Perform the WebSocket handshake and enter the frame loop."""
        accept = _ws_accept_key(key)

        # Send 101 Switching Protocols — write raw to avoid BaseHTTPRequestHandler quirks
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
        self.close_connection = True  # Prevent BaseHTTPRequestHandler keep-alive loop

        raw_sock = self.connection
        with _ws_lock:
            _ws_sockets.add(raw_sock)

        try:
            self._ws_frame_loop(raw_sock)
        finally:
            with _ws_lock:
                _ws_sockets.discard(raw_sock)

    def _ws_frame_loop(self, sock):
        """Read WebSocket frames and dispatch messages."""
        # Drain any data already buffered by rfile (from the HTTP request)
        try:
            buffered = self.rfile.peek()
        except Exception:
            buffered = b''
        if buffered:
            buf = bytes(buffered)
            self.rfile.read(len(buffered))  # consume it so rfile stays in sync
        else:
            buf = b''

        sock.settimeout(30.0)  # Timeout for keepalive checks

        while True:
            # Try to decode frames from current buffer
            while buf:
                opcode, payload, consumed = _ws_decode_frame(buf)
                if consumed == 0:
                    break  # Need more data
                buf = buf[consumed:]

                if payload is None:
                    continue
                if opcode == 0x8:  # Close
                    try:
                        sock.sendall(_ws_encode_frame(b'', opcode=0x8))
                    except Exception:
                        pass
                    return
                elif opcode == 0x9:  # Ping → Pong
                    try:
                        sock.sendall(_ws_encode_frame(payload, opcode=0xA))
                    except Exception:
                        return
                elif opcode == 0x1:  # Text message
                    self._ws_on_message(sock, payload.decode('utf-8', errors='replace'))

            # Read more data from socket
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    return  # Connection closed
                buf += chunk
            except socket.timeout:
                # Send ping to check if client is still alive
                try:
                    sock.sendall(_ws_encode_frame(b'ping', opcode=0x9))
                except Exception:
                    return  # Client gone
            except (ConnectionResetError, BrokenPipeError, OSError):
                return

    def _ws_on_message(self, sock, text: str):
        """Handle an incoming WebSocket text message."""
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return

        if msg.get('type') == 'run-analysis':
            config = msg.get('config', {})
            threading.Thread(
                target=self._run_analysis_inprocess,
                args=(sock, config),
                daemon=True,
            ).start()

    def _ws_send(self, sock, data: dict):
        """Send a JSON message over WebSocket."""
        try:
            frame = _ws_encode_frame(json.dumps(data).encode('utf-8'))
            sock.sendall(frame)
        except (BrokenPipeError, OSError, ConnectionResetError):
            pass

    # ═══════════════════════════════════════════════════
    #  Analysis — runs IN-PROCESS (no subprocess)
    # ═══════════════════════════════════════════════════

    def _run_analysis_inprocess(self, sock, config: dict):
        """Run analysis by importing and calling analyze_file() directly.

        This avoids the critical bug where subprocess.Popen(sys.executable, ...)
        would re-launch the entire packaged app (opening another browser window)
        instead of running Python with main.py.
        """
        def ws(data):
            return self._ws_send(sock, data)

        file_path = config.get('filePath', '')
        if not file_path:
            ws({'type': 'error', 'message': 'No file path provided'})
            return

        ws({'type': 'log', 'line': f'Starting analysis: {file_path}'})
        ws({'type': 'progress', 'pct': 0})

        # Import analysis modules
        try:
            _ensure_analysis_imports()
            analyze_file = _analysis_modules['analyze_file']
            AnalysisConfig = _analysis_modules['AnalysisConfig']
        except ImportError as e:
            ws({'type': 'error', 'message': f'Failed to import analysis modules: {e}'})
            return

        # Build AnalysisConfig from the UI config
        try:
            ac_kwargs = {}

            # Calibration
            if config.get('paPerFS'):
                ac_kwargs['Pa_per_FS'] = float(config['paPerFS'])
            if config.get('sensitivityMv'):
                ac_kwargs['sensitivity_mV_per_Pa'] = float(config['sensitivityMv'])
            if config.get('vPerFS'):
                ac_kwargs['V_per_FS'] = float(config['vPerFS'])
            if config.get('calDesc'):
                ac_kwargs['calibration_description'] = config['calDesc']

            # Shot detection
            if config.get('thresholdDb'):
                ac_kwargs['detection_threshold_dB'] = float(config['thresholdDb'])
            if config.get('refractoryMs'):
                ac_kwargs['refractory_ms'] = float(config['refractoryMs'])
            if config.get('preMs'):
                ac_kwargs['pre_shot_ms'] = float(config['preMs'])
            if config.get('postMs'):
                ac_kwargs['post_shot_ms'] = float(config['postMs'])

            # STFT
            if config.get('nperseg'):
                ac_kwargs['nperseg'] = int(config['nperseg'])

            # Options
            if config.get('noBands'):
                ac_kwargs['compute_bands'] = False
            if config.get('noPerShot'):
                ac_kwargs['save_per_shot_plots'] = False

            # Formats
            if config.get('formats'):
                ac_kwargs['plot_formats'] = config['formats'].split(',')

            analysis_config = AnalysisConfig(**ac_kwargs)

        except Exception as e:
            ws({'type': 'error', 'message': f'Invalid configuration: {e}'})
            return

        # Redirect stdout to capture print() output and stream to WebSocket
        captured_stdout = WebSocketStdoutCapture(ws)
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stdout  # Also capture stderr

            wav_path = Path(file_path)
            output_base = ANALYSIS_DIR

            # Extract audio from video files before analysis
            if wav_path.suffix.lower() in VIDEO_EXTS:
                ws({'type': 'log', 'line': f'[Video detected] Extracting audio from {wav_path.name}...'})
                ws({'type': 'progress', 'pct': 2})
                wav_path = _extract_audio_from_video(wav_path, UPLOAD_DIR)
                ws({'type': 'log', 'line': f'Audio extracted: {wav_path.name}'})

            ws({'type': 'progress', 'pct': 5})

            result = analyze_file(wav_path, analysis_config, output_base)

            # Flush any remaining output
            captured_stdout.flush()

            ws({'type': 'progress', 'pct': 100})
            ws({'type': 'complete', 'outputDir': str(result.output_dir)})

        except Exception as e:
            captured_stdout.flush()
            tb = traceback.format_exc()
            ws({'type': 'log', 'line': f'Error: {tb}'})
            ws({'type': 'error', 'message': str(e)})
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # ── HTTP Helpers ──

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
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
            import subprocess
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split('\n')
            my_pid = str(os.getpid())
            for pid in pids:
                pid = pid.strip()
                if pid and pid != my_pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
            if any(p.strip() and p.strip() != my_pid for p in pids):
                time.sleep(0.5)  # Give the old process time to die
        except Exception:
            pass
    elif sys.platform == 'win32':
        try:
            import subprocess
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True, text=True, timeout=5,
            )
            my_pid = str(os.getpid())
            for line in result.stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1].strip()
                    if pid and pid != my_pid and pid != '0':
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

    # FIRST: set up logging so we can debug crashes in packaged app
    _setup_logging()
    _log(f'SASA starting — frozen={getattr(sys, "frozen", False)}, platform={sys.platform}')
    _log(f'BASE_DIR={BASE_DIR}')
    _log(f'RENDERER_DIR={RENDERER_DIR}')
    _log(f'SOURCE_DIR={SOURCE_DIR}')
    _log(f'DATA_DIR={DATA_DIR}')

    # Ensure data directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Check renderer directory
    if not RENDERER_DIR.is_dir() or not (RENDERER_DIR / 'index.html').is_file():
        _log(f'ERROR: UI renderer not found at {RENDERER_DIR}')
        _log(f'Contents of BASE_DIR: {list(BASE_DIR.iterdir()) if BASE_DIR.is_dir() else "N/A"}')
        return 1

    _log(f'Renderer OK: {list(RENDERER_DIR.iterdir())}')

    # Try to kill any leftover server on our port
    _kill_existing_server(PORT)

    # Find available port
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

    banner = f"""
  ╔══════════════════════════════════════════╗
  ║   SASA — Shot Acoustic Spectral Analysis ║
  ║   Ridgeback Defense                      ║
  ╠══════════════════════════════════════════╣
  ║   UI running at: {url}          ║
  ╚══════════════════════════════════════════╝

  The UI will open in your default browser.
  Close this window or press Ctrl+C to stop.
"""
    _log(banner)

    # Open browser once
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

    # Cleanup on exit
    _cleanup_done = False
    def cleanup():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        _log('Shutting down SASA server...')
        with _ws_lock:
            for sock in list(_ws_sockets):
                try:
                    sock.close()
                except Exception:
                    pass
            _ws_sockets.clear()
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
    except Exception as e:
        # Last-resort crash handler — write to log file even if everything else failed
        try:
            crash_log = Path.home() / '.sasa' / 'logs' / 'crash.log'
            crash_log.parent.mkdir(parents=True, exist_ok=True)
            with open(crash_log, 'a') as f:
                f.write(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} CRASH: {e}\n')
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise
