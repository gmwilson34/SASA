#!/usr/bin/env python3
"""
ExtractAudio.py - Measurement-preserving audio extraction from video containers.

A gunshot measurement lives or dies on the sample rate and bit depth of the
recording. The previous implementation handed the container to MoviePy, whose
defaults resample to 44.1 kHz and mix to mono - which destroys the rise time, the
peak, and every band above 22 kHz before the analysis ever starts. Extraction here
therefore:

  - NEVER passes -ar, -ac or a resampler: the audio leaves the container at its
    original sample rate and channel count;
  - chooses a PCM output format that is at least as wide as the source, so no
    quantisation is introduced;
  - writes to a temporary file and atomically renames it on success, so a
    cancelled, failed or timed-out extraction can never be mistaken for a
    complete one;
  - verifies the extracted duration against the container's declared duration
    and rejects a short file rather than silently analysing a truncated take;
  - caches by CONTENT HASH, not by filename stem, so two different videos that
    happen to share a name cannot serve each other's audio.

ffmpeg is located from imageio-ffmpeg (packaged with the app), falling back to a
system ffmpeg on PATH. MoviePy is no longer required.

Usage:
  python ExtractAudio.py input_video.mp4
  python ExtractAudio.py input_video.mkv -o track.wav
  python ExtractAudio.py input_video.mov --probe
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "ExtractionError",
    "AudioStreamInfo",
    "find_ffmpeg",
    "ensure_moviepy_uses_packaged_ffmpeg",
    "probe_audio_stream",
    "extract_audio",
    "extract_audio_cached",
    "cache_dir",
]

# An extraction that takes longer than this has stalled; the partial file is discarded.
DEFAULT_TIMEOUT_S: float = 1800.0

# Accept an extraction whose duration is at least this fraction of the container's
# declared duration. Container durations are approximate, so a small shortfall is
# normal; anything below this is a truncated file.
MIN_DURATION_RATIO: float = 0.98


class ExtractionError(RuntimeError):
    """Audio could not be extracted, or what was extracted cannot be trusted."""


@dataclass
class AudioStreamInfo:
    """What the container says about its first audio stream."""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    sample_fmt: str = ""
    bit_depth: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---- ffmpeg discovery ----

def find_ffmpeg() -> Optional[str]:
    """
    Locate an ffmpeg binary: the packaged one first, then the system one.

    Returns None when no ffmpeg is available, so the caller can produce a message
    that tells the operator how to install one.
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - optional dependency

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(exe)
    except Exception:  # noqa: BLE001 - a missing optional dependency is not fatal
        pass

    exe = shutil.which("ffmpeg")
    return exe if exe else None


def ensure_moviepy_uses_packaged_ffmpeg() -> str:
    """
    Retained for backward compatibility with older callers.

    MoviePy is no longer used; this simply resolves and publishes the ffmpeg path.
    """
    exe = find_ffmpeg()
    if not exe:
        raise ExtractionError(_no_ffmpeg_message())
    os.environ["FFMPEG_BINARY"] = exe
    return exe


def _no_ffmpeg_message() -> str:
    return (
        "ffmpeg was not found, so audio cannot be extracted from a video file.\n"
        "  Install it with one of:\n"
        "    pip install imageio-ffmpeg      (packaged, recommended)\n"
        "    brew install ffmpeg             (macOS)\n"
        "    winget install ffmpeg           (Windows)\n"
        "  Or supply the audio track directly as a WAV file."
    )


# ---- Probing ----

_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?: Audio:\s*(?P<codec>[A-Za-z0-9_]+)"
    r"(?P<rest>[^\n]*)"
)
_RATE_RE = re.compile(r"(\d+)\s*Hz")
_CHANNELS_RE = re.compile(r"(\d+)\s*channels")
_FMT_RE = re.compile(r",\s*(s16|s32|s64|u8|flt|dbl|s16p|s32p|fltp|dblp)\b")
_BITS_RE = re.compile(r"\((\d+)\s*bit\)")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


def probe_audio_stream(
    input_path: Path,
    ffmpeg: Optional[str] = None,
    *,
    timeout_s: float = 60.0,
) -> AudioStreamInfo:
    """
    Read the first audio stream's parameters straight out of ffmpeg's own report.

    Raises:
        ExtractionError: if ffmpeg is unavailable, the file is unreadable, or the
                         container has no audio stream.
    """
    exe = ffmpeg or find_ffmpeg()
    if not exe:
        raise ExtractionError(_no_ffmpeg_message())
    if not Path(input_path).exists():
        raise ExtractionError(f"Input file not found: {input_path}")

    try:
        proc = subprocess.run(
            [exe, "-nostdin", "-hide_banner", "-i", str(input_path)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractionError(f"ffmpeg timed out probing {input_path.name}") from exc
    except OSError as exc:
        raise ExtractionError(f"Could not run ffmpeg at {exe}: {exc}") from exc

    # `ffmpeg -i` with no output writes its report to stderr and exits non-zero.
    text = (proc.stderr or "") + (proc.stdout or "")
    match = _STREAM_RE.search(text)
    if not match:
        raise ExtractionError(
            f"No audio stream found in {Path(input_path).name}. "
            f"ffmpeg reported:\n{text.strip()[-800:]}"
        )

    rest = match.group("rest")
    info = AudioStreamInfo(codec=match.group("codec"))

    rate = _RATE_RE.search(rest)
    if rate:
        info.sample_rate = int(rate.group(1))

    channels = _CHANNELS_RE.search(rest)
    if channels:
        info.channels = int(channels.group(1))
    elif "stereo" in rest:
        info.channels = 2
    elif "mono" in rest:
        info.channels = 1

    fmt = _FMT_RE.search(rest)
    if fmt:
        info.sample_fmt = fmt.group(1)

    bits = _BITS_RE.search(rest)
    if bits:
        info.bit_depth = int(bits.group(1))

    dur = _DURATION_RE.search(text)
    if dur:
        info.duration_s = (
            int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))
        )

    return info


def _output_codec(info: AudioStreamInfo) -> str:
    """
    Choose a PCM output format at least as wide as the source.

    Widening is safe; narrowing is not. Anything unrecognised falls through to
    32-bit float, which represents every integer depth up to 24 bits exactly and
    never clips a decoded lossy stream.
    """
    codec = (info.codec or "").lower()
    fmt = (info.sample_fmt or "").lower()

    if info.bit_depth == 24 or codec == "pcm_s24le":
        return "pcm_s24le"
    if codec in ("pcm_s16le", "pcm_s16be", "pcm_u8") or fmt.startswith("s16") or fmt == "u8":
        return "pcm_s16le"
    if codec in ("pcm_s32le", "pcm_s32be") or fmt.startswith("s32"):
        return "pcm_s32le"
    if fmt.startswith(("flt", "dbl")) or codec in ("pcm_f32le", "pcm_f64le"):
        return "pcm_f32le"
    return "pcm_f32le"


# ---- Extraction ----

def extract_audio(
    input_path: Path,
    output_path: Path,
    bitrate: Optional[str] = None,     # accepted for backward compatibility; unused for PCM
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    ffmpeg: Optional[str] = None,
) -> Path:
    """
    Extract the first audio stream to WAV without altering the measurement.

    The sample rate, channel count and effective bit depth of the source are all
    preserved. The file appears at output_path only if the extraction completed
    and the result passed verification.

    Args:
        input_path: Video (or any container ffmpeg can read).
        output_path: Destination WAV path.
        bitrate: Ignored - PCM output is not a bitrate-controlled format.
        timeout_s: Abort and discard the partial file after this long.
        ffmpeg: Explicit ffmpeg path; discovered automatically when omitted.

    Returns:
        The output path.

    Raises:
        ExtractionError: on any failure, including a short or empty result.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    exe = ffmpeg or find_ffmpeg()
    if not exe:
        raise ExtractionError(_no_ffmpeg_message())
    if not input_path.exists():
        raise ExtractionError(f"Input file not found: {input_path}")

    info = probe_audio_stream(input_path, exe)
    codec = _output_codec(info)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write beside the destination so the final rename stays on one filesystem
    # and is therefore atomic.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=".partial.wav", dir=str(output_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    cmd = [
        exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        "-vn",                       # no video
        "-map", "0:a:0",             # first audio stream only
        "-c:a", codec,               # PCM, at least as wide as the source
        # No -ar and no -ac: the rate and channel count are the measurement.
        "-f", "wav",
        str(tmp_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        tmp_path.unlink(missing_ok=True)
        raise ExtractionError(
            f"Audio extraction from {input_path.name} timed out after {timeout_s:.0f} s. "
            f"The partial file was discarded rather than analysed."
        ) from exc
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ExtractionError(f"Could not run ffmpeg at {exe}: {exc}") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()[-800:]
        tmp_path.unlink(missing_ok=True)
        raise ExtractionError(
            f"ffmpeg failed to extract audio from {input_path.name} "
            f"(exit {proc.returncode}):\n{detail}"
        )

    _verify_extraction(tmp_path, info, input_path)

    os.replace(tmp_path, output_path)
    return output_path


def _verify_extraction(wav_path: Path, info: AudioStreamInfo, source: Path) -> None:
    """
    Confirm the extracted file is complete and unaltered before it is published.

    Raises:
        ExtractionError: if the file is empty, resampled, or shorter than the
                         container said it should be.
    """
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        wav_path.unlink(missing_ok=True)
        raise ExtractionError(f"ffmpeg produced no audio for {source.name}")

    try:
        import soundfile as sf  # noqa: PLC0415 - imported lazily so probing works without it

        meta = sf.info(str(wav_path))
    except Exception as exc:  # noqa: BLE001
        wav_path.unlink(missing_ok=True)
        raise ExtractionError(
            f"The extracted audio from {source.name} could not be read back: {exc}"
        ) from exc

    if meta.frames <= 0:
        wav_path.unlink(missing_ok=True)
        raise ExtractionError(f"The extracted audio from {source.name} contains no samples")

    if info.sample_rate and meta.samplerate != info.sample_rate:
        wav_path.unlink(missing_ok=True)
        raise ExtractionError(
            f"Extraction changed the sample rate ({info.sample_rate} Hz -> "
            f"{meta.samplerate} Hz). The measurement would be invalid; aborting."
        )

    if info.channels and meta.channels != info.channels:
        wav_path.unlink(missing_ok=True)
        raise ExtractionError(
            f"Extraction changed the channel count ({info.channels} -> {meta.channels}). "
            f"Averaging or dropping channels destroys a multi-microphone measurement."
        )

    if info.duration_s > 0:
        got = meta.frames / float(meta.samplerate)
        if got < info.duration_s * MIN_DURATION_RATIO:
            wav_path.unlink(missing_ok=True)
            raise ExtractionError(
                f"Extraction is truncated: got {got:.2f} s of audio from a "
                f"{info.duration_s:.2f} s container. The partial file was discarded."
            )


# ---- Content-addressed cache ----

def cache_dir() -> Path:
    """
    Directory for extracted audio, overridable with SASA_CACHE_DIR.

    Cached files are named by the CONTENT hash of the source video, so renaming a
    video, or having two different videos with the same stem, cannot cause one
    take's audio to be served for another's.
    """
    env = os.environ.get("SASA_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "SASA" / "extracted"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "SASA" / "Cache" / "extracted"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sasa" / "extracted"


def _content_hash(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_audio_cached(
    input_path: Path,
    directory: Optional[Path] = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    ffmpeg: Optional[str] = None,
) -> tuple[Path, bool]:
    """
    Extract audio, reusing a previous extraction of the same *content*.

    Returns:
        (wav_path, from_cache)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise ExtractionError(f"Input file not found: {input_path}")

    target_dir = Path(directory) if directory is not None else cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    digest = _content_hash(input_path)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem)[:48] or "audio"
    out_path = target_dir / f"{safe_stem}-{digest[:16]}.wav"

    if out_path.is_file() and out_path.stat().st_size > 0:
        try:
            import soundfile as sf  # noqa: PLC0415

            if sf.info(str(out_path)).frames > 0:
                return out_path, True
        except Exception:  # noqa: BLE001 - a bad cache entry is simply re-extracted
            out_path.unlink(missing_ok=True)

    extract_audio(input_path, out_path, timeout_s=timeout_s, ffmpeg=ffmpeg)
    return out_path, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract audio from a video without altering rate, depth or channels."
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output WAV path (default: cached, content-addressed)")
    parser.add_argument("--probe", action="store_true",
                        help="Report the audio stream parameters and exit")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"Extraction timeout in seconds (default: {DEFAULT_TIMEOUT_S:.0f})")
    args = parser.parse_args()

    try:
        if args.probe:
            info = probe_audio_stream(args.input)
            for key, value in info.to_dict().items():
                print(f"{key}: {value}")
            return 0

        if args.output:
            path = extract_audio(args.input, args.output, timeout_s=args.timeout)
            cached = False
        else:
            path, cached = extract_audio_cached(args.input, timeout_s=args.timeout)
        print(f"{'(cached) ' if cached else ''}{path.resolve()}")
        return 0
    except ExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
