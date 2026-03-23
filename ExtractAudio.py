#!/usr/bin/env python3
"""
Portable audio extraction from video without requiring system-installed FFmpeg.

Dependencies:
  pip install moviepy imageio-ffmpeg

How it works:
  - imageio-ffmpeg provides an ffmpeg binary path
  - moviepy uses that ffmpeg binary to read video and write audio (no video rendering)

Usage:
  python ExtractAudio.py input_video.mp4
  python ExtractAudio.py input_video.mkv -o output_audio.mp3 --bitrate 192k
  python ExtractAudio.py input_video.mov -o output_audio.wav
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio_ffmpeg
from moviepy import VideoFileClip


def ensure_moviepy_uses_packaged_ffmpeg() -> str:
    """
    Point MoviePy at the ffmpeg binary provided by imageio-ffmpeg.
    Returns the ffmpeg executable path.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    # MoviePy checks this env var for the ffmpeg binary
    os.environ["FFMPEG_BINARY"] = ffmpeg_exe
    return ffmpeg_exe


def extract_audio(input_path: Path, output_path: Path, bitrate: str | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Open video container; we won't render frames.
    with VideoFileClip(str(input_path)) as clip:
        if clip.audio is None:
            raise RuntimeError("No audio stream found in the input file.")

        output_path.suffix.lower()

        # MoviePy chooses codecs based on extension; you can override if desired.
        # For broad compatibility:
        # - .wav -> PCM
        # - .mp3 -> libmp3lame (if available in ffmpeg build)
        # - .m4a/.aac -> AAC
        write_kwargs: dict[str, object] = {}

        if bitrate:
            write_kwargs["bitrate"] = bitrate

        # Avoid verbose logs unless debugging
        write_kwargs["logger"] = None

        # Extract audio only
        clip.audio.write_audiofile(str(output_path), **write_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract audio from a video file without system-installed FFmpeg.")
    parser.add_argument("input", type=Path, help="Input video file (any format supported by ffmpeg)")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output audio file path (default: input name + .wav)"
    )
    parser.add_argument("--bitrate", default=None, help="Bitrate for encoded outputs (e.g., 192k). Ignored for PCM WAV.")

    args = parser.parse_args()

    ffmpeg_path = ensure_moviepy_uses_packaged_ffmpeg()

    input_path: Path = args.input
    output_path: Path = args.output if args.output else input_path.with_suffix(".wav")

    try:
        extract_audio(input_path, output_path, args.bitrate)
    except Exception as e:
        print(f"Error: {e}")
        print(f"(MoviePy is using ffmpeg at: {ffmpeg_path})")
        return 1

    print(str(output_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
