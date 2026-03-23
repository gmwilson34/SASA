#!/usr/bin/env python3
"""
Load WAV audio into numeric arrays for waveform plots / fitting later.

Outputs:
  - sample rate (Hz)
  - samples as numpy float32 in range [-1, 1] (for PCM WAVs)
  - time axis array (seconds)
  - optional mono mixdown

Usage:
  python WavLoader.py path/to/audio.wav
  python WavLoader.py path/to/audio.wav --mono
  python WavLoader.py path/to/audio.wav --save-npy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class WavData:
    path: Path
    sample_rate: int
    samples: np.ndarray  # shape: (n_frames, n_channels) or (n_frames,) if mono mixdown
    time_s: np.ndarray   # shape: (n_frames,)


def get_wav_info(wav_path: Path) -> tuple[int, int, float, int]:
    """
    Get WAV file metadata without loading samples (memory-efficient).

    Returns:
        (frames, sample_rate, duration_s, channels)
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    info = sf.info(str(wav_path))
    duration_s = info.frames / float(info.samplerate)
    return (info.frames, int(info.samplerate), duration_s, info.channels)


def load_wav_chunk(
    wav_path: Path,
    start_frame: int,
    n_frames: int,
    *,
    dtype: str = "float32",
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load a chunk of WAV (for memory-efficient processing of long files).

    Args:
        wav_path: Path to WAV file.
        start_frame: First frame index to read.
        n_frames: Number of frames to read.
        dtype: Sample dtype (e.g. 'float32').
        mono: If True, average channels to mono.

    Returns:
        (samples_1d, sample_rate). samples_1d is shape (n_frames,) in range [-1, 1].
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    total_frames, sr, _, _ = get_wav_info(wav_path)
    stop_frame = min(start_frame + n_frames, total_frames)
    if stop_frame <= start_frame:
        return (np.array([], dtype=dtype), sr)
    data, sr = sf.read(
        str(wav_path),
        dtype=dtype,
        always_2d=True,
        start=start_frame,
        stop=stop_frame,
    )
    data = np.asarray(data)
    if mono and data.ndim > 1:
        data = data.mean(axis=1)
    elif data.ndim == 2 and data.shape[1] == 1:
        data = data.squeeze(axis=1)
    return (data, int(sr))


def load_wav(
    wav_path: Path,
    *,
    dtype: str = "float32",
    mono: bool = False,
) -> WavData:
    """
    Read a WAV file and return samples + sample rate + time axis.

    dtype:
      - 'float32' is best for analysis/plotting; soundfile will scale PCM to [-1, 1].
      - you can use 'int16' etc. if you want raw PCM integers.

    mono:
      - If True and audio is multichannel, average channels to mono.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    data, sr = sf.read(str(wav_path), dtype=dtype, always_2d=True)  # (frames, channels)

    # Ensure numpy array (soundfile already returns one, but be explicit)
    data = np.asarray(data)

    if mono:
        # Average channels -> shape (frames,)
        data_mono = data.mean(axis=1)
        time_s = np.arange(data_mono.shape[0], dtype=np.float64) / float(sr)
        return WavData(path=wav_path, sample_rate=sr, samples=data_mono, time_s=time_s)

    # Keep channels -> shape (frames, channels)
    time_s = np.arange(data.shape[0], dtype=np.float64) / float(sr)
    return WavData(path=wav_path, sample_rate=sr, samples=data, time_s=time_s)


def rms(samples: np.ndarray) -> float:
    """Compute RMS amplitude (useful quick sanity check)."""
    x = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(x * x)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Load WAV into numpy arrays for waveform plotting/fitting.")
    parser.add_argument("wav", type=Path, help="Input .wav file")
    parser.add_argument("--mono", action="store_true", help="Mix down to mono by averaging channels")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64", "int16", "int32"],
        help="Sample dtype to load (float32 recommended for analysis).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save samples/time/sample_rate next to the wav as .npy/.npz for later use",
    )
    args = parser.parse_args()

    wav_data = load_wav(args.wav, dtype=args.dtype, mono=args.mono)

    # Print summary + a path you can use immediately
    print(f"Loaded: {wav_data.path.resolve()}")
    print(f"Sample rate: {wav_data.sample_rate} Hz")
    print(f"Samples shape: {wav_data.samples.shape}")
    print(f"Duration: {wav_data.time_s[-1]:.3f} s")
    print(f"RMS: {rms(wav_data.samples):.6f}")

    if args.save_npy:
        base = wav_data.path.with_suffix("")
        if wav_data.samples.ndim == 1:
            out = base.with_suffix(".npz")
            np.savez(
                out,
                sample_rate=np.array([wav_data.sample_rate], dtype=np.int32),
                time_s=wav_data.time_s,
                samples=wav_data.samples,
            )
        else:
            out = base.with_suffix(".npz")
            np.savez(
                out,
                sample_rate=np.array([wav_data.sample_rate], dtype=np.int32),
                time_s=wav_data.time_s,
                samples=wav_data.samples,
            )
        print(f"Saved arrays to: {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
