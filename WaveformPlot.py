#!/usr/bin/env python3
"""
Waveform.py

Load a WAV file and generate a waveform plot (time vs amplitude).
- Uses soundfile + numpy to load samples
- Uses matplotlib to plot and save/display

Install:
  python -m pip install soundfile numpy matplotlib

Usage:
  python Waveform.py path/to/audio.wav
  python Waveform.py path/to/audio.wav --mono
  python Waveform.py path/to/audio.wav --out waveform.png
  python Waveform.py path/to/audio.wav --start 5 --duration 10
  python Waveform.py path/to/audio.wav --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def load_wav(wav_path: Path, *, mono: bool = False, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """
    Returns (samples, sample_rate).

    samples:
      - shape (frames, channels) if mono=False
      - shape (frames,) if mono=True
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    data, sr = sf.read(str(wav_path), dtype=dtype, always_2d=True)  # (frames, channels)
    data = np.asarray(data)

    if mono:
        return data.mean(axis=1), sr  # (frames,)
    return data, sr  # (frames, channels)


def slice_by_time(samples: np.ndarray, sr: int, start_s: float | None, duration_s: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (time_axis_seconds, sliced_samples)
    """
    n_frames = samples.shape[0]
    start_i = 0 if start_s is None else int(max(0.0, start_s) * sr)

    if duration_s is None:
        end_i = n_frames
    else:
        end_i = min(n_frames, start_i + int(max(0.0, duration_s) * sr))

    sliced = samples[start_i:end_i]
    t = (np.arange(start_i, end_i, dtype=np.float64) / float(sr))
    return t, sliced


def plot_waveform(
    time_s: np.ndarray,
    samples: np.ndarray,
    *,
    title: str,
    out_path: Path | None,
    show: bool,
) -> None:
    """
    Creates the waveform plot.
    - If samples are multichannel, plots each channel on the same axes.
    """
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)

    if samples.ndim == 1:
        plt.plot(time_s, samples)
    else:
        # Plot each channel separately (same time axis)
        for ch in range(samples.shape[1]):
            plt.plot(time_s, samples[:, ch])

    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)

    if show:
        plt.show()

    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a waveform plot from a WAV file.")
    parser.add_argument("wav", type=Path, help="Input .wav file")
    parser.add_argument("--mono", action="store_true", help="Mix down to mono by averaging channels")
    parser.add_argument("--start", type=float, default=None, help="Start time (seconds) for plotting")
    parser.add_argument("--duration", type=float, default=None, help="Duration (seconds) for plotting")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (e.g., waveform.png). Default: <wavname>_waveform.png",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window")

    args = parser.parse_args()

    samples, sr = load_wav(args.wav, mono=args.mono)
    time_s, sliced = slice_by_time(samples, sr, args.start, args.duration)

    if args.out is None:
        out_path = args.wav.with_suffix("")  # remove .wav
        out_path = out_path.parent / f"{out_path.name}_waveform.png"
    else:
        out_path = args.out

    title = f"Waveform: {args.wav.name} ({sr} Hz)"
    plot_waveform(time_s, sliced, title=title, out_path=out_path, show=args.show)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
