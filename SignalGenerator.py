#!/usr/bin/env python3
"""
SignalGenerator.py

Generate synthetic WAV files containing one or more dominant frequencies
(sine waves), for a specified duration, then save them into an "Audio"
subdirectory with an auto filename based on current date/time.

Dependencies:
  python -m pip install numpy soundfile

Examples:
  # 1 second of 440 Hz
  python SignalGenerator.py --freq 440 --duration 1.0

  # 2 seconds containing 440 Hz + 880 Hz
  python SignalGenerator.py --freq 440 --freq 880 --duration 2.0

  # 3 seconds with 200, 400, 800 Hz, custom sample rate and amplitude
  python SignalGenerator.py --freq 200 --freq 400 --freq 800 --duration 3 --sr 48000 --amp 0.6

  # Add a gentle fade-in/out to avoid clicks
  python SignalGenerator.py --freq 440 --duration 1.5 --fade 0.01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ToneSpec:
    freqs_hz: list[float]
    duration_s: float
    sample_rate: int
    amplitude: float
    fade_s: float
    normalize: bool


def make_time_based_filename(freqs_hz: list[float]) -> str:
    """
    Example:
      2026-01-04_15-22-10_440Hz_880Hz.wav
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [ts] + [f"{int(round(f))}Hz" if abs(f - round(f)) < 1e-6 else f"{f:g}Hz" for f in freqs_hz]
    return "_".join(parts) + ".wav"


def apply_fade(signal: np.ndarray, sr: int, fade_s: float) -> np.ndarray:
    """Apply equal-length linear fade-in and fade-out to avoid clicks."""
    if fade_s <= 0:
        return signal
    n_fade = int(round(fade_s * sr))
    if n_fade <= 0:
        return signal
    n = signal.shape[0]
    n_fade = min(n_fade, n // 2)  # prevent overlap for very short signals
    if n_fade <= 0:
        return signal

    fade_in = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)

    out = signal.copy()
    out[:n_fade] *= fade_in
    out[-n_fade:] *= fade_out
    return out


def synthesize_tone(spec: ToneSpec) -> np.ndarray:
    """
    Synthesize a mono WAV signal containing a sum of sine waves at the given freqs.
    Returns float32 samples (intended range [-1, 1]).
    """
    if spec.duration_s <= 0:
        raise ValueError("duration must be > 0")
    if spec.sample_rate <= 0:
        raise ValueError("sample rate must be > 0")
    if not spec.freqs_hz:
        raise ValueError("At least one frequency must be provided")
    if not (0.0 < spec.amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1].")

    n_samples = int(round(spec.duration_s * spec.sample_rate))
    t = np.arange(n_samples, dtype=np.float64) / float(spec.sample_rate)

    # Sum sines. Divide by count so multi-tone doesn't immediately clip.
    sig = np.zeros(n_samples, dtype=np.float64)
    for f in spec.freqs_hz:
        if f <= 0:
            raise ValueError(f"Frequency must be > 0 Hz. Got: {f}")
        sig += np.sin(2.0 * np.pi * f * t)

    sig /= float(len(spec.freqs_hz))
    sig *= float(spec.amplitude)

    sig = sig.astype(np.float32)

    if spec.fade_s > 0:
        sig = apply_fade(sig, spec.sample_rate, spec.fade_s)

    if spec.normalize:
        peak = float(np.max(np.abs(sig))) if sig.size else 0.0
        if peak > 0:
            # Keep a little headroom
            sig = (sig / peak * 0.98).astype(np.float32)

    return sig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate WAV files with dominant frequencies and save to ./Audio with time-based filenames."
    )
    parser.add_argument(
        "--freq",
        action="append",
        type=float,
        required=True,
        help="Frequency in Hz. Use multiple times for multiple tones: --freq 440 --freq 880",
    )
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds (e.g., 1.5)")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    parser.add_argument("--amp", type=float, default=0.6, help="Amplitude in (0, 1] (default: 0.6)")
    parser.add_argument(
        "--fade",
        type=float,
        default=0.01,
        help="Fade-in/out duration in seconds to avoid clicks (default: 0.01)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize to ~0.98 peak after synthesis (useful for many tones).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many files to generate (default: 1). Each will get a unique timestamp name.",
    )

    args = parser.parse_args()

    audio_dir = Path.cwd() / "Audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    spec = ToneSpec(
        freqs_hz=args.freq,
        duration_s=args.duration,
        sample_rate=args.sr,
        amplitude=args.amp,
        fade_s=args.fade,
        normalize=args.normalize,
    )

    for i in range(args.count):
        samples = synthesize_tone(spec)
        filename = make_time_based_filename(spec.freqs_hz)

        # If generating multiple files quickly, ensure uniqueness by appending an index if needed
        out_path = audio_dir / filename
        if out_path.exists():
            out_path = audio_dir / out_path.with_suffix("").name
            out_path = out_path.with_name(f"{out_path.name}_{i+1}").with_suffix(".wav")

        # Write 16-bit PCM for maximum compatibility
        sf.write(str(out_path), samples, spec.sample_rate, subtype="PCM_16")
        print(out_path.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
