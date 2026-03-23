#!/usr/bin/env python3
"""
stft.py - Short-Time Fourier Transform for Acoustic Analysis

Computes STFT spectrograms with calibrated dB SPL output.
Supports both Z-weighted (flat) and A-weighted spectrograms.

Key features:
  - Proper dB SPL scaling with calibration
  - A-weighting applied in frequency domain
  - Energy-preserving windowing
  - Numerical stability with eps floors

Usage:
    from stft import compute_stft, compute_stft_dB_SPL

    # Basic STFT (magnitude in Pa)
    t, f, mag_Pa = compute_stft(pressure_Pa, sample_rate)

    # STFT in dB SPL (calibrated)
    t, f, mag_dB = compute_stft_dB_SPL(pressure_Pa, sample_rate, weighting='Z')

    # A-weighted spectrogram
    t, f, mag_dB_A = compute_stft_dB_SPL(pressure_Pa, sample_rate, weighting='A')
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

from calibration import P_REF, EPS
from weighting import a_weight_linear

# Supported window types
WINDOW_TYPES = {'hann', 'hamming', 'blackman', 'rectangular'}


def get_window(name: str, length: int) -> np.ndarray:
    """
    Get window function.

    Args:
        name: Window type ('hann', 'hamming', 'blackman', 'rectangular').
        length: Window length in samples.

    Returns:
        Window array.
    """
    name = name.lower()
    if name == 'hann':
        return np.hanning(length).astype(np.float64)
    elif name == 'hamming':
        return np.hamming(length).astype(np.float64)
    elif name == 'blackman':
        return np.blackman(length).astype(np.float64)
    elif name == 'rectangular' or name == 'rect':
        return np.ones(length, dtype=np.float64)
    else:
        raise ValueError(f"Unknown window: {name}. Use: {WINDOW_TYPES}")


def compute_stft(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    scaling: Literal['amplitude', 'power', 'density'] = 'amplitude',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform.

    Args:
        x: Input signal (1D, in Pascals for calibrated analysis).
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size in samples.
        noverlap: Overlap in samples. Default: 75% of nperseg.
        window: Window function name.
        scaling: Output scaling:
                 - 'amplitude': RMS amplitude in same units as input
                 - 'power': Power spectral density (units²)
                 - 'density': Power spectral density normalized by bandwidth

    Returns:
        (time, frequencies, magnitude) where:
          - time: Time axis (seconds), shape (n_frames,)
          - frequencies: Frequency axis (Hz), shape (n_freq_bins,)
          - magnitude: STFT magnitude, shape (n_freq_bins, n_frames)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if nperseg <= 0:
        raise ValueError("nperseg must be > 0")

    if noverlap is None:
        noverlap = int(nperseg * 0.75)  # 75% overlap

    if not (0 <= noverlap < nperseg):
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    hop = nperseg - noverlap

    # Get window
    win = get_window(window, nperseg)

    # Window normalization for amplitude scaling
    # For RMS amplitude: normalize by sum of window
    win_sum = np.sum(win)
    win_amplitude = win / win_sum * 2.0  # Factor of 2 for one-sided spectrum

    # For power scaling: normalize by sum of squared window
    win_power = win / np.sqrt(np.sum(win ** 2) + EPS)

    # Pad signal if needed
    if len(x) < nperseg:
        x = np.pad(x, (0, nperseg - len(x)))

    # Number of frames
    n_frames = max(1, 1 + (len(x) - nperseg) // hop)

    # Create strided view for efficient computation
    shape = (n_frames, nperseg)
    strides = (x.strides[0] * hop, x.strides[0])

    # Ensure we don't exceed array bounds
    x_padded = np.pad(x, (0, max(0, (n_frames - 1) * hop + nperseg - len(x))))
    frames = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    # Apply window based on scaling type
    if scaling == 'amplitude':
        frames_windowed = frames * win_amplitude[None, :]
        # Compute FFT
        X = np.asarray(rfft(frames_windowed, axis=1))
        magnitude = np.abs(X)
        # DC and Nyquist bins are not mirrored in a one-sided spectrum,
        # so undo the factor-of-2 that was applied to all bins via win_amplitude.
        magnitude[:, 0] /= 2.0
        if nperseg % 2 == 0:
            magnitude[:, -1] /= 2.0
    elif scaling == 'power':
        frames_windowed = frames * win_power[None, :]
        X = np.asarray(rfft(frames_windowed, axis=1))
        magnitude = np.abs(X) ** 2
    elif scaling == 'density':
        frames_windowed = frames * win_power[None, :]
        X = np.asarray(rfft(frames_windowed, axis=1))
        # Power spectral density (normalize by frequency resolution)
        df = sample_rate / nperseg
        magnitude = np.abs(X) ** 2 / df
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    # Frequency and time axes
    frequencies = rfftfreq(nperseg, d=1.0 / sample_rate)
    time = (np.arange(n_frames) * hop + nperseg // 2) / sample_rate

    # Transpose to (freq, time)
    return time, frequencies, magnitude.T


def compute_stft_dB_SPL(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    weighting: Literal['Z', 'A', 'C'] = 'Z',
    ref_pressure: float = P_REF,
    db_floor: float = -120.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT magnitude in dB SPL with optional frequency weighting.

    Args:
        x: Input signal in Pascals (calibrated).
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size.
        noverlap: Overlap in samples.
        window: Window function.
        weighting: Frequency weighting ('Z'=flat, 'A'=A-weight, 'C'=C-weight).
        ref_pressure: Reference pressure for dB SPL (default: 20 µPa).
        db_floor: Minimum dB value (floor for numerical stability).

    Returns:
        (time, frequencies, magnitude_dB) where magnitude_dB is in dB SPL.
    """
    # Compute STFT in amplitude
    time, frequencies, magnitude = compute_stft(
        x, sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        scaling='amplitude',
    )

    # Apply frequency weighting
    w = weighting.upper()
    if w == 'A':
        # A-weighting curve
        weights = a_weight_linear(frequencies)
        magnitude = magnitude * weights[:, np.newaxis]
    elif w == 'C':
        # C-weighting curve
        from weighting import c_weight_frequency_response
        c_dB = c_weight_frequency_response(frequencies)
        weights = 10.0 ** (c_dB / 20.0)
        magnitude = magnitude * weights[:, np.newaxis]
    elif w != 'Z':
        raise ValueError(f"Unknown weighting: {weighting}. Use 'Z', 'A', or 'C'.")

    # Convert peak amplitude to RMS for correct dB SPL.
    # Each STFT bin represents a sinusoidal component; RMS = peak / sqrt(2).
    magnitude_rms = magnitude / np.sqrt(2.0)
    magnitude_dB = 20.0 * np.log10(np.maximum(magnitude_rms, EPS) / ref_pressure)

    # Apply floor
    magnitude_dB = np.maximum(magnitude_dB, db_floor)

    return time, frequencies, magnitude_dB


@dataclass
class STFTResult:
    """Container for STFT analysis results."""
    time_s: np.ndarray
    frequencies_Hz: np.ndarray
    magnitude_dB: np.ndarray
    weighting: str
    sample_rate: int
    nperseg: int
    noverlap: int
    window: str

    @property
    def duration_s(self) -> float:
        """Signal duration in seconds."""
        return self.time_s[-1] if len(self.time_s) > 0 else 0.0

    @property
    def freq_resolution_Hz(self) -> float:
        """Frequency resolution in Hz."""
        return self.sample_rate / self.nperseg

    @property
    def time_resolution_s(self) -> float:
        """Time resolution (hop) in seconds."""
        hop = self.nperseg - self.noverlap
        return hop / self.sample_rate

    def get_max_level(self) -> float:
        """Get maximum level in dB."""
        return float(np.max(self.magnitude_dB))

    def get_freq_at_max(self) -> float:
        """Get frequency of maximum level."""
        max_idx = np.unravel_index(np.argmax(self.magnitude_dB), self.magnitude_dB.shape)
        return float(self.frequencies_Hz[max_idx[0]])


def analyze_stft(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    weighting: Literal['Z', 'A', 'C'] = 'Z',
) -> STFTResult:
    """
    Perform STFT analysis and return structured result.

    Args:
        x: Input signal in Pascals.
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size.
        noverlap: Overlap in samples.
        window: Window function.
        weighting: Frequency weighting.

    Returns:
        STFTResult object with all analysis data.
    """
    if noverlap is None:
        noverlap = int(nperseg * 0.75)

    time, freq, mag_dB = compute_stft_dB_SPL(
        x, sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        weighting=weighting,
    )

    return STFTResult(
        time_s=time,
        frequencies_Hz=freq,
        magnitude_dB=mag_dB,
        weighting=weighting,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )


def compute_spectrogram_pair(
    x: np.ndarray,
    sample_rate: int,
    **kwargs,
) -> Tuple[STFTResult, STFTResult]:
    """
    Compute both Z-weighted and A-weighted spectrograms.

    Args:
        x: Input signal in Pascals.
        sample_rate: Sample rate in Hz.
        **kwargs: Additional arguments for analyze_stft.

    Returns:
        (z_weighted_result, a_weighted_result) tuple of STFTResult objects.
    """
    result_z = analyze_stft(x, sample_rate, weighting='Z', **kwargs)
    result_a = analyze_stft(x, sample_rate, weighting='A', **kwargs)
    return result_z, result_a


def save_stft_data(
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    mag_raw: np.ndarray,
    mag_weighted: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save STFT results to NPZ file.

    Args:
        time_axis: Time axis array.
        freq_axis: Frequency axis array.
        mag_raw: Z-weighted magnitude.
        mag_weighted: A-weighted magnitude.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        time_s=time_axis,
        frequency_hz=freq_axis,
        magnitude_raw=mag_raw,
        magnitude_aweighted=mag_weighted,
    )


def load_stft_data(input_path: Path) -> dict:
    """
    Load STFT results from NPZ file.

    Args:
        input_path: Input file path.

    Returns:
        Dictionary with loaded arrays.
    """
    data = np.load(input_path, allow_pickle=True)
    return dict(data)


# ---- Slice helpers ----

def slice_signal(
    x: np.ndarray,
    sr: int,
    start_s: float | None,
    duration_s: float | None,
) -> tuple[np.ndarray, float]:
    """
    Slice signal by time.

    Args:
        x: Input signal.
        sr: Sample rate in Hz.
        start_s: Start time in seconds (default: 0).
        duration_s: Duration in seconds (default: entire signal).

    Returns:
        (sliced_signal, start_time) tuple.
    """
    n = len(x)
    start_i = 0 if start_s is None else int(max(0.0, start_s) * sr)
    start_i = min(start_i, n)

    if duration_s is None:
        end_i = n
    else:
        end_i = min(n, start_i + int(max(0.0, duration_s) * sr))

    return x[start_i:end_i], start_i / float(sr)


# ---- Legacy compatibility functions ----

def stft_amplitude(
    x: np.ndarray,
    sr: int,
    *,
    nperseg: int = 2048,
    noverlap: int = 1536,
    window: str = 'hann',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy function: Compute STFT returning raw magnitude.

    Maintained for backwards compatibility with old main.py.
    """
    return compute_stft(x, sr, nperseg=nperseg, noverlap=noverlap, window=window)


def apply_a_weighting_to_spectrogram(
    mag: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """
    Legacy function: Apply A-weighting to spectrogram.

    Maintained for backwards compatibility.
    """
    weights = a_weight_linear(frequencies)
    return mag * weights[:, np.newaxis]


def plot_stft_comparison(
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    mag_raw: np.ndarray,
    mag_weighted: np.ndarray,
    output_path_raw: Path,
    output_path_weighted: Path,
    title: str = "STFT Analysis",
    db_range: tuple[float, float] = (0, 140),
) -> None:
    """
    Create two plots: Z-weighted and A-weighted spectrograms in dB SPL.

    Args:
        time_axis: Time axis (seconds).
        freq_axis: Frequency axis (Hz).
        mag_raw: Z-weighted magnitude in dB SPL.
        mag_weighted: A-weighted magnitude in dB SPL.
        output_path_raw: Output path for Z-weighted plot.
        output_path_weighted: Output path for A-weighted plot.
        title: Plot title prefix.
        db_range: (min_dB, max_dB) for colorbar.
    """
    output_path_raw.parent.mkdir(parents=True, exist_ok=True)
    output_path_weighted.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = db_range

    # Plot 1: Z-weighted (raw) STFT
    plt.figure(figsize=(12, 5))
    mesh = plt.pcolormesh(time_axis, freq_axis, mag_raw, shading="auto",
                          vmin=vmin, vmax=vmax, cmap="viridis")
    plt.title(f"{title} - Z-Weighted (Unweighted)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Level (dB SPL)")
    plt.ylim(0, min(20000, freq_axis[-1]))
    plt.tight_layout()
    plt.savefig(output_path_raw, dpi=200)
    plt.close()

    # Plot 2: A-weighted (perceptual)
    plt.figure(figsize=(12, 5))
    mesh = plt.pcolormesh(time_axis, freq_axis, mag_weighted, shading="auto",
                          vmin=vmin, vmax=vmax, cmap="viridis")
    plt.title(f"{title} - A-Weighted (Perceptual)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Level (dB SPL)")
    plt.ylim(0, min(20000, freq_axis[-1]))
    plt.tight_layout()
    plt.savefig(output_path_weighted, dpi=200)
    plt.close()


# ---- WAV loader (legacy) ----

@dataclass(frozen=True)
class WavData:
    """Legacy WAV data container."""
    path: Path
    sample_rate: int
    samples: np.ndarray
    time_s: np.ndarray


def load_wav(wav_path: Path, *, dtype: str = "float32", mono: bool = False) -> WavData:
    """Legacy WAV loader for backwards compatibility."""
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    data, sr = sf.read(str(wav_path), dtype=dtype, always_2d=True)
    data = np.asarray(data)

    if mono:
        samples = data.mean(axis=1)
    else:
        samples = data

    time_s = np.arange(samples.shape[0] if samples.ndim == 1 else samples.shape[0],
                       dtype=np.float64) / float(sr)
    return WavData(path=wav_path, sample_rate=sr, samples=samples, time_s=time_s)


# ---- CLI ----

def main() -> int:
    parser = argparse.ArgumentParser(
        description="STFT Analysis with calibrated dB SPL output"
    )
    parser.add_argument("wav", type=Path, help="Input WAV file")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "int16", "int32"],
                        help="Load dtype")
    parser.add_argument("--Pa-per-FS", type=float, default=100.0,
                        help="Calibration: Pascals per full-scale (default: 100)")
    parser.add_argument("--start", type=float, default=None, help="Start time (seconds)")
    parser.add_argument("--duration", type=float, default=None, help="Duration (seconds)")
    parser.add_argument("--nperseg", type=int, default=2048, help="FFT window size")
    parser.add_argument("--noverlap", type=int, default=1536, help="Overlap in samples")
    parser.add_argument("--out-raw", type=Path, default=None, help="Output Z-weighted plot")
    parser.add_argument("--out-weighted", type=Path, default=None, help="Output A-weighted plot")
    parser.add_argument("--out-data", type=Path, default=None, help="Output NPZ data file")
    parser.add_argument("--show", action="store_true", help="Display plots")

    args = parser.parse_args()

    # Load audio
    wav = load_wav(args.wav, dtype=args.dtype, mono=False)
    x = wav.samples
    if x.ndim > 1:
        x = x.mean(axis=1)
    sr = wav.sample_rate

    # Calibrate to Pascals
    x_Pa = x * args.Pa_per_FS

    x_win, t0 = slice_signal(x_Pa, sr, args.start, args.duration)

    if x_win.size == 0:
        raise RuntimeError("Selected time window is empty.")

    # Compute STFT in dB SPL
    t, f, mag_z = compute_stft_dB_SPL(
        x_win, sr,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        window="hann",
        weighting='Z',
    )
    _, _, mag_a = compute_stft_dB_SPL(
        x_win, sr,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        window="hann",
        weighting='A',
    )
    t = t + t0

    # Output paths
    base = args.wav.with_suffix("")
    out_raw = args.out_raw or base.parent / f"{base.name}_stft_z.png"
    out_weighted = args.out_weighted or base.parent / f"{base.name}_stft_a.png"
    out_data = args.out_data or base.parent / f"{base.name}_stft_data.npz"

    # Determine dB range
    max_level = max(np.max(mag_z), np.max(mag_a))
    db_range = (max(0, max_level - 80), max_level + 5)

    # Save plots
    plot_stft_comparison(
        t, f, mag_z, mag_a,
        out_raw, out_weighted,
        title=f"STFT: {args.wav.name}",
        db_range=db_range,
    )

    # Save data
    save_stft_data(t, f, mag_z, mag_a, out_data)

    print(f"Z-weighted plot: {out_raw.resolve()}")
    print(f"A-weighted plot: {out_weighted.resolve()}")
    print(f"Data file: {out_data.resolve()}")
    print(f"Max level: {max_level:.1f} dB SPL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
