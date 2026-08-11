#!/usr/bin/env python3
"""
STFT.py - Short-Time Fourier Transform for Acoustic Analysis

Computes calibrated STFT spectrograms with correct, documented scaling.

    What the numbers mean.

    Two different quantities get called "the spectrogram in dB SPL", and they are
    not interchangeable:

      - BAND LEVEL ('rms'): the level a filter as wide as one FFT bin would read.
        For a pure tone this is exactly the tone's RMS level, which makes it the
        intuitive choice. For BROADBAND content - which is what a gunshot is - it
        scales with the bin width, so halving nperseg adds 3 dB to every value.

      - POWER SPECTRAL DENSITY ('psd'): dB re (20 uPa)^2/Hz. Independent of the FFT
        size, so two analyses at different resolutions are comparable. This is the
        honest choice for impulsive broadband signals.

    Both are provided, the scaling used is recorded on the result, and the colourbar
    label follows the scaling rather than being hard-coded.

Usage:
    from STFT import analyze_stft

    result = analyze_stft(pressure_Pa, sample_rate, weighting='Z')
    result.magnitude_dB     # (n_freq, n_frames)
    result.level_label      # matches what was actually computed
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq

from calibration import EPS, P_REF
from weighting import weighting_linear

# Supported window types
WINDOW_TYPES = {"hann", "hamming", "blackman", "flattop", "rectangular"}

Scaling = Literal["rms", "psd"]


def get_window(name: str, length: int) -> np.ndarray:
    """
    Get a window function.

    Periodic (not symmetric) windows are used, which is correct for spectral
    analysis of a continuous signal via overlapping frames.

    Args:
        name: 'hann', 'hamming', 'blackman', 'flattop' or 'rectangular'.
        length: Window length in samples.

    Returns:
        Window array.
    """
    from scipy.signal import get_window as _scipy_window

    name = name.lower()
    if name in ("rect", "rectangular"):
        return np.ones(length, dtype=np.float64)
    if name == "flattop":
        # Flat-top has very low scalloping loss: a tone between bin centres reads
        # within 0.01 dB instead of Hann's 1.42 dB worst case.
        return np.asarray(_scipy_window("flattop", length, fftbins=True), dtype=np.float64)
    if name in ("hann", "hamming", "blackman"):
        return np.asarray(_scipy_window(name, length, fftbins=True), dtype=np.float64)
    raise ValueError(f"Unknown window: {name}. Use one of: {sorted(WINDOW_TYPES)}")


def window_coherent_gain(win: np.ndarray) -> float:
    """Coherent gain: sum(w)/N. Scales a tone's amplitude."""
    return float(np.sum(win) / len(win))


def window_enbw_bins(win: np.ndarray) -> float:
    """
    Equivalent noise bandwidth of a window, in FFT bins.

    Hann = 1.5 bins, Hamming = 1.36, Blackman = 1.73, rectangular = 1.0.
    """
    s1 = float(np.sum(win))
    s2 = float(np.sum(win ** 2))
    if s1 <= EPS:
        return 1.0
    return float(len(win) * s2 / (s1 ** 2))


def window_enbw_Hz(win: np.ndarray, sample_rate: float, nperseg: int) -> float:
    """Equivalent noise bandwidth of one FFT bin, in Hz."""
    return window_enbw_bins(win) * sample_rate / nperseg


def default_noverlap(nperseg: int, fraction: float = 0.75) -> int:
    """
    Overlap for a given window size.

    Derived from nperseg rather than fixed: a hard-coded overlap silently exceeds
    smaller window sizes and makes the analysis fail outright.
    """
    return int(nperseg * fraction)


def _frame_signal(x: np.ndarray, nperseg: int, hop: int) -> Tuple[np.ndarray, int]:
    """
    Split a signal into overlapping frames using a stride view.

    The stride is taken from the array the view is built on, not from a different
    array that happened to be contiguous, so a non-contiguous input cannot silently
    produce garbage.
    """
    n = len(x)
    if n < nperseg:
        x = np.pad(x, (0, nperseg - n))
        n = len(x)

    n_frames = max(1, 1 + (n - nperseg) // hop)
    needed = (n_frames - 1) * hop + nperseg
    if needed > n:
        x = np.pad(x, (0, needed - n))

    x = np.ascontiguousarray(x, dtype=np.float64)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nperseg),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    return frames, n_frames


def compute_stft(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    window: str = "hann",
    scaling: Scaling = "rms",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Short-Time Fourier Transform with calibrated scaling.

    Args:
        x: Input signal (1D, in Pascals for calibrated analysis).
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size in samples.
        noverlap: Overlap in samples. Defaults to 75% of nperseg.
        window: Window function name.
        scaling: 'rms' for per-bin band level (Pa, RMS) or 'psd' for power spectral
                 density (Pa^2/Hz).

    Returns:
        (time, frequencies, magnitude) with magnitude shaped (n_freq_bins, n_frames).
        Units are Pa (RMS) for 'rms' and Pa^2/Hz for 'psd'.

    Raises:
        ValueError: If parameters are inconsistent.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array")
    if nperseg <= 0:
        raise ValueError(f"nperseg must be > 0, got {nperseg}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    if noverlap is None:
        noverlap = default_noverlap(nperseg)
    noverlap = int(noverlap)
    if not (0 <= noverlap < nperseg):
        raise ValueError(
            f"noverlap ({noverlap}) must satisfy 0 <= noverlap < nperseg ({nperseg})"
        )

    hop = nperseg - noverlap
    win = get_window(window, nperseg)
    frames, n_frames = _frame_signal(x, nperseg, hop)

    X = np.asarray(rfft(frames * win[None, :], axis=1))
    power = np.abs(X) ** 2

    # One-sided spectrum: every bin except DC and (for even N) Nyquist represents
    # a conjugate pair, so its power is doubled.
    doubling = np.full(X.shape[1], 2.0)
    doubling[0] = 1.0
    if nperseg % 2 == 0:
        doubling[-1] = 1.0
    power = power * doubling[None, :]

    s1 = float(np.sum(win))
    s2 = float(np.sum(win ** 2))

    if scaling == "rms":
        # Tone RMS. For a tone of amplitude A on a bin centre, |X| = A*s1/2, so
        # after the one-sided doubling power = 2*|X|^2 = A^2*s1^2/2, and the true
        # mean square is A^2/2 = power / s1^2.
        magnitude = np.sqrt(power) / s1
    elif scaling == "psd":
        # Power spectral density in Pa^2/Hz
        magnitude = power / (sample_rate * s2)
    else:
        raise ValueError(f"Unknown scaling: {scaling}. Use 'rms' or 'psd'.")

    frequencies = rfftfreq(nperseg, d=1.0 / sample_rate)
    # Frame timestamps mark the window CENTRE, which is where the energy in that
    # frame is centred.
    time = (np.arange(n_frames) * hop + nperseg / 2.0) / sample_rate

    return time, frequencies, magnitude.T


def compute_stft_dB_SPL(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    window: str = "hann",
    weighting: Literal["Z", "A", "C"] = "Z",
    scaling: Scaling = "rms",
    ref_pressure: float = P_REF,
    db_floor: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute an STFT in decibels with optional frequency weighting.

    Args:
        x: Input signal in Pascals (calibrated).
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size.
        noverlap: Overlap in samples.
        window: Window function.
        weighting: 'Z' (flat), 'A' or 'C'. Applied per frequency bin using the
                   analytical weighting curve.
        scaling: 'rms' (band level per bin) or 'psd' (density).
        ref_pressure: Reference pressure (default 20 uPa).
        db_floor: Values below this are clamped, keeping the display range usable.

    Returns:
        (time, frequencies, magnitude_dB).
    """
    time, frequencies, magnitude = compute_stft(
        x, sample_rate,
        nperseg=nperseg, noverlap=noverlap, window=window, scaling=scaling,
    )

    w = weighting.upper()
    if w not in ("Z", "A", "C"):
        raise ValueError(f"Unknown weighting: {weighting}. Use 'Z', 'A', or 'C'.")
    if w != "Z":
        gain = weighting_linear(frequencies, w)[:, np.newaxis]
        # 'rms' magnitude is an amplitude, 'psd' is a power density
        magnitude = magnitude * (gain if scaling == "rms" else gain ** 2)

    if scaling == "rms":
        magnitude_dB = 20.0 * np.log10(np.maximum(magnitude, EPS) / ref_pressure)
    else:
        magnitude_dB = 10.0 * np.log10(np.maximum(magnitude, EPS) / (ref_pressure ** 2))

    return time, frequencies, np.maximum(magnitude_dB, db_floor)


@dataclass
class STFTResult:
    """Container for STFT analysis results, including how it was scaled."""
    time_s: np.ndarray
    frequencies_Hz: np.ndarray
    magnitude_dB: np.ndarray
    weighting: str
    sample_rate: int
    nperseg: int
    noverlap: int
    window: str
    scaling: Scaling = "rms"
    enbw_Hz: float = 0.0
    calibrated: bool = True

    @property
    def duration_s(self) -> float:
        """Signal duration in seconds."""
        return float(self.time_s[-1]) if len(self.time_s) > 0 else 0.0

    @property
    def freq_resolution_Hz(self) -> float:
        """Frequency resolution in Hz."""
        return self.sample_rate / self.nperseg

    @property
    def time_resolution_s(self) -> float:
        """Time resolution (hop) in seconds."""
        return (self.nperseg - self.noverlap) / self.sample_rate

    @property
    def level_label(self) -> str:
        """
        Colourbar label matching what was actually computed.

        A label of "dB SPL" on a power spectral density, or on a band level whose
        value depends on the FFT size, misrepresents the plot.
        """
        unit = "dB" if self.calibrated else "dB re FS"
        if self.scaling == "psd":
            return f"PSD ({unit} re (20 uPa)^2/Hz)"
        return f"{self.weighting}-weighted band level ({unit}, {self.enbw_Hz:.1f} Hz bins)"

    def get_max_level(self) -> float:
        """Maximum level in dB."""
        return float(np.max(self.magnitude_dB)) if self.magnitude_dB.size else float("-inf")

    def get_freq_at_max(self) -> float:
        """Frequency of the maximum level."""
        if not self.magnitude_dB.size:
            return 0.0
        idx = np.unravel_index(np.argmax(self.magnitude_dB), self.magnitude_dB.shape)
        return float(self.frequencies_Hz[idx[0]])

    def decimate_frames(self, factor: int) -> "STFTResult":
        """
        Return a copy keeping every Nth frame, for display of long recordings.

        A full-resolution spectrogram of a long high-rate recording is far larger
        than any display or interactive plot can use.
        """
        factor = max(1, int(factor))
        if factor == 1:
            return self
        return STFTResult(
            time_s=self.time_s[::factor],
            frequencies_Hz=self.frequencies_Hz,
            magnitude_dB=self.magnitude_dB[:, ::factor],
            weighting=self.weighting, sample_rate=self.sample_rate,
            nperseg=self.nperseg, noverlap=self.noverlap, window=self.window,
            scaling=self.scaling, enbw_Hz=self.enbw_Hz, calibrated=self.calibrated,
        )

    def to_dict(self) -> dict:
        """Analysis parameters, for the provenance record."""
        return {
            "weighting": self.weighting,
            "sample_rate": self.sample_rate,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "window": self.window,
            "scaling": self.scaling,
            "enbw_Hz": round(self.enbw_Hz, 3),
            "freq_resolution_Hz": round(self.freq_resolution_Hz, 3),
            "time_resolution_ms": round(self.time_resolution_s * 1000.0, 4),
            "level_label": self.level_label,
        }


def analyze_stft(
    x: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    window: str = "hann",
    weighting: Literal["Z", "A", "C"] = "Z",
    scaling: Scaling = "rms",
    db_floor: float = -20.0,
    calibrated: bool = True,
) -> STFTResult:
    """
    Perform STFT analysis and return a structured result.

    Args:
        x: Input signal in Pascals.
        sample_rate: Sample rate in Hz.
        nperseg: FFT window size.
        noverlap: Overlap in samples. Defaults to 75% of nperseg; an overlap that
                  does not fit the window is corrected rather than raising.
        window: Window function.
        weighting: Frequency weighting.
        scaling: 'rms' or 'psd'.
        db_floor: Lower clamp for the returned decibel values.
        calibrated: Whether levels are true SPL, for labelling.

    Returns:
        STFTResult with all analysis data.
    """
    if nperseg <= 0:
        raise ValueError(f"nperseg must be > 0, got {nperseg}")

    if noverlap is None or not (0 <= noverlap < nperseg):
        noverlap = default_noverlap(nperseg)

    time, freq, mag_dB = compute_stft_dB_SPL(
        x, sample_rate,
        nperseg=nperseg, noverlap=noverlap, window=window,
        weighting=weighting, scaling=scaling, db_floor=db_floor,
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
        scaling=scaling,
        enbw_Hz=window_enbw_Hz(get_window(window, nperseg), sample_rate, nperseg),
        calibrated=calibrated,
    )


def compute_spectrogram_pair(
    x: np.ndarray,
    sample_rate: int,
    weightings: Tuple[str, str] = ("Z", "C"),
    **kwargs,
) -> Tuple[STFTResult, STFTResult]:
    """
    Compute two weighted spectrograms from one signal.

    Args:
        x: Input signal in Pascals.
        sample_rate: Sample rate in Hz.
        weightings: The two weightings to compute.
        **kwargs: Additional arguments for analyze_stft.

    Returns:
        Tuple of STFTResult objects.
    """
    kwargs.pop("weighting", None)
    return (
        analyze_stft(x, sample_rate, weighting=weightings[0], **kwargs),
        analyze_stft(x, sample_rate, weighting=weightings[1], **kwargs),
    )


def recommended_nperseg(sample_rate: int, target_ms: float = 2.0) -> int:
    """
    Suggest an FFT size resolving a target time span, rounded to a power of two.

    A muzzle blast lasts a few milliseconds. A 2048-point window is 42.7 ms at
    48 kHz - twenty times longer than the event it is meant to resolve, which smears
    the blast across the whole frame. Scaling the window with the sample rate keeps
    the time resolution physically meaningful.

    Args:
        sample_rate: Sample rate in Hz.
        target_ms: Desired window duration in milliseconds.

    Returns:
        A power-of-two FFT size, at least 128.
    """
    target = max(128, int(round(target_ms * sample_rate / 1000.0)))
    return int(2 ** int(round(np.log2(target))))


def save_stft_data(
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    mag_raw: np.ndarray,
    mag_weighted: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save STFT results to an NPZ file.

    Args:
        time_axis: Time axis array.
        freq_axis: Frequency axis array.
        mag_raw: Unweighted magnitude.
        mag_weighted: Weighted magnitude.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time_s=time_axis,
        frequencies_Hz=freq_axis,
        magnitude_raw_dB=mag_raw,
        magnitude_weighted_dB=mag_weighted,
    )


# ---- CLI for testing ----

def main() -> int:
    """Verify STFT scaling against analytically known signals."""
    parser = argparse.ArgumentParser(description="Test STFT scaling")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate")
    parser.add_argument("--nperseg", type=int, default=2048, help="FFT size")
    args = parser.parse_args()

    fs, nperseg = args.fs, args.nperseg
    t = np.arange(fs) / fs

    print(f"fs={fs} Hz, nperseg={nperseg}")
    for name in ("hann", "hamming", "blackman", "flattop", "rectangular"):
        w = get_window(name, nperseg)
        print(f"  {name:12} coherent gain {window_coherent_gain(w):.4f}, "
              f"ENBW {window_enbw_bins(w):.3f} bins "
              f"({window_enbw_Hz(w, fs, nperseg):.2f} Hz)")

    # A 1 kHz tone at exactly 94 dB SPL RMS must read 94 dB in its bin.
    target_dB = 94.0
    amp = np.sqrt(2.0) * P_REF * 10 ** (target_dB / 20.0)
    x = amp * np.sin(2 * np.pi * 1000.0 * t)

    print(f"\n1 kHz tone at {target_dB:.1f} dB SPL:")
    for scaling in ("rms", "psd"):
        r = analyze_stft(x, fs, nperseg=nperseg, scaling=scaling)
        peak = r.get_max_level()
        if scaling == "rms":
            print(f"  {scaling:4} -> peak bin {peak:7.2f} dB   (expect {target_dB:.2f})")
        else:
            band = peak + 10 * np.log10(r.enbw_Hz)
            print(f"  {scaling:4} -> peak bin {peak:7.2f} dB/Hz, "
                  f"x ENBW = {band:6.2f} dB (expect {target_dB:.2f})")

    # White noise: PSD must be flat and independent of the FFT size.
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, fs) * P_REF * 10 ** (80 / 20.0)
    print("\nWhite noise, PSD independence from FFT size:")
    for n in (512, 1024, 2048, 4096):
        r = analyze_stft(noise, fs, nperseg=n, scaling="psd")
        band = analyze_stft(noise, fs, nperseg=n, scaling="rms")
        mid = slice(len(r.frequencies_Hz) // 4, len(r.frequencies_Hz) * 3 // 4)
        print(f"  nperseg={n:5}: PSD {np.mean(r.magnitude_dB[mid]):7.2f} dB/Hz   "
              f"band level {np.mean(band.magnitude_dB[mid]):7.2f} dB "
              f"(ENBW {r.enbw_Hz:6.2f} Hz)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
