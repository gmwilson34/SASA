#!/usr/bin/env python3
"""
weighting.py - Frequency Weighting Filters for Acoustic Measurements

Implements standardized frequency weighting filters:
  - A-weighting: Matches human hearing sensitivity (IEC 61672-1)
  - Z-weighting: Flat/unweighted (pass-through)
  - C-weighting: For peak measurements (optional, included for completeness)

The A-weighting filter is implemented as a time-domain IIR filter using
cascaded second-order sections (SOS) for numerical stability.

Usage:
    from weighting import apply_a_weight, apply_z_weight, AWeightFilter

    # Simple functional interface
    weighted = apply_a_weight(samples, sample_rate)
    unweighted = apply_z_weight(samples, sample_rate)  # pass-through

    # Stateful filter for streaming/real-time
    filt = AWeightFilter(sample_rate)
    weighted = filt.apply(samples)

References:
    - IEC 61672-1:2013 - Electroacoustics - Sound level meters
    - ANSI S1.4-1983 - Specification for Sound Level Meters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy.signal import bilinear_zpk, zpk2sos, sosfilt, sosfilt_zi, sosfiltfilt

# Numerical stability floor
EPS: float = 1e-30


# ---- A-Weighting IIR Filter Design (IEC 61672-1) ----

# A-weighting analog prototype poles and zeros
# Frequencies in rad/s (analog domain)
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217

# Analog prototype zeros (all at s=0, order 4)
# Analog prototype poles (2nd order at f1, f4; 1st order at f2, f3)


def _design_a_weight_analog() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Design A-weighting analog prototype filter.

    Returns:
        (zeros, poles, gain) in analog domain (rad/s).
    """
    # Convert Hz to rad/s
    w1 = 2 * np.pi * _F1
    w2 = 2 * np.pi * _F2
    w3 = 2 * np.pi * _F3
    w4 = 2 * np.pi * _F4

    # A-weighting has 4 zeros at s=0 (high-pass character)
    zeros = np.array([0, 0, 0, 0], dtype=np.complex128)

    # Poles: double pole at f1, single at f2, single at f3, double at f4
    poles = np.array([
        -w1, -w1,  # double pole
        -w2,       # single pole
        -w3,       # single pole
        -w4, -w4,  # double pole
    ], dtype=np.complex128)

    # Gain normalization: A-weight = 0 dB at 1000 Hz
    # We'll compute this after bilinear transform
    gain = 1.0

    return zeros, poles, gain


def _normalize_a_weight_gain(sos: np.ndarray, fs: float) -> np.ndarray:
    """
    Normalize A-weight filter so gain = 0 dB at 1000 Hz.
    """
    from scipy.signal import sosfreqz

    # Evaluate frequency response at 1000 Hz
    w_norm = 1000.0 / (fs / 2)  # Normalized frequency
    if w_norm >= 1.0:
        w_norm = 0.999  # Avoid aliasing issues for low sample rates

    _, h = sosfreqz(sos, worN=[w_norm * np.pi], fs=2*np.pi)
    h_arr = np.asarray(h)
    gain_at_1k = float(np.abs(h_arr.flat[0]))

    if gain_at_1k > EPS:
        # Adjust gain of first section
        sos = sos.copy()
        sos[0, :3] /= gain_at_1k

    return sos


def design_a_weight_sos(fs: float) -> np.ndarray:
    """
    Design A-weighting digital filter as cascaded second-order sections.

    Args:
        fs: Sample rate in Hz.

    Returns:
        SOS array (N x 6) for use with scipy.signal.sosfilt.
    """
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive, got {fs}")

    # Get analog prototype
    z_a, p_a, k_a = _design_a_weight_analog()

    # Bilinear transform to digital domain
    z_d, p_d, k_d = bilinear_zpk(z_a, p_a, k_a, fs)

    # Convert to second-order sections for numerical stability
    sos = zpk2sos(z_d, p_d, k_d)

    # Normalize gain at 1000 Hz
    sos = _normalize_a_weight_gain(sos, fs)

    return sos


# ---- C-Weighting (for peak measurements) ----

_FC1 = 20.598997
_FC4 = 12194.217


def _design_c_weight_analog() -> tuple[np.ndarray, np.ndarray, float]:
    """Design C-weighting analog prototype."""
    w1 = 2 * np.pi * _FC1
    w4 = 2 * np.pi * _FC4

    # C-weight: 2 zeros at s=0, double poles at f1 and f4
    zeros = np.array([0, 0], dtype=np.complex128)
    poles = np.array([-w1, -w1, -w4, -w4], dtype=np.complex128)
    gain = 1.0

    return zeros, poles, gain


def _normalize_c_weight_gain(sos: np.ndarray, fs: float) -> np.ndarray:
    """Normalize C-weight filter so gain = 0 dB at 1000 Hz."""
    from scipy.signal import sosfreqz

    w_norm = min(1000.0 / (fs / 2), 0.999)
    _, h = sosfreqz(sos, worN=[w_norm * np.pi], fs=2*np.pi)
    h_arr = np.asarray(h)
    gain_at_1k = float(np.abs(h_arr.flat[0]))

    if gain_at_1k > EPS:
        sos = sos.copy()
        sos[0, :3] /= gain_at_1k

    return sos


def design_c_weight_sos(fs: float) -> np.ndarray:
    """
    Design C-weighting digital filter as SOS.

    C-weighting is flatter than A-weighting and is often used for
    peak sound pressure measurements.
    """
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive, got {fs}")

    z_a, p_a, k_a = _design_c_weight_analog()
    z_d, p_d, k_d = bilinear_zpk(z_a, p_a, k_a, fs)
    sos = zpk2sos(z_d, p_d, k_d)
    sos = _normalize_c_weight_gain(sos, fs)

    return sos


# ---- Filter Classes ----

@dataclass
class AWeightFilter:
    """
    Stateful A-weighting filter for streaming/real-time processing.

    Maintains filter state between calls for continuous filtering.
    """
    sample_rate: float
    sos: Optional[np.ndarray] = None
    zi: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.sos = design_a_weight_sos(self.sample_rate)
        self.zi = sosfilt_zi(self.sos)

    def apply(self, x: np.ndarray, reset: bool = False) -> np.ndarray:
        """
        Apply A-weighting filter to signal.

        Args:
            x: Input signal (1D array).
            reset: If True, reset filter state before filtering.

        Returns:
            A-weighted signal.
        """
        if reset:
            self.zi = sosfilt_zi(self.sos)

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Input must be 1D array")

        # Scale initial conditions by first sample for smooth startup
        zi_scaled = self.zi * x[0] if len(x) > 0 else self.zi
        y, self.zi = sosfilt(self.sos, x, zi=zi_scaled)
        return y

    def reset(self) -> None:
        """Reset filter state."""
        self.zi = sosfilt_zi(self.sos)


@dataclass
class CWeightFilter:
    """Stateful C-weighting filter."""
    sample_rate: float
    sos: Optional[np.ndarray] = None
    zi: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.sos = design_c_weight_sos(self.sample_rate)
        self.zi = sosfilt_zi(self.sos)

    def apply(self, x: np.ndarray, reset: bool = False) -> np.ndarray:
        if reset:
            self.zi = sosfilt_zi(self.sos)

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Input must be 1D array")

        zi_scaled = self.zi * x[0] if len(x) > 0 else self.zi
        y, self.zi = sosfilt(self.sos, x, zi=zi_scaled)
        return y

    def reset(self) -> None:
        self.zi = sosfilt_zi(self.sos)


# ---- Functional Interface ----

def apply_a_weight(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply A-weighting filter to signal (stateless, zero initial conditions).

    Args:
        x: Input signal (1D mono array).
        fs: Sample rate in Hz.

    Returns:
        A-weighted signal.

    Note:
        For processing multiple segments from the same recording,
        use AWeightFilter class to maintain state between segments.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    sos = design_a_weight_sos(fs)
    result = sosfilt(sos, x)
    return np.asarray(result)


def apply_c_weight(x: np.ndarray, fs: float) -> np.ndarray:
    """Apply C-weighting filter to signal."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    sos = design_c_weight_sos(fs)
    result = sosfilt(sos, x)
    return np.asarray(result)


def apply_z_weight(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply Z-weighting (flat/unweighted) - pass-through.

    Args:
        x: Input signal.
        fs: Sample rate (unused, for API consistency).

    Returns:
        Input signal unchanged (as float64).
    """
    return np.asarray(x, dtype=np.float64).copy()


# ---- Zero-Phase Filtering (for offline/post-processing analysis) ----
#
# IEC 61672-1 defines causal time-weighting for real-time SLMs, but for offline
# analysis of recorded signals, zero-phase filtering (forward-backward) is superior:
#   - No startup transient (critical for short shot windows)
#   - Zero group delay (peak location is preserved)
#   - Double stopband attenuation
# Use these for per-shot metric computation on extracted windows.


def apply_a_weight_zerophase(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply zero-phase A-weighting filter (offline analysis).

    Uses forward-backward filtering (sosfiltfilt) to eliminate group delay
    and startup transients. Preferred over causal filtering when processing
    short extracted windows (e.g., per-shot analysis).

    Args:
        x: Input signal (1D mono array).
        fs: Sample rate in Hz.

    Returns:
        A-weighted signal with zero phase distortion.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")
    if len(x) < 18:
        # Signal too short for sosfiltfilt; fall back to causal
        return apply_a_weight(x, fs)
    sos = design_a_weight_sos(fs)
    return np.asarray(sosfiltfilt(sos, x))


def apply_c_weight_zerophase(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply zero-phase C-weighting filter (offline analysis).

    Args:
        x: Input signal (1D mono array).
        fs: Sample rate in Hz.

    Returns:
        C-weighted signal with zero phase distortion.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")
    if len(x) < 12:
        return apply_c_weight(x, fs)
    sos = design_c_weight_sos(fs)
    return np.asarray(sosfiltfilt(sos, x))


# ---- Frequency Response Calculation ----

def a_weight_frequency_response(frequencies: np.ndarray) -> np.ndarray:
    """
    Compute theoretical A-weighting curve for given frequencies.

    Args:
        frequencies: Frequency values in Hz.

    Returns:
        A-weighting in dB (relative to 1000 Hz).

    Note:
        This is the analytical formula, useful for applying A-weighting
        in the frequency domain (e.g., to STFT bins).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    f = np.maximum(f, EPS)  # Avoid division by zero

    f2 = f ** 2

    # A-weighting formula (IEC 61672-1)
    num = 12194.217**2 * f2**2
    denom = ((f2 + 20.598997**2) *
             np.sqrt((f2 + 107.65265**2) * (f2 + 737.86223**2)) *
             (f2 + 12194.217**2))

    # Result in dB
    Ra = num / (denom + EPS)
    A_dB = 20.0 * np.log10(Ra + EPS) + 2.0  # +2 dB offset for normalization at 1kHz

    return A_dB


def a_weight_linear(frequencies: np.ndarray) -> np.ndarray:
    """
    Compute A-weighting as linear multipliers for frequency domain.

    Args:
        frequencies: Frequency values in Hz.

    Returns:
        Linear multipliers (apply to magnitude spectrum).
    """
    A_dB = a_weight_frequency_response(frequencies)
    return 10.0 ** (A_dB / 20.0)


def c_weight_frequency_response(frequencies: np.ndarray) -> np.ndarray:
    """Compute theoretical C-weighting curve for given frequencies."""
    f = np.asarray(frequencies, dtype=np.float64)
    f = np.maximum(f, EPS)

    f2 = f ** 2

    num = 12194.217**2 * f2
    denom = ((f2 + 20.598997**2) * (f2 + 12194.217**2))

    Rc = num / (denom + EPS)
    C_dB = 20.0 * np.log10(Rc + EPS) + 0.062  # Normalization offset

    return C_dB


WeightingType = Literal["A", "C", "Z"]


def apply_weighting(
    x: np.ndarray,
    fs: float,
    weighting: WeightingType = "A",
) -> np.ndarray:
    """
    Apply frequency weighting to signal.

    Args:
        x: Input signal (1D).
        fs: Sample rate in Hz.
        weighting: "A", "C", or "Z" (unweighted).

    Returns:
        Weighted signal.
    """
    w = weighting.upper()
    if w == "A":
        return apply_a_weight(x, fs)
    elif w == "C":
        return apply_c_weight(x, fs)
    elif w == "Z":
        return apply_z_weight(x, fs)
    else:
        raise ValueError(f"Unknown weighting: {weighting}. Use 'A', 'C', or 'Z'.")


def get_weighting_curve_dB(
    frequencies: np.ndarray,
    weighting: WeightingType = "A",
) -> np.ndarray:
    """
    Get frequency weighting curve in dB.

    Args:
        frequencies: Frequency values in Hz.
        weighting: "A", "C", or "Z".

    Returns:
        Weighting in dB at each frequency.
    """
    w = weighting.upper()
    if w == "A":
        return a_weight_frequency_response(frequencies)
    elif w == "C":
        return c_weight_frequency_response(frequencies)
    elif w == "Z":
        return np.zeros_like(frequencies, dtype=np.float64)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")


# ---- CLI for testing ----

def main() -> int:
    """Test and visualize weighting filters."""
    import argparse

    parser = argparse.ArgumentParser(description="Test frequency weighting filters")
    parser.add_argument("--fs", type=float, default=96000, help="Sample rate (Hz)")
    parser.add_argument("--plot", action="store_true", help="Plot frequency response")
    args = parser.parse_args()

    fs = args.fs
    print(f"Sample rate: {fs} Hz")

    # Design filters
    sos_a = design_a_weight_sos(fs)
    sos_c = design_c_weight_sos(fs)

    print(f"\nA-weight filter: {sos_a.shape[0]} biquad sections")
    print(f"C-weight filter: {sos_c.shape[0]} biquad sections")

    # Test at key frequencies
    test_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    print("\nTheoretical A-weighting at standard frequencies:")
    print("Freq (Hz)    A-weight (dB)")
    for f in test_freqs:
        if f < fs / 2:
            a_dB = a_weight_frequency_response(np.array([f]))[0]
            print(f"{f:8.1f}    {a_dB:+.1f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from scipy.signal import sosfreqz

            # Compute digital filter response
            w, h_a = sosfreqz(sos_a, worN=2048, fs=fs)
            _, h_c = sosfreqz(sos_c, worN=2048, fs=fs)

            # Compute theoretical curves
            freqs = np.logspace(np.log10(20), np.log10(fs/2 * 0.99), 500)
            a_theory = a_weight_frequency_response(freqs)
            c_theory = c_weight_frequency_response(freqs)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # A-weighting
            ax = axes[0]
            ax.semilogx(w, 20 * np.log10(np.abs(h_a) + EPS), 'b-', label='Digital filter', linewidth=2)
            ax.semilogx(freqs, a_theory, 'r--', label='Theoretical', linewidth=1)
            ax.set_xlim(20, fs/2)
            ax.set_ylim(-80, 10)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Gain (dB)')
            ax.set_title(f'A-Weighting Filter (fs={fs} Hz)')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

            # C-weighting
            ax = axes[1]
            ax.semilogx(w, 20 * np.log10(np.abs(h_c) + EPS), 'b-', label='Digital filter', linewidth=2)
            ax.semilogx(freqs, c_theory, 'r--', label='Theoretical', linewidth=1)
            ax.set_xlim(20, fs/2)
            ax.set_ylim(-30, 5)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Gain (dB)')
            ax.set_title(f'C-Weighting Filter (fs={fs} Hz)')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

            plt.tight_layout()
            plt.savefig('weighting_response.png', dpi=150)
            print("\nPlot saved to weighting_response.png")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for plotting")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
