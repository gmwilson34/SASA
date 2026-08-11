#!/usr/bin/env python3
"""
weighting.py - Frequency Weighting Filters for Acoustic Measurements

Implements standardized frequency weighting filters:
  - A-weighting: Matches human hearing sensitivity (IEC 61672-1)
  - C-weighting: For peak measurements (IEC 61672-1)
  - Z-weighting: Flat/unweighted (pass-through)

The weighting filters are implemented as time-domain IIR filters using cascaded
second-order sections (SOS) for numerical stability, with bilinear pre-warping so
that each analog corner frequency lands on its intended digital frequency.

    IMPORTANT - weighting must be applied ONCE, causally.

    A sound level meter's weighting network is a causal, single-pass filter, and
    IEC 61672-1 defines every weighted quantity in those terms. Forward-backward
    ("zero-phase") filtering squares the magnitude response, which DOUBLES the
    weighting curve in decibels: A-weighting at 125 Hz becomes -32.4 dB instead of
    -16.2 dB. Use apply_a_weight()/apply_c_weight(), or apply_weighting_with_context()
    when filtering a short extracted window, and never sosfiltfilt.

Usage:
    from weighting import apply_weighting, apply_weighting_with_context

    weighted = apply_weighting(samples, sample_rate, "A")

    # Per-shot window: use the pre-trigger samples as filter warm-up
    shot_a = apply_weighting_with_context(pressure, fs, "A",
                                          start=win_start, stop=win_stop)

References:
    - IEC 61672-1:2013 - Electroacoustics - Sound level meters
    - ANSI S1.4-1983 - Specification for Sound Level Meters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.signal import bilinear_zpk, sosfilt, sosfilt_zi, sosfreqz, zpk2sos

# Numerical stability floor
EPS: float = 1e-30

WeightingType = Literal["A", "C", "Z"]


# ---- Weighting filter pole frequencies (IEC 61672-1) ----

# A- and C-weighting share the outer pole pair; A adds the 107.7 / 737.9 Hz poles.
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217

# C-weighting normalisation offset at 1 kHz (dB). A-weighting's is 2.0 dB.
_A_OFFSET_dB = 2.0
_C_OFFSET_dB = 0.062


# ---- IEC 61672-1:2013 Table 3: nominal weightings and acceptance limits ----
#
# The high-frequency limits are ASYMMETRIC: the standard allows a weighting filter
# to roll off well below nominal above 8 kHz, but tightly bounds any excess. A
# symmetric tolerance check will report false failures.
#
#   frequency Hz: (A dB, C dB, class1 +, class1 -, class2 +, class2 -)
_INF = float("inf")
IEC_61672_TABLE_3: Dict[float, Tuple[float, float, float, float, float, float]] = {
    10.0:    (-70.4, -14.3,  3.5, _INF,  5.5, _INF),
    12.5:    (-63.4, -11.2,  3.0, _INF,  5.5, _INF),
    16.0:    (-56.7,  -8.5,  2.5, _INF,  5.5, _INF),
    20.0:    (-50.5,  -6.2,  2.5, 2.5,   3.5, _INF),
    25.0:    (-44.7,  -4.4,  2.5, 2.5,   3.5, _INF),
    31.5:    (-39.4,  -3.0,  2.0, 2.0,   3.5, 3.5),
    40.0:    (-34.6,  -2.0,  1.5, 1.5,   2.5, 2.5),
    50.0:    (-30.2,  -1.3,  1.5, 1.5,   2.5, 2.5),
    63.0:    (-26.2,  -0.8,  1.5, 1.5,   2.5, 2.5),
    80.0:    (-22.5,  -0.5,  1.5, 1.5,   2.5, 2.5),
    100.0:   (-19.1,  -0.3,  1.5, 1.5,   2.0, 2.0),
    125.0:   (-16.1,  -0.2,  1.5, 1.5,   2.0, 2.0),
    160.0:   (-13.4,  -0.1,  1.5, 1.5,   2.0, 2.0),
    200.0:   (-10.9,   0.0,  1.5, 1.5,   2.0, 2.0),
    250.0:    (-8.6,   0.0,  1.4, 1.4,   1.9, 1.9),
    315.0:    (-6.6,   0.0,  1.4, 1.4,   1.9, 1.9),
    400.0:    (-4.8,   0.0,  1.4, 1.4,   1.9, 1.9),
    500.0:    (-3.2,   0.0,  1.4, 1.4,   1.9, 1.9),
    630.0:    (-1.9,   0.0,  1.4, 1.4,   1.9, 1.9),
    800.0:    (-0.8,   0.0,  1.4, 1.4,   1.9, 1.9),
    1000.0:    (0.0,   0.0,  1.1, 1.1,   1.4, 1.4),
    1250.0:    (0.6,   0.0,  1.4, 1.4,   1.9, 1.9),
    1600.0:    (1.0,  -0.1,  1.6, 1.6,   2.6, 2.6),
    2000.0:    (1.2,  -0.2,  1.6, 1.6,   2.6, 2.6),
    2500.0:    (1.3,  -0.3,  1.6, 1.6,   3.1, 3.1),
    3150.0:    (1.2,  -0.5,  1.6, 1.6,   3.1, 3.1),
    4000.0:    (1.0,  -0.8,  1.6, 1.6,   3.6, 3.6),
    5000.0:    (0.5,  -1.3,  2.1, 2.1,   4.1, 4.1),
    6300.0:   (-0.1,  -2.0,  2.1, 2.6,   5.1, _INF),
    8000.0:   (-1.1,  -3.0,  2.1, 3.1,   5.6, _INF),
    10000.0:  (-2.5,  -4.4,  2.6, 3.6,   5.6, _INF),
    12500.0:  (-4.3,  -6.2,  3.0, 6.0,   6.0, _INF),
    16000.0:  (-6.6,  -8.5,  3.5, 17.0,  6.0, _INF),
    20000.0:  (-9.3, -11.2,  4.0, _INF,  6.0, _INF),
}


# ---- Filter design ----

def _prewarp(frequency_Hz: float, fs: float, enabled: bool) -> float:
    """
    Pre-warp an analog corner frequency for the bilinear transform.

    The bilinear transform compresses the analog frequency axis onto the digital
    one, so a pole placed at 12194 Hz in the analog prototype lands well below
    12194 Hz digitally. Pre-warping cancels that for each corner:

        w_analog' = 2 * fs * tan(w_digital / (2 * fs))

    At 48 kHz this reduces A-weighting error at 12.5 kHz from -2.62 dB to -0.11 dB.
    """
    w = 2.0 * np.pi * frequency_Hz
    if not enabled:
        return w
    # Guard against corners at or above Nyquist, where tan() blows up.
    if frequency_Hz >= fs / 2.0:
        return w
    return 2.0 * fs * np.tan(w / (2.0 * fs))


def _normalize_at_1k(sos: np.ndarray, fs: float) -> np.ndarray:
    """Scale the cascade so its gain is exactly 0 dB at 1000 Hz."""
    ref = min(1000.0, fs / 2.0 * 0.999)
    _, h = sosfreqz(sos, worN=[ref], fs=fs)
    gain = float(np.abs(np.asarray(h).flat[0]))
    if gain > EPS:
        sos = sos.copy()
        sos[0, :3] /= gain
    return sos


def design_a_weight_sos(fs: float, *, prewarp: bool = True) -> np.ndarray:
    """
    Design the A-weighting digital filter as cascaded second-order sections.

    A-weighting has four zeros at s=0, a double pole at 20.6 Hz, single poles at
    107.7 Hz and 737.9 Hz, and a double pole at 12194 Hz.

    Args:
        fs: Sample rate in Hz.
        prewarp: Pre-warp corner frequencies for the bilinear transform.

    Returns:
        SOS array (N x 6) for scipy.signal.sosfilt.
    """
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive, got {fs}")

    w1 = _prewarp(_F1, fs, prewarp)
    w2 = _prewarp(_F2, fs, prewarp)
    w3 = _prewarp(_F3, fs, prewarp)
    w4 = _prewarp(_F4, fs, prewarp)

    zeros = np.zeros(4, dtype=np.complex128)
    poles = np.array([-w1, -w1, -w2, -w3, -w4, -w4], dtype=np.complex128)

    z_d, p_d, k_d = bilinear_zpk(zeros, poles, 1.0, fs)
    sos = zpk2sos(z_d, p_d, k_d)
    return _normalize_at_1k(sos, fs)


def design_c_weight_sos(fs: float, *, prewarp: bool = True) -> np.ndarray:
    """
    Design the C-weighting digital filter as cascaded second-order sections.

    C-weighting has two zeros at s=0 and double poles at 20.6 Hz and 12194 Hz.
    It is flatter than A-weighting and is the weighting IEC 61672-1 specifies for
    peak sound level (LCpeak), which is the primary impulse-noise quantity.
    """
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive, got {fs}")

    w1 = _prewarp(_F1, fs, prewarp)
    w4 = _prewarp(_F4, fs, prewarp)

    zeros = np.zeros(2, dtype=np.complex128)
    poles = np.array([-w1, -w1, -w4, -w4], dtype=np.complex128)

    z_d, p_d, k_d = bilinear_zpk(zeros, poles, 1.0, fs)
    sos = zpk2sos(z_d, p_d, k_d)
    return _normalize_at_1k(sos, fs)


def design_weighting_sos(fs: float, weighting: WeightingType, *, prewarp: bool = True) -> Optional[np.ndarray]:
    """Design the SOS cascade for a weighting, or None for Z (pass-through)."""
    w = weighting.upper()
    if w == "A":
        return design_a_weight_sos(fs, prewarp=prewarp)
    if w == "C":
        return design_c_weight_sos(fs, prewarp=prewarp)
    if w == "Z":
        return None
    raise ValueError(f"Unknown weighting: {weighting}. Use 'A', 'C', or 'Z'.")


# ---- Stateful filters (streaming / chunked processing) ----

@dataclass
class WeightingFilter:
    """
    Stateful weighting filter for streaming or chunked processing.

    Filter state persists across calls, so a long recording can be processed in
    chunks without discontinuities at the boundaries.
    """
    sample_rate: float
    weighting: WeightingType = "A"
    prewarp: bool = True
    sos: Optional[np.ndarray] = field(default=None, repr=False)
    zi: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.sos = design_weighting_sos(self.sample_rate, self.weighting, prewarp=self.prewarp)
        self.reset()

    def reset(self) -> None:
        """Reset filter state to zero (silence)."""
        self.zi = None if self.sos is None else np.zeros((self.sos.shape[0], 2))

    def apply(self, x: np.ndarray, reset: bool = False) -> np.ndarray:
        """
        Filter a block, carrying state forward to the next call.

        Args:
            x: Input block (1D).
            reset: Reset state before filtering this block.

        Returns:
            Weighted block, same length as input.
        """
        if reset:
            self.reset()

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Input must be 1D array")
        if self.sos is None:
            return x.copy()

        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return np.asarray(y)


# Backwards-compatible aliases
def AWeightFilter(sample_rate: float) -> WeightingFilter:
    """A-weighting streaming filter."""
    return WeightingFilter(sample_rate=sample_rate, weighting="A")


def CWeightFilter(sample_rate: float) -> WeightingFilter:
    """C-weighting streaming filter."""
    return WeightingFilter(sample_rate=sample_rate, weighting="C")


# ---- Functional interface (causal, single pass) ----

def apply_a_weight(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply A-weighting (causal, single pass, zero initial conditions).

    Args:
        x: Input signal (1D mono array).
        fs: Sample rate in Hz.

    Returns:
        A-weighted signal.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")
    return np.asarray(sosfilt(design_a_weight_sos(fs), x))


def apply_c_weight(x: np.ndarray, fs: float) -> np.ndarray:
    """Apply C-weighting (causal, single pass, zero initial conditions)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")
    return np.asarray(sosfilt(design_c_weight_sos(fs), x))


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


def apply_weighting(
    x: np.ndarray,
    fs: float,
    weighting: WeightingType = "A",
) -> np.ndarray:
    """
    Apply frequency weighting to a signal, causally and once.

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
    if w == "C":
        return apply_c_weight(x, fs)
    if w == "Z":
        return apply_z_weight(x, fs)
    raise ValueError(f"Unknown weighting: {weighting}. Use 'A', 'C', or 'Z'.")


def weighting_settling_samples(fs: float, weighting: WeightingType = "A") -> int:
    """
    Samples of warm-up needed before a weighting filter's output is trustworthy.

    The A-weighting cascade's slowest mode is the 20.6 Hz double pole, whose time
    constant is 1/(2*pi*20.6) = 7.7 ms. Five time constants puts the startup
    transient more than 40 dB down.

    Returns:
        Number of warm-up samples (0 for Z-weighting, which has no state).
    """
    if weighting.upper() == "Z":
        return 0
    tau = 1.0 / (2.0 * np.pi * _F1)
    return int(np.ceil(5.0 * tau * fs))


def apply_weighting_with_context(
    signal: np.ndarray,
    fs: float,
    weighting: WeightingType,
    start: int,
    stop: int,
    *,
    context_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Weight an extracted window using surrounding samples as filter warm-up.

    Filtering a short window in isolation starts the filter from silence, so the
    first several milliseconds are a startup transient rather than signal. That
    transient lands exactly where a gunshot's shock front is. This runs the causal
    filter from earlier in the recording and returns only the requested window, so
    the filter is fully settled by the time the window begins - the accuracy that
    motivated forward-backward filtering, without doubling the weighting curve.

    Args:
        signal: The FULL signal the window is drawn from.
        fs: Sample rate in Hz.
        weighting: "A", "C" or "Z".
        start: First sample of the window of interest.
        stop: One past the last sample of the window of interest.
        context_samples: Warm-up length. Defaults to the filter's settling time.

    Returns:
        The weighted window, length (stop - start) clipped to the signal bounds.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    n = len(x)
    start = max(0, min(int(start), n))
    stop = max(start, min(int(stop), n))

    sos = design_weighting_sos(fs, weighting)
    if sos is None:
        return x[start:stop].copy()

    if context_samples is None:
        context_samples = weighting_settling_samples(fs, weighting)

    ctx_start = max(0, start - int(context_samples))
    block = x[ctx_start:stop]
    if block.size == 0:
        return block.astype(np.float64)

    filtered = np.asarray(sosfilt(sos, block))
    return filtered[start - ctx_start:]


# ---- Analytical frequency response ----

def a_weight_frequency_response(frequencies: np.ndarray) -> np.ndarray:
    """
    Compute the theoretical A-weighting curve.

    Args:
        frequencies: Frequency values in Hz.

    Returns:
        A-weighting in dB (0 dB at 1000 Hz).

    Note:
        This is the analytical formula, used for applying A-weighting in the
        frequency domain (e.g. to STFT bins). It agrees with the IEC 61672-1
        tabulated values to better than 0.13 dB across 31.5 Hz - 16 kHz.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    f = np.maximum(f, EPS)
    f2 = f ** 2

    num = _F4**2 * f2**2
    denom = ((f2 + _F1**2) *
             np.sqrt((f2 + _F2**2) * (f2 + _F3**2)) *
             (f2 + _F4**2))

    return 20.0 * np.log10(num / (denom + EPS) + EPS) + _A_OFFSET_dB


def c_weight_frequency_response(frequencies: np.ndarray) -> np.ndarray:
    """
    Compute the theoretical C-weighting curve.

    Args:
        frequencies: Frequency values in Hz.

    Returns:
        C-weighting in dB (0 dB at 1000 Hz).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    f = np.maximum(f, EPS)
    f2 = f ** 2

    num = _F4**2 * f2
    denom = (f2 + _F1**2) * (f2 + _F4**2)

    return 20.0 * np.log10(num / (denom + EPS) + EPS) + _C_OFFSET_dB


def a_weight_linear(frequencies: np.ndarray) -> np.ndarray:
    """A-weighting as linear multipliers for a magnitude spectrum."""
    return 10.0 ** (a_weight_frequency_response(frequencies) / 20.0)


def c_weight_linear(frequencies: np.ndarray) -> np.ndarray:
    """C-weighting as linear multipliers for a magnitude spectrum."""
    return 10.0 ** (c_weight_frequency_response(frequencies) / 20.0)


def get_weighting_curve_dB(
    frequencies: np.ndarray,
    weighting: WeightingType = "A",
) -> np.ndarray:
    """
    Get a frequency weighting curve in dB.

    Args:
        frequencies: Frequency values in Hz.
        weighting: "A", "C", or "Z".

    Returns:
        Weighting in dB at each frequency.
    """
    w = weighting.upper()
    if w == "A":
        return a_weight_frequency_response(frequencies)
    if w == "C":
        return c_weight_frequency_response(frequencies)
    if w == "Z":
        return np.zeros_like(np.asarray(frequencies, dtype=np.float64))
    raise ValueError(f"Unknown weighting: {weighting}")


def weighting_linear(frequencies: np.ndarray, weighting: WeightingType = "A") -> np.ndarray:
    """Weighting as linear multipliers for a magnitude spectrum."""
    return 10.0 ** (get_weighting_curve_dB(frequencies, weighting) / 20.0)


# ---- Standards conformance ----

@dataclass
class ConformancePoint:
    """Conformance of a designed filter at one tabulated frequency."""
    frequency_Hz: float
    nominal_dB: float
    measured_dB: float
    error_dB: float
    limit_plus_dB: float
    limit_minus_dB: float
    passed: bool


@dataclass
class ConformanceReport:
    """Result of checking a designed weighting filter against IEC 61672-1."""
    weighting: str
    sample_rate: float
    sound_class: int
    points: List[ConformancePoint]

    @property
    def passed(self) -> bool:
        return all(p.passed for p in self.points)

    @property
    def max_abs_error_dB(self) -> float:
        return max((abs(p.error_dB) for p in self.points), default=0.0)

    def failures(self) -> List[ConformancePoint]:
        return [p for p in self.points if not p.passed]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"IEC 61672-1 class {self.sound_class} conformance, "
            f"{self.weighting}-weighting @ {self.sample_rate:.0f} Hz: {status} "
            f"(max error {self.max_abs_error_dB:.2f} dB)"
        ]
        for p in self.failures():
            lines.append(
                f"    {p.frequency_Hz:8.0f} Hz: {p.measured_dB:+.2f} dB vs nominal "
                f"{p.nominal_dB:+.2f} (error {p.error_dB:+.2f}, "
                f"limit +{p.limit_plus_dB}/-{p.limit_minus_dB})"
            )
        return "\n".join(lines)


def check_iec_61672_conformance(
    fs: float,
    weighting: WeightingType = "A",
    sound_class: int = 1,
    *,
    prewarp: bool = True,
) -> ConformanceReport:
    """
    Verify a designed weighting filter against IEC 61672-1:2013 Table 3.

    The standard's acceptance limits are asymmetric above 6.3 kHz - a filter may
    roll off far below nominal but must not exceed it. A symmetric tolerance check
    produces false failures, so the two directions are checked separately.

    Frequencies at or above Nyquist are skipped: they cannot be represented at this
    sample rate and are not a property of the filter design.

    Args:
        fs: Sample rate in Hz.
        weighting: "A" or "C" ("Z" is trivially conformant).
        sound_class: 1 or 2.
        prewarp: Whether the filter under test uses bilinear pre-warping.

    Returns:
        ConformanceReport with a per-frequency breakdown.
    """
    w = weighting.upper()
    if w == "Z":
        return ConformanceReport(weighting="Z", sample_rate=fs, sound_class=sound_class, points=[])
    if sound_class not in (1, 2):
        raise ValueError(f"sound_class must be 1 or 2, got {sound_class}")

    sos = design_weighting_sos(fs, w, prewarp=prewarp)
    assert sos is not None

    nominal_idx = 0 if w == "A" else 1
    plus_idx, minus_idx = (2, 3) if sound_class == 1 else (4, 5)

    points: List[ConformancePoint] = []
    for freq in sorted(IEC_61672_TABLE_3):
        if freq >= fs / 2.0:
            continue
        row = IEC_61672_TABLE_3[freq]
        nominal = row[nominal_idx]
        lim_p, lim_m = row[plus_idx], row[minus_idx]

        _, h = sosfreqz(sos, worN=[freq], fs=fs)
        measured = float(20.0 * np.log10(np.abs(np.asarray(h).flat[0]) + EPS))
        error = measured - nominal

        points.append(ConformancePoint(
            frequency_Hz=freq,
            nominal_dB=nominal,
            measured_dB=measured,
            error_dB=error,
            limit_plus_dB=lim_p,
            limit_minus_dB=lim_m,
            passed=(-lim_m <= error <= lim_p),
        ))

    return ConformanceReport(
        weighting=w, sample_rate=fs, sound_class=sound_class, points=points
    )


# ---- CLI for testing ----

def main() -> int:
    """Test and visualize weighting filters."""
    import argparse

    parser = argparse.ArgumentParser(description="Test frequency weighting filters")
    parser.add_argument("--fs", type=float, default=96000, help="Sample rate (Hz)")
    parser.add_argument("--class", dest="sound_class", type=int, default=1, choices=(1, 2),
                        help="IEC 61672-1 sound class to check against")
    parser.add_argument("--plot", action="store_true", help="Plot frequency response")
    args = parser.parse_args()

    fs = args.fs
    print(f"Sample rate: {fs} Hz\n")

    for w in ("A", "C"):
        report = check_iec_61672_conformance(fs, w, args.sound_class)
        print(report.summary())

    print("\nTheoretical weighting at standard frequencies:")
    print(f"{'Freq (Hz)':>10} {'A (dB)':>10} {'C (dB)':>10}")
    for f in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
        if f < fs / 2:
            a = a_weight_frequency_response(np.array([f]))[0]
            c = c_weight_frequency_response(np.array([f]))[0]
            print(f"{f:10.1f} {a:+10.2f} {c:+10.2f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            sos_a = design_a_weight_sos(fs)
            sos_c = design_c_weight_sos(fs)
            w_ax, h_a = sosfreqz(sos_a, worN=4096, fs=fs)
            _, h_c = sosfreqz(sos_c, worN=4096, fs=fs)
            freqs = np.logspace(np.log10(10), np.log10(fs / 2 * 0.99), 500)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            for ax, h, theory, name, ylim in (
                (axes[0], h_a, a_weight_frequency_response(freqs), "A", (-80, 10)),
                (axes[1], h_c, c_weight_frequency_response(freqs), "C", (-30, 5)),
            ):
                ax.semilogx(w_ax, 20 * np.log10(np.abs(h) + EPS), "b-",
                            label="Digital filter", linewidth=2)
                ax.semilogx(freqs, theory, "r--", label="Theoretical", linewidth=1)
                tab_f = [f for f in sorted(IEC_61672_TABLE_3) if f < fs / 2]
                idx = 0 if name == "A" else 1
                ax.plot(tab_f, [IEC_61672_TABLE_3[f][idx] for f in tab_f], "g.",
                        markersize=8, label="IEC 61672-1 Table 3")
                ax.set_xlim(10, fs / 2)
                ax.set_ylim(*ylim)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Gain (dB)")
                ax.set_title(f"{name}-Weighting Filter (fs={fs:.0f} Hz)")
                ax.legend()
                ax.grid(True, which="both", alpha=0.3)

            plt.tight_layout()
            plt.savefig("weighting_response.png", dpi=150)
            print("\nPlot saved to weighting_response.png")

        except ImportError:
            print("\nMatplotlib not available for plotting")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
