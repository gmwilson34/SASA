#!/usr/bin/env python3
"""
metrics.py - Acoustic Metrics for Gunshot and Suppressor Analysis

Computes per-shot and aggregate acoustic metrics following ISO/IEC and MIL-STD
definitions:

  Levels
    - Lpeak (Z/A/C): maximum instantaneous pressure level. LCpeak is the quantity
      IEC 61672-1 and hearing-conservation regulation specify for impulse noise.
    - LZE / LAE / LCE: sound exposure level (SEL), energy normalised to 1 second.
    - LAFmax / LASmax / LAImax: maxima of the Fast, Slow and Impulse detectors.

  Blast waveform (MIL-STD-1474E / free-field blast convention)
    - Rise time: 10-90% of the leading edge of the positive phase.
    - A-duration: duration of the initial positive overpressure phase.
    - B-duration: total time the envelope stays within 20 dB of peak.
    - Specific impulse: integral of positive overpressure (Pa*s).

  Character
    - Crest factor, spectral centroid, kurtosis - all scoped to the blast itself
      rather than to the extraction window, so they do not change when the operator
      changes the window length.

  Hazard
    - LAeq8h, NIOSH dose, and allowable rounds per day.

    Weighting is applied CAUSALLY and ONCE.

    Forward-backward filtering squares the magnitude response, doubling the
    weighting curve in dB. Where a per-shot window is analysed, the surrounding
    recording is used as filter warm-up (see weighting.apply_weighting_with_context)
    so there is no startup transient and no doubled curve.

Usage:
    from metrics import compute_shot_metrics

    m = compute_shot_metrics(pressure_Pa, sample_rate=96000)
    print(f"LCpeak: {m.Lpeak_C:.1f} dB   LAE: {m.LAE:.1f} dB")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert

from calibration import (
    EPS,
    P_REF,
    amplitude_to_dB_SPL,
    compute_peak,
    energy_average_dB,
    power_to_dB_SPL,
    remove_dc_offset,
)
from weighting import (
    apply_weighting,
    apply_weighting_with_context,
    weighting_settling_samples,
)
from bands import ThirdOctaveAnalyzer, exponential_detector, impulse_detector

# Time constants for level computation (IEC 61672-1:2013)
TIME_CONSTANT_FAST = 0.125      # 125 ms
TIME_CONSTANT_SLOW = 1.0        # 1000 ms
TIME_CONSTANT_IMPULSE_ATTACK = 0.035   # 35 ms
TIME_CONSTANT_IMPULSE_DECAY = 1.5      # 1500 ms

# Hearing-conservation constants
EIGHT_HOURS_S: float = 8 * 3600.0       # 28800 s
NIOSH_CRITERION_dB: float = 85.0        # LAeq8h limit, 3 dB exchange rate
MIL_STD_1474E_LIMIT_dB: float = 85.0    # unprotected LAeq8h limit


# ---- Per-shot metrics ----

@dataclass
class ShotMetrics:
    """
    Acoustic metrics for a single gunshot event.

    All levels are in dB re 20 µPa (or dB re FS when the analysis is uncalibrated).
    """
    # Peak levels (instantaneous)
    Lpeak_Z: float       # Peak SPL, Z-weighted (unweighted)
    Lpeak_A: float       # Peak SPL, A-weighted
    Lpeak_C: float       # Peak SPL, C-weighted (the IEC/regulatory impulse quantity)

    # Exposure levels (integrated energy)
    LAE: float           # A-weighted Sound Exposure Level (SEL)
    LZE: float           # Z-weighted Sound Exposure Level
    LCE: float           # C-weighted Sound Exposure Level

    # Maximum time-weighted levels
    LAFmax: float        # Max A-weighted, Fast (125 ms)
    LASmax: float        # Max A-weighted, Slow (1 s)
    LZFmax: float        # Max Z-weighted, Fast
    LZSmax: float        # Max Z-weighted, Slow
    LAImax: float = 0.0  # Max A-weighted, Impulse (35 ms / 1500 ms)
    LZImax: float = 0.0  # Max Z-weighted, Impulse

    # Blast waveform metrics
    rise_time_us: float = 0.0            # 10-90% rise of the positive phase
    rise_time_resolved: bool = True      # False when the sample rate cannot resolve it
    a_duration_ms: float = 0.0           # initial positive overpressure phase
    b_duration_ms: float = 0.0           # envelope within 20 dB of peak
    specific_impulse_Pa_s: float = 0.0   # integral of positive overpressure
    peak_overpressure_Pa: float = 0.0    # signed peak of the positive phase

    # Character
    crest_factor_dB: float = 0.0
    spectral_centroid_Hz: float = 0.0
    kurtosis: float = 0.0

    # Analysis provenance
    duration_s: float = 0.0              # extraction window length
    blast_duration_s: float = 0.0        # span the character metrics were computed over
    integration_window_s: float = 0.0    # span SEL was integrated over
    window_truncated: bool = False       # window hit a file or chunk boundary
    clipped: bool = False                # source samples were at full scale
    noise_floor_dB: float = 0.0
    snr_dB: float = 0.0

    # Time series (optional, for plotting)
    time_s: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    LAF: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    LAS: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    LZF: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    LZS: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # Band analysis
    band_frequencies: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    band_exposure_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # Metadata
    shot_number: int = 0
    valid: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'shot_number': self.shot_number,
            'valid': self.valid,
            'notes': list(self.notes),
            'Lpeak_Z_dB': round(self.Lpeak_Z, 1),
            'Lpeak_A_dB': round(self.Lpeak_A, 1),
            'Lpeak_C_dB': round(self.Lpeak_C, 1),
            'LAE_dB': round(self.LAE, 1),
            'LZE_dB': round(self.LZE, 1),
            'LCE_dB': round(self.LCE, 1),
            'LAFmax_dB': round(self.LAFmax, 1),
            'LASmax_dB': round(self.LASmax, 1),
            'LZFmax_dB': round(self.LZFmax, 1),
            'LZSmax_dB': round(self.LZSmax, 1),
            'LAImax_dB': round(self.LAImax, 1),
            'LZImax_dB': round(self.LZImax, 1),
            'rise_time_us': round(self.rise_time_us, 2),
            'rise_time_resolved': self.rise_time_resolved,
            'a_duration_ms': round(self.a_duration_ms, 3),
            'b_duration_ms': round(self.b_duration_ms, 3),
            'specific_impulse_Pa_s': float(f"{self.specific_impulse_Pa_s:.6g}"),
            'peak_overpressure_Pa': round(self.peak_overpressure_Pa, 3),
            'crest_factor_dB': round(self.crest_factor_dB, 1),
            'spectral_centroid_Hz': round(self.spectral_centroid_Hz, 0),
            'kurtosis': round(self.kurtosis, 1),
            'duration_s': round(self.duration_s, 4),
            'blast_duration_s': round(self.blast_duration_s, 4),
            'integration_window_s': round(self.integration_window_s, 4),
            'window_truncated': self.window_truncated,
            'clipped': self.clipped,
            'noise_floor_dB': round(self.noise_floor_dB, 1),
            'snr_dB': round(self.snr_dB, 1),
            'band_frequencies_Hz': self.band_frequencies.tolist() if self.band_frequencies.size else [],
            'band_exposure_dB': [round(float(x), 1) for x in self.band_exposure_dB] if self.band_exposure_dB.size else [],
        }


# ---- Time weighting ----

def compute_exponential_average(
    x_squared: np.ndarray,
    sample_rate: int,
    time_constant: float,
) -> np.ndarray:
    """
    IEC 61672-1 exponential (RC) time weighting on a squared signal.

    Args:
        x_squared: Squared pressure signal (Pa²).
        sample_rate: Sample rate in Hz.
        time_constant: Time constant in seconds (0.125 Fast, 1.0 Slow).

    Returns:
        Time-weighted mean-square pressure (Pa²).
    """
    return exponential_detector(x_squared, float(sample_rate), time_constant)


def compute_impulse_exponential_average(
    x_squared: np.ndarray,
    sample_rate: int,
    tau_attack: float = TIME_CONSTANT_IMPULSE_ATTACK,
    tau_decay: float = TIME_CONSTANT_IMPULSE_DECAY,
) -> np.ndarray:
    """
    IEC 61672-1 Impulse time weighting: 35 ms average followed by a decay-limited hold.

    Args:
        x_squared: Squared pressure signal (Pa²).
        sample_rate: Sample rate in Hz.
        tau_attack: Attack (averaging) time constant.
        tau_decay: Decay time constant of the hold stage.

    Returns:
        Time-weighted mean-square pressure (Pa²).
    """
    return impulse_detector(x_squared, float(sample_rate), tau_attack, tau_decay)


def compute_time_weighted_levels(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    hop_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Fast and Slow time-weighted levels.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.
        hop_samples: Output decimation factor.

    Returns:
        (time_axis, L_fast, L_slow) where levels are in dB SPL.
    """
    x_sq = np.asarray(pressure_Pa, dtype=np.float64) ** 2
    fast_sq = compute_exponential_average(x_sq, sample_rate, TIME_CONSTANT_FAST)
    slow_sq = compute_exponential_average(x_sq, sample_rate, TIME_CONSTANT_SLOW)

    hop = max(1, int(hop_samples))
    indices = np.arange(0, len(fast_sq), hop)
    return (
        indices / sample_rate,
        np.asarray(power_to_dB_SPL(fast_sq[indices])),
        np.asarray(power_to_dB_SPL(slow_sq[indices])),
    )


# ---- Blast waveform analysis ----

def signal_envelope(x: np.ndarray) -> np.ndarray:
    """
    Analytic-signal (Hilbert) envelope of a waveform.

    Threshold-crossing durations must be measured on the envelope, not on |p|. An
    oscillating waveform's absolute value dips below any threshold twice per cycle,
    so counting samples above a threshold systematically understates the duration -
    by about 25% for a decaying sinusoid at the 20 dB-down point.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return np.abs(x)
    return np.abs(hilbert(x))


def find_blast_span(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    down_dB: float = 20.0,
    guard_ms: float = 1.0,
) -> Tuple[int, int]:
    """
    Locate the blast within an extraction window.

    Character metrics (kurtosis, crest factor, spectral centroid) computed over a
    fixed extraction window measure the window, not the shot: a window that is 90%
    silence yields a kurtosis dominated by the silence-to-impulse ratio, so the same
    shot scores differently when the operator changes pre/post margins.

    The blast span runs from the first to the last point at which the envelope is
    within `down_dB` of the peak, plus a small guard.

    Returns:
        (start_index, stop_index) into pressure_Pa.
    """
    env = signal_envelope(pressure_Pa)
    n = len(env)
    if n == 0:
        return 0, 0

    peak = float(env.max())
    if peak <= EPS:
        return 0, n

    above = np.flatnonzero(env >= peak * (10.0 ** (-abs(down_dB) / 20.0)))
    if above.size == 0:
        return 0, n

    guard = int(guard_ms * sample_rate / 1000.0)
    return max(0, int(above[0]) - guard), min(n, int(above[-1]) + guard + 1)


def compute_rise_time(
    pressure_Pa: np.ndarray,
    sample_rate: int,
) -> Tuple[float, bool]:
    """
    Compute the 10-90% rise time of the blast's leading edge.

    Blast rise time is defined on the LEADING EDGE OF THE POSITIVE PHASE, from
    ambient to peak overpressure. Two things matter:

      - It must be measured on the signed pressure, not |p|. A negative-going global
        extremum is the rarefaction phase, not the shock front.
      - It must be measured on the FIRST rise, not from the global argmax. Searching
        backward from the global peak of an oscillating waveform lands in the middle
        of the ringing and returns a quarter-cycle instead of the shock front.

    Sub-sample linear interpolation is applied at each crossing, because at 48 kHz
    one sample is 20.8 µs while real muzzle-blast rise times are 1-50 µs.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        (rise_time_us, resolved) where resolved is False when the rise spans fewer
        than two samples and the sample rate therefore cannot resolve it.
    """
    p = np.asarray(pressure_Pa, dtype=np.float64)
    if p.size < 3:
        return 0.0, False

    # The shock front is the first excursion that reaches near-peak amplitude.
    env = signal_envelope(p)
    peak_env = float(env.max())
    if peak_env <= EPS:
        return 0.0, False

    onset_candidates = np.flatnonzero(env >= 0.5 * peak_env)
    if onset_candidates.size == 0:
        return 0.0, False
    onset = int(onset_candidates[0])

    # Peak of the positive phase local to that onset
    search_stop = min(p.size, onset + max(4, int(0.005 * sample_rate)))
    local = p[max(0, onset - int(0.001 * sample_rate)):search_stop]
    if local.size == 0:
        return 0.0, False

    base = max(0, onset - int(0.001 * sample_rate))
    peak_idx = base + int(np.argmax(local))
    peak_val = float(p[peak_idx])
    if peak_val <= EPS:
        # Rarefaction-first waveform: measure the negative excursion instead
        peak_idx = base + int(np.argmin(local))
        peak_val = float(p[peak_idx])
        if abs(peak_val) <= EPS:
            return 0.0, False
        p = -p
        peak_val = -peak_val

    t10, t90 = 0.1 * peak_val, 0.9 * peak_val

    def _crossing(threshold: float) -> Optional[float]:
        """Last index before peak_idx where the signal crosses threshold upward."""
        for i in range(peak_idx, 0, -1):
            if p[i] >= threshold > p[i - 1]:
                denom = p[i] - p[i - 1]
                frac = (threshold - p[i - 1]) / denom if abs(denom) > EPS else 0.0
                return (i - 1) + frac
        return None

    x90 = _crossing(t90)
    x10 = _crossing(t10)
    if x90 is None or x10 is None or x90 <= x10:
        return 0.0, False

    samples = x90 - x10
    return float(samples / sample_rate * 1e6), bool(samples >= 2.0)


def compute_a_duration(pressure_Pa: np.ndarray, sample_rate: int) -> Tuple[float, float, float]:
    """
    Compute A-duration, specific impulse and peak overpressure of the positive phase.

    A-duration is the length of the initial positive overpressure phase: from the
    onset of the shock front to the first return to ambient pressure. It, and the
    impulse carried by that phase, are the primary free-field blast descriptors and
    are what distinguishes suppressed from unsuppressed muzzle blast.

    Returns:
        (a_duration_ms, specific_impulse_Pa_s, peak_overpressure_Pa)
    """
    p = np.asarray(pressure_Pa, dtype=np.float64)
    if p.size < 3:
        return 0.0, 0.0, 0.0

    env = signal_envelope(p)
    peak_env = float(env.max())
    if peak_env <= EPS:
        return 0.0, 0.0, 0.0

    cand = np.flatnonzero(env >= 0.5 * peak_env)
    if cand.size == 0:
        return 0.0, 0.0, 0.0
    onset = int(cand[0])

    # The phase boundaries are found by anchoring on the PEAK of the excursion, not
    # on its onset. The analytic-signal envelope leads the waveform, so the sample at
    # the envelope onset is usually still pre-blast baseline; anchoring there makes
    # the result depend on the sign of the last noise sample before the shock front,
    # and a single negative noise sample then terminates the "positive phase"
    # immediately. On a 3 ms Friedlander wave with 0.01 Pa of noise that returned an
    # A-duration of 0.021 ms and a specific impulse of zero.
    #
    # From the peak, the surrounding zero crossings are unambiguous: noise near the
    # baseline cannot truncate a phase that is entered from its own maximum.
    search_stop = min(p.size, onset + max(4, int(0.005 * sample_rate)))
    local = p[onset:search_stop]
    if local.size == 0:
        return 0.0, 0.0, 0.0

    sign = 1.0 if abs(float(local.max())) >= abs(float(local.min())) else -1.0
    signed = p * sign
    peak_idx = onset + int(np.argmax(signed[onset:search_stop]))
    if signed[peak_idx] <= EPS:
        return 0.0, 0.0, 0.0

    # Back to the zero crossing that opens the phase
    start = peak_idx
    while start > 0 and signed[start - 1] > 0:
        start -= 1

    # Forward to the first genuine return to ambient
    stop = peak_idx
    while stop < p.size - 1 and signed[stop] > 0:
        stop += 1

    if stop <= start:
        return 0.0, 0.0, 0.0

    seg = signed[start:stop]
    dt = 1.0 / sample_rate
    impulse = float(np.sum(np.maximum(seg, 0.0)) * dt)
    peak_over = float(sign * np.max(seg))

    return float((stop - start) * dt * 1000.0), impulse, peak_over


def compute_b_duration(pressure_Pa: np.ndarray, sample_rate: int, down_dB: float = 20.0) -> float:
    """
    Compute B-duration: total time the ENVELOPE stays within 20 dB of the peak.

    Measured on the analytic-signal envelope. Counting individual samples whose |p|
    exceeds the threshold understates the duration by roughly 25% for a decaying
    sinusoid, because |p| dips below the threshold twice per cycle.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.
        down_dB: Level below peak defining the duration (20 dB by convention).

    Returns:
        B-duration in milliseconds.
    """
    env = signal_envelope(pressure_Pa)
    if env.size == 0:
        return 0.0
    peak = float(env.max())
    if peak <= EPS:
        return 0.0

    above = env >= peak * (10.0 ** (-abs(down_dB) / 20.0))
    return float(np.count_nonzero(above) / sample_rate * 1000.0)


def compute_crest_factor(pressure_Pa: np.ndarray) -> float:
    """
    Compute crest factor (peak-to-RMS ratio) in dB.

    A pure sine has a crest factor of 3.01 dB; gunshots typically run 15-30 dB.
    Compute this over the blast span, not the extraction window, or the value scales
    with however much silence the window happens to contain.
    """
    p = np.asarray(pressure_Pa, dtype=np.float64)
    if p.size == 0:
        return 0.0
    peak = float(np.max(np.abs(p)))
    rms = float(np.sqrt(np.mean(p ** 2)))
    if rms < EPS or peak < EPS:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def compute_spectral_centroid(pressure_Pa: np.ndarray, sample_rate: int) -> float:
    """
    Compute the spectral centroid (frequency centre of mass) of the blast.

    Suppressors shift energy downward in frequency, so the centroid is a useful
    single-number summary - provided it is computed over the blast rather than over
    a window whose silent portion is dominated by the noise floor.

    Args:
        pressure_Pa: Pressure waveform in Pascals (blast span).
        sample_rate: Sample rate in Hz.

    Returns:
        Spectral centroid in Hz.
    """
    p = np.asarray(pressure_Pa, dtype=np.float64)
    n = p.size
    if n < 4:
        return 0.0

    X = np.abs(np.fft.rfft(p * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    power = X ** 2

    # Ignore sub-20 Hz content: it is wind/handling, not blast, and it drags the
    # centroid down without carrying acoustic information.
    mask = freqs >= 20.0
    power, freqs = power[mask], freqs[mask]

    total = float(np.sum(power))
    if total < EPS:
        return 0.0
    return float(np.sum(freqs * power) / total)


def compute_kurtosis(pressure_Pa: np.ndarray) -> float:
    """
    Compute excess kurtosis (Fisher definition; Gaussian = 0).

    Kurtosis is MIL-STD-1474E's impulsiveness indicator. It is strongly dependent on
    the analysis window, so it must be computed over the blast span to be comparable
    between shots and between test sessions.
    """
    x = np.asarray(pressure_Pa, dtype=np.float64)
    if x.size < 4:
        return 0.0
    centered = x - np.mean(x)
    m2 = float(np.mean(centered ** 2))
    m4 = float(np.mean(centered ** 4))
    if m2 < EPS:
        return 0.0
    return float(m4 / (m2 ** 2) - 3.0)


def compute_exposure_level(
    pressure_Pa: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Compute Sound Exposure Level (SEL / LE).

        SEL = 10 * log10( integral p²(t) dt / (p_ref² * T_ref) ),  T_ref = 1 s

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        Sound Exposure Level in dB.
    """
    p = np.asarray(pressure_Pa, dtype=np.float64)
    if p.size == 0:
        return float("-inf")
    energy = float(np.sum(p ** 2)) / sample_rate
    return float(10.0 * np.log10(max(energy, EPS) / (P_REF ** 2 * 1.0)))


def find_integration_span(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    capture_fraction: float = 0.99,
) -> Tuple[int, int]:
    """
    Find the span carrying `capture_fraction` of the event's energy.

    SEL integrated over a fixed extraction window changes whenever the operator
    changes the window length or the noise floor moves, which makes suppressed-vs-
    unsuppressed comparison meaningless. Integrating over the span that actually
    holds the energy makes SEL a property of the shot instead.

    Returns:
        (start_index, stop_index)
    """
    p = np.asarray(pressure_Pa, dtype=np.float64)
    n = p.size
    if n == 0:
        return 0, 0

    cumulative = np.cumsum(p ** 2)
    total = float(cumulative[-1])
    if total <= EPS:
        return 0, n

    normalized = cumulative / total
    margin = (1.0 - capture_fraction) / 2.0
    start = int(np.searchsorted(normalized, margin))
    stop = int(np.searchsorted(normalized, 1.0 - margin)) + 1
    return max(0, start), min(n, max(stop, start + 1))


# ---- Main per-shot computation ----

def compute_shot_metrics(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    compute_bands: bool = True,
    compute_time_series: bool = True,
    shot_number: int = 0,
    full_signal: Optional[np.ndarray] = None,
    window_start: Optional[int] = None,
    window_truncated: bool = False,
    clipped: bool = False,
    high_pass: bool = True,
) -> ShotMetrics:
    """
    Compute comprehensive acoustic metrics for a gunshot event.

    Args:
        pressure_Pa: Calibrated pressure waveform for the shot window, in Pascals.
        sample_rate: Sample rate in Hz.
        compute_bands: Whether to compute 1/3-octave band exposure.
        compute_time_series: Whether to retain level time series for plotting.
        shot_number: Shot identifier for metadata.
        full_signal: The complete recording. When supplied together with
                     window_start, weighting filters are warmed up on the samples
                     preceding the window so there is no startup transient.
        window_start: Index of pressure_Pa[0] within full_signal.
        window_truncated: True when the window hit a file or chunk boundary, so
                          energy metrics under-report and must be flagged.
        clipped: True when the source samples were at digital full scale.
        high_pass: Remove DC and sub-10 Hz content before computing levels.

    Returns:
        ShotMetrics with all computed metrics.
    """
    x = np.asarray(pressure_Pa, dtype=np.float64)
    n = x.size
    duration_s = n / sample_rate
    notes: List[str] = []

    if n == 0:
        return ShotMetrics(
            Lpeak_Z=float("-inf"), Lpeak_A=float("-inf"), Lpeak_C=float("-inf"),
            LAE=float("-inf"), LZE=float("-inf"), LCE=float("-inf"),
            LAFmax=float("-inf"), LASmax=float("-inf"),
            LZFmax=float("-inf"), LZSmax=float("-inf"),
            shot_number=shot_number, valid=False, notes=["Empty shot window"],
        )

    if high_pass:
        x = remove_dc_offset(x, sample_rate, cutoff_Hz=10.0)

    # ---- Frequency weighting: causal, single pass, warmed up on real context ----
    can_warm = (
        full_signal is not None
        and window_start is not None
        and len(full_signal) >= window_start + n
    )
    if can_warm:
        src = np.asarray(full_signal, dtype=np.float64)
        if high_pass:
            # Warm-up context must receive the same conditioning as the window
            ctx = weighting_settling_samples(sample_rate, "A")
            lo = max(0, int(window_start) - ctx)
            src = remove_dc_offset(src[lo:int(window_start) + n], sample_rate, cutoff_Hz=10.0)
            rel_start = int(window_start) - lo
        else:
            rel_start = int(window_start)
        x_a = apply_weighting_with_context(src, sample_rate, "A", rel_start, rel_start + n)
        x_c = apply_weighting_with_context(src, sample_rate, "C", rel_start, rel_start + n)
    else:
        x_a = apply_weighting(x, sample_rate, "A")
        x_c = apply_weighting(x, sample_rate, "C")
        notes.append("Weighting filters started cold (no pre-shot context available)")
    x_z = x

    # ---- Peak levels ----
    Lpeak_Z = float(amplitude_to_dB_SPL(compute_peak(x_z)))
    Lpeak_A = float(amplitude_to_dB_SPL(compute_peak(x_a)))
    Lpeak_C = float(amplitude_to_dB_SPL(compute_peak(x_c)))

    # ---- Exposure levels over the span that carries the energy ----
    i_start, i_stop = find_integration_span(x_z, sample_rate)
    integration_window_s = (i_stop - i_start) / sample_rate
    LZE = compute_exposure_level(x_z[i_start:i_stop], sample_rate)
    LAE = compute_exposure_level(x_a[i_start:i_stop], sample_rate)
    LCE = compute_exposure_level(x_c[i_start:i_stop], sample_rate)

    # ---- Time-weighted levels ----
    hop = max(1, sample_rate // 1000)  # ~1 ms resolution
    time_a, LAF, LAS = compute_time_weighted_levels(x_a, sample_rate, hop)
    _, LZF, LZS = compute_time_weighted_levels(x_z, sample_rate, hop)

    LAFmax = float(np.max(LAF)) if LAF.size else float("-inf")
    LASmax = float(np.max(LAS)) if LAS.size else float("-inf")
    LZFmax = float(np.max(LZF)) if LZF.size else float("-inf")
    LZSmax = float(np.max(LZS)) if LZS.size else float("-inf")

    impulse_a = compute_impulse_exponential_average(x_a ** 2, sample_rate)
    impulse_z = compute_impulse_exponential_average(x_z ** 2, sample_rate)
    LAImax = float(power_to_dB_SPL(np.max(impulse_a))) if impulse_a.size else float("-inf")
    LZImax = float(power_to_dB_SPL(np.max(impulse_z))) if impulse_z.size else float("-inf")

    # ---- Blast waveform metrics (on unweighted pressure) ----
    rise_time_us, rise_resolved = compute_rise_time(x_z, sample_rate)
    a_duration_ms, specific_impulse, peak_overpressure = compute_a_duration(x_z, sample_rate)
    b_duration_ms = compute_b_duration(x_z, sample_rate)

    if not rise_resolved and rise_time_us > 0:
        notes.append(
            f"Rise time ({rise_time_us:.1f} us) spans under two samples at "
            f"{sample_rate} Hz; treat as an upper bound"
        )

    # ---- Character metrics, scoped to the blast ----
    b_start, b_stop = find_blast_span(x_z, sample_rate)
    blast = x_z[b_start:b_stop]
    blast_duration_s = blast.size / sample_rate
    crest_factor_dB = compute_crest_factor(blast)
    spectral_centroid_Hz = compute_spectral_centroid(blast, sample_rate)
    kurtosis_val = compute_kurtosis(blast)

    # ---- Noise floor / SNR within this window ----
    noise_floor_dB, snr_dB = _window_noise_floor(x_z, sample_rate, b_start, b_stop)

    # ---- Band analysis ----
    band_frequencies = np.array([])
    band_exposure_dB = np.array([])
    if compute_bands:
        try:
            analyzer = ThirdOctaveAnalyzer(sample_rate=sample_rate)
            band_frequencies = analyzer.center_frequencies
            band_exposure_dB = analyzer.compute_band_exposure(x_z[i_start:i_stop])
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            notes.append(f"Band analysis failed: {exc}")

    if not compute_time_series:
        time_a = np.array([])
        LAF = LAS = LZF = LZS = np.array([])

    if clipped:
        notes.append(
            "Source samples were clipped: peak, rise time, crest factor and kurtosis "
            "are invalid and levels are understated"
        )
    if window_truncated:
        notes.append(
            "Extraction window was truncated at a boundary; SEL under-reports the event"
        )

    return ShotMetrics(
        Lpeak_Z=Lpeak_Z, Lpeak_A=Lpeak_A, Lpeak_C=Lpeak_C,
        LAE=LAE, LZE=LZE, LCE=LCE,
        LAFmax=LAFmax, LASmax=LASmax, LZFmax=LZFmax, LZSmax=LZSmax,
        LAImax=LAImax, LZImax=LZImax,
        rise_time_us=rise_time_us,
        rise_time_resolved=rise_resolved,
        a_duration_ms=a_duration_ms,
        b_duration_ms=b_duration_ms,
        specific_impulse_Pa_s=specific_impulse,
        peak_overpressure_Pa=peak_overpressure,
        crest_factor_dB=crest_factor_dB,
        spectral_centroid_Hz=spectral_centroid_Hz,
        kurtosis=kurtosis_val,
        duration_s=duration_s,
        blast_duration_s=blast_duration_s,
        integration_window_s=integration_window_s,
        window_truncated=window_truncated,
        clipped=clipped,
        noise_floor_dB=noise_floor_dB,
        snr_dB=snr_dB,
        time_s=time_a, LAF=LAF, LAS=LAS, LZF=LZF, LZS=LZS,
        band_frequencies=band_frequencies,
        band_exposure_dB=band_exposure_dB,
        shot_number=shot_number,
        valid=not clipped,
        notes=notes,
    )


def _window_noise_floor(
    x: np.ndarray,
    sample_rate: int,
    blast_start: int,
    blast_stop: int,
) -> Tuple[float, float]:
    """Estimate the noise floor from the parts of the window outside the blast."""
    quiet = np.concatenate([x[:blast_start], x[blast_stop:]])
    peak_dB = float(amplitude_to_dB_SPL(np.max(np.abs(x)))) if x.size else float("-inf")
    if quiet.size < sample_rate // 1000:
        return float("-inf"), float("inf")
    floor_dB = float(power_to_dB_SPL(max(float(np.mean(quiet ** 2)), EPS)))
    return floor_dB, peak_dB - floor_dB


# ---- Hazard assessment ----

@dataclass
class HazardAssessment:
    """
    Hearing-hazard assessment for a string of shots.

    Uses the energy-based (3 dB exchange rate) criterion shared by MIL-STD-1474E's
    LAeq8h limit and the NIOSH recommended exposure limit. This is the widely
    accepted screening method; a full MIL-STD-1474E determination for impulse noise
    uses AHAAH, which models the middle-ear reflex and is outside this scope.
    """
    n_rounds: int
    LAE_mean: float
    LAeq8h_dB: float
    criterion_dB: float
    dose_percent: float
    allowable_rounds: float
    protection_NRR_dB: float = 0.0
    exceeds_limit: bool = False

    def to_dict(self) -> Dict:
        return {
            'n_rounds': self.n_rounds,
            'LAE_mean_dB': round(self.LAE_mean, 1),
            'LAeq8h_dB': round(self.LAeq8h_dB, 1),
            'criterion_dB': self.criterion_dB,
            'dose_percent': round(self.dose_percent, 1),
            'allowable_rounds': round(self.allowable_rounds, 1),
            'protection_NRR_dB': self.protection_NRR_dB,
            'exceeds_limit': self.exceeds_limit,
            'method': 'Energy-based LAeq8h, 3 dB exchange rate (MIL-STD-1474E / NIOSH)',
        }


def compute_hazard(
    LAE_values: Sequence[float],
    *,
    criterion_dB: float = NIOSH_CRITERION_dB,
    protection_NRR_dB: float = 0.0,
) -> HazardAssessment:
    """
    Assess daily hearing hazard from a set of per-shot A-weighted exposure levels.

        LAeq8h = LAE_energy_mean + 10*log10(N) - 10*log10(28800)
        allowable N = 10^((criterion - LAE + 10*log10(28800)) / 10)

    Args:
        LAE_values: Per-shot A-weighted SEL, in dB.
        criterion_dB: 8-hour equivalent limit (85 dB for NIOSH / MIL-STD-1474E).
        protection_NRR_dB: Noise reduction rating of hearing protection, if worn.

    Returns:
        HazardAssessment including the allowable number of rounds per day.
    """
    finite = [float(v) for v in LAE_values if np.isfinite(v)]
    n = len(finite)
    if n == 0:
        return HazardAssessment(
            n_rounds=0, LAE_mean=float("nan"), LAeq8h_dB=float("nan"),
            criterion_dB=criterion_dB, dose_percent=0.0, allowable_rounds=float("inf"),
            protection_NRR_dB=protection_NRR_dB,
        )

    lae = energy_average_dB(finite) - float(protection_NRR_dB)
    laeq8h = lae + 10.0 * np.log10(n) - 10.0 * np.log10(EIGHT_HOURS_S)
    dose = 100.0 * (10.0 ** ((laeq8h - criterion_dB) / 10.0))
    allowable = 10.0 ** ((criterion_dB - lae + 10.0 * np.log10(EIGHT_HOURS_S)) / 10.0)

    return HazardAssessment(
        n_rounds=n,
        LAE_mean=lae,
        LAeq8h_dB=float(laeq8h),
        criterion_dB=criterion_dB,
        dose_percent=float(dose),
        allowable_rounds=float(allowable),
        protection_NRR_dB=float(protection_NRR_dB),
        exceeds_limit=bool(laeq8h > criterion_dB),
    )


# ---- Aggregate statistics ----

@dataclass
class MetricStats:
    """Distribution summary for one metric across a shot string."""
    name: str
    unit: str
    n: int
    mean: float           # energy mean for levels, arithmetic otherwise
    std: float
    minimum: float
    maximum: float
    median: float
    ci95_half_width: float

    def to_dict(self) -> Dict:
        return {
            'name': self.name, 'unit': self.unit, 'n': self.n,
            'mean': round(self.mean, 2), 'std': round(self.std, 2),
            'min': round(self.minimum, 2), 'max': round(self.maximum, 2),
            'median': round(self.median, 2),
            'ci95_half_width': round(self.ci95_half_width, 2),
        }


def _summarize(values: Sequence[float], name: str, unit: str, *, is_level: bool) -> MetricStats:
    """
    Summarise a metric across shots.

    Levels are averaged on an energy basis (ISO convention); dispersion is always
    reported on the decibel values, since that is what shot-to-shot variability
    means to a user. The sample standard deviation (ddof=1) is used because the
    shots are a sample of the weapon's behaviour, not the entire population.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = arr.size
    if n == 0:
        nan = float("nan")
        return MetricStats(name, unit, 0, nan, nan, nan, nan, nan, nan)

    mean = energy_average_dB(arr) if is_level else float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0

    return MetricStats(
        name=name, unit=unit, n=n, mean=mean, std=std,
        minimum=float(np.min(arr)), maximum=float(np.max(arr)),
        median=float(np.median(arr)), ci95_half_width=ci,
    )


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple shots."""
    n_shots: int
    n_valid: int
    stats: Dict[str, MetricStats] = field(default_factory=dict)
    hazard: Optional[HazardAssessment] = None
    band_frequencies: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    band_exposure_mean_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    shot_metrics: List[ShotMetrics] = field(default_factory=list, repr=False)

    # Convenience accessors retained for existing callers
    @property
    def Lpeak_Z_max(self) -> float:
        s = self.stats.get('Lpeak_Z')
        return s.maximum if s else float("nan")

    @property
    def Lpeak_A_max(self) -> float:
        s = self.stats.get('Lpeak_A')
        return s.maximum if s else float("nan")

    @property
    def LAE_mean(self) -> float:
        s = self.stats.get('LAE')
        return s.mean if s else float("nan")

    @property
    def LAFmax_mean(self) -> float:
        s = self.stats.get('LAFmax')
        return s.mean if s else float("nan")

    def to_dict(self) -> Dict:
        return {
            'n_shots': self.n_shots,
            'n_valid': self.n_valid,
            'statistics': {k: v.to_dict() for k, v in self.stats.items()},
            'hazard': self.hazard.to_dict() if self.hazard else None,
            'band_frequencies_Hz': self.band_frequencies.tolist() if self.band_frequencies.size else [],
            'band_exposure_mean_dB': [round(float(x), 1) for x in self.band_exposure_mean_dB]
                                      if self.band_exposure_mean_dB.size else [],
            'shots': [m.to_dict() for m in self.shot_metrics],
        }


# Which metrics are levels (energy-averaged) vs plain quantities (arithmetic mean)
_AGGREGATE_SPEC: List[Tuple[str, str, str, bool]] = [
    ('Lpeak_Z', 'Lpeak_Z', 'dB', True),
    ('Lpeak_A', 'Lpeak_A', 'dB', True),
    ('Lpeak_C', 'Lpeak_C', 'dB', True),
    ('LAE', 'LAE', 'dB', True),
    ('LZE', 'LZE', 'dB', True),
    ('LCE', 'LCE', 'dB', True),
    ('LAFmax', 'LAFmax', 'dB', True),
    ('LASmax', 'LASmax', 'dB', True),
    ('LAImax', 'LAImax', 'dB', True),
    ('LZImax', 'LZImax', 'dB', True),
    ('rise_time_us', 'rise_time_us', 'us', False),
    ('a_duration_ms', 'a_duration_ms', 'ms', False),
    ('b_duration_ms', 'b_duration_ms', 'ms', False),
    ('specific_impulse_Pa_s', 'specific_impulse_Pa_s', 'Pa*s', False),
    ('crest_factor_dB', 'crest_factor_dB', 'dB', False),
    ('spectral_centroid_Hz', 'spectral_centroid_Hz', 'Hz', False),
    ('kurtosis', 'kurtosis', '', False),
]


def compute_aggregate_metrics(
    shot_metrics_list: List[ShotMetrics],
    *,
    include_invalid: bool = False,
    protection_NRR_dB: float = 0.0,
) -> AggregateMetrics:
    """
    Compute aggregate statistics across multiple shots.

    Args:
        shot_metrics_list: Per-shot metrics.
        include_invalid: Include shots flagged invalid (e.g. clipped). Off by default
                         so that a saturated shot cannot drag the reported mean.
        protection_NRR_dB: Hearing protection rating for the hazard assessment.

    Returns:
        AggregateMetrics with per-metric distributions and a hazard assessment.
    """
    all_shots = list(shot_metrics_list)
    used = all_shots if include_invalid else [m for m in all_shots if m.valid]

    if not used:
        return AggregateMetrics(n_shots=len(all_shots), n_valid=0, shot_metrics=all_shots)

    stats: Dict[str, MetricStats] = {}
    for key, attr, unit, is_level in _AGGREGATE_SPEC:
        stats[key] = _summarize([getattr(m, attr) for m in used], key, unit, is_level=is_level)

    hazard = compute_hazard([m.LAE for m in used], protection_NRR_dB=protection_NRR_dB)

    # Mean band exposure across shots, energy-averaged per band
    band_freqs = np.array([])
    band_mean = np.array([])
    band_sets = [m.band_exposure_dB for m in used if m.band_exposure_dB.size]
    if band_sets and all(b.shape == band_sets[0].shape for b in band_sets):
        band_freqs = next(m.band_frequencies for m in used if m.band_exposure_dB.size)
        stacked = np.vstack(band_sets)
        band_mean = 10.0 * np.log10(np.mean(10.0 ** (stacked / 10.0), axis=0))

    return AggregateMetrics(
        n_shots=len(all_shots),
        n_valid=len(used),
        stats=stats,
        hazard=hazard,
        band_frequencies=band_freqs,
        band_exposure_mean_dB=band_mean,
        shot_metrics=all_shots,
    )


# ---- Suppressor comparison ----

@dataclass
class InsertionLoss:
    """
    Net suppression of a test configuration against an unsuppressed reference.

    This is the number a suppressor test exists to produce. Reporting it requires
    both recordings to share a calibration, a microphone position and a sample rate;
    those preconditions are checked rather than assumed.
    """
    metric: str
    reference_dB: float
    test_dB: float
    reduction_dB: float
    reference_n: int
    test_n: int
    reference_ci95: float = 0.0
    test_ci95: float = 0.0

    @property
    def combined_ci95(self) -> float:
        """95% confidence half-width on the reduction, combining both samples."""
        return float(np.sqrt(self.reference_ci95 ** 2 + self.test_ci95 ** 2))

    def to_dict(self) -> Dict:
        return {
            'metric': self.metric,
            'reference_dB': round(self.reference_dB, 1),
            'test_dB': round(self.test_dB, 1),
            'reduction_dB': round(self.reduction_dB, 1),
            'ci95_dB': round(self.combined_ci95, 2),
            'reference_n': self.reference_n,
            'test_n': self.test_n,
        }


def compute_insertion_loss(
    reference: AggregateMetrics,
    test: AggregateMetrics,
    metrics: Sequence[str] = ("Lpeak_Z", "Lpeak_A", "Lpeak_C", "LAE", "LZE", "LAImax"),
) -> List[InsertionLoss]:
    """
    Compute net suppression (insertion loss) for each metric.

        reduction = L_reference - L_test

    Positive values mean the test configuration is quieter than the reference.

    Args:
        reference: Aggregate metrics for the UNSUPPRESSED reference string.
        test: Aggregate metrics for the suppressed string.
        metrics: Which metrics to compare.

    Returns:
        One InsertionLoss per requested metric that exists in both.
    """
    out: List[InsertionLoss] = []
    for name in metrics:
        r, t = reference.stats.get(name), test.stats.get(name)
        if r is None or t is None or not (np.isfinite(r.mean) and np.isfinite(t.mean)):
            continue
        out.append(InsertionLoss(
            metric=name,
            reference_dB=r.mean, test_dB=t.mean,
            reduction_dB=r.mean - t.mean,
            reference_n=r.n, test_n=t.n,
            reference_ci95=r.ci95_half_width, test_ci95=t.ci95_half_width,
        ))
    return out


def format_metrics_summary(metrics: ShotMetrics, prefix: str = "", unit: str = "dB") -> str:
    """
    Format metrics as a human-readable summary string.

    Args:
        metrics: ShotMetrics object.
        prefix: Optional prefix for each line.
        unit: Level unit label ("dB SPL" when calibrated, "dB re FS" otherwise).

    Returns:
        Formatted string.
    """
    lines = [
        f"{prefix}Shot {metrics.shot_number} Metrics:",
        f"{prefix}  Peak (Z):            {metrics.Lpeak_Z:.1f} {unit}",
        f"{prefix}  Peak (A):            {metrics.Lpeak_A:.1f} {unit}",
        f"{prefix}  Peak (C) [LCpeak]:   {metrics.Lpeak_C:.1f} {unit}",
        f"{prefix}  LAE (A-weighted SEL):{metrics.LAE:.1f} {unit}",
        f"{prefix}  LAFmax:              {metrics.LAFmax:.1f} {unit}",
        f"{prefix}  LAImax:              {metrics.LAImax:.1f} {unit}",
        f"{prefix}  Rise time:           {metrics.rise_time_us:.1f} us"
        + ("" if metrics.rise_time_resolved else "  (UNRESOLVED at this sample rate)"),
        f"{prefix}  A-duration:          {metrics.a_duration_ms:.3f} ms",
        f"{prefix}  B-duration:          {metrics.b_duration_ms:.2f} ms",
        f"{prefix}  Specific impulse:    {metrics.specific_impulse_Pa_s:.4g} Pa*s",
        f"{prefix}  Crest factor:        {metrics.crest_factor_dB:.1f} dB",
        f"{prefix}  Spectral centroid:   {metrics.spectral_centroid_Hz:.0f} Hz",
        f"{prefix}  Kurtosis:            {metrics.kurtosis:.1f}",
        f"{prefix}  SNR:                 {metrics.snr_dB:.1f} dB",
    ]
    for note in metrics.notes:
        lines.append(f"{prefix}  ! {note}")
    return "\n".join(lines)


# ---- CLI for testing ----

def main() -> int:
    """Test metrics computation."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute acoustic metrics for audio")
    parser.add_argument("wav", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--Pa-per-FS", type=float, default=1.0, help="Calibration factor")
    parser.add_argument("--start", type=float, default=None, help="Start time (s)")
    parser.add_argument("--duration", type=float, default=None, help="Duration (s)")
    parser.add_argument("--test-impulse", action="store_true", help="Use a synthetic blast")
    args = parser.parse_args()

    if args.test_impulse or args.wav is None:
        sr = 96000
        t = np.arange(int(sr * 0.25)) / sr
        # Friedlander wave: the canonical free-field blast model
        T = 0.001
        pressure_Pa = 200.0 * (1 - t / T) * np.exp(-t / T)
        print(f"Synthetic Friedlander blast: 200 Pa peak, T={T*1000:.1f} ms, {sr} Hz")
    else:
        import soundfile as sf
        data, sr = sf.read(str(args.wav), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        pressure_Pa = data * args.Pa_per_FS
        if args.start is not None or args.duration is not None:
            i0 = int((args.start or 0) * sr)
            i1 = i0 + int((args.duration or (len(data) / sr)) * sr)
            pressure_Pa = pressure_Pa[i0:i1]
        print(f"Loaded: {args.wav}  ({sr} Hz)")

    m = compute_shot_metrics(pressure_Pa, sr, shot_number=1)
    print("\n" + format_metrics_summary(m))

    if m.band_frequencies.size:
        print("\n  1/3-Octave Band Exposure (SEL):")
        print(f"  {'Freq (Hz)':>10} {'SEL (dB)':>10}")
        print("  " + "-" * 22)
        for f, sel in zip(m.band_frequencies, m.band_exposure_dB):
            print(f"  {f:10.0f} {sel:10.1f}")

    hazard = compute_hazard([m.LAE] * 50)
    print(f"\n  Hazard for 50 rounds: LAeq8h = {hazard.LAeq8h_dB:.1f} dB, "
          f"dose {hazard.dose_percent:.0f}%, allowable {hazard.allowable_rounds:.0f} rounds/day")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
