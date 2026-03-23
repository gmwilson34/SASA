#!/usr/bin/env python3
"""
metrics.py - Acoustic Metrics for Gunshot Analysis

Computes per-shot and aggregate acoustic metrics following ISO/IEC standards:
  - Peak SPL (Lpeak, Z-weighted): Maximum instantaneous pressure level
  - LAE (Sound Exposure Level, A-weighted): Total A-weighted energy normalized to 1s
  - LAFmax (Maximum Fast A-weighted): Peak of 125ms exponential average
  - LASmax (Maximum Slow A-weighted): Peak of 1s exponential average
  - Per-band exposure: SEL for each 1/3-octave band
  - Loudness: Placeholder for perceptual loudness models (sone)

Standard metrics definitions:
  - Lpeak = 20 * log10(p_peak / p_ref)  where p_ref = 20 µPa
  - LAE = 10 * log10(∫pA²(t)dt / (p_ref² * T_ref))  where T_ref = 1 s
  - LAFmax = max(LAF(t))  where LAF is A-weighted with Fast (125ms) time constant
  - LASmax = max(LAS(t))  where LAS is A-weighted with Slow (1s) time constant

Usage:
    from metrics import compute_shot_metrics, ShotMetrics

    metrics = compute_shot_metrics(
        pressure_Pa,
        sample_rate=96000,
    )

    print(f"Peak SPL: {metrics.Lpeak_Z:.1f} dB")
    print(f"LAE: {metrics.LAE:.1f} dB")
    print(f"LAFmax: {metrics.LAFmax:.1f} dB")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from calibration import (
    amplitude_to_dB_SPL,
    power_to_dB_SPL,
    P_REF,
    EPS,
    compute_peak,
)
from weighting import (
    apply_z_weight,
    apply_a_weight_zerophase, apply_c_weight_zerophase,
)
from bands import ThirdOctaveAnalyzer, compute_band_exposure


# Time constants for level computation (IEC 61672-1:2013)
TIME_CONSTANT_FAST = 0.125      # 125 ms
TIME_CONSTANT_SLOW = 1.0        # 1000 ms
TIME_CONSTANT_IMPULSE_ATTACK = 0.035   # 35 ms (rise)
TIME_CONSTANT_IMPULSE_DECAY = 1.5      # 1500 ms (decay)


@dataclass
class ShotMetrics:
    """
    Acoustic metrics for a single gunshot event.

    All levels are in dB re 20 µPa unless otherwise noted.
    """
    # Peak levels (instantaneous)
    Lpeak_Z: float       # Peak SPL, Z-weighted (unweighted)
    Lpeak_A: float       # Peak SPL, A-weighted
    Lpeak_C: float       # Peak SPL, C-weighted

    # Exposure levels (integrated energy)
    LAE: float           # A-weighted Sound Exposure Level (SEL)
    LZE: float           # Z-weighted Sound Exposure Level
    LCE: float           # C-weighted Sound Exposure Level

    # Maximum time-weighted levels
    LAFmax: float        # Max A-weighted, Fast (125ms)
    LASmax: float        # Max A-weighted, Slow (1s)
    LZFmax: float        # Max Z-weighted, Fast
    LZSmax: float        # Max Z-weighted, Slow

    # Impulse time-weighted max (IEC 61672-1: 35ms attack / 1500ms decay)
    LAImax: float = 0.0  # Max A-weighted, Impulse
    LZImax: float = 0.0  # Max Z-weighted, Impulse

    # Gunshot-specific metrics
    rise_time_us: float = 0.0       # 10-90% rise time in microseconds
    b_duration_ms: float = 0.0      # B-duration: time within 20 dB of peak (ms)
    crest_factor_dB: float = 0.0    # Peak-to-RMS ratio in dB
    spectral_centroid_Hz: float = 0.0  # Frequency center of spectral mass (Hz)
    kurtosis: float = 0.0           # Excess kurtosis (impulsiveness; normal=0, gunshot>>10)

    # Time series (optional, for plotting)
    time_s: np.ndarray = field(default_factory=lambda: np.array([]))
    LAF: np.ndarray = field(default_factory=lambda: np.array([]))  # A-weighted Fast
    LAS: np.ndarray = field(default_factory=lambda: np.array([]))  # A-weighted Slow
    LZF: np.ndarray = field(default_factory=lambda: np.array([]))  # Z-weighted Fast
    LZS: np.ndarray = field(default_factory=lambda: np.array([]))  # Z-weighted Slow

    # Band analysis
    band_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    band_exposure_dB: np.ndarray = field(default_factory=lambda: np.array([]))  # Per-band SEL

    # Duration
    duration_s: float = 0.0

    # Loudness (placeholder)
    loudness_sone_max: Optional[float] = None  # TODO: Implement loudness model

    # Metadata
    shot_number: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'Lpeak_Z': round(self.Lpeak_Z, 1),
            'Lpeak_A': round(self.Lpeak_A, 1),
            'Lpeak_C': round(self.Lpeak_C, 1),
            'LAE': round(self.LAE, 1),
            'LZE': round(self.LZE, 1),
            'LCE': round(self.LCE, 1),
            'LAFmax': round(self.LAFmax, 1),
            'LASmax': round(self.LASmax, 1),
            'LZFmax': round(self.LZFmax, 1),
            'LZSmax': round(self.LZSmax, 1),
            'LAImax': round(self.LAImax, 1),
            'LZImax': round(self.LZImax, 1),
            'rise_time_us': round(self.rise_time_us, 1),
            'b_duration_ms': round(self.b_duration_ms, 2),
            'crest_factor_dB': round(self.crest_factor_dB, 1),
            'spectral_centroid_Hz': round(self.spectral_centroid_Hz, 0),
            'kurtosis': round(self.kurtosis, 1),
            'duration_s': round(self.duration_s, 4),
            'loudness_sone_max': self.loudness_sone_max,
            'shot_number': self.shot_number,
            'band_frequencies': self.band_frequencies.tolist() if len(self.band_frequencies) > 0 else [],
            'band_exposure_dB': [round(x, 1) for x in self.band_exposure_dB] if len(self.band_exposure_dB) > 0 else [],
        }


def compute_exponential_average(
    x_squared: np.ndarray,
    sample_rate: int,
    time_constant: float,
) -> np.ndarray:
    """
    Compute exponential (RC) time weighting on squared signal.

    This implements the standard IEC 61672-1 time weighting.

    Args:
        x_squared: Squared pressure signal (Pa²).
        sample_rate: Sample rate in Hz.
        time_constant: Time constant in seconds (0.125 for Fast, 1.0 for Slow).

    Returns:
        Time-weighted mean-square pressure (Pa²).
    """
    # Exponential smoothing coefficient
    dt = 1.0 / sample_rate
    alpha = 1.0 - np.exp(-dt / time_constant)

    # Apply exponential filter
    n = len(x_squared)
    y = np.zeros(n, dtype=np.float64)

    state = 0.0
    for i in range(n):
        state = alpha * x_squared[i] + (1.0 - alpha) * state
        y[i] = state

    return y


def compute_impulse_exponential_average(
    x_squared: np.ndarray,
    sample_rate: int,
    tau_attack: float = TIME_CONSTANT_IMPULSE_ATTACK,
    tau_decay: float = TIME_CONSTANT_IMPULSE_DECAY,
) -> np.ndarray:
    """
    Compute IEC 61672-1 Impulse time weighting (asymmetric attack/decay).

    The Impulse detector uses a fast attack (35 ms) to capture transient peaks
    and a slow decay (1500 ms) to hold the reading, making it ideal for
    impulsive noise like gunshots.

    Args:
        x_squared: Squared pressure signal (Pa²).
        sample_rate: Sample rate in Hz.
        tau_attack: Attack time constant (default 35 ms per IEC 61672-1).
        tau_decay: Decay time constant (default 1500 ms per IEC 61672-1).

    Returns:
        Time-weighted mean-square pressure (Pa²).
    """
    dt = 1.0 / sample_rate
    alpha_attack = 1.0 - np.exp(-dt / tau_attack)
    alpha_decay = 1.0 - np.exp(-dt / tau_decay)

    n = len(x_squared)
    y = np.zeros(n, dtype=np.float64)

    state = 0.0
    for i in range(n):
        if x_squared[i] > state:
            state = alpha_attack * x_squared[i] + (1.0 - alpha_attack) * state
        else:
            state = alpha_decay * x_squared[i] + (1.0 - alpha_decay) * state
        y[i] = state

    return y


# ---- Gunshot-Specific Metrics ----


def compute_rise_time(pressure_Pa: np.ndarray, sample_rate: int) -> float:
    """
    Compute 10-90% rise time of the pressure waveform.

    Rise time measures how quickly the acoustic impulse develops, from 10%
    to 90% of peak absolute pressure. Typical gunshot rise times are 1-50 µs
    for muzzle blast, longer for suppressed weapons.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        Rise time in microseconds (µs).
    """
    abs_p = np.abs(pressure_Pa)
    peak_val = float(np.max(abs_p))
    peak_idx = int(np.argmax(abs_p))

    if peak_val < EPS:
        return 0.0

    threshold_10 = 0.1 * peak_val
    threshold_90 = 0.9 * peak_val

    # Search backwards from peak for 10% crossing (onset of impulse)
    i_10 = 0
    for i in range(peak_idx, -1, -1):
        if abs_p[i] <= threshold_10:
            i_10 = i
            break

    # Search backwards from peak for 90% crossing (near-peak)
    i_90 = peak_idx
    for i in range(peak_idx, -1, -1):
        if abs_p[i] <= threshold_90:
            i_90 = i + 1
            break

    rise_samples = max(0, i_90 - i_10)
    return rise_samples / sample_rate * 1e6  # microseconds


def compute_b_duration(pressure_Pa: np.ndarray, sample_rate: int) -> float:
    """
    Compute B-duration of the acoustic event.

    B-duration is the total time the signal envelope remains within 20 dB
    of the peak level. It measures the effective duration of the impulse
    including oscillatory components. Typical gunshot B-durations: 2-20 ms.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        B-duration in milliseconds (ms).
    """
    abs_p = np.abs(pressure_Pa)
    peak_val = float(np.max(abs_p))

    if peak_val < EPS:
        return 0.0

    # -20 dB below peak in linear amplitude = factor of 0.1
    threshold = peak_val * 0.1
    n_above = int(np.sum(abs_p >= threshold))
    return n_above / sample_rate * 1000.0  # milliseconds


def compute_crest_factor(pressure_Pa: np.ndarray) -> float:
    """
    Compute crest factor (peak-to-RMS ratio) in dB.

    Crest factor quantifies the "peakiness" of the signal. A pure sine wave
    has a crest factor of 3.01 dB. Gunshots typically have crest factors of
    15-30 dB, indicating highly impulsive character.

    Args:
        pressure_Pa: Pressure waveform in Pascals.

    Returns:
        Crest factor in dB.
    """
    peak = float(np.max(np.abs(pressure_Pa)))
    rms = float(np.sqrt(np.mean(pressure_Pa ** 2)))

    if rms < EPS:
        return 0.0

    return 20.0 * np.log10(peak / rms)


def compute_spectral_centroid(pressure_Pa: np.ndarray, sample_rate: int) -> float:
    """
    Compute spectral centroid (frequency center of mass).

    The spectral centroid indicates where the "center of gravity" of the
    spectrum lies. Higher values indicate more high-frequency energy.
    Useful for distinguishing weapon types and suppressor effects.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        Spectral centroid in Hz.
    """
    N = len(pressure_Pa)
    if N < 2:
        return 0.0

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(N)
    X = np.abs(np.fft.rfft(pressure_Pa * window))
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    power = X ** 2
    total_power = float(np.sum(power))

    if total_power < EPS:
        return 0.0

    return float(np.sum(freqs * power) / total_power)


def compute_kurtosis(pressure_Pa: np.ndarray) -> float:
    """
    Compute excess kurtosis (measure of impulsiveness).

    Kurtosis quantifies how "peaked" the amplitude distribution is relative
    to a Gaussian. A Gaussian has excess kurtosis of 0. Gunshots typically
    have kurtosis >> 10, indicating extreme impulsiveness. This metric is
    used in MIL-STD-1474E for impulsive noise assessment.

    Args:
        pressure_Pa: Pressure waveform in Pascals.

    Returns:
        Excess kurtosis (Fisher definition: normal distribution = 0).
    """
    x = np.asarray(pressure_Pa, dtype=np.float64)
    mu = np.mean(x)
    centered = x - mu
    m2 = float(np.mean(centered ** 2))
    m4 = float(np.mean(centered ** 4))

    if m2 < EPS:
        return 0.0

    return m4 / (m2 ** 2) - 3.0


def compute_exposure_level(
    pressure_Pa: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Compute Sound Exposure Level (SEL/LE).

    SEL = 10 * log10(∫p²(t)dt / (p_ref² * T_ref))

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        Sound Exposure Level in dB.
    """
    # Integrate squared pressure
    dt = 1.0 / sample_rate
    energy = np.sum(pressure_Pa ** 2) * dt

    # Reference: p_ref² * T_ref where T_ref = 1 second
    ref_energy = P_REF ** 2 * 1.0

    sel = 10.0 * np.log10(energy / ref_energy + EPS)
    return sel


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
    x_sq = pressure_Pa ** 2

    # Compute exponential averages
    fast_sq = compute_exponential_average(x_sq, sample_rate, TIME_CONSTANT_FAST)
    slow_sq = compute_exponential_average(x_sq, sample_rate, TIME_CONSTANT_SLOW)

    # Decimate for efficiency
    indices = np.arange(0, len(fast_sq), hop_samples)
    fast_sq_dec = fast_sq[indices]
    slow_sq_dec = slow_sq[indices]

    # Convert to dB SPL
    L_fast = np.asarray(power_to_dB_SPL(fast_sq_dec))
    L_slow = np.asarray(power_to_dB_SPL(slow_sq_dec))

    time_axis = indices / sample_rate

    return time_axis, L_fast, L_slow


def compute_shot_metrics(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    compute_bands: bool = True,
    compute_time_series: bool = True,
    shot_number: int = 0,
) -> ShotMetrics:
    """
    Compute comprehensive acoustic metrics for a gunshot event.

    Args:
        pressure_Pa: Calibrated pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.
        compute_bands: Whether to compute 1/3-octave band analysis.
        compute_time_series: Whether to store time series for plotting.
        shot_number: Shot identifier for metadata.

    Returns:
        ShotMetrics object with all computed metrics.
    """
    x = np.asarray(pressure_Pa, dtype=np.float64)
    duration_s = len(x) / sample_rate

    # Apply frequency weightings using zero-phase filtering for offline analysis.
    # Zero-phase (sosfiltfilt) eliminates startup transient and group delay,
    # giving more accurate peak and energy measurements on short shot windows.
    x_z = apply_z_weight(x, sample_rate)              # Z = unweighted (pass-through)
    x_a = apply_a_weight_zerophase(x, sample_rate)     # A-weighted (zero-phase)
    x_c = apply_c_weight_zerophase(x, sample_rate)     # C-weighted (zero-phase)

    # Peak levels (instantaneous)
    Lpeak_Z = float(amplitude_to_dB_SPL(compute_peak(x_z)))
    Lpeak_A = float(amplitude_to_dB_SPL(compute_peak(x_a)))
    Lpeak_C = float(amplitude_to_dB_SPL(compute_peak(x_c)))

    # Exposure levels (integrated energy)
    LZE = compute_exposure_level(x_z, sample_rate)
    LAE = compute_exposure_level(x_a, sample_rate)
    LCE = compute_exposure_level(x_c, sample_rate)

    # Time-weighted levels (Fast + Slow, causal per IEC 61672-1)
    hop = max(1, sample_rate // 1000)  # ~1ms resolution

    time_z, LZF, LZS = compute_time_weighted_levels(x_z, sample_rate, hop)
    time_a, LAF, LAS = compute_time_weighted_levels(x_a, sample_rate, hop)

    # Maximum time-weighted levels (Fast, Slow)
    LAFmax = float(np.max(LAF))
    LASmax = float(np.max(LAS))
    LZFmax = float(np.max(LZF))
    LZSmax = float(np.max(LZS))

    # Impulse time-weighted max (IEC 61672-1: 35ms attack, 1500ms decay)
    # The asymmetric detector captures fast transient peaks and holds them,
    # making it specifically suited for impulsive noise like gunshots.
    impulse_a = compute_impulse_exponential_average(x_a ** 2, sample_rate)
    impulse_z = compute_impulse_exponential_average(x_z ** 2, sample_rate)
    LAImax = float(power_to_dB_SPL(np.max(impulse_a)))
    LZImax = float(power_to_dB_SPL(np.max(impulse_z)))

    # Gunshot-specific metrics (computed on Z-weighted signal for physical accuracy)
    rise_time_us = compute_rise_time(x_z, sample_rate)
    b_duration_ms = compute_b_duration(x_z, sample_rate)
    crest_factor_dB = compute_crest_factor(x_z)
    spectral_centroid_Hz = compute_spectral_centroid(x_z, sample_rate)
    kurtosis_val = compute_kurtosis(x_z)

    # Band analysis
    band_frequencies = np.array([])
    band_exposure_dB = np.array([])

    if compute_bands:
        try:
            analyzer = ThirdOctaveAnalyzer(sample_rate=sample_rate)
            results = analyzer.analyze(x, time_weighting='fast', hop_ms=10.0)

            band_frequencies = results['center_frequencies']
            band_exposure_dB = compute_band_exposure(
                results['band_levels_dB'],
                results['time_s'],
            )
        except Exception as e:
            print(f"Warning: Band analysis failed: {e}")

    # Store time series if requested
    if not compute_time_series:
        time_a = np.array([])
        LAF = np.array([])
        LAS = np.array([])
        LZF = np.array([])
        LZS = np.array([])

    # TODO: Loudness model (e.g., ISO 532-1 or Moore-Glasberg)
    # This requires a proper implementation of temporal loudness
    loudness_sone_max = None  # Placeholder

    return ShotMetrics(
        Lpeak_Z=Lpeak_Z,
        Lpeak_A=Lpeak_A,
        Lpeak_C=Lpeak_C,
        LAE=LAE,
        LZE=LZE,
        LCE=LCE,
        LAFmax=LAFmax,
        LASmax=LASmax,
        LZFmax=LZFmax,
        LZSmax=LZSmax,
        LAImax=LAImax,
        LZImax=LZImax,
        rise_time_us=rise_time_us,
        b_duration_ms=b_duration_ms,
        crest_factor_dB=crest_factor_dB,
        spectral_centroid_Hz=spectral_centroid_Hz,
        kurtosis=kurtosis_val,
        time_s=time_a,
        LAF=LAF,
        LAS=LAS,
        LZF=LZF,
        LZS=LZS,
        band_frequencies=band_frequencies,
        band_exposure_dB=band_exposure_dB,
        duration_s=duration_s,
        loudness_sone_max=loudness_sone_max,
        shot_number=shot_number,
    )


@dataclass
class AggregateMetrics:
    """
    Aggregate metrics across multiple shots.
    """
    n_shots: int

    # Peak levels (max across shots)
    Lpeak_Z_max: float
    Lpeak_A_max: float

    # Mean levels
    LAE_mean: float
    LAFmax_mean: float

    # Standard deviations
    LAE_std: float
    LAFmax_std: float

    # Per-shot metrics
    shot_metrics: List[ShotMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'n_shots': self.n_shots,
            'Lpeak_Z_max': round(self.Lpeak_Z_max, 1),
            'Lpeak_A_max': round(self.Lpeak_A_max, 1),
            'LAE_mean': round(self.LAE_mean, 1),
            'LAE_std': round(self.LAE_std, 2),
            'LAFmax_mean': round(self.LAFmax_mean, 1),
            'LAFmax_std': round(self.LAFmax_std, 2),
            'shots': [m.to_dict() for m in self.shot_metrics],
        }


def compute_aggregate_metrics(
    shot_metrics_list: List[ShotMetrics],
) -> AggregateMetrics:
    """
    Compute aggregate statistics across multiple shots.

    Args:
        shot_metrics_list: List of ShotMetrics objects.

    Returns:
        AggregateMetrics with summary statistics.
    """
    if not shot_metrics_list:
        return AggregateMetrics(
            n_shots=0,
            Lpeak_Z_max=0.0,
            Lpeak_A_max=0.0,
            LAE_mean=0.0,
            LAFmax_mean=0.0,
            LAE_std=0.0,
            LAFmax_std=0.0,
        )

    n = len(shot_metrics_list)

    # Extract values
    Lpeak_Z_values = [m.Lpeak_Z for m in shot_metrics_list]
    Lpeak_A_values = [m.Lpeak_A for m in shot_metrics_list]
    LAE_values = np.array([m.LAE for m in shot_metrics_list])
    LAFmax_values = np.array([m.LAFmax for m in shot_metrics_list])

    # Energy-average for sound levels (ISO standard):
    #   L_avg = 10 * log10( mean( 10^(L_i / 10) ) )
    # Arithmetic mean of dB underestimates true energy; energy-average is correct.
    LAE_energy_mean = float(10.0 * np.log10(np.mean(10.0 ** (LAE_values / 10.0))))
    LAFmax_energy_mean = float(10.0 * np.log10(np.mean(10.0 ** (LAFmax_values / 10.0))))

    return AggregateMetrics(
        n_shots=n,
        Lpeak_Z_max=max(Lpeak_Z_values),
        Lpeak_A_max=max(Lpeak_A_values),
        LAE_mean=LAE_energy_mean,
        LAFmax_mean=LAFmax_energy_mean,
        LAE_std=float(np.std(LAE_values)) if n > 1 else 0.0,
        LAFmax_std=float(np.std(LAFmax_values)) if n > 1 else 0.0,
        shot_metrics=shot_metrics_list,
    )


def format_metrics_summary(metrics: ShotMetrics, prefix: str = "") -> str:
    """
    Format metrics as human-readable summary string.

    Args:
        metrics: ShotMetrics object.
        prefix: Optional prefix for each line.

    Returns:
        Formatted string.
    """
    lines = [
        f"{prefix}Shot {metrics.shot_number} Metrics:",
        f"{prefix}  Peak SPL (Z): {metrics.Lpeak_Z:.1f} dB",
        f"{prefix}  Peak SPL (A): {metrics.Lpeak_A:.1f} dB",
        f"{prefix}  Peak SPL (C): {metrics.Lpeak_C:.1f} dB",
        f"{prefix}  LAE (A-weighted SEL): {metrics.LAE:.1f} dB",
        f"{prefix}  LAFmax (Fast A-weighted max): {metrics.LAFmax:.1f} dB",
        f"{prefix}  LASmax (Slow A-weighted max): {metrics.LASmax:.1f} dB",
        f"{prefix}  LAImax (Impulse A-weighted max): {metrics.LAImax:.1f} dB",
        f"{prefix}  Rise time: {metrics.rise_time_us:.1f} us",
        f"{prefix}  B-duration: {metrics.b_duration_ms:.2f} ms",
        f"{prefix}  Crest factor: {metrics.crest_factor_dB:.1f} dB",
        f"{prefix}  Spectral centroid: {metrics.spectral_centroid_Hz:.0f} Hz",
        f"{prefix}  Kurtosis: {metrics.kurtosis:.1f}",
        f"{prefix}  Duration: {metrics.duration_s*1000:.1f} ms",
    ]

    if metrics.loudness_sone_max is not None:
        lines.append(f"{prefix}  Loudness (max): {metrics.loudness_sone_max:.1f} sone")

    return "\n".join(lines)


# ---- TODO: Loudness Model ----
#
# The following is a placeholder for a proper loudness implementation.
# A full implementation would require:
#   - ISO 532-1:2017 (Zwicker method for stationary sounds)
#   - ISO 532-2:2017 (Moore-Glasberg method)
#   - For impulsive sounds like gunshots, temporal integration is critical
#
# Recommended libraries for loudness:
#   - mosqito: https://github.com/Eomys/MoSQITo (ISO 532-1, 532-2)
#   - python-acoustics: https://github.com/python-acoustics/python-acoustics
#
# Until implemented, LAFmax serves as a reasonable perceptual correlate.

def compute_loudness_placeholder(
    pressure_Pa: np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    """
    Placeholder for loudness calculation.

    TODO: Implement proper loudness model (ISO 532-1 or Moore-Glasberg).

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.

    Returns:
        None (placeholder) or loudness in sone if implemented.
    """
    # Try to use mosqito if available
    try:
        from mosqito.sq_metrics import loudness_zwst

        # Compute stationary loudness (approximation for impulsive)
        # Note: This is not ideal for gunshots - use time-varying loudness
        N, N_spec, _ = loudness_zwst(pressure_Pa, sample_rate)
        return float(N)
    except ImportError:
        pass

    # Fallback: Return None
    return None


# ---- CLI for testing ----

def main() -> int:
    """Test metrics computation."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute acoustic metrics for audio")
    parser.add_argument("wav", type=Path, nargs='?', help="Input WAV file")
    parser.add_argument("--Pa-per-FS", type=float, default=100.0,
                        help="Calibration factor (default: 100)")
    parser.add_argument("--start", type=float, default=None, help="Start time (s)")
    parser.add_argument("--duration", type=float, default=None, help="Duration (s)")
    parser.add_argument("--test-impulse", action="store_true", help="Use test impulse")
    args = parser.parse_args()

    if args.test_impulse or args.wav is None:
        # Generate test impulse
        sr = 96000
        duration = 0.1  # 100 ms
        t = np.arange(int(sr * duration)) / sr

        # Exponentially decaying impulse
        decay = 0.01  # decay time constant
        signal = np.exp(-t / decay) * np.sin(2 * np.pi * 1000 * t)
        signal[0] = 1.0  # Sharp attack

        # Calibrate to ~140 dB peak
        pressure_Pa = signal * 200.0  # ~140 dB SPL peak

        print("Test impulse: exponential decay, 1kHz, ~140 dB peak")
        print(f"Sample rate: {sr} Hz")
    else:
        import soundfile as sf
        data, sr = sf.read(str(args.wav), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)

        pressure_Pa = data * args.Pa_per_FS

        if args.start is not None or args.duration is not None:
            start_i = int((args.start or 0) * sr)
            end_i = start_i + int((args.duration or len(data)/sr) * sr)
            pressure_Pa = pressure_Pa[start_i:end_i]

        print(f"Loaded: {args.wav}")
        print(f"Sample rate: {sr} Hz")

    # Compute metrics
    metrics = compute_shot_metrics(
        pressure_Pa,
        sr,
        compute_bands=True,
        compute_time_series=True,
        shot_number=1,
    )

    # Print summary
    print("\n" + format_metrics_summary(metrics))

    # Print band exposure
    if len(metrics.band_frequencies) > 0:
        print("\n  1/3-Octave Band Exposure (SEL):")
        print(f"  {'Freq (Hz)':>10} {'SEL (dB)':>10}")
        print("  " + "-" * 25)
        for f, sel in zip(metrics.band_frequencies, metrics.band_exposure_dB):
            print(f"  {f:10.0f} {sel:10.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
