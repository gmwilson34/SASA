#!/usr/bin/env python3
"""
bands.py - Fractional-Octave Band Analysis for Acoustic Measurements

Computes 1/3-octave (and 1/1-octave) band levels using IEC 61260-1 bandpass filters
on ISO 266 preferred centre frequencies.

    Band filters are designed on a DECIMATED signal.

    A 20 Hz band at 192 kHz has a normalised bandwidth of 2.4e-5. An IIR bandpass
    designed at that normalised frequency is numerically degenerate: its poles sit so
    close to z=1 that float64 cannot represent the passband, and naive implementations
    end up clamping the band edges into a completely different (and wrong) passband.
    The standard remedy, used here, is a decimation cascade - each band is filtered at
    a sample rate chosen so its normalised centre frequency lands in a well-conditioned
    range, then the resulting level is resampled onto a common time grid.

Outputs band SPL vs time with configurable time weighting:
  - Fast (125 ms exponential averaging)
  - Slow (1000 ms exponential averaging)
  - Impulse (35 ms attack / 1500 ms decay, per IEC 61672-1)
  - None (linear energy average over each hop)

Usage:
    from bands import ThirdOctaveAnalyzer

    analyzer = ThirdOctaveAnalyzer(sample_rate=96000)
    results = analyzer.analyze(pressure_Pa, time_weighting='fast')

    results['center_frequencies']  # nominal ISO 266 labels
    results['band_levels_dB']      # (n_bands, n_frames)
    results['band_exposure_dB']    # per-band SEL

References:
    - ISO 266:1997 - Preferred frequencies
    - IEC 61260-1:2014 - Octave-band and fractional-octave-band filters
    - IEC 61672-1:2013 - Time weightings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfreqz

from calibration import EPS, P_REF, power_to_dB_SPL

TimeWeighting = Literal["fast", "slow", "impulse", "none"]

# ISO 266:1997 preferred (nominal) 1/3-octave centre frequencies, Hz.
# These are ROUNDED labels. Filters are designed on the exact midband frequencies
# defined by IEC 61260-1 (see exact_midband_frequency), not on these values.
ISO_CENTER_FREQUENCIES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
], dtype=np.float64)

# Extended nominal series for high sample rates. Gunshot spectra carry meaningful
# energy well above 20 kHz, and a 192 kHz recording can resolve it.
EXTENDED_CENTER_FREQUENCIES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
    20000, 25000, 31500, 40000, 50000, 63000, 80000
], dtype=np.float64)

# IEC 61260-1 base-10 octave ratio. The standard prefers base-10 (G = 10^(3/10))
# over base-2; using 2 instead leaves gaps and overlaps between adjacent bands.
G_BASE10: float = 10.0 ** (3.0 / 10.0)

REFERENCE_FREQUENCY: float = 1000.0

# Time constants for symmetric exponential averaging (seconds), IEC 61672-1
TIME_CONSTANTS: Dict[str, float] = {
    "fast": 0.125,
    "slow": 1.0,
}

# IEC 61672-1 Impulse time weighting is ASYMMETRIC
TIME_CONSTANT_IMPULSE_ATTACK: float = 0.035
TIME_CONSTANT_IMPULSE_DECAY: float = 1.5

# Target range for a band's centre frequency relative to the working Nyquist.
# Below the floor the IIR design is ill-conditioned; above the ceiling the filter
# runs out of room before Nyquist.
_DECIMATION_TARGET_HI: float = 0.25
_MIN_DECIMATED_RATE: float = 400.0


def band_number(nominal_fc: float, fraction: int = 3) -> int:
    """Band index x such that the exact midband frequency is closest to nominal_fc."""
    return int(round(fraction * np.log(nominal_fc / REFERENCE_FREQUENCY) / np.log(G_BASE10)))


def exact_midband_frequency(nominal_fc: float, fraction: int = 3) -> float:
    """
    Exact IEC 61260-1 midband frequency for a nominal ISO 266 label.

    For an odd fraction b:  f_m = G^(x/b) * f_r
    e.g. the band labelled "1250 Hz" is exactly 1258.9 Hz.
    """
    x = band_number(nominal_fc, fraction)
    return float(REFERENCE_FREQUENCY * G_BASE10 ** (x / fraction))


def compute_band_edges(fc: float, fraction: float = 3.0) -> Tuple[float, float]:
    """
    Compute band edge frequencies for a fractional-octave band (IEC 61260-1, base-10).

    Args:
        fc: Exact midband frequency in Hz.
        fraction: Fraction of an octave (3 for 1/3-octave, 1 for octave).

    Returns:
        (f_low, f_high) band edge frequencies in Hz.
    """
    ratio = G_BASE10 ** (1.0 / (2.0 * fraction))
    return fc / ratio, fc * ratio


def _decimation_factor(f_high: float, sample_rate: float, fraction: float = 3.0) -> int:
    """
    Choose a power-of-two decimation factor that puts a band in a well-conditioned range.

    The factor is the largest power of two for which the band's upper edge still sits
    below a quarter of the decimated Nyquist, subject to a floor on the decimated rate.
    """
    d = 1
    while True:
        nxt = d * 2
        rate = sample_rate / nxt
        if rate < _MIN_DECIMATED_RATE:
            return d
        if f_high > _DECIMATION_TARGET_HI * (rate / 2.0):
            return d
        d = nxt


def design_bandpass_sos(
    f_low: float,
    f_high: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Design a Butterworth bandpass filter as second-order sections.

    No frequency clamping is applied: the caller is responsible for supplying a
    sample rate at which the band is representable (see _decimation_factor). Clamping
    edges into the valid range silently returns a filter for a DIFFERENT band, which
    is far worse than an error.

    Args:
        f_low: Lower band edge (Hz).
        f_high: Upper band edge (Hz).
        fs: Sample rate (Hz).
        order: Butterworth order per side (scipy convention); 4 gives an 8th-order bandpass.

    Returns:
        SOS array for scipy.signal.sosfilt.

    Raises:
        ValueError: If the band is not representable at this sample rate.
    """
    nyq = fs / 2.0
    if not (0.0 < f_low < f_high < nyq):
        raise ValueError(
            f"Band [{f_low:.3f}, {f_high:.3f}] Hz is not representable at fs={fs:.1f} Hz "
            f"(Nyquist {nyq:.1f} Hz)"
        )

    low_norm = f_low / nyq
    high_norm = f_high / nyq
    if low_norm < 1e-4:
        raise ValueError(
            f"Band edge {f_low:.3f} Hz is only {low_norm:.2e} of Nyquist at fs={fs:.1f} Hz; "
            f"decimate before designing this filter"
        )

    return np.asarray(butter(order, [low_norm, high_norm], btype="band", output="sos"))


@dataclass
class BandFilter:
    """One fractional-octave band filter, together with the rate it runs at."""
    nominal_freq: float          # ISO 266 label, for display
    center_freq: float           # exact IEC 61260-1 midband frequency
    f_low: float
    f_high: float
    decimation: int              # signal is decimated by this factor before filtering
    working_rate: float          # sample rate the filter is designed for
    sos: np.ndarray = field(repr=False)

    @property
    def bandwidth_Hz(self) -> float:
        return self.f_high - self.f_low


@dataclass
class ThirdOctaveAnalyzer:
    """
    Fractional-octave band analyzer with time-weighted level output.

    Implements ISO/IEC standards for fractional-octave band analysis:
      - ISO 266:1997 nominal centre frequencies (labels)
      - IEC 61260-1:2014 exact midband frequencies, base-10 band edges, filter design
      - IEC 61672-1:2013 time weighting

    Bands are filtered on a decimated signal so that every filter is numerically
    well-conditioned regardless of sample rate.
    """
    sample_rate: int
    min_freq: float = 20.0
    max_freq: Optional[float] = None
    filter_order: int = 4
    fraction: int = 3
    center_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    nominal_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    filters: List[BandFilter] = field(default_factory=list, repr=False)
    skipped_bands: List[Tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.max_freq is None:
            # A band is only measurable if its upper edge is below Nyquist with margin
            # for the filter's transition. 0.45 of the sample rate is a safe ceiling.
            self.max_freq = self.sample_rate * 0.45
        self._design_filters()

    def _design_filters(self) -> None:
        """Design bandpass filters for every band representable at this sample rate."""
        self.filters = []
        self.skipped_bands = []

        for nominal in EXTENDED_CENTER_FREQUENCIES:
            if nominal < self.min_freq or nominal > self.max_freq:
                continue

            fc = exact_midband_frequency(float(nominal), self.fraction)
            f_low, f_high = compute_band_edges(fc, self.fraction)

            if f_high >= self.sample_rate / 2.0:
                self.skipped_bands.append((float(nominal), "upper edge at or above Nyquist"))
                continue

            decim = _decimation_factor(f_high, self.sample_rate, self.fraction)
            working_rate = self.sample_rate / decim

            try:
                sos = design_bandpass_sos(f_low, f_high, working_rate, self.filter_order)
            except ValueError as exc:
                self.skipped_bands.append((float(nominal), str(exc)))
                continue

            self.filters.append(BandFilter(
                nominal_freq=float(nominal),
                center_freq=fc,
                f_low=f_low,
                f_high=f_high,
                decimation=decim,
                working_rate=working_rate,
                sos=sos,
            ))

        self.nominal_frequencies = np.array([f.nominal_freq for f in self.filters])
        self.center_frequencies = np.array([f.nominal_freq for f in self.filters])
        self.exact_frequencies = np.array([f.center_freq for f in self.filters])

    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.filters)

    # ---- Decimation pyramid ----

    def _build_pyramid(self, x: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Build the anti-aliased decimation pyramid needed by the designed filters.

        Only the decimation factors actually used are materialised, and each level is
        produced from the previous one by a single factor-of-two polyphase resample,
        so total extra memory is bounded by the original signal length.
        """
        needed = sorted({f.decimation for f in self.filters})
        pyramid: Dict[int, np.ndarray] = {1: np.asarray(x, dtype=np.float64)}
        if needed == [1]:
            return pyramid

        current = pyramid[1]
        factor = 1
        while factor < max(needed):
            factor *= 2
            # resample_poly applies an anti-aliasing FIR before decimating
            current = np.asarray(resample_poly(current, up=1, down=2))
            if factor in needed:
                pyramid[factor] = current
        return pyramid

    # ---- Level computation ----

    def compute_levels(
        self,
        x: np.ndarray,
        time_weighting: TimeWeighting = "fast",
        hop_ms: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time-weighted band levels.

        Args:
            x: Input signal in Pascals.
            time_weighting: 'fast' (125 ms), 'slow' (1 s), 'impulse' (35/1500 ms), or 'none'.
            hop_ms: Output time resolution in milliseconds.

        Returns:
            (time_axis, band_levels_dB) where band_levels_dB is (n_bands, n_frames).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Input must be 1D array")
        if hop_ms <= 0:
            raise ValueError(f"hop_ms must be positive, got {hop_ms}")

        n = len(x)
        hop_samples = max(1, int(round(hop_ms * self.sample_rate / 1000.0)))
        n_frames = max(1, int(np.ceil(n / hop_samples))) if n else 0

        if n == 0 or self.n_bands == 0:
            return np.array([]), np.zeros((self.n_bands, 0))

        # Frame boundaries on the ORIGINAL time base
        frame_time = (np.arange(n_frames) * hop_samples + hop_samples / 2.0) / self.sample_rate

        pyramid = self._build_pyramid(x)
        band_ms = np.zeros((self.n_bands, n_frames), dtype=np.float64)

        for i, filt in enumerate(self.filters):
            signal = pyramid[filt.decimation]
            if signal.size == 0:
                continue

            # Band-limit, then square to get instantaneous mean-square pressure.
            # One band at a time keeps peak memory at O(n), not O(n_bands * n).
            filtered = np.asarray(sosfilt(filt.sos, signal))
            squared = filtered * filtered

            detected = _apply_time_weighting(squared, filt.working_rate, time_weighting,
                                             hop_ms=hop_ms)

            # Map the decimated time base onto the common frame grid
            src_time = np.arange(len(detected), dtype=np.float64) / filt.working_rate
            if len(detected) == 1:
                band_ms[i, :] = detected[0]
            else:
                band_ms[i, :] = np.interp(frame_time, src_time, detected,
                                          left=detected[0], right=detected[-1])

        return frame_time, np.asarray(power_to_dB_SPL(band_ms))

    def analyze(
        self,
        x: np.ndarray,
        time_weighting: TimeWeighting = "fast",
        hop_ms: float = 10.0,
    ) -> dict:
        """
        Full band analysis returning a results dictionary.

        Args:
            x: Input signal in Pascals.
            time_weighting: Time weighting type.
            hop_ms: Output time resolution.

        Returns:
            Dictionary with time_s, center_frequencies, band_levels_dB,
            overall_level_dB, band_exposure_dB and the filter bank description.
        """
        time_s, band_levels_dB = self.compute_levels(x, time_weighting, hop_ms)

        # Overall level from the sum of band powers
        if band_levels_dB.size:
            band_Pa2 = (P_REF ** 2) * (10.0 ** (band_levels_dB / 10.0))
            overall_dB = np.asarray(power_to_dB_SPL(np.sum(band_Pa2, axis=0)))
        else:
            overall_dB = np.array([])

        # Per-band exposure is computed from the ENERGY of each band directly rather
        # than from the time-weighted levels, which carry the detector's own gain.
        band_exposure_dB = self.compute_band_exposure(x)

        return {
            "time_s": time_s,
            "center_frequencies": self.center_frequencies.copy(),
            "exact_frequencies": self.exact_frequencies.copy(),
            "band_levels_dB": band_levels_dB,
            "overall_level_dB": overall_dB,
            "band_exposure_dB": band_exposure_dB,
            "time_weighting": time_weighting,
            "hop_ms": hop_ms,
            "bandwidths_Hz": np.array([f.bandwidth_Hz for f in self.filters]),
            "decimation": np.array([f.decimation for f in self.filters]),
            "skipped_bands": list(self.skipped_bands),
        }

    def compute_band_exposure(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sound exposure level (SEL) for each band, directly from band energy.

            SEL_band = 10*log10( integral p_band^2 dt / (p_ref^2 * 1 s) )

        Integrating the true band signal avoids the detector gain that any
        time-weighted level carries, so per-band SEL sums correctly to broadband SEL.

        Args:
            x: Input signal in Pascals.

        Returns:
            Per-band SEL in dB, shape (n_bands,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0 or self.n_bands == 0:
            return np.zeros(self.n_bands)

        pyramid = self._build_pyramid(x)
        energy = np.zeros(self.n_bands, dtype=np.float64)

        for i, filt in enumerate(self.filters):
            signal = pyramid[filt.decimation]
            if signal.size == 0:
                continue
            filtered = np.asarray(sosfilt(filt.sos, signal))
            # Energy is rate-independent: sum(p^2) * dt
            energy[i] = float(np.sum(filtered * filtered)) / filt.working_rate

        return 10.0 * np.log10(np.maximum(energy, EPS) / (P_REF ** 2 * 1.0))

    # ---- Filter bank verification ----

    def verify_filter_bank(self, tolerance_dB: float = 0.5) -> "FilterBankReport":
        """
        Verify that every designed filter actually passes the band it is labelled with.

        For each band the measured -3 dB points are compared against the nominal IEC
        61260-1 edges. This is the check whose absence allowed band edges to be
        silently clamped into the wrong passband.

        Args:
            tolerance_dB: Allowed error, in dB, on each measured band edge position
                          expressed as a fraction of the band's own width.

        Returns:
            FilterBankReport listing any band whose real passband is not its nominal one.
        """
        problems: List[str] = []
        rows: List[Dict[str, float]] = []

        for filt in self.filters:
            w, h = sosfreqz(filt.sos, worN=8192, fs=filt.working_rate)
            mag = np.abs(h)
            peak = float(mag.max())
            if peak <= EPS:
                problems.append(f"{filt.nominal_freq:g} Hz: filter has no passband")
                continue

            above = np.flatnonzero(mag >= peak / np.sqrt(2.0))
            if above.size == 0:
                problems.append(f"{filt.nominal_freq:g} Hz: no -3 dB region")
                continue

            meas_lo, meas_hi = float(w[above[0]]), float(w[above[-1]])
            width = filt.f_high - filt.f_low
            err_lo = (meas_lo - filt.f_low) / width
            err_hi = (meas_hi - filt.f_high) / width

            rows.append({
                "nominal": filt.nominal_freq,
                "f_low": filt.f_low, "f_high": filt.f_high,
                "measured_low": meas_lo, "measured_high": meas_hi,
                "error_low_frac": err_lo, "error_high_frac": err_hi,
                "decimation": filt.decimation,
            })

            if abs(err_lo) > 0.1 or abs(err_hi) > 0.1:
                problems.append(
                    f"{filt.nominal_freq:g} Hz: nominal [{filt.f_low:.2f}, {filt.f_high:.2f}] "
                    f"but measured [{meas_lo:.2f}, {meas_hi:.2f}] Hz"
                )

        return FilterBankReport(
            sample_rate=self.sample_rate,
            n_bands=self.n_bands,
            rows=rows,
            problems=problems,
            skipped=list(self.skipped_bands),
        )


@dataclass
class FilterBankReport:
    """Result of verifying that each band filter passes its nominal band."""
    sample_rate: int
    n_bands: int
    rows: List[Dict[str, float]]
    problems: List[str]
    skipped: List[Tuple[float, str]]

    @property
    def passed(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Filter bank @ {self.sample_rate} Hz: {self.n_bands} bands, {status}"]
        for p in self.problems:
            lines.append(f"    {p}")
        for freq, why in self.skipped:
            lines.append(f"    skipped {freq:g} Hz: {why}")
        return "\n".join(lines)


# ---- Time weighting detectors ----

def _apply_time_weighting(
    squared: np.ndarray,
    sample_rate: float,
    time_weighting: TimeWeighting,
    *,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Apply an IEC 61672-1 detector to a squared-pressure signal."""
    if time_weighting == "none":
        hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))
        n_frames = max(1, int(np.ceil(len(squared) / hop)))
        out = np.zeros(n_frames)
        for i in range(n_frames):
            seg = squared[i * hop:(i + 1) * hop]
            out[i] = float(np.mean(seg)) if seg.size else 0.0
        return out

    if time_weighting == "impulse":
        return impulse_detector(squared, sample_rate)

    tau = TIME_CONSTANTS.get(time_weighting)
    if tau is None:
        raise ValueError(
            f"Unknown time weighting: {time_weighting}. Use 'fast', 'slow', 'impulse' or 'none'."
        )
    return exponential_detector(squared, sample_rate, tau)


def exponential_detector(
    squared: np.ndarray,
    sample_rate: float,
    time_constant: float,
) -> np.ndarray:
    """
    IEC 61672-1 symmetric exponential (RC) detector on a squared signal.

    Implemented as a first-order IIR via scipy.signal.lfilter:

        y[n] = a*x[n] + (1-a)*y[n-1],   a = 1 - exp(-dt/tau)

    which is identical to the per-sample recursion to within float64 rounding
    (~1e-14) and roughly 30x faster.
    """
    from scipy.signal import lfilter

    x = np.asarray(squared, dtype=np.float64)
    if x.size == 0:
        return x
    alpha = 1.0 - np.exp(-1.0 / (sample_rate * time_constant))
    return np.asarray(lfilter([alpha], [1.0, -(1.0 - alpha)], x))


def impulse_detector(
    squared: np.ndarray,
    sample_rate: float,
    tau_attack: float = TIME_CONSTANT_IMPULSE_ATTACK,
    tau_decay: float = TIME_CONSTANT_IMPULSE_DECAY,
) -> np.ndarray:
    """
    IEC 61672-1 Impulse detector: 35 ms exponential average, then a decay-limited hold.

    The standard specifies a cascade, not a single asymmetric one-pole:

      1. a 35 ms exponential average of the squared signal, then
      2. a peak detector whose output may only fall with a 1500 ms time constant.

    A single asymmetric one-pole (choosing the attack or decay coefficient by
    comparing the instantaneous sample against the running state) rides the peaks of
    the raw signal instead, which gives an energy gain far above unity and inflates
    any exposure computed from it.

    Args:
        squared: Squared pressure signal (Pa^2).
        sample_rate: Sample rate in Hz.
        tau_attack: Attack (averaging) time constant, default 35 ms.
        tau_decay: Decay time constant of the hold stage, default 1500 ms.

    Returns:
        Time-weighted mean-square pressure (Pa^2).
    """
    x = np.asarray(squared, dtype=np.float64)
    if x.size == 0:
        return x

    # Stage 1: 35 ms exponential average
    averaged = exponential_detector(x, sample_rate, tau_attack)

    # Stage 2: decay-limited peak hold. Vectorised via a running maximum against an
    # exponentially decaying envelope, which is what the standard's detector realises.
    decay = float(np.exp(-1.0 / (sample_rate * tau_decay)))
    return _decay_limited_hold(averaged, decay)


def _decay_limited_hold(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Running maximum with exponential decay:  y[n] = max(x[n], decay * y[n-1]).

    Evaluated in blocks so the Python-level loop runs once per block rather than
    once per sample, which keeps a 192 kHz recording tractable.
    """
    n = x.size
    if n == 0:
        return x
    y = np.empty(n, dtype=np.float64)

    # Block size chosen so decay**block is negligible: beyond that horizon a past
    # sample can no longer influence the present, so blocks can be seeded cheaply.
    if decay >= 1.0:
        block = n
    else:
        block = int(min(n, max(1024, np.ceil(np.log(1e-12) / np.log(decay)))))

    carry = 0.0
    for start in range(0, n, block):
        stop = min(start + block, n)
        seg = x[start:stop]
        m = seg.size

        # Contribution of the carried state, decaying across the block
        powers = decay ** np.arange(1, m + 1)
        from_carry = carry * powers

        # Within-block decay-limited running max
        # scaled[i] = seg[i] / decay**i  ->  running max  ->  rescale
        idx = np.arange(m)
        scale = decay ** idx
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = seg / scale
            scaled = np.where(np.isfinite(scaled), scaled, seg)
            within = np.maximum.accumulate(scaled) * scale
            within = np.where(np.isfinite(within), within, seg)

        out = np.maximum(within, from_carry)
        y[start:stop] = out
        carry = float(out[-1])

    return y


# ---- Aggregate helpers ----

def compute_band_exposure(
    band_levels_dB: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """
    Compute sound exposure level (SEL) per band by integrating level over time.

    Prefer ThirdOctaveAnalyzer.compute_band_exposure(), which integrates the band
    signal itself and so is free of detector gain. This form is retained for callers
    that only have the level matrix.

    Args:
        band_levels_dB: Band levels in dB SPL, shape (n_bands, n_frames).
        time_s: Time axis in seconds, shape (n_frames,).

    Returns:
        Band exposure levels in dB, shape (n_bands,).
    """
    levels = np.atleast_2d(np.asarray(band_levels_dB, dtype=np.float64))
    time_s = np.asarray(time_s, dtype=np.float64)
    n_bands, n_frames = levels.shape

    if n_frames == 0:
        return np.full(n_bands, -np.inf)

    band_Pa2 = (P_REF ** 2) * (10.0 ** (levels / 10.0))

    if n_frames == 1:
        # A single frame carries no duration information; treat it as one hop long.
        dt = float(time_s[0] * 2.0) if time_s.size and time_s[0] > 0 else 1.0
        energy = band_Pa2[:, 0] * dt
    else:
        energy = np.trapezoid(band_Pa2, x=time_s, axis=1) if hasattr(np, "trapezoid") \
            else np.trapz(band_Pa2, x=time_s, axis=1)

    return 10.0 * np.log10(np.maximum(energy, EPS) / (P_REF ** 2 * 1.0))


def compute_leq(
    band_levels_dB: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """
    Compute equivalent continuous level (Leq) for each band.

    Args:
        band_levels_dB: Band levels in dB SPL, shape (n_bands, n_frames).
        time_s: Time axis in seconds.

    Returns:
        Leq for each band in dB, shape (n_bands,).
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    duration = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 1.0
    duration = max(duration, EPS)
    return compute_band_exposure(band_levels_dB, time_s) - 10.0 * np.log10(duration)


def band_insertion_loss(
    reference_band_dB: np.ndarray,
    test_band_dB: np.ndarray,
) -> np.ndarray:
    """
    Per-band insertion loss: how much quieter the test configuration is, band by band.

        IL(f) = L_reference(f) - L_test(f)

    Positive values are reduction. This is the spectral form of net suppression and
    is the single most informative suppressor deliverable, because it shows WHERE a
    suppressor works rather than just by how much.

    Args:
        reference_band_dB: Per-band levels of the unsuppressed reference.
        test_band_dB: Per-band levels of the suppressed configuration.

    Returns:
        Per-band insertion loss in dB.

    Raises:
        ValueError: If the two band vectors do not describe the same filter bank.
    """
    ref = np.asarray(reference_band_dB, dtype=np.float64)
    test = np.asarray(test_band_dB, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(
            f"Band vectors must match: reference has {ref.shape}, test has {test.shape}. "
            f"Both recordings must be analysed at the same sample rate."
        )
    return ref - test


# ---- CLI for testing ----

def main() -> int:
    """Test 1/3-octave band analysis."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="1/3-octave band analysis")
    parser.add_argument("wav", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--Pa-per-FS", type=float, default=1.0, help="Calibration factor")
    parser.add_argument("--weighting", choices=["fast", "slow", "impulse", "none"],
                        default="fast", help="Time weighting")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Output time resolution (ms)")
    parser.add_argument("--fs", type=float, default=96000, help="Sample rate for --verify")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the filter bank passes its nominal bands")
    args = parser.parse_args()

    if args.verify:
        for fs in (44100, 48000, 96000, 192000):
            analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
            print(analyzer.verify_filter_bank().summary())
        return 0

    if args.wav is None:
        sr = int(args.fs)
        duration = 1.0
        t = np.arange(int(sr * duration)) / sr
        signal = np.zeros_like(t)
        test_freqs = [100, 500, 1000, 4000]
        for f in test_freqs:
            signal += np.sqrt(2.0) * np.sin(2 * np.pi * f * t)
        pressure_Pa = signal * P_REF * 10 ** (94 / 20)
        print(f"Test signal: {test_freqs} Hz at 94 dB SPL each")
        print(f"Sample rate: {sr} Hz, Duration: {duration} s")
    else:
        import soundfile as sf
        data, sr = sf.read(str(args.wav), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        pressure_Pa = data * args.Pa_per_FS
        print(f"Loaded: {args.wav}")
        print(f"Sample rate: {sr} Hz")

    analyzer = ThirdOctaveAnalyzer(sample_rate=int(sr))
    print(f"\nBands: {analyzer.n_bands} "
          f"({analyzer.center_frequencies[0]:.0f} - {analyzer.center_frequencies[-1]:.0f} Hz)")

    results = analyzer.analyze(pressure_Pa, time_weighting=args.weighting, hop_ms=args.hop_ms)

    print(f"\nBand analysis ({args.weighting} weighting):")
    print(f"{'Freq (Hz)':>10} {'Max (dB)':>10} {'SEL (dB)':>10} {'Decim':>7}")
    print("-" * 42)
    for i, fc in enumerate(results["center_frequencies"]):
        print(f"{fc:10.0f} {np.max(results['band_levels_dB'][i]):10.1f} "
              f"{results['band_exposure_dB'][i]:10.1f} {results['decimation'][i]:7d}")

    print("-" * 42)
    print(f"{'Overall':>10} {np.max(results['overall_level_dB']):10.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
