#!/usr/bin/env python3
"""
calibration.py - Microphone/Recording Chain Calibration for Acoustic Measurements

Converts digital full-scale (FS) waveform samples to physical pressure units (Pascals).
Provides helper functions for SPL calculations using the standard reference pressure.

Calibration Methods (in descending order of traceability):
  1) Calibrator tone: measure a recording of a 94 dB / 114 dB acoustic calibrator.
     This is the only method that captures the WHOLE chain (mic + preamp + gain + ADC)
     as it was actually configured, and it is what an accredited lab does.
  2) Recording chain: mic sensitivity + preamp gain + ADC full-scale voltage.
  3) Direct: Pa_per_FS, if the factor is already known.
  4) Uncalibrated: results are RELATIVE (dB re FS) and are labelled as such everywhere.

Reference: 20 µPa (threshold of human hearing) for dB SPL calculations.

Usage:
    from calibration import Calibration, assess_signal_quality

    # Method 1 (preferred): from a calibrator tone recording
    cal = Calibration.from_calibrator_tone(tone_samples, sample_rate=48000,
                                           calibrator_level_dB=114.0)

    # Method 2: from the recording chain
    cal = Calibration.from_recording_chain(
        sensitivity_mV_per_Pa=10.0,   # microphone datasheet
        preamp_gain_dB=20.0,          # preamp / recorder input gain
        adc_full_scale_V=1.0,         # what +/-1.0 FS means in volts
    )

    # Method 4: explicitly uncalibrated (relative units)
    cal = Calibration.uncalibrated()

    pressure_Pa = cal.to_pascals(samples)
    spl_dB = amplitude_to_dB_SPL(pressure_rms)

    # Always check the recording before trusting the numbers
    qa = assess_signal_quality(samples, sample_rate, cal)
    if not qa.is_valid:
        print(qa.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Standard reference pressure for dB SPL (threshold of hearing)
P_REF: float = 20e-6  # 20 µPa

# Numerical floor to avoid log(0)
EPS: float = 1e-30

# Acoustic calibrator levels in common use (IEC 60942)
CALIBRATOR_LEVELS_dB = (94.0, 114.0, 124.0)

# A sample is treated as at-full-scale within this tolerance of 1.0.
# 24-bit PCM quantises to ~1.2e-7, so 1e-4 is comfortably above quantisation
# while still only catching genuinely pinned samples.
CLIP_TOLERANCE: float = 1e-4

# Consecutive at-full-scale samples that constitute a real clipping event
# rather than a single sample that happened to land on the rail.
CLIP_RUN_SAMPLES: int = 2

# ---- Ceiling (limiter / AGC) clipping ------------------------------------
#
# Clipping does NOT have to happen at digital full scale. Phone and camera
# recorders, field recorders with AGC, and anything that has been through a
# broadcast limiter flat-top the waveform at whatever ceiling the limiter was
# set to, and the file then arrives with comfortable-looking headroom. The
# rail test above sees nothing; the measurement is still ruined, because the
# peak is understated and rise time, crest factor and kurtosis are computed
# from a plateau that the microphone never saw.
#
# The signature is the same one ahaah._detect_clipping() looks for in a
# calibrated pressure history: several samples in a row pinned at the
# waveform's OWN extreme. These constants set what "pinned" and "several" mean.

# How close to the extreme counts as sitting on it, as a fraction of the
# extreme. 1e-4 is ~1.5 LSB at 16-bit and ~1700 LSB at 24-bit, so it absorbs
# a limiter's own dither without merging genuinely distinct sample values.
CEILING_TOLERANCE: float = 1e-4

# Consecutive at-ceiling samples that make a plateau. Three rather than the
# rail test's two: a rail is a hard boundary a signal can only touch, but a
# waveform's own maximum is by definition attained, and at high sample rates
# a rounded peak can put two samples within tolerance of it by curvature
# alone. Three in a row is flat-topping.
CEILING_RUN_SAMPLES: int = 3

# Distinct plateaus required before ceiling clipping is declared. One plateau
# anywhere in a recording is not evidence; a limiter that engaged once will
# engage on every shot in the string.
CEILING_MIN_RUNS: int = 2

# A waveform whose peak is less than this above its RMS is periodic (a square
# wave is 0 dB, a sine 3 dB) rather than impulsive, and its flat top is its
# shape rather than damage. Real gunshot recordings sit above 15 dB even
# across a whole file of mostly silence.
CEILING_MIN_CREST_dB: float = 6.0


def _validate_positive_finite(value: float, name: str) -> float:
    """
    Validate that a calibration quantity is a positive, finite real number.

    NaN and infinity both pass a naive ``<= 0`` check and then silently poison
    every downstream dB value, so they are rejected explicitly.
    """
    v = float(value)
    if math.isnan(v):
        raise ValueError(f"{name} must be a real number, got NaN")
    if math.isinf(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if v <= 0.0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


@dataclass
class Calibration:
    """
    Calibration data for converting digital samples to physical pressure (Pascals).

    Attributes:
        Pa_per_FS: Pascals per full-scale unit. Multiply float waveform [-1, 1] by this
                   to get pressure in Pascals.
        calibrated: True only when Pa_per_FS is traceable to a real measurement or
                    a real recording chain. Uncalibrated results are relative (dB re FS)
                    and must never be presented as dB SPL.
        method: Machine-readable provenance: "calibrator_tone", "recording_chain",
                "direct", "preset" or "uncalibrated".
        description: Human-readable description of the calibration source.
        reference_level_dB: For calibrator-tone calibration, the calibrator's stated level.
        residual_dB: For calibrator-tone calibration, drift between pre- and post-test
                     calibration, when both are supplied.
    """
    Pa_per_FS: float
    calibrated: bool = True
    method: str = "direct"
    description: str = ""
    reference_level_dB: Optional[float] = None
    residual_dB: Optional[float] = None

    def __post_init__(self) -> None:
        self.Pa_per_FS = _validate_positive_finite(self.Pa_per_FS, "Pa_per_FS")
        if not self.description:
            self.description = f"Direct: {self.Pa_per_FS:.6g} Pa/FS"

    # ---- Constructors ----

    @classmethod
    def from_calibrator_tone(
        cls,
        tone_samples: np.ndarray,
        sample_rate: int,
        calibrator_level_dB: float = 114.0,
        *,
        tone_frequency_Hz: float = 1000.0,
        description: str = "",
        post_test_samples: Optional[np.ndarray] = None,
    ) -> "Calibration":
        """
        Derive calibration from a recording of an acoustic calibrator (pistonphone).

        This is the preferred method: it measures the entire acquisition chain exactly
        as it was configured for the test, including any gain the operator forgot about.

        The calibrator produces a known SPL at a known frequency. Measuring the digital
        RMS of that tone gives the conversion factor directly:

            Pa_per_FS = p_ref * 10^(L_cal/20) / rms_digital

        Args:
            tone_samples: Digital samples of the calibrator tone, float in [-1, 1].
            sample_rate: Sample rate in Hz.
            calibrator_level_dB: Calibrator's stated output level (typically 94 or 114 dB).
            tone_frequency_Hz: Calibrator tone frequency, used to validate the recording.
            description: Optional description string.
            post_test_samples: Optional post-test calibrator recording. When supplied,
                               the drift between pre- and post-test is recorded as
                               residual_dB; a drift over 0.5 dB invalidates the test
                               under most measurement protocols.

        Returns:
            Calibration instance with method="calibrator_tone".

        Raises:
            ValueError: If the tone is absent, clipped, or not at the expected frequency.
        """
        rms = _measure_tone_rms(tone_samples, sample_rate, tone_frequency_Hz)
        level = float(calibrator_level_dB)
        if math.isnan(level) or math.isinf(level):
            raise ValueError(f"calibrator_level_dB must be finite, got {calibrator_level_dB}")

        target_Pa = P_REF * (10.0 ** (level / 20.0))
        Pa_per_FS = target_Pa / rms

        residual = None
        if post_test_samples is not None and len(post_test_samples) > 0:
            rms_post = _measure_tone_rms(post_test_samples, sample_rate, tone_frequency_Hz)
            residual = float(20.0 * np.log10(rms_post / rms))

        if not description:
            description = f"Calibrator tone: {level:.1f} dB @ {tone_frequency_Hz:.0f} Hz"
            if residual is not None:
                description += f" (post-test drift {residual:+.2f} dB)"

        return cls(
            Pa_per_FS=Pa_per_FS,
            calibrated=True,
            method="calibrator_tone",
            description=description,
            reference_level_dB=level,
            residual_dB=residual,
        )

    @classmethod
    def from_recording_chain(
        cls,
        sensitivity_mV_per_Pa: float,
        adc_full_scale_V: float = 1.0,
        preamp_gain_dB: float = 0.0,
        description: str = "",
    ) -> "Calibration":
        """
        Create calibration from the physical recording chain.

        This models what a user can actually read off their equipment:
          - the microphone datasheet gives sensitivity in mV/Pa (or dB re 1 V/Pa),
          - the preamp/recorder front panel gives a gain in dB,
          - the recorder specification gives the input voltage that corresponds to 0 dBFS.

        Chain:
            V_at_ADC = p_Pa * (S_mV/Pa / 1000) * 10^(gain_dB/20)
            FS       = V_at_ADC / adc_full_scale_V
        so
            Pa_per_FS = adc_full_scale_V / (S_V/Pa * 10^(gain_dB/20))

        Args:
            sensitivity_mV_per_Pa: Microphone sensitivity in mV/Pa (e.g. 10 mV/Pa = -40 dB re 1V/Pa).
            adc_full_scale_V: Input voltage corresponding to digital full scale (0 dBFS).
            preamp_gain_dB: Preamp / recorder input gain applied between mic and ADC.
            description: Optional description string.

        Returns:
            Calibration instance with method="recording_chain".
        """
        sens_mV = _validate_positive_finite(sensitivity_mV_per_Pa, "sensitivity_mV_per_Pa")
        fs_V = _validate_positive_finite(adc_full_scale_V, "adc_full_scale_V")
        gain = float(preamp_gain_dB)
        if math.isnan(gain) or math.isinf(gain):
            raise ValueError(f"preamp_gain_dB must be finite, got {preamp_gain_dB}")

        sensitivity_V_per_Pa = (sens_mV / 1000.0) * (10.0 ** (gain / 20.0))
        Pa_per_FS = fs_V / sensitivity_V_per_Pa

        if not description:
            description = (
                f"Chain: {sens_mV:g} mV/Pa, {gain:+g} dB gain, {fs_V:g} V full scale"
            )

        return cls(
            Pa_per_FS=Pa_per_FS,
            calibrated=True,
            method="recording_chain",
            description=description,
        )

    @classmethod
    def from_sensitivity(
        cls,
        sensitivity_mV_per_Pa: float,
        V_per_FS: float,
        description: str = "",
    ) -> "Calibration":
        """
        Create calibration from microphone sensitivity and recorder full-scale voltage.

        Retained for backward compatibility; equivalent to from_recording_chain() with
        no preamp gain. Prefer from_recording_chain() or from_calibrator_tone().
        """
        return cls.from_recording_chain(
            sensitivity_mV_per_Pa=sensitivity_mV_per_Pa,
            adc_full_scale_V=V_per_FS,
            preamp_gain_dB=0.0,
            description=description,
        )

    @classmethod
    def from_dB_sensitivity(
        cls,
        sensitivity_dB_re_1V_per_Pa: float,
        V_per_FS: float,
        preamp_gain_dB: float = 0.0,
        description: str = "",
    ) -> "Calibration":
        """
        Create calibration from microphone sensitivity in dB re 1V/Pa.

        Args:
            sensitivity_dB_re_1V_per_Pa: Sensitivity in dB re 1V/Pa.
                                         Typical values: -40 to -26 dB for measurement mics.
            V_per_FS: Recorder full-scale voltage.
            preamp_gain_dB: Preamp / recorder input gain.
            description: Optional description string.
        """
        s = float(sensitivity_dB_re_1V_per_Pa)
        if math.isnan(s) or math.isinf(s):
            raise ValueError(f"sensitivity_dB_re_1V_per_Pa must be finite, got {s}")

        sensitivity_mV_per_Pa = (10.0 ** (s / 20.0)) * 1000.0

        if not description:
            description = (
                f"Chain: {s:g} dB re 1V/Pa, {preamp_gain_dB:+g} dB gain, {V_per_FS:g} V full scale"
            )

        return cls.from_recording_chain(
            sensitivity_mV_per_Pa=sensitivity_mV_per_Pa,
            adc_full_scale_V=V_per_FS,
            preamp_gain_dB=preamp_gain_dB,
            description=description,
        )

    @classmethod
    def preset(cls, Pa_per_FS: float, name: str, provenance: str) -> "Calibration":
        """
        Create a named, dated calibration preset.

        A preset is a real calibration that was measured at some point for a specific
        rig. It is only valid for that rig, so its provenance travels with it into
        every report.

        Args:
            Pa_per_FS: The measured conversion factor.
            name: Short preset name shown in the UI.
            provenance: Where this number came from (source recording, date, hardware).
        """
        return cls(
            Pa_per_FS=Pa_per_FS,
            calibrated=True,
            method="preset",
            description=f"Preset '{name}': {provenance}",
        )

    @classmethod
    def uncalibrated(cls) -> "Calibration":
        """
        Return a unit calibration (Pa_per_FS=1.0) for uncalibrated analysis.

        Results are RELATIVE (dB re FS), not dB SPL. The calibrated flag is False so
        that every consumer can label output correctly rather than inferring it from
        a description string.
        """
        return cls(
            Pa_per_FS=1.0,
            calibrated=False,
            method="uncalibrated",
            description="UNCALIBRATED - relative units (dB re FS)",
        )

    # ---- Use ----

    def to_pascals(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert digital samples (float, nominally [-1, 1]) to pressure in Pascals.

        For an uncalibrated instance this is a pass-through and the result is in
        full-scale units, not Pascals.
        """
        return np.asarray(samples, dtype=np.float64) * self.Pa_per_FS

    def is_calibrated(self) -> bool:
        """Whether results from this calibration are true dB SPL rather than relative."""
        return self.calibrated

    @property
    def level_unit(self) -> str:
        """Unit string for levels derived from this calibration."""
        return "dB SPL" if self.calibrated else "dB re FS"

    @property
    def full_scale_dB(self) -> float:
        """
        Level corresponding to a full-scale sample.

        This is the ceiling of the instrument: no measurement from this recording can
        legitimately exceed it, and a detection threshold above it can never trigger.
        """
        return float(20.0 * np.log10(self.Pa_per_FS / P_REF))

    def to_dict(self) -> dict:
        """Serialise for the analysis provenance record."""
        return {
            "Pa_per_FS": self.Pa_per_FS,
            "calibrated": self.calibrated,
            "method": self.method,
            "description": self.description,
            "level_unit": self.level_unit,
            "full_scale_dB": round(self.full_scale_dB, 2),
            "reference_level_dB": self.reference_level_dB,
            "residual_dB": self.residual_dB,
        }


def _measure_tone_rms(
    samples: np.ndarray,
    sample_rate: int,
    expected_frequency_Hz: float,
    *,
    tolerance_ratio: float = 0.05,
) -> float:
    """
    Measure the RMS of a steady calibrator tone, validating that it really is one.

    The middle 60% of the recording is used so that switch-on transients and the
    operator removing the calibrator do not bias the result.

    Raises:
        ValueError: If the recording is too short, clipped, silent, or the dominant
                    frequency is not the expected calibrator frequency.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    if n < sample_rate // 10:
        raise ValueError(
            f"Calibrator recording is too short ({n} samples); need at least 100 ms"
        )

    # Use the steady middle portion
    lo, hi = int(n * 0.2), int(n * 0.8)
    seg = x[lo:hi]

    peak = float(np.max(np.abs(seg)))
    if peak >= 1.0 - CLIP_TOLERANCE:
        raise ValueError(
            "Calibrator tone is clipped; reduce input gain and re-record the calibration"
        )
    if peak < 1e-6:
        raise ValueError("No calibrator tone found (recording is silent)")

    # Validate the dominant frequency really is the calibrator
    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sample_rate)
    dominant = float(freqs[int(np.argmax(spectrum))])
    if abs(dominant - expected_frequency_Hz) > tolerance_ratio * expected_frequency_Hz:
        raise ValueError(
            f"Dominant frequency is {dominant:.1f} Hz, expected {expected_frequency_Hz:.0f} Hz "
            f"(+/-{tolerance_ratio*100:.0f}%). This does not look like a calibrator tone."
        )

    rms = float(np.sqrt(np.mean(seg ** 2)))
    if rms < EPS:
        raise ValueError("Calibrator tone RMS is zero")
    return rms


# ---- Recording quality assessment ----

@dataclass
class SignalQuality:
    """
    Measurement-validity assessment of a recording.

    A gunshot recording that is clipped, DC-offset, or barely above the noise floor
    produces numbers that look precise and are wrong. This captures the checks that
    decide whether the measurement is admissible at all.
    """
    n_samples: int
    sample_rate: int
    duration_s: float

    peak_FS: float                # peak absolute sample, in full-scale units
    headroom_dB: float            # dB between peak and full scale
    clipped_samples: int
    clipped_runs: int
    clipping_ratio: float         # fraction of samples at full scale

    dc_offset_FS: float
    dc_offset_dB: float           # DC relative to signal RMS

    noise_floor_dB: float         # estimated from the quietest 5% of the recording
    peak_level_dB: float
    snr_dB: float

    lf_energy_fraction: float     # fraction of energy below 20 Hz (wind/handling)

    nyquist_Hz: float
    sample_rate_adequate: bool

    # Flat-topping below full scale. Kept separate from clipped_samples so the
    # two causes stay distinguishable in the record: one is a converter that
    # ran out of range, the other is a limiter that was left switched on.
    ceiling: "CeilingClipping" = field(default_factory=lambda: CeilingClipping())

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """False when a hard validity check failed and results must not be trusted."""
        return not self.errors

    @property
    def is_clipped(self) -> bool:
        """True for either cause: the digital rail, or a limiter below it."""
        return self.clipped_runs > 0 or self.ceiling.detected

    def summary(self) -> str:
        """Human-readable QA summary."""
        lines = [
            f"  Peak:          {self.peak_level_dB:.1f} dB  ({self.headroom_dB:.1f} dB headroom)",
            f"  Noise floor:   {self.noise_floor_dB:.1f} dB  (SNR {self.snr_dB:.1f} dB)",
            f"  DC offset:     {self.dc_offset_FS:+.2e} FS  ({self.dc_offset_dB:.1f} dB re signal)",
            f"  Sub-20 Hz:     {self.lf_energy_fraction*100:.1f}% of total energy",
            f"  Sample rate:   {self.sample_rate} Hz (Nyquist {self.nyquist_Hz:.0f} Hz)",
        ]
        if self.clipped_runs:
            lines.append(
                f"  CLIPPING:      {self.clipped_samples} samples in {self.clipped_runs} runs "
                f"({self.clipping_ratio*100:.4f}%)"
            )
        if self.ceiling.detected:
            lines.append(f"  LIMITER:       {self.ceiling.describe()}")
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "peak_FS": round(self.peak_FS, 6),
            "peak_level_dB": round(self.peak_level_dB, 2),
            "headroom_dB": round(self.headroom_dB, 2),
            "clipped_samples": self.clipped_samples,
            "clipped_runs": self.clipped_runs,
            "clipping_ratio": self.clipping_ratio,
            "is_clipped": self.is_clipped,
            "ceiling_clipped": self.ceiling.detected,
            "ceiling_dBFS": round(self.ceiling.ceiling_dBFS, 2) if self.ceiling.detected else None,
            "ceiling_samples": self.ceiling.samples,
            "ceiling_runs": self.ceiling.runs,
            "dc_offset_FS": self.dc_offset_FS,
            "dc_offset_dB": round(self.dc_offset_dB, 2),
            "noise_floor_dB": round(self.noise_floor_dB, 2),
            "snr_dB": round(self.snr_dB, 2),
            "lf_energy_fraction": round(self.lf_energy_fraction, 4),
            "sample_rate": self.sample_rate,
            "sample_rate_adequate": self.sample_rate_adequate,
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def detect_clipping(
    samples_FS: np.ndarray,
    *,
    tolerance: float = CLIP_TOLERANCE,
    min_run: int = CLIP_RUN_SAMPLES,
) -> tuple[int, int]:
    """
    Detect digital clipping in a full-scale waveform.

    Clipping is identified as runs of consecutive samples pinned at full scale. A
    single sample touching the rail is legitimate; two or more in a row is the
    flat-topping signature of a saturated converter.

    Args:
        samples_FS: Samples in full-scale units (NOT Pascals), nominally [-1, 1].
        tolerance: How close to 1.0 counts as full scale.
        min_run: Minimum consecutive at-rail samples to count as a clipping event.

    Returns:
        (clipped_sample_count, clipped_run_count)
    """
    x = np.abs(np.asarray(samples_FS, dtype=np.float64))
    if x.size == 0:
        return 0, 0

    at_rail = x >= (1.0 - tolerance)
    if not np.any(at_rail):
        return 0, 0

    # Find run boundaries
    padded = np.concatenate(([False], at_rail, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    lengths = ends - starts

    qualifying = lengths >= min_run
    return int(lengths[qualifying].sum()), int(qualifying.sum())


@dataclass
class CeilingClipping:
    """
    Flat-topping at a ceiling BELOW digital full scale.

    ``detected`` is the only thing a caller should branch on; the rest is the
    evidence, so a report can say what was seen rather than just asserting it.
    """
    ceiling_FS: float = 0.0        # the level the waveform is pinned at
    ceiling_dBFS: float = 0.0      # the same, as dB below full scale
    samples: int = 0               # samples inside qualifying plateaus
    runs: int = 0                  # number of qualifying plateaus
    longest_run: int = 0
    crest_dB: float = 0.0
    detected: bool = False

    def describe(self) -> str:
        """The evidence, as a phrase that can be dropped into a sentence."""
        return (
            f"{self.samples} samples pinned there, in {self.runs} plateaus, "
            f"the longest {self.longest_run} samples long"
        )


def _pinned_runs(x_abs: np.ndarray, ceiling: float, tolerance: float) -> np.ndarray:
    """Lengths of the maximal runs of samples sitting within tolerance of ceiling."""
    at = np.abs(x_abs - ceiling) <= ceiling * tolerance
    if not np.any(at):
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(([False], at, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return (edges[1::2] - edges[0::2]).astype(np.int64)


def detect_ceiling_clipping(
    samples_FS: np.ndarray,
    *,
    ceiling: Optional[float] = None,
    tolerance: float = CEILING_TOLERANCE,
    min_run: int = CEILING_RUN_SAMPLES,
    min_runs: int = CEILING_MIN_RUNS,
    min_crest_dB: float = CEILING_MIN_CREST_dB,
) -> CeilingClipping:
    """
    Detect a limiter or AGC ceiling: plateaus at the waveform's own extreme.

    ``detect_clipping`` above only sees the digital rail. A recorder that
    limits at, say, -3 dBFS produces a file with 3 dB of apparent headroom and
    a destroyed peak, and passes the rail test untouched. This finds that.

    Args:
        samples_FS: Samples in full-scale units, nominally [-1, 1].
        ceiling: The level to test, when the caller already knows the global
                 extreme (a chunked reader does). Defaults to this block's own.
        tolerance: Fractional distance from the ceiling that still counts as on it.
        min_run: Consecutive at-ceiling samples that make a plateau.
        min_runs: Plateaus required before clipping is declared.
        min_crest_dB: Peak-to-RMS below which the waveform is periodic rather
                      than impulsive, and its flat top is its shape.

    Returns:
        CeilingClipping. ``detected`` is False for a signal that is empty,
        silent, periodic, or already pinned at the digital rail — the rail is
        ``detect_clipping``'s job and reporting it twice helps nobody.
    """
    x = np.asarray(samples_FS, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < min_run:
        return CeilingClipping()

    x_abs = np.abs(x)
    peak = float(x_abs.max())
    top = float(ceiling) if ceiling is not None else peak
    if top <= EPS:
        return CeilingClipping()

    rms = float(np.sqrt(np.mean(x ** 2)))
    crest_dB = float(20.0 * np.log10(max(peak, EPS) / max(rms, EPS)))

    lengths = _pinned_runs(x_abs, top, tolerance)
    qualifying = lengths[lengths >= min_run]

    found = CeilingClipping(
        ceiling_FS=top,
        ceiling_dBFS=float(20.0 * np.log10(max(top, EPS))),
        samples=int(qualifying.sum()),
        runs=int(qualifying.size),
        longest_run=int(lengths.max()) if lengths.size else 0,
        crest_dB=crest_dB,
    )
    found.detected = (
        found.runs >= min_runs
        and top < 1.0 - CLIP_TOLERANCE          # the rail is the other test's
        and crest_dB >= min_crest_dB
    )
    return found


class CeilingClippingScan:
    """
    Streaming form of detect_ceiling_clipping, for a file read in chunks.

    A chunked reader cannot know the global extreme until it has read the
    whole file, and the ceiling has to be global: a quiet chunk's own maximum
    is not a limiter ceiling. So each chunk is measured against the highest
    extreme seen so far, and the moment a louder chunk arrives the tally
    starts again from it. The result is identical to running the whole-file
    detector once the last chunk has been fed.
    """

    def __init__(self, **options) -> None:
        self._options = options
        self._ceiling = 0.0
        self._samples = 0
        self._runs = 0
        self._longest = 0
        self._peak = 0.0
        self._sq_sum = 0.0
        self._n = 0

    def feed(self, block: np.ndarray) -> None:
        x = np.asarray(block, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size == 0:
            return
        self._sq_sum += float(np.sum(x ** 2))
        self._n += x.size

        x_abs = np.abs(x)
        block_peak = float(x_abs.max())
        self._peak = max(self._peak, block_peak)

        tolerance = self._options.get("tolerance", CEILING_TOLERANCE)
        min_run = self._options.get("min_run", CEILING_RUN_SAMPLES)

        # A louder chunk redefines the ceiling; everything counted against the
        # old, lower one was not a ceiling at all.
        if block_peak > self._ceiling * (1.0 + tolerance):
            self._ceiling = block_peak
            self._samples = self._runs = self._longest = 0
        elif block_peak < self._ceiling * (1.0 - tolerance):
            return      # nothing in this chunk can reach the ceiling

        lengths = _pinned_runs(x_abs, self._ceiling, tolerance)
        if not lengths.size:
            return
        qualifying = lengths[lengths >= min_run]
        self._samples += int(qualifying.sum())
        self._runs += int(qualifying.size)
        self._longest = max(self._longest, int(lengths.max()))

    def result(self) -> CeilingClipping:
        if self._n == 0 or self._ceiling <= EPS:
            return CeilingClipping()
        rms = math.sqrt(self._sq_sum / self._n)
        crest_dB = float(20.0 * math.log10(max(self._peak, EPS) / max(rms, EPS)))
        found = CeilingClipping(
            ceiling_FS=self._ceiling,
            ceiling_dBFS=float(20.0 * math.log10(max(self._ceiling, EPS))),
            samples=self._samples,
            runs=self._runs,
            longest_run=self._longest,
            crest_dB=crest_dB,
        )
        found.detected = (
            found.runs >= self._options.get("min_runs", CEILING_MIN_RUNS)
            and self._ceiling < 1.0 - CLIP_TOLERANCE
            and crest_dB >= self._options.get("min_crest_dB", CEILING_MIN_CREST_dB)
        )
        return found


def ceiling_clipping_error(found: CeilingClipping) -> str:
    """The operator-facing sentence for a detected limiter ceiling."""
    return (
        f"Recording is CLIPPED at {found.ceiling_dBFS:.1f} dBFS - {found.describe()}. "
        f"The flat top is below digital full scale, so this is a limiter or automatic "
        f"gain control in the recording chain rather than a converter that ran out of "
        f"range: the file appears to have {abs(found.ceiling_dBFS):.1f} dB of headroom "
        f"and has none. Peak level is understated by an unknown amount, and rise time, "
        f"crest factor and kurtosis describe the plateau rather than the blast. "
        f"Re-record with limiting and AGC switched off."
    )


def assess_signal_quality(
    samples_FS: np.ndarray,
    sample_rate: int,
    calibration: Calibration,
    *,
    min_snr_dB: float = 20.0,
    min_headroom_dB: float = 1.0,
    min_sample_rate: int = 48000,
) -> SignalQuality:
    """
    Assess whether a recording can support a defensible acoustic measurement.

    Args:
        samples_FS: Digital samples in full-scale units, nominally [-1, 1].
        sample_rate: Sample rate in Hz.
        calibration: Calibration used to express levels.
        min_snr_dB: Below this peak-to-noise-floor ratio the measurement is flagged.
        min_headroom_dB: Below this headroom the recording is flagged as at risk.
        min_sample_rate: Rate below which muzzle-blast rise time cannot be resolved.

    Returns:
        SignalQuality with populated warnings and errors.
    """
    x = np.asarray(samples_FS, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    n = len(x)
    warnings: List[str] = []
    errors: List[str] = []

    if n == 0:
        return SignalQuality(
            n_samples=0, sample_rate=sample_rate, duration_s=0.0,
            peak_FS=0.0, headroom_dB=0.0, clipped_samples=0, clipped_runs=0,
            clipping_ratio=0.0, dc_offset_FS=0.0, dc_offset_dB=0.0,
            noise_floor_dB=0.0, peak_level_dB=0.0, snr_dB=0.0,
            lf_energy_fraction=0.0, nyquist_Hz=sample_rate / 2.0,
            sample_rate_adequate=False,
            errors=["Recording is empty"],
        )

    peak_FS = float(np.max(np.abs(x)))
    headroom_dB = float(-20.0 * np.log10(max(peak_FS, EPS)))

    clipped_samples, clipped_runs = detect_clipping(x)
    clipping_ratio = clipped_samples / n
    ceiling = detect_ceiling_clipping(x)

    # DC offset
    dc = float(np.mean(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    dc_offset_dB = float(20.0 * np.log10(max(abs(dc), EPS) / max(rms, EPS)))

    # Level estimates, expressed through the calibration
    pressure = x * calibration.Pa_per_FS
    peak_level_dB = float(amplitude_to_dB_SPL(np.max(np.abs(pressure))))

    # Noise floor from the quietest 5% of short frames
    frame = max(1, sample_rate // 100)  # 10 ms
    n_frames = n // frame
    if n_frames >= 20:
        frames = pressure[: n_frames * frame].reshape(n_frames, frame)
        frame_ms = np.mean(frames ** 2, axis=1)
        quiet = np.percentile(frame_ms, 5.0)
        noise_floor_dB = float(power_to_dB_SPL(max(quiet, EPS)))
    else:
        noise_floor_dB = float(power_to_dB_SPL(max(float(np.mean(pressure ** 2)), EPS)))
    snr_dB = peak_level_dB - noise_floor_dB

    # Sub-20 Hz energy fraction (wind, handling, mic-mount rumble)
    lf_fraction = _low_frequency_energy_fraction(x, sample_rate, cutoff_Hz=20.0)

    nyquist = sample_rate / 2.0
    sample_rate_adequate = sample_rate >= min_sample_rate

    # ---- Validity rules ----
    if clipped_runs > 0:
        errors.append(
            f"Recording is CLIPPED ({clipped_samples} samples in {clipped_runs} runs). "
            f"Peak levels are understated and rise time, crest factor and kurtosis are invalid. "
            f"Re-record with lower input gain."
        )
    elif ceiling.detected:
        errors.append(ceiling_clipping_error(ceiling))
    elif headroom_dB < min_headroom_dB:
        warnings.append(
            f"Only {headroom_dB:.1f} dB of headroom; the recording is close to clipping."
        )

    if not calibration.calibrated:
        warnings.append(
            "No calibration supplied. All levels are RELATIVE (dB re FS), not dB SPL."
        )

    if snr_dB < min_snr_dB:
        warnings.append(
            f"Peak is only {snr_dB:.1f} dB above the noise floor "
            f"(want >= {min_snr_dB:.0f} dB); energy metrics will be noise-biased."
        )

    if abs(dc) > 1e-3:
        warnings.append(
            f"DC offset of {dc:+.2e} FS detected; it inflates Z-weighted levels and "
            f"corrupts rise-time detection. A high-pass will be applied."
        )

    if lf_fraction > 0.5:
        warnings.append(
            f"{lf_fraction*100:.0f}% of signal energy is below 20 Hz, which usually means "
            f"wind or handling noise rather than blast. Use a windscreen."
        )

    if not sample_rate_adequate:
        warnings.append(
            f"Sample rate {sample_rate} Hz cannot resolve muzzle-blast rise time "
            f"(one sample = {1e6/sample_rate:.1f} us vs typical 1-50 us rise). "
            f"Use >= {min_sample_rate} Hz."
        )

    return SignalQuality(
        n_samples=n,
        sample_rate=sample_rate,
        duration_s=n / sample_rate,
        peak_FS=peak_FS,
        headroom_dB=headroom_dB,
        clipped_samples=clipped_samples,
        clipped_runs=clipped_runs,
        clipping_ratio=clipping_ratio,
        dc_offset_FS=dc,
        dc_offset_dB=dc_offset_dB,
        noise_floor_dB=noise_floor_dB,
        peak_level_dB=peak_level_dB,
        snr_dB=snr_dB,
        lf_energy_fraction=lf_fraction,
        nyquist_Hz=nyquist,
        sample_rate_adequate=sample_rate_adequate,
        ceiling=ceiling,
        warnings=warnings,
        errors=errors,
    )


def _low_frequency_energy_fraction(
    x: np.ndarray,
    sample_rate: int,
    cutoff_Hz: float = 20.0,
) -> float:
    """Fraction of total signal energy below cutoff_Hz, via Parseval on the FFT."""
    n = len(x)
    if n < 16:
        return 0.0
    # Decimate long signals; sub-20 Hz content survives heavy decimation.
    step = max(1, n // 1_000_000)
    seg = x[::step]
    eff_rate = sample_rate / step
    if eff_rate <= 2 * cutoff_Hz:
        return 0.0

    seg = seg - np.mean(seg)
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / eff_rate)
    total = float(np.sum(spectrum))
    if total < EPS:
        return 0.0
    return float(np.sum(spectrum[freqs < cutoff_Hz]) / total)


def remove_dc_offset(
    x: np.ndarray,
    sample_rate: int,
    cutoff_Hz: float = 10.0,
) -> np.ndarray:
    """
    Remove DC and infrasonic content ahead of level computation.

    Z-weighting is a bare pass-through, so any DC offset or sub-audio rumble is
    integrated straight into Z-weighted levels and shifts the |p| baseline that
    rise-time and B-duration detection depend on.

    A 2nd-order zero-phase Butterworth high-pass is used so the shock front is not
    delayed or asymmetrically distorted.

    The signal is extended with silence before filtering rather than using
    sosfiltfilt's default odd reflection. A shot window that begins at or near its
    peak presents a step at the array edge, and reflecting it fabricates a large
    transient: on a synthetic Friedlander blast the default padding cost 3.4 dB of
    peak and left a tail 60 orders of magnitude above the true one. Silence is also
    the physically correct extension, since pressure was ambient before the blast.

    Args:
        x: Input signal.
        sample_rate: Sample rate in Hz.
        cutoff_Hz: High-pass corner. 10 Hz sits below the 20 Hz measurement band.

    Returns:
        High-passed signal.
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if x.size < 32 or cutoff_Hz <= 0:
        return x - np.mean(x)

    nyq = sample_rate / 2.0
    wn = cutoff_Hz / nyq
    if not (0 < wn < 1):
        return x - np.mean(x)

    # Remove DC exactly first: the zero extension below cannot represent a genuine
    # offset, so subtracting the mean up front keeps both cases correct.
    x = x - np.mean(x)

    sos = butter(2, wn, btype="highpass", output="sos")

    # Lead-in long enough for the filter to settle: five time constants of the corner.
    pad = int(np.ceil(5.0 * sample_rate / (2.0 * np.pi * cutoff_Hz)))
    pad = max(pad, 3 * (2 * sos.shape[0] + 1))
    if pad >= 10 * x.size:
        pad = max(1, x.size)

    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    filtered = np.asarray(sosfiltfilt(sos, padded, padtype=None))
    return filtered[pad:pad + x.size]


# ---- Level conversion helpers ----

def amplitude_to_dB_SPL(amplitude_Pa: np.ndarray | float, eps: float = EPS) -> np.ndarray | float:
    """
    Convert instantaneous or RMS pressure amplitude to dB SPL.

    Args:
        amplitude_Pa: Pressure in Pascals (RMS or instantaneous magnitude).
        eps: Small floor value to avoid log(0).

    Returns:
        Sound pressure level in dB re 20 µPa.

    Note:
        For true SPL measurements, use RMS pressure over an appropriate
        time window (e.g. Fast = 125ms, Slow = 1s).
    """
    amp = np.asarray(amplitude_Pa, dtype=np.float64)
    return 20.0 * np.log10(np.maximum(np.abs(amp), eps) / P_REF)


def power_to_dB_SPL(power_Pa2: np.ndarray | float, eps: float = EPS) -> np.ndarray | float:
    """
    Convert mean-square pressure (Pa²) to dB SPL.

    Args:
        power_Pa2: Mean-square pressure in Pa².
        eps: Small floor value to avoid log(0).

    Returns:
        Sound pressure level in dB re 20 µPa.
    """
    pwr = np.asarray(power_Pa2, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(pwr, eps) / (P_REF ** 2))


def dB_SPL_to_amplitude(dB_SPL: np.ndarray | float) -> np.ndarray | float:
    """
    Convert dB SPL back to RMS pressure amplitude in Pascals.

    Args:
        dB_SPL: Sound pressure level in dB re 20 µPa.

    Returns:
        RMS pressure in Pascals.
    """
    return P_REF * (10.0 ** (np.asarray(dB_SPL, dtype=np.float64) / 20.0))


def compute_rms(samples: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """
    Compute RMS (root-mean-square) of samples.

    Args:
        samples: Input array.
        axis: Axis along which to compute RMS. None = entire array.

    Returns:
        A scalar when axis is None, otherwise one RMS value per slice.
    """
    x = np.asarray(samples, dtype=np.float64)
    return np.sqrt(np.mean(x ** 2, axis=axis))


def compute_peak(samples: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """
    Compute peak absolute value of samples.

    Args:
        samples: Input array.
        axis: Axis along which to compute peak. None = entire array.

    Returns:
        A scalar when axis is None, otherwise one peak per slice.
    """
    x = np.asarray(samples, dtype=np.float64)
    return np.max(np.abs(x), axis=axis)


def energy_average_dB(levels_dB: np.ndarray | List[float]) -> float:
    """
    Energy-average a set of levels, per ISO convention.

        L_avg = 10 * log10( mean( 10^(L_i / 10) ) )

    Arithmetic averaging of decibels understates the true energy mean and is not
    a valid summary of sound levels.
    """
    arr = np.asarray(levels_dB, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(10.0 ** (arr / 10.0))))


# ---- CLI for testing ----

def main() -> int:
    """Test calibration module."""
    import argparse

    parser = argparse.ArgumentParser(description="Test calibration calculations")
    parser.add_argument("--Pa-per-FS", type=float, default=None, help="Direct Pa/FS calibration")
    parser.add_argument("--sensitivity-mV", type=float, default=None, help="Mic sensitivity in mV/Pa")
    parser.add_argument("--sensitivity-dB", type=float, default=None, help="Mic sensitivity in dB re 1V/Pa")
    parser.add_argument("--preamp-gain-dB", type=float, default=0.0, help="Preamp gain in dB")
    parser.add_argument("--V-per-FS", type=float, default=1.0, help="ADC full-scale voltage (default: 1.0)")
    parser.add_argument("--tone", type=str, default=None, help="Calibrator tone WAV file")
    parser.add_argument("--tone-level-dB", type=float, default=114.0, help="Calibrator level (default: 114)")
    args = parser.parse_args()

    if args.tone is not None:
        import soundfile as sf
        data, sr = sf.read(args.tone, dtype="float64")
        cal = Calibration.from_calibrator_tone(data, sr, args.tone_level_dB)
    elif args.Pa_per_FS is not None:
        cal = Calibration(Pa_per_FS=args.Pa_per_FS)
    elif args.sensitivity_mV is not None:
        cal = Calibration.from_recording_chain(args.sensitivity_mV, args.V_per_FS, args.preamp_gain_dB)
    elif args.sensitivity_dB is not None:
        cal = Calibration.from_dB_sensitivity(args.sensitivity_dB, args.V_per_FS, args.preamp_gain_dB)
    else:
        cal = Calibration.uncalibrated()

    print(f"Method:        {cal.method}")
    print(f"Description:   {cal.description}")
    print(f"Pa per FS:     {cal.Pa_per_FS:.6g}")
    print(f"Calibrated:    {cal.is_calibrated()}")
    print(f"Level unit:    {cal.level_unit}")
    print(f"Full scale:    {cal.full_scale_dB:.1f} {cal.level_unit}")

    print("\nReference levels:")
    print(f"  P_REF = {P_REF:.2e} Pa (0 dB SPL)")
    print(f"  1 Pa   = {amplitude_to_dB_SPL(1.0):.1f} dB SPL")
    print(f"  20 Pa  = {amplitude_to_dB_SPL(20.0):.1f} dB SPL")
    print(f"  200 Pa = {amplitude_to_dB_SPL(200.0):.1f} dB SPL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
