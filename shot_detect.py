#!/usr/bin/env python3
"""
shot_detect.py - Gunshot Event Detection for Acoustic Analysis

Detects impulsive events (gunshots) in pressure waveforms using an onset detector
with hysteresis, prominence-based peak picking, and a refractory period.

Detection strategy:
  1. Band-limit to the blast band (rejects wind, handling and mains hum).
  2. Compute a short-term RMS envelope.
  3. Pick candidate peaks by prominence, not by contiguous-region maxima.
  4. Resolve the refractory period by KEEPING THE LOUDEST candidate in each
     neighbourhood, and report how many candidates it suppressed.
  5. Refine each peak to the true sample maximum and extract a window.

    Why prominence-based picking rather than region maxima.

    Collapsing each contiguous above-threshold region to a single peak silently
    merges shots whose reverberant tails overlap - exactly what happens in a
    rapid-fire string, which is the normal case here. Prominence-based picking
    finds each blast independently, and the refractory stage then reports its
    suppressions instead of discarding them without trace.

Usage:
    from shot_detect import detect_shots

    shots = detect_shots(
        pressure_Pa,
        sample_rate=96000,
        threshold_relative_dB=25.0,   # 25 dB below the loudest event
        refractory_ms=60.0,
    )
    for shot in shots:
        print(f"Shot {shot.shot_number} at {shot.time_s:.3f}s, peak={shot.peak_dB:.1f} dB")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from calibration import EPS, P_REF, amplitude_to_dB_SPL, detect_clipping

# Blast band for detection. Muzzle blast energy lives well inside this; wind,
# handling noise and mains hum sit below it, and it excludes ultrasonic hiss.
DETECTION_BAND_LOW_HZ: float = 50.0
DETECTION_BAND_HIGH_HZ: float = 8000.0

# Default refractory. A 200 ms refractory cannot represent semi-auto fire, let alone
# automatic fire: 600 rpm is one round every 100 ms.
DEFAULT_REFRACTORY_MS: float = 50.0

# Default detection threshold, in dB below the loudest event in the recording.
# A RELATIVE threshold is the default because an absolute dB SPL threshold is
# meaningless without calibration: with Pa_per_FS = 1.0 the loudest representable
# sample is only 94 dB, so any absolute threshold above that detects nothing at all.
DEFAULT_THRESHOLD_RELATIVE_DB: float = 30.0


@dataclass
class ShotEvent:
    """
    Detected gunshot event with timing and window information.

    Attributes:
        index: Sample index of the detected peak in the original signal.
        time_s: Time of the peak in seconds.
        peak_Pa: Peak absolute pressure in Pascals.
        peak_dB: Peak level (dB SPL when calibrated, dB re FS otherwise).
        window_start: Start sample index of the extraction window.
        window_end: End sample index of the extraction window (exclusive).
        shot_number: Sequential shot number (1-based).
        truncated: Window hit the start or end of the available signal.
        clipped: Samples within the window were at digital full scale.
        snr_dB: Peak level above the local noise floor.
        arrivals: Additional distinct arrivals within the window (e.g. the ballistic
                  crack of a supersonic round alongside the muzzle blast).
    """
    index: int
    time_s: float
    peak_Pa: float
    peak_dB: float
    window_start: int
    window_end: int
    shot_number: int = 0
    truncated: bool = False
    clipped: bool = False
    snr_dB: float = float("inf")
    arrivals: List["Arrival"] = field(default_factory=list)

    # Retained for backward compatibility with older callers
    @property
    def peak_dB_SPL(self) -> float:
        return self.peak_dB

    def extract_window(self, signal: np.ndarray) -> np.ndarray:
        """Extract the windowed signal for this shot."""
        return np.asarray(signal)[self.window_start:self.window_end].copy()

    def window_duration_s(self, sample_rate: int) -> float:
        """Duration of the extraction window in seconds."""
        return (self.window_end - self.window_start) / sample_rate

    @property
    def has_multiple_arrivals(self) -> bool:
        return len(self.arrivals) > 1

    def to_dict(self) -> dict:
        return {
            "shot_number": self.shot_number,
            "time_s": round(self.time_s, 5),
            "peak_Pa": round(self.peak_Pa, 4),
            "peak_dB": round(self.peak_dB, 1),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "truncated": self.truncated,
            "clipped": self.clipped,
            "snr_dB": round(self.snr_dB, 1) if math.isfinite(self.snr_dB) else None,
            "arrivals": [a.to_dict() for a in self.arrivals],
        }


@dataclass
class Arrival:
    """
    One distinct acoustic arrival within a shot window.

    A supersonic round produces TWO events at most microphone positions: the
    ballistic crack (the projectile's N-wave) and the muzzle blast. A suppressor
    acts only on the muzzle blast, so reporting a single peak that happens to be the
    crack credits the suppressor with nothing and misrepresents the measurement.
    """
    offset_s: float          # relative to the window start
    peak_Pa: float
    peak_dB: float
    label: str = "unclassified"   # "crack", "blast" or "unclassified"

    # How the label was arrived at, so a guess is never mistaken for a deduction.
    #   "geometric" - the separation matched the Mach-cone delay computed from
    #                 the recorded distance, angle and muzzle velocity
    #   "subsonic"  - the round cannot produce a crack, so the arrival is blast
    #   "spectral"  - the two arrivals differ in BOTH centre frequency and decay
    #                 in the directions an N-wave and a blast must differ
    #   "ordering"  - only the loudest-arrival heuristic was available
    #   "none"      - not labelled
    basis: str = "none"
    ambiguity: str = ""      # why a label was withheld, when it was

    # Character measurements the classification is based on.
    centroid_Hz: float = float("nan")
    decay_ms: float = float("nan")

    @property
    def classified(self) -> bool:
        return self.label in ("crack", "blast")

    def to_dict(self) -> dict:
        def num(x):
            return round(float(x), 3) if math.isfinite(x) else None
        return {
            "offset_s": round(self.offset_s, 6),
            "peak_Pa": round(self.peak_Pa, 4),
            "peak_dB": round(self.peak_dB, 1),
            "label": self.label,
            "basis": self.basis,
            "ambiguity": self.ambiguity,
            "centroid_Hz": num(self.centroid_Hz),
            "decay_ms": num(self.decay_ms),
        }


@dataclass
class DetectionReport:
    """Diagnostics describing how detection behaved, so a shot count can be trusted."""
    n_detected: int
    n_candidates: int
    n_suppressed_by_refractory: int
    threshold_dB: float
    threshold_mode: str
    peak_level_dB: float
    noise_floor_dB: float
    full_scale_dB: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_detected": self.n_detected,
            "n_candidates": self.n_candidates,
            "n_suppressed_by_refractory": self.n_suppressed_by_refractory,
            "threshold_dB": round(self.threshold_dB, 1),
            "threshold_mode": self.threshold_mode,
            "peak_level_dB": round(self.peak_level_dB, 1),
            "noise_floor_dB": round(self.noise_floor_dB, 1),
            "full_scale_dB": round(self.full_scale_dB, 1) if self.full_scale_dB is not None else None,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        lines = [
            f"  Threshold:     {self.threshold_dB:.1f} dB ({self.threshold_mode})",
            f"  Peak level:    {self.peak_level_dB:.1f} dB",
            f"  Noise floor:   {self.noise_floor_dB:.1f} dB",
            f"  Candidates:    {self.n_candidates}",
            f"  Detected:      {self.n_detected}",
        ]
        if self.n_suppressed_by_refractory:
            lines.append(
                f"  Suppressed:    {self.n_suppressed_by_refractory} candidate(s) fell inside "
                f"the refractory period"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---- Envelope and conditioning ----

def bandpass_for_detection(
    x: np.ndarray,
    sample_rate: int,
    low_Hz: float = DETECTION_BAND_LOW_HZ,
    high_Hz: float = DETECTION_BAND_HIGH_HZ,
) -> np.ndarray:
    """
    Band-limit a signal to the blast band before detection.

    Detection on raw broadband pressure triggers on wind gusts, mic-handling thumps
    and action noise, none of which are blast. Restricting to 50 Hz - 8 kHz keeps the
    muzzle-blast energy and rejects the rest.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = sample_rate / 2.0
    high = min(high_Hz, nyq * 0.95)
    if x.size < 64 or low_Hz <= 0 or high <= low_Hz:
        return x

    sos = butter(2, [low_Hz / nyq, high / nyq], btype="band", output="sos")
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.size <= padlen:
        return x
    return np.asarray(sosfiltfilt(sos, x))


def compute_envelope(
    x: np.ndarray,
    window_samples: int = 96,
    hop_samples: int = 48,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a short-term RMS envelope of a signal.

    Args:
        x: Input signal (1D).
        window_samples: RMS window size in samples.
        hop_samples: Hop size in samples.

    Returns:
        (envelope, indices) where indices are the centre sample positions.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    window_samples = max(1, int(window_samples))
    hop_samples = max(1, int(hop_samples))

    if n < window_samples:
        return np.array([float(np.sqrt(np.mean(x ** 2)))]) if n else np.array([]), \
               np.array([n // 2]) if n else np.array([], dtype=np.int64)

    n_frames = 1 + (n - window_samples) // hop_samples
    starts = np.arange(n_frames) * hop_samples

    # Vectorised framing via cumulative sum of squares: O(n) regardless of overlap.
    cumsum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    sums = cumsum[starts + window_samples] - cumsum[starts]
    envelope = np.sqrt(np.maximum(sums / window_samples, 0.0) + EPS)
    indices = starts + window_samples // 2

    return envelope, indices.astype(np.int64)


def refine_peak_location(
    x: np.ndarray,
    approx_idx: int,
    sample_rate: int,
    search_ms: float = 5.0,
) -> int:
    """
    Refine a peak location to the true sample maximum.

    The search span is specified in TIME, not samples: a fixed sample count spans
    4.4x more time at 44.1 kHz than at 192 kHz, so the same recording analysed at two
    rates would refine over different physical windows.

    Args:
        x: Input signal.
        approx_idx: Approximate peak index from the envelope.
        sample_rate: Sample rate in Hz.
        search_ms: Search half-width in milliseconds.

    Returns:
        Refined peak sample index.
    """
    half = max(1, int(search_ms * sample_rate / 1000.0))
    start = max(0, approx_idx - half)
    end = min(len(x), approx_idx + half)
    if end <= start:
        return int(approx_idx)
    return int(start + np.argmax(np.abs(x[start:end])))


def _select_with_refractory(
    candidate_indices: Sequence[int],
    candidate_values: Sequence[float],
    refractory_samples: int,
) -> Tuple[List[int], int]:
    """
    Resolve the refractory period by keeping the LOUDEST candidate in each neighbourhood.

    Scanning left-to-right and dropping anything too soon after the last accepted peak
    biases against loud shots that follow quiet ones, and quietly loses events. Greedy
    selection by descending amplitude keeps the physically dominant event in each
    neighbourhood and makes the number of suppressed candidates reportable.

    Returns:
        (accepted_indices_sorted_by_time, n_suppressed)
    """
    if not len(candidate_indices):
        return [], 0

    order = np.argsort(np.asarray(candidate_values))[::-1]
    accepted: List[int] = []
    for k in order:
        idx = int(candidate_indices[k])
        if all(abs(idx - a) >= refractory_samples for a in accepted):
            accepted.append(idx)

    return sorted(accepted), len(candidate_indices) - len(accepted)


def find_arrivals(
    window: np.ndarray,
    sample_rate: int,
    *,
    min_separation_ms: float = 0.5,
    within_dB: float = 20.0,
) -> List[Arrival]:
    """
    Find distinct acoustic arrivals within one shot window.

    A supersonic round yields both a ballistic crack and a muzzle blast. They are
    separated by the difference in path and propagation time and are typically
    0.5-10 ms apart. The EARLIER arrival is the crack at a downrange microphone; at
    the shooter's position the muzzle blast usually arrives first and dominates.
    Classification here is deliberately conservative: the arrivals are reported with
    timing so the operator can judge, and only labelled when the ordering is
    unambiguous.

    Arrivals are located on the analytic-signal ENVELOPE. Peak-picking the raw
    waveform instead finds every individual cycle of the ringdown: a 900 Hz decay
    stays within 20 dB of peak for about eight cycles, each of which is a local
    maximum, so a single blast would be reported as eight arrivals.

    Args:
        window: Pressure window for one shot.
        sample_rate: Sample rate in Hz.
        min_separation_ms: Minimum spacing to count arrivals as distinct.
        within_dB: Only report secondary arrivals within this range of the main peak.

    Returns:
        Arrivals in time order.
    """
    from scipy.signal import hilbert

    raw = np.asarray(window, dtype=np.float64)
    if raw.size < 64:
        return []

    x = np.abs(hilbert(raw))
    peak = float(x.max())
    if peak <= EPS:
        return []

    distance = max(1, int(min_separation_ms * sample_rate / 1000.0))
    height = peak * (10.0 ** (-abs(within_dB) / 20.0))
    # A genuine second arrival rises well clear of the preceding decay, so require
    # prominence comparable to the arrival's own height rather than a fraction of
    # the detection floor.
    idx, _ = find_peaks(x, height=height, distance=distance, prominence=peak * 0.1)
    if idx.size == 0:
        return []

    # Keep at most the four strongest arrivals; more than that is reverberation.
    strongest = idx[np.argsort(x[idx])[::-1][:4]]
    arrivals = [
        Arrival(
            offset_s=float(i) / sample_rate,
            peak_Pa=float(x[i]),
            peak_dB=float(amplitude_to_dB_SPL(x[i])),
        )
        for i in sorted(strongest)
    ]

    for arrival in arrivals:
        centroid, decay = _arrival_character(raw, x, sample_rate, arrival.offset_s)
        arrival.centroid_Hz = centroid
        arrival.decay_ms = decay

    classify_arrivals(arrivals)
    return arrivals


# ---- Arrival classification ----

# Span over which an arrival's spectral character is measured. A ballistic crack
# is an N-wave a few hundred microseconds long; a muzzle blast runs to several
# milliseconds. Two milliseconds covers the crack entirely and captures enough of
# the blast to separate them, without reaching the next arrival at the 0.5 ms
# minimum separation used above.
CHARACTER_SPAN_MS: float = 2.0

# Decibels of envelope decay used to define an arrival's decay time.
CHARACTER_DECAY_DB: float = 10.0

# How far the observed crack-to-blast separation may sit from the delay computed
# from geometry and still be treated as that delay. Muzzle velocity is quoted to
# a few percent and the microphone position to a few centimetres, so a fixed
# window in absolute time is the honest form of this tolerance.
GEOMETRIC_DELAY_TOLERANCE_MS: float = 0.5

# How much higher in centre frequency, and how much faster to decay, an arrival
# must be before "it looks like an N-wave" counts as evidence.
#
# Without a margin these comparisons are just orderings, and two acoustically
# IDENTICAL arrivals satisfy an ordering by chance about a quarter of the time -
# which would stamp a spectral basis on a rounding difference. A ballistic crack
# carries most of its energy above 2 kHz while a muzzle blast peaks in the low
# hundreds of hertz, and an N-wave is a few hundred microseconds against the
# blast's several milliseconds. Both true ratios are therefore well beyond 2, so
# requiring a factor of 2 rejects coincidence without rejecting real cracks.
CHARACTER_CENTROID_RATIO: float = 2.0
CHARACTER_DECAY_RATIO: float = 2.0


def _arrival_character(
    raw: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int,
    offset_s: float,
) -> Tuple[float, float]:
    """
    Measure the spectral centroid and decay time of one arrival.

    Returns:
        (centroid_Hz, decay_ms), either of which may be NaN when the span is too
        short to measure.
    """
    start = int(round(offset_s * sample_rate))
    span = max(8, int(round(CHARACTER_SPAN_MS * sample_rate / 1000.0)))
    stop = min(raw.size, start + span)
    segment = raw[start:stop]

    centroid = float("nan")
    if segment.size >= 8:
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
        freqs = np.fft.rfftfreq(segment.size, 1.0 / sample_rate)
        total = float(spectrum.sum())
        if total > EPS:
            centroid = float(np.dot(freqs, spectrum) / total)

    decay = float("nan")
    if start < envelope.size:
        peak = float(envelope[start])
        if peak > EPS:
            floor = peak * (10.0 ** (-CHARACTER_DECAY_DB / 20.0))
            tail = envelope[start:]
            below = np.flatnonzero(tail <= floor)
            if below.size:
                decay = float(below[0]) * 1000.0 / sample_rate

    return centroid, decay


def classify_arrivals(
    arrivals: List[Arrival],
    *,
    expected_delay_s: Optional[float] = None,
    projectile_supersonic: Optional[bool] = None,
    tolerance_ms: float = GEOMETRIC_DELAY_TOLERANCE_MS,
) -> List[Arrival]:
    """
    Label arrivals as ballistic crack or muzzle blast, in place.

    A suppressor acts only on the muzzle blast. Reporting the ballistic crack as
    if it were the blast credits the suppressor with nothing it did, so getting
    this wrong is a measurement error, not a presentation detail. The label is
    therefore only applied when something actually establishes it, and the basis
    is recorded alongside so a weak inference cannot be read as a strong one.

    In descending order of strength:

      geometric   The separation between two arrivals matches the Mach-cone
                  delay computed from the recorded microphone distance and
                  angle and the round's muzzle velocity. This identifies the
                  crack outright.
      subsonic    The round is known to be subsonic, so no crack exists and a
                  single arrival must be the muzzle blast.
      spectral    One arrival is both higher in centre frequency AND faster to
                  decay than the other, which is what separates an N-wave from
                  a blast. Requiring both indicators to agree is what keeps a
                  reflection from being labelled a crack.
      ordering    Nothing but the loudest-arrival heuristic was available.

    Args:
        arrivals: Arrivals in time order, with character already measured.
        expected_delay_s: Crack-to-blast delay from geometry, if computable.
                          See atmosphere.crack_blast_delay_s.
        projectile_supersonic: True/False if known, None if not.
        tolerance_ms: Window around `expected_delay_s` treated as a match.

    Returns:
        The same list, labelled.
    """
    for arrival in arrivals:
        arrival.label = "unclassified"
        arrival.basis = "none"
        arrival.ambiguity = ""

    if not arrivals:
        return arrivals

    # A subsonic round produces no crack, so whatever is there is blast.
    if projectile_supersonic is False:
        loudest = max(range(len(arrivals)), key=lambda k: arrivals[k].peak_Pa)
        arrivals[loudest].label = "blast"
        arrivals[loudest].basis = "subsonic"
        for k, arrival in enumerate(arrivals):
            if k != loudest:
                arrival.ambiguity = (
                    "the round is subsonic, so this is neither crack nor muzzle "
                    "blast; it is most likely a reflection"
                )
        return arrivals

    if len(arrivals) == 1:
        arrivals[0].ambiguity = (
            "only one arrival was found, and a single arrival cannot be shown to "
            "be the muzzle blast rather than the ballistic crack without either "
            "the shot geometry or a known subsonic round"
        )
        return arrivals

    # Geometry: does any pair sit at the delay the Mach cone predicts?
    if expected_delay_s is not None and math.isfinite(expected_delay_s):
        tolerance_s = abs(tolerance_ms) / 1000.0
        for i in range(len(arrivals) - 1):
            for j in range(i + 1, len(arrivals)):
                separation = arrivals[j].offset_s - arrivals[i].offset_s
                if abs(separation - expected_delay_s) <= tolerance_s:
                    arrivals[i].label = "crack"
                    arrivals[i].basis = "geometric"
                    arrivals[j].label = "blast"
                    arrivals[j].basis = "geometric"
                    for k, arrival in enumerate(arrivals):
                        if k not in (i, j):
                            arrival.ambiguity = (
                                "arrived outside the crack/blast pair identified "
                                "from geometry; most likely a reflection"
                            )
                    return arrivals

    # Spectral character: both indicators must agree.
    loudest = max(range(len(arrivals)), key=lambda k: arrivals[k].peak_Pa)
    earlier = [k for k in range(len(arrivals)) if k < loudest]
    if earlier:
        candidate = earlier[-1]
        a, b = arrivals[candidate], arrivals[loudest]
        have_character = all(
            math.isfinite(v) for v in (a.centroid_Hz, b.centroid_Hz, a.decay_ms, b.decay_ms)
        )
        decisive = (
            have_character
            and b.centroid_Hz > 0.0
            and a.decay_ms > 0.0
            and a.centroid_Hz >= CHARACTER_CENTROID_RATIO * b.centroid_Hz
            and b.decay_ms >= CHARACTER_DECAY_RATIO * a.decay_ms
        )
        if decisive:
            a.label, a.basis = "crack", "spectral"
            b.label, b.basis = "blast", "spectral"
        else:
            a.label, a.basis = "crack", "ordering"
            b.label, b.basis = "blast", "ordering"
            if have_character:
                a.ambiguity = (
                    f"labelled from arrival order alone: this arrival is not both "
                    f"{CHARACTER_CENTROID_RATIO:g}x higher in centre frequency and "
                    f"{CHARACTER_DECAY_RATIO:g}x faster to decay than the loudest "
                    f"one, so it may be a reflection rather than a ballistic crack"
                )
            else:
                a.ambiguity = (
                    "labelled from arrival order alone: the arrivals were too short "
                    "to measure spectral character"
                )
        for k, arrival in enumerate(arrivals):
            if k not in (candidate, loudest):
                arrival.ambiguity = (
                    "additional arrival beyond the crack/blast pair; most likely a "
                    "reflection"
                )
    else:
        arrivals[loudest].label = "blast"
        arrivals[loudest].basis = "ordering"
        for k, arrival in enumerate(arrivals):
            if k != loudest:
                arrival.ambiguity = (
                    "arrived after the loudest arrival, so it is a reflection or "
                    "reverberation rather than a ballistic crack"
                )

    return arrivals


# ---- Main detection ----

def detect_shots(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    threshold_dB: Optional[float] = None,
    threshold_relative_dB: Optional[float] = None,
    pre_samples: Optional[int] = None,
    post_samples: Optional[int] = None,
    pre_ms: float = 50.0,
    post_ms: float = 200.0,
    refractory_ms: float = DEFAULT_REFRACTORY_MS,
    envelope_window_ms: float = 1.0,
    envelope_hop_ms: float = 0.25,
    min_snr_dB: float = 15.0,
    bandpass: bool = True,
    min_shots: int = 0,
    max_shots: int = 1000,
    full_scale_dB: Optional[float] = None,
    samples_FS: Optional[np.ndarray] = None,
    report: Optional[List[DetectionReport]] = None,
) -> List[ShotEvent]:
    """
    Detect gunshot events in a pressure waveform.

    Args:
        pressure_Pa: Pressure waveform in Pascals (or full-scale units if uncalibrated).
        sample_rate: Sample rate in Hz.
        threshold_dB: ABSOLUTE detection threshold. Only meaningful with a real
                      calibration; if the value exceeds what the recording can
                      represent, a warning is issued and the relative threshold is used.
        threshold_relative_dB: Threshold in dB BELOW the loudest event. This is the
                               default mode because it works with or without calibration.
        pre_samples: Samples before the peak to include. Overrides pre_ms.
        post_samples: Samples after the peak to include. Overrides post_ms.
        pre_ms: Milliseconds before the peak (pre-trigger context and filter warm-up).
        post_ms: Milliseconds after the peak (decay and reverberation).
        refractory_ms: Minimum spacing between detected shots.
        envelope_window_ms: RMS envelope window size.
        envelope_hop_ms: RMS envelope hop size.
        min_snr_dB: Reject candidates that are not this far above the noise floor.
        bandpass: Band-limit to the blast band before detection.
        min_shots: Warn if fewer than this many shots are found.
        max_shots: Safety limit on the number of detections.
        full_scale_dB: Level of a full-scale sample, used to sanity-check thresholds.
        samples_FS: Original full-scale samples, used to flag clipped shots.
        report: If a list is supplied, a DetectionReport is appended to it.

    Returns:
        List of ShotEvent objects, sorted by time.

    Raises:
        ValueError: If the signal or parameters are invalid.
    """
    x = np.asarray(pressure_Pa, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("pressure_Pa must be a 1D array")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    for name, value in (("refractory_ms", refractory_ms), ("pre_ms", pre_ms),
                        ("post_ms", post_ms), ("envelope_window_ms", envelope_window_ms),
                        ("envelope_hop_ms", envelope_hop_ms)):
        v = float(value)
        if math.isnan(v) or math.isinf(v) or v < 0:
            raise ValueError(f"{name} must be a non-negative finite number, got {value}")

    n = x.size
    warnings: List[str] = []
    if n == 0:
        if report is not None:
            report.append(DetectionReport(0, 0, 0, 0.0, "none", float("-inf"),
                                          float("-inf"), full_scale_dB,
                                          ["Recording is empty"]))
        return []

    if pre_samples is None:
        pre_samples = int(pre_ms * sample_rate / 1000.0)
    if post_samples is None:
        post_samples = int(post_ms * sample_rate / 1000.0)

    refractory_samples = max(1, int(refractory_ms * sample_rate / 1000.0))
    env_window = max(1, int(envelope_window_ms * sample_rate / 1000.0))
    env_hop = max(1, int(envelope_hop_ms * sample_rate / 1000.0))

    if post_samples > refractory_samples:
        warnings.append(
            f"post_ms ({post_ms:.0f} ms) exceeds refractory_ms ({refractory_ms:.0f} ms); "
            f"adjacent shot windows will overlap and count the same energy twice"
        )

    # ---- Envelope on the band-limited signal ----
    detect_signal = bandpass_for_detection(x, sample_rate) if bandpass else x
    envelope, indices = compute_envelope(detect_signal, env_window, env_hop)
    if envelope.size == 0:
        if report is not None:
            report.append(DetectionReport(0, 0, 0, 0.0, "none", float("-inf"),
                                          float("-inf"), full_scale_dB,
                                          ["Signal too short for envelope"]))
        return []

    peak_envelope = float(envelope.max())
    peak_dB = float(amplitude_to_dB_SPL(peak_envelope))
    noise_floor = float(np.percentile(envelope, 10.0))
    noise_floor_dB = float(amplitude_to_dB_SPL(max(noise_floor, EPS)))

    # ---- Threshold resolution ----
    mode = "relative"
    if threshold_dB is not None and threshold_relative_dB is None:
        ceiling = full_scale_dB if full_scale_dB is not None else peak_dB
        if threshold_dB > ceiling:
            warnings.append(
                f"Absolute threshold {threshold_dB:.0f} dB is above the highest level this "
                f"recording can represent ({ceiling:.1f} dB), so it can never trigger. "
                f"Falling back to {DEFAULT_THRESHOLD_RELATIVE_DB:.0f} dB below peak. "
                f"An absolute threshold requires a real calibration."
            )
            threshold_level_dB = peak_dB - DEFAULT_THRESHOLD_RELATIVE_DB
        else:
            threshold_level_dB = float(threshold_dB)
            mode = "absolute"
    else:
        rel = DEFAULT_THRESHOLD_RELATIVE_DB if threshold_relative_dB is None \
            else abs(float(threshold_relative_dB))
        threshold_level_dB = peak_dB - rel

    # Never let the threshold fall into the noise
    floor_guard = noise_floor_dB + min_snr_dB
    if threshold_level_dB < floor_guard:
        threshold_level_dB = floor_guard
        warnings.append(
            f"Threshold raised to {floor_guard:.1f} dB to stay {min_snr_dB:.0f} dB above "
            f"the noise floor"
        )

    threshold_Pa = P_REF * (10.0 ** (threshold_level_dB / 20.0))

    # ---- Prominence-based candidate picking ----
    min_distance = max(1, refractory_samples // (2 * env_hop))
    prominence = max(threshold_Pa * 0.5, peak_envelope * 1e-4)
    cand_idx, _ = find_peaks(
        envelope,
        height=threshold_Pa,
        distance=min_distance,
        prominence=prominence,
    )

    if cand_idx.size == 0:
        if report is not None:
            report.append(DetectionReport(
                0, 0, 0, threshold_level_dB, mode, peak_dB, noise_floor_dB,
                full_scale_dB,
                warnings + ["No events found above the detection threshold"],
            ))
        return []

    cand_samples = indices[cand_idx]
    cand_values = envelope[cand_idx]

    accepted, n_suppressed = _select_with_refractory(
        cand_samples, cand_values, refractory_samples
    )

    if n_suppressed:
        warnings.append(
            f"{n_suppressed} candidate event(s) were within {refractory_ms:.0f} ms of a "
            f"louder event and were suppressed. Lower refractory_ms if the weapon's "
            f"cyclic rate is faster than {60000.0/max(refractory_ms,1e-9):.0f} rpm."
        )

    if len(accepted) > max_shots:
        warnings.append(f"Found {len(accepted)} events, limiting to {max_shots}")
        accepted = accepted[:max_shots]

    # ---- Build events ----
    #
    # Every candidate must clear the SNR gate to become a shot. A RELATIVE threshold
    # is defined against the recording's own peak, so on a recording that contains no
    # gunfire at all it simply selects the loudest thing present - and a Gaussian
    # noise peak sits roughly 4.8 sigma above the RMS, which is about 13 dB. Without a
    # meaningful gate a shot-free recording therefore yields a confident "shot" with a
    # full metrics record. Real muzzle blast clears its noise floor by 30-60 dB, so a
    # 15 dB gate rejects noise while remaining far below any genuine shot.
    shots: List[ShotEvent] = []
    n_rejected_snr = 0
    rejected_peak_dB: List[float] = []

    for approx_idx in accepted:
        refined = refine_peak_location(x, approx_idx, sample_rate)
        peak_Pa = float(abs(x[refined]))
        peak_dB = float(amplitude_to_dB_SPL(peak_Pa))

        snr = peak_dB - noise_floor_dB
        if snr < min_snr_dB:
            n_rejected_snr += 1
            rejected_peak_dB.append(peak_dB)
            continue

        win_start = refined - pre_samples
        win_end = refined + post_samples
        truncated = win_start < 0 or win_end > n
        win_start, win_end = max(0, win_start), min(n, win_end)

        clipped = False
        if samples_FS is not None:
            seg = np.asarray(samples_FS)[win_start:win_end]
            clipped = detect_clipping(seg)[1] > 0

        shot = ShotEvent(
            index=refined,
            time_s=refined / sample_rate,
            peak_Pa=peak_Pa,
            peak_dB=peak_dB,
            window_start=win_start,
            window_end=win_end,
            shot_number=len(shots) + 1,
            truncated=truncated,
            clipped=clipped,
            snr_dB=snr,
            arrivals=find_arrivals(x[win_start:win_end], sample_rate),
        )
        shots.append(shot)

    if n_rejected_snr:
        loudest = max(rejected_peak_dB) if rejected_peak_dB else float("-inf")
        warnings.append(
            f"{n_rejected_snr} candidate event(s) were rejected for insufficient "
            f"impulsiveness: the loudest reached {loudest:.1f} dB, only "
            f"{loudest - noise_floor_dB:.1f} dB above the noise floor "
            f"({noise_floor_dB:.1f} dB), against a {min_snr_dB:.0f} dB requirement. "
            f"This usually means the recording contains no gunfire."
        )

    n_multi = sum(1 for s in shots if s.has_multiple_arrivals)
    if n_multi:
        warnings.append(
            f"{n_multi} shot(s) contain more than one distinct arrival, which usually "
            f"means a supersonic round's ballistic crack alongside the muzzle blast. "
            f"A suppressor acts only on the muzzle blast."
        )

    n_clipped = sum(1 for s in shots if s.clipped)
    if n_clipped:
        warnings.append(
            f"{n_clipped} shot(s) are clipped; their peak levels are understated"
        )

    if len(shots) < min_shots:
        warnings.append(f"Expected at least {min_shots} shots, found {len(shots)}")

    if report is not None:
        report.append(DetectionReport(
            n_detected=len(shots),
            n_candidates=int(cand_idx.size),
            n_suppressed_by_refractory=n_suppressed,
            threshold_dB=threshold_level_dB,
            threshold_mode=mode,
            peak_level_dB=peak_dB,
            noise_floor_dB=noise_floor_dB,
            full_scale_dB=full_scale_dB,
            warnings=warnings,
        ))

    return shots


def detect_shots_adaptive(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    target_count: int = 1,
    start_relative_dB: float = 15.0,
    max_relative_dB: float = 45.0,
    step_dB: float = 3.0,
    **kwargs,
) -> List[ShotEvent]:
    """
    Detect shots, widening the relative threshold until the target count is reached.

    Useful when the expected round count is known. The threshold is expressed
    relative to the recording's own peak so the search works without calibration.

    The threshold keywords are owned by this function; passing them through kwargs
    would collide with the values it sets and raise TypeError, so they are rejected
    explicitly.

    Args:
        pressure_Pa: Pressure waveform.
        sample_rate: Sample rate in Hz.
        target_count: Number of shots expected.
        start_relative_dB: Initial threshold, in dB below peak.
        max_relative_dB: Widest threshold to try.
        step_dB: Step size.
        **kwargs: Additional arguments forwarded to detect_shots().

    Returns:
        The first result reaching target_count, or the widest attempt.
    """
    for reserved in ("threshold_dB", "threshold_relative_dB"):
        if reserved in kwargs:
            raise TypeError(
                f"detect_shots_adaptive() controls '{reserved}'; pass "
                f"start_relative_dB/max_relative_dB instead"
            )

    relative = float(start_relative_dB)
    best: List[ShotEvent] = []

    while relative <= max_relative_dB:
        shots = detect_shots(
            pressure_Pa, sample_rate, threshold_relative_dB=relative, **kwargs
        )
        if len(shots) >= target_count:
            return shots
        if len(shots) > len(best):
            best = shots
        relative += step_dB

    return best


# ---- Auto-trim ----

# Context kept either side of the shot string. Long enough to carry the noise
# floor the shots are measured against and the tail of the last shot's
# reverberation, short enough that a ten-minute range recording collapses to the
# part that holds the string.
DEFAULT_TRIM_MARGIN_S: float = 1.0


@dataclass
class TrimSpan:
    """The part of a recording that actually holds the shot string."""
    start: int
    end: int
    sample_rate: int
    n_shots: int = 0
    original_samples: int = 0
    applied: bool = False
    reason: str = ""

    @property
    def start_s(self) -> float:
        return self.start / self.sample_rate if self.sample_rate else 0.0

    @property
    def end_s(self) -> float:
        return self.end / self.sample_rate if self.sample_rate else 0.0

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / self.sample_rate if self.sample_rate else 0.0

    @property
    def removed_s(self) -> float:
        if not self.sample_rate:
            return 0.0
        return (self.original_samples - (self.end - self.start)) / self.sample_rate

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Slice a signal to this span. Returns the input unchanged if not applied."""
        if not self.applied:
            return np.asarray(signal)
        return np.asarray(signal)[self.start:self.end]

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "start_sample": self.start,
            "end_sample": self.end,
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "duration_s": round(self.duration_s, 4),
            "removed_s": round(self.removed_s, 4),
            "n_shots": self.n_shots,
            "reason": self.reason,
        }

    def summary(self) -> str:
        if not self.applied:
            return f"  Auto-trim not applied: {self.reason}"
        return (
            f"  Auto-trimmed to {self.start_s:.2f}-{self.end_s:.2f} s "
            f"({self.duration_s:.2f} s holding {self.n_shots} shot(s); "
            f"{self.removed_s:.2f} s of silence removed)"
        )


def find_shot_string_span(
    shots: Sequence[ShotEvent],
    n_samples: int,
    sample_rate: int,
    *,
    margin_s: float = DEFAULT_TRIM_MARGIN_S,
) -> TrimSpan:
    """
    Find the span of a recording that holds the shot string.

    A range recording is mostly not shooting. Trimming to the string removes the
    walk to the line, the conversation and the pack-up, which is what makes a
    whole-session batch affordable and what stops a distant string in the next
    bay from being detected as part of this one.

    The span runs from the first shot's extraction window to the last shot's,
    plus a margin either side. It never trims INSIDE a shot window, so no
    detected event can lose part of its integration span to the trim.

    Args:
        shots: Detected shots. An empty list produces an unapplied span.
        n_samples: Length of the recording in samples.
        sample_rate: Sample rate in Hz.
        margin_s: Context to keep either side of the string.

    Returns:
        TrimSpan. Check `.applied` before using it; `.reason` explains a refusal.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")

    span = TrimSpan(
        start=0, end=int(n_samples), sample_rate=int(sample_rate),
        n_shots=len(shots), original_samples=int(n_samples),
    )

    if not shots:
        span.reason = (
            "no shots were detected, so there is no string to trim to and the "
            "whole recording is kept"
        )
        return span

    margin = max(0, int(round(abs(margin_s) * sample_rate)))
    first = min(int(s.window_start) for s in shots)
    last = max(int(s.window_end) for s in shots)

    start = max(0, first - margin)
    end = min(int(n_samples), last + margin)

    if start >= end:
        span.reason = (
            "the detected shot windows do not lie inside the recording, so the "
            "trim would be empty"
        )
        return span

    if start == 0 and end == n_samples:
        span.reason = (
            "the shot string already spans the whole recording once the margin is "
            "included, so there is nothing to remove"
        )
        return span

    span.start, span.end, span.applied = start, end, True
    span.reason = (
        f"trimmed to the {len(shots)} detected shot(s) with a {abs(margin_s):g} s margin"
    )
    return span


def get_shot_windows(
    signal: np.ndarray,
    shots: List[ShotEvent],
) -> List[np.ndarray]:
    """
    Extract signal windows for each detected shot.

    Args:
        signal: Full signal array.
        shots: Detected ShotEvent objects.

    Returns:
        One array per shot.
    """
    return [shot.extract_window(signal) for shot in shots]


def summarize_shots(shots: List[ShotEvent], sample_rate: int) -> dict:
    """
    Generate summary statistics for detected shots.

    Args:
        shots: List of ShotEvent objects.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary with summary statistics.
    """
    if not shots:
        return {
            "count": 0,
            "peak_dB_max": None, "peak_dB_min": None, "peak_dB_mean": None,
            "intervals_ms": [], "mean_interval_ms": None,
            "n_truncated": 0, "n_clipped": 0,
        }

    peaks = [s.peak_dB for s in shots]
    times = [s.time_s for s in shots]
    intervals = [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))]

    return {
        "count": len(shots),
        "peak_dB_max": max(peaks),
        "peak_dB_min": min(peaks),
        "peak_dB_mean": sum(peaks) / len(peaks),
        "peak_dB_std": float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0,
        "intervals_ms": intervals,
        "mean_interval_ms": sum(intervals) / len(intervals) if intervals else None,
        "cyclic_rate_rpm": (60000.0 / (sum(intervals) / len(intervals))) if intervals else None,
        "first_shot_time_s": times[0],
        "last_shot_time_s": times[-1],
        "n_truncated": sum(1 for s in shots if s.truncated),
        "n_clipped": sum(1 for s in shots if s.clipped),
        "n_multi_arrival": sum(1 for s in shots if s.has_multiple_arrivals),
    }


# ---- CLI for testing ----

def main() -> int:
    """Test shot detection on a WAV file."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Detect gunshots in audio")
    parser.add_argument("wav", type=Path, nargs="?", help="Input WAV file")
    parser.add_argument("--Pa-per-FS", type=float, default=1.0, help="Calibration factor")
    parser.add_argument("--threshold-dB", type=float, default=None,
                        help="Absolute threshold in dB SPL (needs calibration)")
    parser.add_argument("--threshold-relative-dB", type=float, default=None,
                        help=f"Threshold in dB below peak (default {DEFAULT_THRESHOLD_RELATIVE_DB:.0f})")
    parser.add_argument("--refractory-ms", type=float, default=DEFAULT_REFRACTORY_MS,
                        help=f"Refractory period in ms (default {DEFAULT_REFRACTORY_MS:.0f})")
    args = parser.parse_args()

    if args.wav is None:
        sr = 96000
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.02, int(sr * 2.0))
        for t0 in (0.3, 0.42, 0.54, 0.66, 0.78):
            i0 = int(t0 * sr)
            tt = np.arange(min(len(x) - i0, int(sr * 0.3))) / sr
            x[i0:i0 + len(tt)] += 200 * np.exp(-tt / 0.004) * np.sin(2 * np.pi * 900 * tt)
        pressure_Pa = x
        print(f"Synthetic string: 5 shots 120 ms apart, {sr} Hz")
    else:
        import soundfile as sf
        data, sr = sf.read(str(args.wav), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        pressure_Pa = data * args.Pa_per_FS
        print(f"Loaded: {args.wav}  ({sr} Hz, {len(data)/sr:.2f} s)")

    report: List[DetectionReport] = []
    shots = detect_shots(
        pressure_Pa, sr,
        threshold_dB=args.threshold_dB,
        threshold_relative_dB=args.threshold_relative_dB,
        refractory_ms=args.refractory_ms,
        report=report,
    )

    print(f"\nDetected {len(shots)} shot(s):")
    for s in shots:
        extra = f", {len(s.arrivals)} arrivals" if s.has_multiple_arrivals else ""
        print(f"  Shot {s.shot_number}: t={s.time_s:.4f}s, peak={s.peak_dB:.1f} dB, "
              f"SNR={s.snr_dB:.1f} dB{extra}")

    if report:
        print("\nDetection report:")
        print(report[0].summary())

    summary = summarize_shots(shots, sr)
    if summary["mean_interval_ms"]:
        print(f"\n  Mean interval: {summary['mean_interval_ms']:.1f} ms "
              f"({summary['cyclic_rate_rpm']:.0f} rpm)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
