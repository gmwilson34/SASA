#!/usr/bin/env python3
"""
shot_detect.py - Gunshot Event Detection for Acoustic Analysis

Detects impulsive events (gunshots) in calibrated pressure waveforms using
a combination of peak detection and energy-based methods with refractory period.

Detection Algorithm:
  1. Compute short-term energy envelope (RMS over ~1ms windows)
  2. Apply threshold relative to peak or absolute dB SPL
  3. Find peaks in envelope above threshold
  4. Enforce refractory period (minimum time between shots)
  5. Extract windows around each detected event

Usage:
    from shot_detect import detect_shots, ShotEvent

    # Detect shots in calibrated pressure waveform
    shots = detect_shots(
        pressure_Pa,
        sample_rate=96000,
        threshold_dB=100.0,       # Absolute threshold in dB SPL
        pre_samples=4800,         # 50ms pre-trigger
        post_samples=19200,      # 200ms post-trigger at 96 kHz
        refractory_ms=200.0,      # Min time between shots
    )

    for shot in shots:
        print(f"Shot at {shot.time_s:.3f}s, peak={shot.peak_dB_SPL:.1f} dB")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from calibration import amplitude_to_dB_SPL, P_REF, EPS


@dataclass
class ShotEvent:
    """
    Detected gunshot event with timing and window information.

    Attributes:
        index: Sample index of detected peak in original signal.
        time_s: Time of peak in seconds.
        peak_Pa: Peak absolute pressure in Pascals.
        peak_dB_SPL: Peak level in dB SPL (instantaneous).
        window_start: Start sample index of extraction window.
        window_end: End sample index of extraction window (exclusive).
        shot_number: Sequential shot number (1-based).
    """
    index: int
    time_s: float
    peak_Pa: float
    peak_dB_SPL: float
    window_start: int
    window_end: int
    shot_number: int = 0

    def extract_window(self, signal: np.ndarray) -> np.ndarray:
        """Extract the windowed signal for this shot."""
        return signal[self.window_start:self.window_end].copy()

    @property
    def window_duration_s(self) -> float:
        """Duration of extraction window in seconds (requires sample_rate)."""
        # This is computed at detection time and stored elsewhere
        return 0.0  # Placeholder


def compute_envelope(
    x: np.ndarray,
    window_samples: int = 96,
    hop_samples: int = 48,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute short-term RMS envelope of signal.

    Args:
        x: Input signal (1D).
        window_samples: RMS window size in samples.
        hop_samples: Hop size in samples.

    Returns:
        (envelope, indices) where envelope is RMS values and indices
        are the center sample positions.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n < window_samples:
        # Signal too short, return single RMS
        rms = np.sqrt(np.mean(x ** 2))
        return np.array([rms]), np.array([n // 2])

    # Number of frames
    n_frames = 1 + (n - window_samples) // hop_samples

    # Compute RMS for each frame
    envelope = np.zeros(n_frames, dtype=np.float64)
    indices = np.zeros(n_frames, dtype=np.int64)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + window_samples
        frame = x[start:end]
        envelope[i] = np.sqrt(np.mean(frame ** 2) + EPS)
        indices[i] = start + window_samples // 2

    return envelope, indices


def detect_peaks_above_threshold(
    envelope: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    refractory_samples: int,
) -> List[int]:
    """
    Find peaks in envelope above threshold with refractory period.

    Args:
        envelope: RMS envelope values.
        indices: Sample indices corresponding to envelope values.
        threshold: Minimum envelope value for detection.
        refractory_samples: Minimum samples between detections.

    Returns:
        List of sample indices where peaks were detected.
    """
    # Find all samples above threshold
    above = envelope > threshold

    if not np.any(above):
        return []

    peaks = []
    last_peak_idx = -refractory_samples - 1

    # Scan through envelope
    i = 0
    while i < len(envelope):
        if above[i]:
            # Found region above threshold
            # Find the peak within this region
            region_start = i
            while i < len(envelope) and above[i]:
                i += 1
            region_end = i

            # Find maximum in this region
            region_max_idx = region_start + np.argmax(envelope[region_start:region_end])
            peak_sample_idx = indices[region_max_idx]

            # Check refractory period
            if peak_sample_idx - last_peak_idx >= refractory_samples:
                peaks.append(int(peak_sample_idx))
                last_peak_idx = peak_sample_idx
        else:
            i += 1

    return peaks


def refine_peak_location(
    x: np.ndarray,
    approx_idx: int,
    search_window: int = 500,
) -> int:
    """
    Refine peak location to find exact maximum absolute value.

    Args:
        x: Input signal.
        approx_idx: Approximate peak index from envelope.
        search_window: Samples to search around approximate peak.

    Returns:
        Refined peak sample index.
    """
    start = max(0, approx_idx - search_window)
    end = min(len(x), approx_idx + search_window)

    local_max_idx = np.argmax(np.abs(x[start:end]))
    return int(start + local_max_idx)


def detect_shots(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    threshold_dB: float = 100.0,
    threshold_relative_dB: Optional[float] = None,
    pre_samples: Optional[int] = None,
    post_samples: Optional[int] = None,
    pre_ms: float = 50.0,
    post_ms: float = 200.0,
    refractory_ms: float = 200.0,
    envelope_window_ms: float = 1.0,
    envelope_hop_ms: float = 0.5,
    min_shots: int = 0,
    max_shots: int = 1000,
) -> List[ShotEvent]:
    """
    Detect gunshot events in calibrated pressure waveform.

    Args:
        pressure_Pa: Pressure waveform in Pascals (calibrated).
        sample_rate: Sample rate in Hz.
        threshold_dB: Absolute detection threshold in dB SPL (for RMS envelope).
                      Typical gunshots: 140-170 dB peak, so 100-120 dB envelope.
        threshold_relative_dB: If set, use threshold relative to peak level (dB below peak).
                               Overrides threshold_dB if set.
        pre_samples: Samples before peak to include in window. Overrides pre_ms.
        post_samples: Samples after peak to include in window. Overrides post_ms.
        pre_ms: Milliseconds before peak (default 50ms for pre-shot context).
        post_ms: Milliseconds after peak (default 200ms for reverb/decay).
        refractory_ms: Minimum time between detected shots (default 200ms).
        envelope_window_ms: RMS envelope window size (default 1ms).
        envelope_hop_ms: RMS envelope hop size (default 0.5ms).
        min_shots: Minimum expected shots (warning if fewer detected).
        max_shots: Maximum shots to detect (safety limit).

    Returns:
        List of ShotEvent objects, sorted by time.

    Raises:
        ValueError: If signal is too short or parameters invalid.
    """
    x = np.asarray(pressure_Pa, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("pressure_Pa must be 1D array")

    n = len(x)
    if n == 0:
        return []

    # Convert time parameters to samples
    if pre_samples is None:
        pre_samples = int(pre_ms * sample_rate / 1000.0)
    if post_samples is None:
        post_samples = int(post_ms * sample_rate / 1000.0)

    refractory_samples = int(refractory_ms * sample_rate / 1000.0)
    envelope_window = max(1, int(envelope_window_ms * sample_rate / 1000.0))
    envelope_hop = max(1, int(envelope_hop_ms * sample_rate / 1000.0))

    # Compute envelope
    envelope, indices = compute_envelope(x, envelope_window, envelope_hop)

    # Determine threshold
    if threshold_relative_dB is not None:
        # Threshold relative to peak envelope
        peak_envelope = float(np.max(envelope))
        peak_dB = amplitude_to_dB_SPL(peak_envelope)
        threshold_dB = float(peak_dB) - abs(threshold_relative_dB)

    # Convert dB threshold to Pa
    threshold_Pa = P_REF * (10.0 ** (threshold_dB / 20.0))

    # Detect peaks
    peak_indices = detect_peaks_above_threshold(
        envelope, indices, threshold_Pa, refractory_samples
    )

    # Limit number of detections
    if len(peak_indices) > max_shots:
        print(f"Warning: Found {len(peak_indices)} peaks, limiting to {max_shots}")
        peak_indices = peak_indices[:max_shots]

    # Refine peaks and create events
    shots = []
    for i, approx_idx in enumerate(peak_indices):
        # Refine to exact peak
        refined_idx = refine_peak_location(x, approx_idx)

        # Get peak value
        peak_Pa = float(abs(x[refined_idx]))
        peak_dB_SPL_val = float(amplitude_to_dB_SPL(peak_Pa))

        # Compute window
        window_start = max(0, refined_idx - pre_samples)
        window_end = min(n, refined_idx + post_samples)

        shot = ShotEvent(
            index=refined_idx,
            time_s=refined_idx / sample_rate,
            peak_Pa=peak_Pa,
            peak_dB_SPL=peak_dB_SPL_val,
            window_start=window_start,
            window_end=window_end,
            shot_number=i + 1,
        )
        shots.append(shot)

    # Warning if fewer than expected
    if len(shots) < min_shots:
        print(f"Warning: Expected at least {min_shots} shots, found {len(shots)}")

    return shots


def detect_shots_adaptive(
    pressure_Pa: np.ndarray,
    sample_rate: int,
    *,
    target_count: int = 1,
    initial_threshold_dB: float = 120.0,
    min_threshold_dB: float = 80.0,
    threshold_step_dB: float = 5.0,
    **kwargs,
) -> List[ShotEvent]:
    """
    Detect shots with adaptive threshold to find target number of events.

    Useful when expected shot count is known but optimal threshold is not.

    Args:
        pressure_Pa: Pressure waveform in Pascals.
        sample_rate: Sample rate in Hz.
        target_count: Target number of shots to detect.
        initial_threshold_dB: Starting threshold (high).
        min_threshold_dB: Minimum threshold to try.
        threshold_step_dB: Step size for lowering threshold.
        **kwargs: Additional arguments passed to detect_shots().

    Returns:
        List of detected ShotEvent objects.
    """
    threshold = initial_threshold_dB

    while threshold >= min_threshold_dB:
        shots = detect_shots(
            pressure_Pa,
            sample_rate,
            threshold_dB=threshold,
            **kwargs,
        )

        if len(shots) >= target_count:
            return shots

        threshold -= threshold_step_dB

    # Return best effort
    return detect_shots(
        pressure_Pa,
        sample_rate,
        threshold_dB=min_threshold_dB,
        **kwargs,
    )


def get_shot_windows(
    signal: np.ndarray,
    shots: List[ShotEvent],
) -> List[np.ndarray]:
    """
    Extract signal windows for each detected shot.

    Args:
        signal: Full signal array (can be different from detection signal,
                e.g., A-weighted version).
        shots: List of detected ShotEvent objects.

    Returns:
        List of numpy arrays, one per shot.
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
            "peak_dB_SPL_max": None,
            "peak_dB_SPL_min": None,
            "peak_dB_SPL_mean": None,
            "intervals_ms": [],
            "mean_interval_ms": None,
        }

    peaks = [s.peak_dB_SPL for s in shots]
    times = [s.time_s for s in shots]

    intervals = []
    for i in range(1, len(times)):
        intervals.append((times[i] - times[i-1]) * 1000.0)

    return {
        "count": len(shots),
        "peak_dB_SPL_max": max(peaks),
        "peak_dB_SPL_min": min(peaks),
        "peak_dB_SPL_mean": sum(peaks) / len(peaks),
        "peak_dB_SPL_std": float(np.std(peaks)) if len(peaks) > 1 else 0.0,
        "intervals_ms": intervals,
        "mean_interval_ms": sum(intervals) / len(intervals) if intervals else None,
        "first_shot_time_s": times[0],
        "last_shot_time_s": times[-1],
    }


# ---- CLI for testing ----

def main() -> int:
    """Test shot detection on a WAV file."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Detect gunshots in audio file")
    parser.add_argument("wav", type=Path, help="Input WAV file")
    parser.add_argument("--Pa-per-FS", type=float, default=100.0,
                        help="Calibration: Pascals per full-scale (default: 100)")
    parser.add_argument("--threshold-dB", type=float, default=100.0,
                        help="Detection threshold in dB SPL (default: 110)")
    parser.add_argument("--refractory-ms", type=float, default=200.0,
                        help="Refractory period in ms (default: 200)")
    parser.add_argument("--plot", action="store_true", help="Plot detections")
    args = parser.parse_args()

    # Load WAV
    import soundfile as sf
    data, sr = sf.read(str(args.wav), dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)

    print(f"Loaded: {args.wav}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(data)/sr:.2f} s")

    # Calibrate
    pressure_Pa = data * args.Pa_per_FS
    peak_dB = amplitude_to_dB_SPL(np.max(np.abs(pressure_Pa)))
    print(f"Peak level: {peak_dB:.1f} dB SPL")

    # Detect shots
    shots = detect_shots(
        pressure_Pa,
        sr,
        threshold_dB=args.threshold_dB,
        refractory_ms=args.refractory_ms,
    )

    print(f"\nDetected {len(shots)} shot(s):")
    for shot in shots:
        print(f"  Shot {shot.shot_number}: t={shot.time_s:.3f}s, "
              f"peak={shot.peak_dB_SPL:.1f} dB SPL, "
              f"window=[{shot.window_start}:{shot.window_end}]")

    # Summary
    summary = summarize_shots(shots, sr)
    print("\nSummary:")
    print(f"  Peak SPL range: {summary['peak_dB_SPL_min']:.1f} - {summary['peak_dB_SPL_max']:.1f} dB")
    if summary['mean_interval_ms']:
        print(f"  Mean interval: {summary['mean_interval_ms']:.1f} ms")

    if args.plot and shots:
        try:
            import matplotlib.pyplot as plt

            time = np.arange(len(pressure_Pa)) / sr

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # Waveform with markers
            ax = axes[0]
            ax.plot(time, pressure_Pa, 'b-', linewidth=0.5)
            for shot in shots:
                ax.axvline(shot.time_s, color='r', linestyle='--', alpha=0.7)
                ax.plot(shot.time_s, shot.peak_Pa, 'ro', markersize=8)
            ax.set_ylabel('Pressure (Pa)')
            ax.set_title('Waveform with Shot Detections')
            ax.grid(True, alpha=0.3)

            # dB SPL
            ax = axes[1]
            dB_inst = amplitude_to_dB_SPL(np.abs(pressure_Pa))
            ax.plot(time, dB_inst, 'g-', linewidth=0.5)
            ax.axhline(args.threshold_dB, color='r', linestyle=':', label=f'Threshold ({args.threshold_dB} dB)')
            for shot in shots:
                ax.axvline(shot.time_s, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Level (dB SPL)')
            ax.set_title('Instantaneous dB SPL')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('shot_detection.png', dpi=150)
            print("\nPlot saved to shot_detection.png")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for plotting")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
