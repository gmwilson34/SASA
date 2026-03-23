#!/usr/bin/env python3
"""
bands.py - 1/3-Octave Band Analysis for Acoustic Measurements

Computes 1/3-octave band time histories using bandpass filters.
Implements ISO 266:1997 preferred center frequencies and IEC 61260-1:2014
fractional-octave band filters.

Outputs band SPL vs time with configurable time weighting:
  - Fast (125 ms exponential averaging)
  - Slow (1000 ms exponential averaging)
  - Impulse (35 ms rise, 1500 ms decay - for transients)

Usage:
    from bands import ThirdOctaveAnalyzer, ISO_CENTER_FREQUENCIES

    analyzer = ThirdOctaveAnalyzer(sample_rate=96000)
    band_levels = analyzer.analyze(pressure_Pa, time_weighting='fast')

    # band_levels shape: (n_bands, n_time_frames)
    # Access center frequencies: analyzer.center_frequencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from calibration import power_to_dB_SPL, P_REF, EPS


# ISO 266:1997 preferred 1/3-octave band center frequencies (Hz)
# From 20 Hz to 20 kHz (audible range)
ISO_CENTER_FREQUENCIES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
], dtype=np.float64)

# Extended range for high sample rates (up to 100 kHz)
EXTENDED_CENTER_FREQUENCIES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
    20000, 25000, 31500, 40000, 50000, 63000, 80000, 100000
], dtype=np.float64)

# Time constants for symmetric exponential averaging (seconds)
TIME_CONSTANTS = {
    'fast': 0.125,      # 125 ms
    'slow': 1.0,        # 1000 ms
}

# IEC 61672-1 Impulse time weighting is ASYMMETRIC:
#   attack = 35 ms (fast rise to capture transient peaks)
#   decay  = 1500 ms (slow decay to "hold" the reading)
TIME_CONSTANT_IMPULSE_ATTACK = 0.035    # 35 ms
TIME_CONSTANT_IMPULSE_DECAY = 1.5       # 1500 ms


def compute_band_edges(fc: float, fraction: float = 3.0) -> Tuple[float, float]:
    """
    Compute band edge frequencies for a fractional-octave band.

    Args:
        fc: Center frequency in Hz.
        fraction: Fraction of octave (3 for 1/3-octave, 1 for octave).

    Returns:
        (f_low, f_high) band edge frequencies in Hz.
    """
    # Band edge ratio: 2^(1/(2*fraction))
    ratio = 2.0 ** (1.0 / (2.0 * fraction))
    f_low = fc / ratio
    f_high = fc * ratio
    return f_low, f_high


def design_bandpass_sos(
    f_low: float,
    f_high: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Design bandpass filter as second-order sections.

    Args:
        f_low: Lower cutoff frequency (Hz).
        f_high: Upper cutoff frequency (Hz).
        fs: Sample rate (Hz).
        order: Filter order (default 4 for IEC 61260 Class 1).

    Returns:
        SOS array for scipy.signal.sosfilt.
    """
    nyq = fs / 2.0

    # Normalize frequencies
    low_norm = f_low / nyq
    high_norm = f_high / nyq

    # Clamp to valid range
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(low_norm + 0.001, min(high_norm, 0.999))

    try:
        sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
    except ValueError:
        # Fallback for edge cases
        sos = np.array([[1, 0, 0, 1, 0, 0]])  # Pass-through

    return np.asarray(sos)


@dataclass
class BandFilter:
    """Single 1/3-octave band filter."""
    center_freq: float
    f_low: float
    f_high: float
    sos: np.ndarray
    zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset filter state."""
        if self.sos is not None:
            self.zi = sosfilt_zi(self.sos)


@dataclass
class ThirdOctaveAnalyzer:
    """
    1/3-octave band analyzer with time-weighted level output.

    Implements ISO/IEC standards for fractional-octave band analysis:
      - ISO 266:1997 center frequencies
      - IEC 61260-1:2014 filter design
      - IEC 61672-1:2013 time weighting
    """
    sample_rate: int
    min_freq: float = 20.0
    max_freq: Optional[float] = None
    filter_order: int = 4
    center_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    filters: List[BandFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_freq is None:
            self.max_freq = self.sample_rate / 2.0 * 0.9  # 90% of Nyquist

        self._design_filters()

    def _design_filters(self) -> None:
        """Design bandpass filters for all valid bands."""
        # Select center frequencies within valid range
        all_freqs = EXTENDED_CENTER_FREQUENCIES
        max_f = self.max_freq if self.max_freq is not None else self.sample_rate / 2.0
        valid_mask = (all_freqs >= self.min_freq) & (all_freqs <= max_f)
        self.center_frequencies = all_freqs[valid_mask]

        self.filters = []
        for fc in self.center_frequencies:
            f_low, f_high = compute_band_edges(fc)

            # Skip if band edges exceed Nyquist
            if f_high >= self.sample_rate / 2.0:
                continue

            sos = design_bandpass_sos(f_low, f_high, self.sample_rate, self.filter_order)

            filt = BandFilter(
                center_freq=fc,
                f_low=f_low,
                f_high=f_high,
                sos=sos,
            )
            self.filters.append(filt)

        # Update center frequencies to match actual filters
        self.center_frequencies = np.array([f.center_freq for f in self.filters])

    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.filters)

    def filter_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Apply all bandpass filters to signal.

        Args:
            x: Input signal (1D, in Pascals).

        Returns:
            Band-filtered signals, shape (n_bands, n_samples).
        """
        x = np.asarray(x, dtype=np.float64)
        n_samples = len(x)

        band_signals = np.zeros((self.n_bands, n_samples), dtype=np.float64)

        for i, filt in enumerate(self.filters):
            band_signals[i] = sosfilt(filt.sos, x)

        return band_signals

    def compute_levels(
        self,
        x: np.ndarray,
        time_weighting: Literal['fast', 'slow', 'impulse', 'none'] = 'fast',
        hop_ms: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time-weighted band levels.

        Args:
            x: Input signal in Pascals.
            time_weighting: 'fast' (125ms), 'slow' (1s), 'impulse' (35ms), or 'none'.
            hop_ms: Output time resolution in milliseconds.

        Returns:
            (time_axis, band_levels) where band_levels is shape (n_bands, n_frames)
            in dB SPL.
        """
        x = np.asarray(x, dtype=np.float64)

        # Filter into bands
        band_signals = self.filter_signal(x)

        # Compute squared pressure (mean-square for each band)
        band_squared = band_signals ** 2

        # Time weighting via exponential averaging
        hop_samples = max(1, int(hop_ms * self.sample_rate / 1000.0))
        n_frames = len(x) // hop_samples

        if time_weighting == 'none':
            # No weighting - compute RMS in each hop window
            band_levels = np.zeros((self.n_bands, n_frames), dtype=np.float64)
            for i in range(n_frames):
                start = i * hop_samples
                end = min(start + hop_samples, len(x))
                band_levels[:, i] = np.mean(band_squared[:, start:end], axis=1)
        elif time_weighting == 'impulse':
            # IEC 61672-1 Impulse: asymmetric detector (35ms attack, 1500ms decay).
            # Cannot use a single IIR — requires conditional per-sample logic.
            dt = 1.0 / self.sample_rate
            alpha_attack = 1.0 - np.exp(-dt / TIME_CONSTANT_IMPULSE_ATTACK)
            alpha_decay = 1.0 - np.exp(-dt / TIME_CONSTANT_IMPULSE_DECAY)

            band_levels = np.zeros((self.n_bands, n_frames), dtype=np.float64)
            state = np.zeros(self.n_bands, dtype=np.float64)
            frame_idx = 0
            for i in range(len(x)):
                instant = band_squared[:, i]
                # Per-band: use fast attack when signal rises, slow decay when it falls
                rising = instant > state
                alpha = np.where(rising, alpha_attack, alpha_decay)
                state = alpha * instant + (1.0 - alpha) * state
                if (i + 1) % hop_samples == 0 and frame_idx < n_frames:
                    band_levels[:, frame_idx] = state
                    frame_idx += 1
        else:
            # IEC 61672-1 symmetric exponential averaging (Fast / Slow) at full
            # sample rate, then decimate.  Using sosfilt preserves sub-millisecond
            # temporal detail for transient signals like gunshots.
            tau = TIME_CONSTANTS.get(time_weighting, 0.125)
            dt = 1.0 / self.sample_rate
            alpha_s = 1.0 - np.exp(-dt / tau)

            # Express exponential average as a 1st-order IIR in SOS format:
            #   y[n] = alpha_s * x[n] + (1-alpha_s) * y[n-1]
            #   H(z) = alpha_s / (1 - (1-alpha_s)*z^-1)
            sos_exp = np.array([[alpha_s, 0.0, 0.0, 1.0, -(1.0 - alpha_s), 0.0]])

            # Apply at full sample rate across all bands simultaneously (axis=1)
            smoothed = sosfilt(sos_exp, band_squared, axis=1)

            # Decimate: take value at end of each hop window
            smoothed = np.asarray(smoothed)
            decimate_indices = np.arange(hop_samples - 1, hop_samples * n_frames, hop_samples)
            decimate_indices = np.minimum(decimate_indices, smoothed.shape[1] - 1)
            band_levels = np.take(smoothed, decimate_indices, axis=1)

        # Convert to dB SPL
        band_levels_dB = power_to_dB_SPL(band_levels)

        # Time axis
        time_axis = (np.arange(n_frames) * hop_samples + hop_samples // 2) / self.sample_rate

        return time_axis, np.asarray(band_levels_dB)

    def analyze(
        self,
        x: np.ndarray,
        time_weighting: Literal['fast', 'slow', 'impulse', 'none'] = 'fast',
        hop_ms: float = 10.0,
    ) -> dict:
        """
        Full band analysis returning results dictionary.

        Args:
            x: Input signal in Pascals.
            time_weighting: Time weighting type.
            hop_ms: Output time resolution.

        Returns:
            Dictionary with:
              - time_s: time axis
              - center_frequencies: band center frequencies
              - band_levels_dB: (n_bands, n_frames) level matrix
              - overall_level_dB: (n_frames,) total level
        """
        time_s, band_levels_dB = self.compute_levels(x, time_weighting, hop_ms)

        # Compute overall level (sum of band powers)
        band_levels_Pa2 = (P_REF ** 2) * (10.0 ** (band_levels_dB / 10.0))
        overall_Pa2 = np.sum(band_levels_Pa2, axis=0)
        overall_dB = power_to_dB_SPL(overall_Pa2)

        return {
            'time_s': time_s,
            'center_frequencies': self.center_frequencies.copy(),
            'band_levels_dB': band_levels_dB,
            'overall_level_dB': overall_dB,
            'time_weighting': time_weighting,
            'hop_ms': hop_ms,
        }


def compute_band_exposure(
    band_levels_dB: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """
    Compute sound exposure level (SEL/LE) for each band.

    SEL integrates squared pressure over time.

    Args:
        band_levels_dB: Band levels in dB SPL, shape (n_bands, n_frames).
        time_s: Time axis in seconds, shape (n_frames,).

    Returns:
        Band exposure levels in dB, shape (n_bands,).
    """
    # Convert to Pa²
    band_Pa2 = (P_REF ** 2) * (10.0 ** (band_levels_dB / 10.0))

    n_bands = band_Pa2.shape[0]
    n_frames = band_Pa2.shape[1] if band_Pa2.ndim > 1 else 1

    if n_frames < 2:
        # Single frame: exposure = level × assumed frame duration.
        # Use the hop duration implied by time_s[0] as best estimate.
        frame_dt = float(time_s[0]) if len(time_s) > 0 else 1.0
        energy = band_Pa2[:, 0] * frame_dt if n_frames == 1 else np.zeros(n_bands)
    else:
        # Trapezoidal integration over time
        dt = np.diff(time_s)
        energy = np.zeros(n_bands)
        for i in range(len(dt)):
            energy += 0.5 * (band_Pa2[:, i] + band_Pa2[:, i + 1]) * dt[i]

    # SEL = 10 * log10(E / (p_ref² * T_ref))  where T_ref = 1 second
    T_ref = 1.0
    sel_dB = 10.0 * np.log10(energy / (P_REF ** 2 * T_ref) + EPS)

    return sel_dB


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
    duration = time_s[-1] - time_s[0] if len(time_s) > 1 else 1.0
    sel_dB = compute_band_exposure(band_levels_dB, time_s)

    # Leq = SEL - 10*log10(T)
    leq_dB = sel_dB - 10.0 * np.log10(duration + EPS)

    return leq_dB


# ---- CLI for testing ----

def main() -> int:
    """Test 1/3-octave band analysis."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="1/3-octave band analysis")
    parser.add_argument("wav", type=Path, nargs='?', help="Input WAV file")
    parser.add_argument("--Pa-per-FS", type=float, default=100.0,
                        help="Calibration factor (default: 100)")
    parser.add_argument("--weighting", choices=['fast', 'slow', 'impulse', 'none'],
                        default='fast', help="Time weighting")
    parser.add_argument("--hop-ms", type=float, default=10.0,
                        help="Output time resolution (ms)")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--test-tone", action="store_true",
                        help="Generate test tone instead of loading file")
    args = parser.parse_args()

    if args.test_tone or args.wav is None:
        # Generate test tone
        sr = 96000
        duration = 1.0
        t = np.arange(int(sr * duration)) / sr

        # Multi-frequency test signal
        signal = np.zeros_like(t)
        test_freqs = [100, 500, 1000, 4000]
        for f in test_freqs:
            signal += 0.25 * np.sin(2 * np.pi * f * t)

        # Convert to Pa (simulate 94 dB SPL)
        pressure_Pa = signal * P_REF * 10 ** (94 / 20)

        print(f"Test signal: {test_freqs} Hz, 94 dB SPL each")
        print(f"Sample rate: {sr} Hz, Duration: {duration} s")
    else:
        # Load WAV
        import soundfile as sf
        data, sr = sf.read(str(args.wav), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)

        pressure_Pa = data * args.Pa_per_FS
        print(f"Loaded: {args.wav}")
        print(f"Sample rate: {sr} Hz")

    # Analyze
    analyzer = ThirdOctaveAnalyzer(sample_rate=sr)
    print(f"\nBands: {analyzer.n_bands} ({analyzer.center_frequencies[0]:.0f} - "
          f"{analyzer.center_frequencies[-1]:.0f} Hz)")

    results = analyzer.analyze(
        pressure_Pa,
        time_weighting=args.weighting,
        hop_ms=args.hop_ms,
    )

    # Print band levels
    print(f"\nBand analysis ({args.weighting} weighting):")
    print(f"{'Freq (Hz)':>10} {'Max (dB)':>10} {'Mean (dB)':>10}")
    print("-" * 35)

    for i, fc in enumerate(results['center_frequencies']):
        levels = results['band_levels_dB'][i]
        max_level = np.max(levels)
        mean_level = np.mean(levels)
        print(f"{fc:10.0f} {max_level:10.1f} {mean_level:10.1f}")

    overall_max = np.max(results['overall_level_dB'])
    overall_mean = np.mean(results['overall_level_dB'])
    print("-" * 35)
    print(f"{'Overall':>10} {overall_max:10.1f} {overall_mean:10.1f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Time-frequency heatmap
            ax = axes[0]
            extent = [results['time_s'][0], results['time_s'][-1],
                      0, len(results['center_frequencies'])]
            im = ax.imshow(
                results['band_levels_dB'],
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap='viridis',
                vmin=-20,
                vmax=np.max(results['band_levels_dB']) + 5,
            )
            ax.set_ylabel('Band Index')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'1/3-Octave Band Levels ({args.weighting} weighting)')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Level (dB SPL)')

            # Set y-tick labels to frequencies
            n_labels = min(10, len(results['center_frequencies']))
            tick_idx = np.linspace(0, len(results['center_frequencies'])-1, n_labels, dtype=int)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([f"{results['center_frequencies'][i]:.0f}" for i in tick_idx])
            ax.set_ylabel('Frequency (Hz)')

            # Overall level
            ax = axes[1]
            ax.plot(results['time_s'], results['overall_level_dB'], 'b-', linewidth=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Level (dB SPL)')
            ax.set_title('Overall Level')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('band_analysis.png', dpi=150)
            print("\nPlot saved to band_analysis.png")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for plotting")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
