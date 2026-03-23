#!/usr/bin/env python3
"""
calibration.py - Microphone/Recording Chain Calibration for Acoustic Measurements

Converts digital full-scale (FS) waveform samples to physical pressure units (Pascals).
Provides helper functions for SPL calculations using the standard reference pressure.

Calibration Methods:
  1) Direct: Pa_per_FS - multiply float waveform directly
  2) Derived: sensitivity_mV_per_Pa + V_per_FS - compute Pa_per_FS from mic specs

Reference: 20 µPa (threshold of human hearing) for dB SPL calculations.

Usage:
    from calibration import Calibration

    # Method 1: Direct calibration factor
    cal = Calibration(Pa_per_FS=50.0)

    # Method 2: From microphone sensitivity specs
    cal = Calibration.from_sensitivity(
        sensitivity_mV_per_Pa=10.0,  # e.g., 10 mV/Pa (-40 dB re 1V/Pa)
        V_per_FS=1.0                  # recorder full-scale voltage
    )

    # Convert samples to Pascals
    pressure_Pa = cal.to_pascals(samples)

    # Convert to dB SPL
    spl_dB = cal.to_dB_SPL(pressure_rms)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Standard reference pressure for dB SPL (threshold of hearing)
P_REF: float = 20e-6  # 20 µPa

# Numerical floor to avoid log(0)
EPS: float = 1e-30


@dataclass
class Calibration:
    """
    Calibration data for converting digital samples to physical pressure (Pascals).

    Attributes:
        Pa_per_FS: Pascals per full-scale unit. Multiply float waveform [-1, 1] by this
                   to get pressure in Pascals.
        description: Optional description of calibration source/method.
    """
    Pa_per_FS: float
    description: str = ""

    def __post_init__(self) -> None:
        if self.Pa_per_FS <= 0:
            raise ValueError(f"Pa_per_FS must be positive, got {self.Pa_per_FS}")

    @classmethod
    def from_sensitivity(
        cls,
        sensitivity_mV_per_Pa: float,
        V_per_FS: float,
        description: str = "",
    ) -> "Calibration":
        """
        Create calibration from microphone sensitivity and recorder specs.

        Args:
            sensitivity_mV_per_Pa: Microphone sensitivity in mV/Pa.
                                   Example: 10 mV/Pa = -40 dB re 1V/Pa
            V_per_FS: Recorder full-scale voltage (what ±1.0 in float maps to).
                      For many pro recorders this might be ~1-10V depending on gain.
            description: Optional description string.

        Returns:
            Calibration instance with computed Pa_per_FS.

        Example:
            # Mic: 10 mV/Pa sensitivity, recorder: 1V = FS
            # 1.0 FS = 1000 mV → 1000 mV / 10 mV/Pa = 100 Pa
            cal = Calibration.from_sensitivity(10.0, 1.0)
            # Pa_per_FS = 100.0
        """
        if sensitivity_mV_per_Pa <= 0:
            raise ValueError(f"sensitivity_mV_per_Pa must be positive, got {sensitivity_mV_per_Pa}")
        if V_per_FS <= 0:
            raise ValueError(f"V_per_FS must be positive, got {V_per_FS}")

        # Convert sensitivity to V/Pa
        sensitivity_V_per_Pa = sensitivity_mV_per_Pa / 1000.0

        # Pa_per_FS = V_per_FS / sensitivity_V_per_Pa
        # Because: FS_value * V_per_FS = voltage, and voltage / sensitivity = pressure
        Pa_per_FS = V_per_FS / sensitivity_V_per_Pa

        if not description:
            description = f"Derived: {sensitivity_mV_per_Pa} mV/Pa, {V_per_FS} V/FS"

        return cls(Pa_per_FS=Pa_per_FS, description=description)

    @classmethod
    def from_dB_sensitivity(
        cls,
        sensitivity_dB_re_1V_per_Pa: float,
        V_per_FS: float,
        description: str = "",
    ) -> "Calibration":
        """
        Create calibration from microphone sensitivity in dB re 1V/Pa.

        Args:
            sensitivity_dB_re_1V_per_Pa: Sensitivity in dB re 1V/Pa.
                                         Typical values: -40 to -26 dB for measurement mics.
            V_per_FS: Recorder full-scale voltage.
            description: Optional description string.

        Returns:
            Calibration instance.

        Example:
            # Mic: -40 dB re 1V/Pa (= 10 mV/Pa), recorder: 1V = FS
            cal = Calibration.from_dB_sensitivity(-40.0, 1.0)
        """
        # Convert dB to linear: sensitivity_V_per_Pa = 10^(dB/20)
        sensitivity_V_per_Pa = 10.0 ** (sensitivity_dB_re_1V_per_Pa / 20.0)
        sensitivity_mV_per_Pa = sensitivity_V_per_Pa * 1000.0

        if not description:
            description = f"Derived: {sensitivity_dB_re_1V_per_Pa} dB re 1V/Pa, {V_per_FS} V/FS"

        return cls.from_sensitivity(sensitivity_mV_per_Pa, V_per_FS, description)

    @classmethod
    def uncalibrated(cls) -> "Calibration":
        """
        Return a unit calibration (Pa_per_FS=1.0) for uncalibrated analysis.

        Note: Results will be in "relative dB" not true dB SPL.
        """
        return cls(Pa_per_FS=1.0, description="UNCALIBRATED (relative units)")

    def to_pascals(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert digital samples (float, nominally [-1, 1]) to pressure in Pascals.

        Args:
            samples: Audio samples as float array.

        Returns:
            Pressure waveform in Pascals.
        """
        return np.asarray(samples, dtype=np.float64) * self.Pa_per_FS

    def is_calibrated(self) -> bool:
        """Check if this is a real calibration (not the unit placeholder)."""
        return "UNCALIBRATED" not in self.description


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
        time window (e.g., Fast = 125ms, Slow = 1s).
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
        RMS value(s).
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
        Peak absolute value(s).
    """
    x = np.asarray(samples, dtype=np.float64)
    return np.max(np.abs(x), axis=axis)


# ---- CLI for testing ----

def main() -> int:
    """Test calibration module."""
    import argparse

    parser = argparse.ArgumentParser(description="Test calibration calculations")
    parser.add_argument("--Pa-per-FS", type=float, default=None, help="Direct Pa/FS calibration")
    parser.add_argument("--sensitivity-mV", type=float, default=None, help="Mic sensitivity in mV/Pa")
    parser.add_argument("--sensitivity-dB", type=float, default=None, help="Mic sensitivity in dB re 1V/Pa")
    parser.add_argument("--V-per-FS", type=float, default=1.0, help="Recorder V/FS (default: 1.0)")
    args = parser.parse_args()

    if args.Pa_per_FS is not None:
        cal = Calibration(Pa_per_FS=args.Pa_per_FS)
    elif args.sensitivity_mV is not None:
        cal = Calibration.from_sensitivity(args.sensitivity_mV, args.V_per_FS)
    elif args.sensitivity_dB is not None:
        cal = Calibration.from_dB_sensitivity(args.sensitivity_dB, args.V_per_FS)
    else:
        cal = Calibration.uncalibrated()

    print(f"Calibration: {cal}")
    print(f"Pa per FS: {cal.Pa_per_FS:.6g}")
    print(f"Is calibrated: {cal.is_calibrated()}")

    # Example: 1.0 FS = ?
    test_val = np.array([1.0])
    Pa = cal.to_pascals(test_val)
    dB = amplitude_to_dB_SPL(float(Pa[0]))
    print(f"\nExample: {test_val[0]} FS → {Pa[0]:.3g} Pa → {dB:.1f} dB SPL")

    # Reference levels
    print("\nReference levels:")
    print(f"  P_REF = {P_REF:.2e} Pa (0 dB SPL)")
    print(f"  1 Pa = {amplitude_to_dB_SPL(1.0):.1f} dB SPL")
    print(f"  20 Pa = {amplitude_to_dB_SPL(20.0):.1f} dB SPL (approx. 120 dB SPL)")
    print(f"  200 Pa = {amplitude_to_dB_SPL(200.0):.1f} dB SPL (approx. 140 dB SPL)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
