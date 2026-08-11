"""
test_calibration.py - pins the pressure/level conversion chain to closed-form values.

Everything the instrument reports is a decibel value derived from Pa_per_FS and
P_REF. If this file is right, every level in the application has a defensible
zero point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibration import (
    P_REF,
    Calibration,
    amplitude_to_dB_SPL,
    dB_SPL_to_amplitude,
    detect_clipping,
    energy_average_dB,
    power_to_dB_SPL,
    remove_dc_offset,
)
from conftest import make_friedlander, make_sine


# ---------------------------------------------------------------------------
# Reference level
# ---------------------------------------------------------------------------

def test_one_pascal_is_93_9794_dB_SPL():
    # L = 20*log10(p / p_ref) = 20*log10(1 / 20e-6) = 20*log10(50000)
    #   = 20 * 4.698970004336019 = 93.97940008672038 dB
    expected = 20.0 * math.log10(1.0 / 20e-6)
    assert expected == pytest.approx(93.979400, abs=1e-6)
    assert float(amplitude_to_dB_SPL(1.0)) == pytest.approx(expected, abs=1e-6)


def test_power_to_dB_SPL_agrees_with_amplitude_form():
    # 10*log10(p^2/p_ref^2) == 20*log10(p/p_ref) identically.
    for p in (1e-3, 0.02, 1.0, 200.0, 20000.0):
        assert float(power_to_dB_SPL(p ** 2)) == pytest.approx(
            float(amplitude_to_dB_SPL(p)), abs=1e-9
        )


def test_dB_SPL_round_trip():
    # dB_SPL_to_amplitude is the exact inverse of amplitude_to_dB_SPL on (0, inf).
    for level in (0.0, 20.0, 60.0, 93.9794, 94.0, 114.0, 140.0, 180.0):
        amp = float(dB_SPL_to_amplitude(level))
        assert float(amplitude_to_dB_SPL(amp)) == pytest.approx(level, abs=1e-9)

    for amp in (1e-5, 2e-5, 1.0, 200.0):
        level = float(amplitude_to_dB_SPL(amp))
        assert float(dB_SPL_to_amplitude(level)) == pytest.approx(amp, rel=1e-12)


def test_dB_SPL_to_amplitude_known_points():
    # 0 dB SPL is the reference pressure by definition; 94 dB is 1.0023 Pa.
    assert float(dB_SPL_to_amplitude(0.0)) == pytest.approx(P_REF, rel=1e-15)
    assert float(dB_SPL_to_amplitude(94.0)) == pytest.approx(
        20e-6 * 10 ** 4.7, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Calibrator-tone calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calibrator_dB", [94.0, 114.0])
@pytest.mark.parametrize("digital_amplitude", [0.1, 0.5, 0.8])
def test_from_calibrator_tone_recovers_known_Pa_per_FS(calibrator_dB, digital_amplitude):
    fs = 48000
    duration = 1.0  # 1000 Hz * 1.0 s = 1000 whole cycles -> RMS is exact
    t = np.arange(int(fs * duration)) / fs
    tone = digital_amplitude * np.sin(2 * np.pi * 1000.0 * t)

    cal = Calibration.from_calibrator_tone(tone, fs, calibrator_dB)

    # Pa_per_FS = p_ref * 10^(L/20) / rms_digital, rms_digital = A/sqrt(2)
    expected = P_REF * 10.0 ** (calibrator_dB / 20.0) / (digital_amplitude / np.sqrt(2.0))
    error_dB = 20.0 * np.log10(cal.Pa_per_FS / expected)
    assert abs(error_dB) < 0.01
    assert cal.method == "calibrator_tone"
    assert cal.calibrated is True
    assert cal.reference_level_dB == calibrator_dB


def test_calibrator_tone_makes_the_tone_read_its_stated_level():
    # End-to-end: applying the calibration to the tone must reproduce the
    # calibrator's stated SPL, since that is the definition of the calibration.
    fs = 48000
    t = np.arange(fs) / fs
    tone = 0.25 * np.sin(2 * np.pi * 1000.0 * t)
    cal = Calibration.from_calibrator_tone(tone, fs, 114.0)

    pressure = cal.to_pascals(tone)
    measured = float(amplitude_to_dB_SPL(np.sqrt(np.mean(pressure ** 2))))
    assert measured == pytest.approx(114.0, abs=0.01)


def test_from_calibrator_tone_rejects_clipped_tone():
    fs = 48000
    t = np.arange(fs) / fs
    # Amplitude 1.0 pins samples at full scale -> peak >= 1 - CLIP_TOLERANCE.
    tone = 1.0 * np.sin(2 * np.pi * 1000.0 * t)
    with pytest.raises(ValueError, match="clipped"):
        Calibration.from_calibrator_tone(tone, fs, 114.0)


def test_from_calibrator_tone_rejects_silence():
    fs = 48000
    with pytest.raises(ValueError, match="silent"):
        Calibration.from_calibrator_tone(np.zeros(fs), fs, 114.0)


def test_from_calibrator_tone_rejects_wrong_frequency():
    fs = 48000
    t = np.arange(fs) / fs
    tone = 0.5 * np.sin(2 * np.pi * 500.0 * t)  # 500 Hz, not the expected 1000
    with pytest.raises(ValueError, match="Dominant frequency"):
        Calibration.from_calibrator_tone(tone, fs, 114.0, tone_frequency_Hz=1000.0)


def test_from_calibrator_tone_rejects_too_short_recording():
    fs = 48000
    t = np.arange(fs // 20) / fs  # 50 ms, below the 100 ms minimum
    tone = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    with pytest.raises(ValueError, match="too short"):
        Calibration.from_calibrator_tone(tone, fs, 114.0)


def test_calibrator_post_test_drift_is_reported():
    # A post-test tone 1 dB quieter must be reported as -1.00 dB of drift:
    # residual = 20*log10(rms_post / rms_pre) = 20*log10(10^(-1/20)) = -1.
    fs = 48000
    t = np.arange(fs) / fs
    pre = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    post = pre * (10.0 ** (-1.0 / 20.0))
    cal = Calibration.from_calibrator_tone(pre, fs, 114.0, post_test_samples=post)
    assert cal.residual_dB == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Recording-chain calibration
# ---------------------------------------------------------------------------

def test_from_recording_chain_10mV_per_Pa_unity_gain():
    # Pa_per_FS = adc_full_scale_V / (S_V/Pa * 10^(gain/20))
    #           = 1.0 / (0.010 * 1) = 100 Pa per full scale, exactly.
    cal = Calibration.from_recording_chain(
        sensitivity_mV_per_Pa=10.0, adc_full_scale_V=1.0, preamp_gain_dB=0.0
    )
    assert cal.Pa_per_FS == pytest.approx(100.0, rel=1e-12)
    assert cal.calibrated is True
    assert cal.method == "recording_chain"


def test_from_recording_chain_20dB_preamp_gain():
    # 10^(20/20) = 10 exactly, so 1.0 / (0.010 * 10) = 10 Pa per full scale.
    cal = Calibration.from_recording_chain(
        sensitivity_mV_per_Pa=10.0, adc_full_scale_V=1.0, preamp_gain_dB=20.0
    )
    assert cal.Pa_per_FS == pytest.approx(10.0, rel=1e-12)


def test_from_dB_sensitivity_matches_mV_form():
    # -40 dB re 1 V/Pa == 10 mV/Pa exactly (10^(-40/20) V = 0.01 V).
    a = Calibration.from_dB_sensitivity(-40.0, V_per_FS=1.0, preamp_gain_dB=0.0)
    b = Calibration.from_recording_chain(10.0, adc_full_scale_V=1.0)
    assert a.Pa_per_FS == pytest.approx(b.Pa_per_FS, rel=1e-12)


def test_full_scale_dB_is_the_level_of_a_full_scale_sample():
    # full_scale_dB = 20*log10(Pa_per_FS / p_ref); for Pa_per_FS = 1 that is
    # 93.9794 dB, which is why an absolute 120 dB threshold detects nothing
    # in an uncalibrated recording.
    assert Calibration.uncalibrated().full_scale_dB == pytest.approx(
        20.0 * math.log10(1.0 / P_REF), abs=1e-9
    )
    assert Calibration(Pa_per_FS=100.0).full_scale_dB == pytest.approx(
        20.0 * math.log10(100.0 / P_REF), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_Pa_per_FS_raises(bad):
    with pytest.raises(ValueError):
        Calibration(Pa_per_FS=bad)


def test_uncalibrated_flag_and_units():
    uncal = Calibration.uncalibrated()
    assert uncal.calibrated is False
    assert uncal.is_calibrated() is False
    assert uncal.level_unit == "dB re FS"
    assert uncal.Pa_per_FS == 1.0

    cal = Calibration(Pa_per_FS=2.5)
    assert cal.calibrated is True
    assert cal.level_unit == "dB SPL"


def test_preset_records_provenance():
    cal = Calibration.preset(12.5, name="Rig A", provenance="2026-01-02 pistonphone")
    assert cal.method == "preset"
    assert cal.calibrated is True
    assert "Rig A" in cal.description and "pistonphone" in cal.description


# ---------------------------------------------------------------------------
# Clipping detection
# ---------------------------------------------------------------------------

def test_detect_clipping_counts_known_runs():
    # Three runs of 5 at-rail samples (15 samples, 3 runs) plus one isolated
    # single at-rail sample, which is legitimate and must NOT be counted
    # (min_run = 2).
    x = np.zeros(1000)
    for start in (100, 300, 500):
        x[start:start + 5] = 1.0
    x[700] = -1.0  # single sample on the rail: not a clipping event
    samples, runs = detect_clipping(x)
    assert (samples, runs) == (15, 3)


def test_detect_clipping_counts_negative_rail_runs():
    x = np.zeros(500)
    x[10:14] = -1.0
    x[200:203] = 1.0
    assert detect_clipping(x) == (7, 2)


def test_detect_clipping_finds_none_at_0_99_peak():
    # 0.99 is 0.01 from the rail, far outside CLIP_TOLERANCE (1e-4).
    fs = 48000
    t = np.arange(fs) / fs
    x = 0.99 * np.sin(2 * np.pi * 1000.0 * t)
    assert float(np.max(np.abs(x))) <= 0.99
    assert detect_clipping(x) == (0, 0)


def test_detect_clipping_empty_signal():
    assert detect_clipping(np.array([])) == (0, 0)


# ---------------------------------------------------------------------------
# Energy averaging
# ---------------------------------------------------------------------------

def test_energy_average_of_80_and_90_dB():
    # 10*log10((10^8 + 10^9)/2) = 10*log10(5.5e8) = 87.40362689494244 dB.
    # The arithmetic mean, 85 dB, is not a valid summary of sound levels.
    expected = 10.0 * math.log10((1e8 + 1e9) / 2.0)
    assert expected == pytest.approx(87.4036, abs=1e-4)
    got = energy_average_dB([80.0, 90.0])
    assert got == pytest.approx(expected, abs=1e-12)
    assert abs(got - 85.0) > 2.0


def test_energy_average_of_identical_levels_is_that_level():
    assert energy_average_dB([100.0] * 37) == pytest.approx(100.0, abs=1e-12)


def test_energy_average_of_empty_is_nan():
    assert math.isnan(energy_average_dB([]))


# ---------------------------------------------------------------------------
# DC removal
# ---------------------------------------------------------------------------

def test_remove_dc_offset_removes_a_known_offset():
    fs = 48000
    t = np.arange(fs) / fs
    signal = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    offset = 0.2
    out = remove_dc_offset(signal + offset, fs, cutoff_Hz=10.0)
    # A 10 Hz high-pass must drive the mean to ~0 while leaving the 1 kHz tone
    # untouched (its gain at 1 kHz is 1 to within 1e-6).
    assert abs(float(np.mean(out))) < 1e-6
    assert float(np.sqrt(np.mean(out ** 2))) == pytest.approx(
        float(np.sqrt(np.mean(signal ** 2))), rel=1e-4
    )


def test_remove_dc_offset_preserves_friedlander_peak():
    # Regression: sosfiltfilt's default odd-reflection padding fabricates a huge
    # transient when a window starts at its own peak. The Friedlander wave has
    # exactly zero net impulse, so a 10 Hz high-pass must not move its 200 Pa
    # peak by more than a small fraction of a dB.
    fs = 96000
    p = make_friedlander(P0=200.0, T=0.001, fs=fs, duration=0.05)
    assert float(np.max(p)) == pytest.approx(200.0, rel=1e-12)

    out = remove_dc_offset(p, fs, cutoff_Hz=10.0)
    peak_error_dB = 20.0 * np.log10(float(np.max(np.abs(out))) / 200.0)
    assert abs(peak_error_dB) < 0.05


def test_remove_dc_offset_preserves_friedlander_peak_at_window_start():
    # The hardest case: the blast is the very first sample of the window.
    for fs in (48000, 192000):
        p = make_friedlander(P0=200.0, T=0.001, fs=fs, duration=0.02)
        out = remove_dc_offset(p, fs, cutoff_Hz=10.0)
        peak_error_dB = 20.0 * np.log10(float(np.max(np.abs(out))) / 200.0)
        assert abs(peak_error_dB) < 0.05, f"fs={fs}"


def test_remove_dc_offset_leaves_a_clean_sine_alone():
    fs = 48000
    x = make_sine(1000.0, 94.0, fs, 1.0)
    out = remove_dc_offset(x, fs, cutoff_Hz=10.0)
    # Level change must be under 0.01 dB three decades above the corner.
    change_dB = 20.0 * np.log10(
        float(np.sqrt(np.mean(out ** 2))) / float(np.sqrt(np.mean(x ** 2)))
    )
    assert abs(change_dB) < 0.01
