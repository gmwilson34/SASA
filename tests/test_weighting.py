"""
test_weighting.py - pins A/C weighting to IEC 61672-1:2013 Table 3.

The headline test here is test_apply_weighting_is_single_pass_not_doubled: a
forward-backward ("zero-phase") implementation applies |H|^2 and therefore
DOUBLES the weighting curve in decibels, turning A-weighting at 125 Hz from
-16.2 dB into -32.4 dB. That error is invisible in a plot of a spectrogram and
fatal to every A-weighted number the instrument reports.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import sosfreqz

from weighting import (
    IEC_61672_TABLE_3,
    a_weight_frequency_response,
    apply_weighting,
    apply_weighting_with_context,
    c_weight_frequency_response,
    check_iec_61672_conformance,
    design_a_weight_sos,
    design_c_weight_sos,
    get_weighting_curve_dB,
    weighting_settling_samples,
)

SAMPLE_RATES = (44100, 48000, 96000, 192000)

# Class 1 acceptance limits live in columns 2 (plus) and 3 (minus) of the table.
_CLASS1_PLUS, _CLASS1_MINUS = 2, 3


# ---------------------------------------------------------------------------
# Analytical curves vs the standard's table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq", sorted(IEC_61672_TABLE_3))
def test_analytical_A_curve_within_class1_limits(freq):
    """
    IEC 61672-1 Table 3 limits are ASYMMETRIC above 6.3 kHz: a weighting network
    may roll off arbitrarily far below nominal but must not exceed it. Checking
    a symmetric tolerance would both hide real over-response and report false
    failures, so the two directions are checked separately.
    """
    row = IEC_61672_TABLE_3[freq]
    nominal = row[0]
    lim_plus, lim_minus = row[_CLASS1_PLUS], row[_CLASS1_MINUS]

    measured = float(a_weight_frequency_response(np.array([freq]))[0])
    error = measured - nominal
    assert -lim_minus <= error <= lim_plus, (
        f"A-weighting at {freq} Hz: {measured:+.3f} dB vs nominal {nominal:+.1f} "
        f"(error {error:+.3f}, class 1 limit +{lim_plus}/-{lim_minus})"
    )


@pytest.mark.parametrize("freq", sorted(IEC_61672_TABLE_3))
def test_analytical_C_curve_within_class1_limits(freq):
    row = IEC_61672_TABLE_3[freq]
    nominal = row[1]
    lim_plus, lim_minus = row[_CLASS1_PLUS], row[_CLASS1_MINUS]

    measured = float(c_weight_frequency_response(np.array([freq]))[0])
    error = measured - nominal
    assert -lim_minus <= error <= lim_plus, (
        f"C-weighting at {freq} Hz: {measured:+.3f} dB vs nominal {nominal:+.1f} "
        f"(error {error:+.3f}, class 1 limit +{lim_plus}/-{lim_minus})"
    )


def test_A_weighting_is_zero_at_1kHz():
    # 1 kHz is the normalisation point of both weightings by definition.
    assert float(a_weight_frequency_response(np.array([1000.0]))[0]) == pytest.approx(
        0.0, abs=0.01
    )


def test_C_weighting_is_zero_at_1kHz():
    assert float(c_weight_frequency_response(np.array([1000.0]))[0]) == pytest.approx(
        0.0, abs=0.01
    )


def test_Z_weighting_curve_is_flat_zero():
    freqs = np.array([10.0, 100.0, 1000.0, 10000.0])
    assert np.allclose(get_weighting_curve_dB(freqs, "Z"), 0.0)


# ---------------------------------------------------------------------------
# Designed digital filters vs the standard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", SAMPLE_RATES)
@pytest.mark.parametrize("weighting", ["A", "C"])
def test_designed_filter_meets_class1_conformance(fs, weighting):
    report = check_iec_61672_conformance(fs, weighting, sound_class=1)
    assert report.points, "no tabulated frequencies were checked"
    assert report.passed, report.summary()


@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_designed_filter_normalised_to_0dB_at_1kHz(fs):
    # _normalize_at_1k scales the cascade so |H(1 kHz)| == 1 exactly.
    for design in (design_a_weight_sos, design_c_weight_sos):
        sos = design(fs)
        _, h = sosfreqz(sos, worN=[1000.0], fs=fs)
        gain_dB = 20.0 * np.log10(abs(complex(np.asarray(h).flat[0])))
        assert gain_dB == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# THE CRITICAL REGRESSION TEST
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq", [31.5, 125.0, 500.0, 2000.0])
@pytest.mark.parametrize("weighting", ["A", "C"])
def test_apply_weighting_is_single_pass_not_doubled(freq, weighting):
    """
    REGRESSION: apply_weighting must filter ONCE, causally.

    A steady tone's steady-state gain through the filter must equal the
    analytical weighting curve. The previous implementation used sosfiltfilt,
    which runs the filter forwards then backwards and so applies |H(f)|^2 -
    exactly twice the weighting in decibels. At 125 Hz that is -32.4 dB instead
    of -16.2 dB, and every A-weighted level the instrument reports is wrong by
    the size of the weighting itself.

    Both facts are asserted: the gain matches the curve, and it is NOT the
    doubled curve.
    """
    fs = 48000
    n = int(6.0 * fs)
    t = np.arange(n) / fs
    x = np.sin(2.0 * np.pi * freq * t)

    y = apply_weighting(x, fs, weighting)
    settle = int(1.0 * fs)  # >> the 7.7 ms time constant of the 20.6 Hz pole
    measured_dB = 20.0 * np.log10(
        float(np.sqrt(np.mean(y[settle:] ** 2)))
        / float(np.sqrt(np.mean(x[settle:] ** 2)))
    )

    expected_dB = float(
        (a_weight_frequency_response if weighting == "A" else c_weight_frequency_response)(
            np.array([freq])
        )[0]
    )

    assert measured_dB == pytest.approx(expected_dB, abs=0.3), (
        f"{weighting}-weighting at {freq} Hz measured {measured_dB:+.3f} dB, "
        f"analytical {expected_dB:+.3f} dB"
    )

    # And explicitly NOT the squared (forward-backward) response.
    if abs(expected_dB) > 1.0:
        doubled = 2.0 * expected_dB
        assert abs(measured_dB - doubled) > 3 * 0.3, (
            f"{weighting}-weighting at {freq} Hz measured {measured_dB:+.3f} dB, "
            f"which is the DOUBLED curve ({doubled:+.3f} dB): the filter is being "
            f"applied forwards and backwards."
        )


def test_Z_weighting_is_an_exact_pass_through():
    fs = 48000
    x = np.random.default_rng(0).normal(0, 1, 4096)
    assert np.array_equal(apply_weighting(x, fs, "Z"), x)


def test_unknown_weighting_raises():
    with pytest.raises(ValueError):
        apply_weighting(np.zeros(1024), 48000, "B")


# ---------------------------------------------------------------------------
# Windowed filtering with warm-up context
# ---------------------------------------------------------------------------

def test_apply_weighting_with_context_returns_exact_window_length():
    fs = 48000
    x = np.random.default_rng(2).normal(0, 1, int(2.0 * fs))
    start, stop = int(0.5 * fs), int(0.5 * fs) + 4800
    out = apply_weighting_with_context(x, fs, "A", start, stop)
    assert out.shape == (stop - start,)


def test_apply_weighting_with_context_beats_cold_start():
    """
    Filtering an extracted window in isolation starts the filter from silence,
    so the first several milliseconds are a startup transient sitting exactly
    where the shock front is. Warming the filter on the preceding samples must
    reproduce a full-signal filtering of the same region far more closely.
    """
    fs = 48000
    x = np.random.default_rng(3).normal(0, 1, int(2.0 * fs))
    start, stop = int(0.5 * fs), int(0.6 * fs)

    reference = apply_weighting(x, fs, "A")[start:stop]
    warmed = apply_weighting_with_context(x, fs, "A", start, stop)
    cold = apply_weighting(x[start:stop], fs, "A")

    # Over the whole window (transient included) the cold start is orders of
    # magnitude worse.
    err_warm = float(np.max(np.abs(warmed - reference)))
    err_cold = float(np.max(np.abs(cold - reference)))
    assert err_warm < err_cold / 100.0

    # And in the steady-state portion (past the 5-time-constant settling point)
    # it is still clearly better.
    settle = weighting_settling_samples(fs, "A")
    err_warm_ss = float(np.max(np.abs(warmed[settle:] - reference[settle:])))
    err_cold_ss = float(np.max(np.abs(cold[settle:] - reference[settle:])))
    assert err_warm_ss < err_cold_ss


def test_apply_weighting_with_context_Z_is_pass_through_slice():
    fs = 48000
    x = np.random.default_rng(4).normal(0, 1, 10000)
    out = apply_weighting_with_context(x, fs, "Z", 1000, 2000)
    assert np.array_equal(out, x[1000:2000])


def test_weighting_settling_samples_is_five_time_constants():
    # tau = 1 / (2*pi*20.598997) = 7.7266 ms; 5*tau at 48 kHz is 1854 samples.
    fs = 48000
    tau = 1.0 / (2.0 * np.pi * 20.598997)
    assert weighting_settling_samples(fs, "A") == int(np.ceil(5.0 * tau * fs))
    assert weighting_settling_samples(fs, "Z") == 0
