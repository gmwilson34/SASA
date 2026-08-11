"""
test_bands.py - pins the 1/3-octave filter bank to IEC 61260-1 and to Parseval.

The headline test is test_filter_bank_passes_its_nominal_bands: the previous
implementation clamped band edges into the representable range instead of
decimating, so at 192 kHz the 20, 25, 31.5, 40, 50, 63, 80 and 100 Hz bands were
all the SAME 96-192 Hz filter. Eight different band levels, one filter, no error
raised.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bands import (
    G_BASE10,
    ThirdOctaveAnalyzer,
    band_insertion_loss,
    compute_band_edges,
    design_bandpass_sos,
    exact_midband_frequency,
    exponential_detector,
    impulse_detector,
)
from calibration import P_REF
from conftest import make_sine

SAMPLE_RATES = (44100, 48000, 96000, 192000)


# ---------------------------------------------------------------------------
# THE CRITICAL REGRESSION TEST
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_filter_bank_passes_its_nominal_bands(fs):
    """
    REGRESSION: every designed filter must actually pass the band it is labelled
    with, at every supported sample rate.
    """
    analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
    report = analyzer.verify_filter_bank()
    assert report.passed, report.summary()
    assert analyzer.n_bands >= 30


@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_no_two_bands_share_a_passband(fs):
    """
    REGRESSION: clamping band edges silently produced identical filters for
    every band below the clamp. Distinct nominal bands must have distinct
    measured passbands.
    """
    analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
    rows = analyzer.verify_filter_bank().rows
    measured = [
        (round(r["measured_low"], 6), round(r["measured_high"], 6)) for r in rows
    ]
    assert len(set(measured)) == len(measured), (
        f"fs={fs}: {len(measured) - len(set(measured))} band(s) share a passband"
    )

    # Passbands must also be strictly increasing with nominal frequency.
    lows = [r["measured_low"] for r in rows]
    assert all(b > a for a, b in zip(lows, lows[1:]))


@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_measured_minus_3dB_points_match_IEC_61260_edges(fs):
    """
    Each band's measured -3 dB points must sit within 10% of the nominal
    IEC 61260-1 base-10 edges f_m * G^(-1/6) and f_m * G^(+1/6).
    """
    analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
    for row in analyzer.verify_filter_bank().rows:
        rel_lo = abs(row["measured_low"] - row["f_low"]) / row["f_low"]
        rel_hi = abs(row["measured_high"] - row["f_high"]) / row["f_high"]
        assert rel_lo < 0.10, f"fs={fs} band {row['nominal']} Hz lower edge off by {rel_lo:.1%}"
        assert rel_hi < 0.10, f"fs={fs} band {row['nominal']} Hz upper edge off by {rel_hi:.1%}"


def test_design_bandpass_refuses_unrepresentable_band():
    # Clamping into the valid range would silently return a filter for a
    # DIFFERENT band; an error is the correct behaviour.
    with pytest.raises(ValueError, match="decimate"):
        design_bandpass_sos(1.0, 1.26, fs=192000)  # 1.04e-5 of Nyquist
    with pytest.raises(ValueError, match="not representable"):
        design_bandpass_sos(1000.0, 30000.0, fs=48000)  # above Nyquist


@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_low_bands_are_designed_on_a_decimated_signal(fs):
    """
    The 20 Hz band's edges are 17.8-22.4 Hz. At 192 kHz that is 1.9e-4 of
    Nyquist, where an IIR bandpass is numerically degenerate. The decimation
    cascade must put every band's centre frequency in a well-conditioned range,
    which is the mechanism that replaced edge clamping.
    """
    analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
    low = analyzer.filters[0]
    assert low.nominal_freq == 20.0
    # Working Nyquist must not be more than ~1000x the band's upper edge.
    assert low.working_rate / 2.0 < 1000.0 * low.f_high
    assert low.f_low / (low.working_rate / 2.0) > 1e-3
    # Every filter runs at a rate where its band is comfortably representable.
    for filt in analyzer.filters:
        assert filt.f_high < filt.working_rate / 2.0
        assert filt.f_low / (filt.working_rate / 2.0) >= 1e-4


# ---------------------------------------------------------------------------
# Exact midband frequencies and band edges (IEC 61260-1, base 10)
# ---------------------------------------------------------------------------

def test_exact_midband_frequency_of_reference_band():
    # x = 0 -> f_m = G^0 * 1000 = 1000 Hz exactly.
    assert exact_midband_frequency(1000.0) == pytest.approx(1000.0, rel=1e-12)


def test_exact_midband_frequency_of_1250_band_is_1258_9():
    # The ISO 266 label "1250" is band x = 1, whose exact midband frequency is
    # G^(1/3) * 1000 with G = 10^(3/10):  10^(0.1) * 1000 = 1258.925 Hz.
    expected = 1000.0 * 10.0 ** 0.1
    assert expected == pytest.approx(1258.9254, abs=1e-4)
    assert exact_midband_frequency(1250.0) == pytest.approx(expected, rel=1e-12)
    # It is emphatically NOT the rounded label.
    assert abs(exact_midband_frequency(1250.0) - 1250.0) > 8.0


@pytest.mark.parametrize(
    "nominal,exponent",
    [
        (20.0, -17), (31.5, -15), (100.0, -10), (250.0, -6),
        (500.0, -3), (1000.0, 0), (2000.0, 3), (8000.0, 9), (16000.0, 12),
    ],
)
def test_exact_midband_frequency_follows_G_power(nominal, exponent):
    # f_m = G^(x/3) * 1000, G = 10^(3/10).
    expected = 1000.0 * G_BASE10 ** (exponent / 3.0)
    assert exact_midband_frequency(nominal) == pytest.approx(expected, rel=1e-12)


def test_band_edges_use_the_base10_ratio():
    # f_high / f_low = G^(1/(2*3)) / G^(-1/(2*3)) = G^(1/3) = 10^(0.1).
    for fc in (20.0, 100.0, 1000.0, 1258.925, 16000.0):
        f_low, f_high = compute_band_edges(fc, fraction=3.0)
        assert f_high / f_low == pytest.approx(G_BASE10 ** (1.0 / 3.0), rel=1e-14)
        assert f_high / f_low == pytest.approx(10.0 ** 0.1, rel=1e-14)
        # Geometric mean of the edges is the midband frequency.
        assert math.sqrt(f_low * f_high) == pytest.approx(fc, rel=1e-14)


def test_octave_band_edges_use_base10_not_base2():
    # fraction = 1 -> f_high/f_low = G = 10^0.3 = 1.995262, NOT 2. Using base-2
    # for the edges while placing midband frequencies on the base-10 grid leaves
    # gaps and overlaps between adjacent bands.
    f_low, f_high = compute_band_edges(1000.0, fraction=1.0)
    assert f_high / f_low == pytest.approx(G_BASE10, rel=1e-14)
    assert f_high / f_low == pytest.approx(1.9952623149688795, rel=1e-14)
    assert abs(f_high / f_low - 2.0) > 4e-3


def test_adjacent_third_octave_bands_meet_exactly_at_their_shared_edge():
    # Contiguity is the property base-10 edges are chosen for: the upper edge of
    # band x must be the lower edge of band x+1, exactly.
    for lower, upper in ((500.0, 630.0), (1000.0, 1250.0), (4000.0, 5000.0)):
        _, hi = compute_band_edges(exact_midband_frequency(lower), 3.0)
        lo, _ = compute_band_edges(exact_midband_frequency(upper), 3.0)
        assert hi == pytest.approx(lo, rel=1e-14)


# ---------------------------------------------------------------------------
# Parseval / energy conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_tone_energy_lands_in_its_own_band_and_sums_to_the_true_level(fs):
    """
    A 1 kHz sine at exactly 94 dB SPL, one second long, has SEL = 94 dB
    (SEL = 10*log10(p_rms^2 * 1 s / (p_ref^2 * 1 s))).

    That energy must appear in the 1 kHz band, and summing the energy of every
    band must reconstruct 94 dB - the filter bank must neither lose nor
    manufacture energy.
    """
    # 1000 Hz * 1.0 s = 1000 whole cycles, so the RMS level is exact.
    x = make_sine(1000.0, 94.0, fs, 1.0)
    assert 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) / P_REF) == pytest.approx(94.0, abs=1e-9)

    analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
    sel = analyzer.compute_band_exposure(x)
    idx_1k = int(np.argmin(np.abs(analyzer.center_frequencies - 1000.0)))

    assert sel[idx_1k] == pytest.approx(94.0, abs=0.2)

    total_dB = 10.0 * np.log10(np.sum(10.0 ** (sel / 10.0)))
    assert total_dB == pytest.approx(94.0, abs=0.2)


@pytest.mark.parametrize("fs", SAMPLE_RATES)
def test_tone_leaks_less_than_20dB_into_bands_two_steps_away(fs):
    x = make_sine(1000.0, 94.0, fs, 1.0)
    analyzer = ThirdOctaveAnalyzer(sample_rate=fs)
    sel = analyzer.compute_band_exposure(x)

    freqs = analyzer.center_frequencies
    idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
    for offset in (-2, +2):  # 630 Hz and 1600 Hz
        j = idx_1k + offset
        assert sel[j] < sel[idx_1k] - 20.0, (
            f"fs={fs}: band {freqs[j]:g} Hz is only "
            f"{sel[idx_1k] - sel[j]:.1f} dB below the 1 kHz band"
        )


def test_overall_level_from_analyze_matches_the_tone_level():
    # The 'none' detector is a plain energy average over each hop, so the peak
    # overall level of a steady tone is its true level.
    fs = 48000
    x = make_sine(1000.0, 94.0, fs, 1.0)
    results = ThirdOctaveAnalyzer(sample_rate=fs).analyze(x, time_weighting="none", hop_ms=100.0)
    # Skip the first frame, which contains the filters' startup transient.
    assert float(np.max(results["overall_level_dB"][1:])) == pytest.approx(94.0, abs=0.3)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def test_exponential_detector_matches_explicit_recursion():
    """
    The vectorised lfilter form must equal the textbook recursion

        y[n] = a*x[n] + (1-a)*y[n-1],   a = 1 - exp(-dt/tau),  y[-1] = 0

    to float64 rounding.
    """
    fs = 48000
    tau = 0.125
    x = np.abs(np.random.default_rng(7).normal(0.0, 1.0, 20000)) ** 2

    alpha = 1.0 - np.exp(-1.0 / (fs * tau))
    reference = np.empty_like(x)
    state = 0.0
    for i, v in enumerate(x):
        state = alpha * v + (1.0 - alpha) * state
        reference[i] = state

    got = exponential_detector(x, fs, tau)
    assert np.max(np.abs(got - reference)) < 1e-9


def test_exponential_detector_step_response_reaches_1_minus_1_over_e_at_tau():
    # A unit step into a one-pole RC detector reads 1 - exp(-1) = 0.632 at t=tau.
    fs = 48000
    tau = 0.125
    y = exponential_detector(np.ones(int(2.0 * fs)), fs, tau)
    assert y[int(tau * fs) - 1] == pytest.approx(1.0 - math.exp(-1.0), abs=1e-4)
    assert y[-1] == pytest.approx(1.0, abs=1e-6)


def test_impulse_detector_attack_is_faster_than_decay():
    """
    IEC 61672-1 Impulse weighting is ASYMMETRIC: a 35 ms attack and a 1500 ms
    decay. The attack must reach 1 - 1/e in ~35 ms while the decay takes
    ~1500 ms to fall to 1/e.
    """
    fs = 48000
    step = np.ones(int(1.0 * fs))
    rising = impulse_detector(step, fs)
    attack_idx = int(np.argmax(rising >= 1.0 - math.exp(-1.0)))
    attack_s = attack_idx / fs
    assert attack_s == pytest.approx(0.035, abs=0.002)

    # Burst then silence: measure the decay time constant.
    n = int(4.0 * fs)
    burst = np.zeros(n)
    burst[: int(0.05 * fs)] = 1.0
    out = impulse_detector(burst, fs)
    i1, i2 = int(0.5 * fs), int(3.5 * fs)
    tau_measured = -(i2 - i1) / fs / math.log(out[i2] / out[i1])
    assert tau_measured == pytest.approx(1.5, rel=0.02)
    assert tau_measured > 10 * attack_s


def test_impulse_detector_decays_at_the_1500ms_time_constant():
    # y(t) = y(t0) * exp(-(t-t0)/1.5) exactly, since stage 2 is a decay-limited
    # hold with per-sample factor exp(-1/(fs*1.5)).
    fs = 48000
    n = int(3.0 * fs)
    burst = np.zeros(n)
    burst[: int(0.05 * fs)] = 1.0
    out = impulse_detector(burst, fs)

    t0, t1 = int(0.5 * fs), int(2.5 * fs)
    expected_ratio = math.exp(-(t1 - t0) / fs / 1.5)
    assert out[t1] / out[t0] == pytest.approx(expected_ratio, rel=1e-6)


def test_detectors_handle_empty_input():
    assert exponential_detector(np.array([]), 48000, 0.125).size == 0
    assert impulse_detector(np.array([]), 48000).size == 0


# ---------------------------------------------------------------------------
# Insertion loss
# ---------------------------------------------------------------------------

def test_band_insertion_loss_is_reference_minus_test():
    ref = np.array([120.0, 130.0, 140.0])
    test = np.array([100.0, 105.0, 138.0])
    assert np.allclose(band_insertion_loss(ref, test), [20.0, 25.0, 2.0])


def test_band_insertion_loss_rejects_mismatched_bands():
    # Two recordings analysed at different sample rates yield different numbers
    # of bands; silently broadcasting them would compare unrelated frequencies.
    with pytest.raises(ValueError, match="must match"):
        band_insertion_loss(np.zeros(31), np.zeros(34))
