"""
test_metrics.py - pins the reported acoustic metrics to analytic oracles.

The Friedlander blast p(t) = P0 (1 - t/T) exp(-t/T) is the workhorse here: its
peak, positive-phase duration and specific impulse are all known exactly, so it
turns rise time, A-duration and impulse into closed-form assertions rather than
plausibility checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibration import P_REF, energy_average_dB
from conftest import make_decaying_sinusoid, make_friedlander, make_sine
from metrics import (
    AggregateMetrics,
    ShotMetrics,
    compute_a_duration,
    compute_aggregate_metrics,
    compute_b_duration,
    compute_crest_factor,
    compute_exposure_level,
    compute_hazard,
    compute_insertion_loss,
    compute_kurtosis,
    compute_rise_time,
    compute_shot_metrics,
    find_blast_span,
    signal_envelope,
    EIGHT_HOURS_S,
)


def _friedlander_window(P0, T, fs, pre_ms=10.0, duration=0.05):
    """Friedlander blast preceded by silence, as an extracted shot window."""
    return np.concatenate(
        [np.zeros(int(pre_ms / 1000.0 * fs)), make_friedlander(P0, T, fs, duration)]
    )


# ---------------------------------------------------------------------------
# Peak level
# ---------------------------------------------------------------------------

def test_friedlander_peak_level_is_exactly_140_dB():
    # 200 Pa -> 20*log10(200 / 20e-6) = 20*log10(1e7) = 140.000 dB exactly.
    fs = 96000
    p = _friedlander_window(200.0, 0.001, fs)
    assert float(np.max(np.abs(p))) == pytest.approx(200.0, rel=1e-12)

    m = compute_shot_metrics(p, fs, compute_bands=False)
    assert m.Lpeak_Z == pytest.approx(140.0, abs=0.05)


@pytest.mark.parametrize("P0,expected_dB", [(2.0, 100.0), (20.0, 120.0), (2000.0, 160.0)])
def test_peak_level_scales_20_log10(P0, expected_dB):
    # Each factor of 10 in pressure is exactly 20 dB.
    fs = 96000
    m = compute_shot_metrics(_friedlander_window(P0, 0.001, fs), fs, compute_bands=False)
    assert m.Lpeak_Z == pytest.approx(expected_dB, abs=0.05)


# ---------------------------------------------------------------------------
# A-duration and specific impulse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", [48000, 96000, 192000])
@pytest.mark.parametrize("T", [0.0005, 0.001, 0.003])
def test_friedlander_a_duration_equals_T(fs, T):
    """
    p(t) = P0 (1 - t/T) e^{-t/T} is positive on [0, T) and crosses zero at
    exactly t = T, so the initial positive overpressure phase lasts exactly T.
    """
    p = _friedlander_window(200.0, T, fs)
    a_ms, _, _ = compute_a_duration(p, fs)
    assert a_ms == pytest.approx(T * 1000.0, rel=0.05)


@pytest.mark.parametrize("fs", [96000, 192000])
@pytest.mark.parametrize("T", [0.0005, 0.001, 0.003])
def test_friedlander_specific_impulse_is_P0_T_over_e(fs, T):
    """
    int_0^T P0 (1 - t/T) e^{-t/T} dt
        = P0*T * int_0^1 (1-u) e^{-u} du
        = P0*T * [(1 - e^-1) - (1 - 2 e^-1)]
        = P0*T / e
    """
    P0 = 200.0
    p = _friedlander_window(P0, T, fs)
    _, impulse, peak = compute_a_duration(p, fs)

    expected = P0 * T / math.e
    assert impulse == pytest.approx(expected, rel=0.05)
    assert peak == pytest.approx(P0, rel=1e-3)


@pytest.mark.parametrize("T", [0.0005, 0.001, 0.003])
def test_friedlander_specific_impulse_at_48kHz_within_quadrature_bias(T):
    """
    The integral is a LEFT Riemann sum, which over-estimates a monotonically
    decreasing integrand by about (p(0) - p(T))/2 * dt = P0*dt/2. Relative to
    the true impulse P0*T/e that is e*dt/(2T), i.e. 5.7% for T = 0.5 ms at
    48 kHz. The bound below is that analytic bias, doubled for headroom.
    """
    fs, P0 = 48000, 200.0
    dt = 1.0 / fs
    expected = P0 * T / math.e
    bias = math.e * dt / (2.0 * T)

    _, impulse, _ = compute_a_duration(_friedlander_window(P0, T, fs), fs)
    assert impulse >= expected * (1.0 - 1e-6)         # left sum is an upper bound
    assert impulse <= expected * (1.0 + 2.0 * bias)


def test_a_duration_is_not_decided_by_one_pre_blast_noise_sample():
    """
    REGRESSION: the phase sign used to be read from the single sample sitting at
    the envelope onset. Because the analytic-signal envelope leads the waveform,
    that sample is pre-blast baseline, so the sign of the last noise sample
    before the shock front decided whether the positive or the negative phase
    was measured. With silence it returned 0 ms; with 0.01 Pa of noise it
    returned 1.01 ms or 0.01 ms depending on the seed.
    """
    fs, P0, T = 96000, 200.0, 0.001
    clean = _friedlander_window(P0, T, fs)
    for seed in range(8):
        noisy = clean + np.random.default_rng(seed).normal(0.0, 0.01, clean.size)
        a_ms, impulse, peak = compute_a_duration(noisy, fs)
        assert a_ms == pytest.approx(T * 1000.0, rel=0.05), f"seed {seed}"
        assert impulse == pytest.approx(P0 * T / math.e, rel=0.05), f"seed {seed}"
        assert peak == pytest.approx(P0, rel=0.01), f"seed {seed}"


def test_a_duration_handles_a_rarefaction_first_waveform():
    # An inverted blast has the same positive-phase length; the reported peak
    # overpressure carries the sign of the phase that was measured.
    fs, T = 96000, 0.001
    p = -_friedlander_window(200.0, T, fs)
    a_ms, impulse, peak = compute_a_duration(p, fs)
    assert a_ms == pytest.approx(T * 1000.0, rel=0.05)
    assert impulse == pytest.approx(200.0 * T / math.e, rel=0.05)
    assert peak == pytest.approx(-200.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Rise time
# ---------------------------------------------------------------------------

def test_linear_ramp_rise_time_is_80_percent_of_the_ramp():
    # A linear ramp from 0 to P0 over R seconds crosses 10% at 0.1R and 90% at
    # 0.9R, so the 10-90% rise time is exactly 0.8R.
    fs, R = 192000, 0.001
    n_ramp = int(R * fs)
    t = np.arange(n_ramp) / fs
    ramp = 200.0 * t / R
    decay = 200.0 * np.exp(-(np.arange(int(0.02 * fs)) / fs) / 0.002)
    p = np.concatenate([np.zeros(int(0.005 * fs)), ramp, decay])

    rise_us, resolved = compute_rise_time(p, fs)
    assert rise_us == pytest.approx(0.8 * R * 1e6, rel=0.02)
    assert resolved is True


def test_instantaneous_shock_front_is_flagged_unresolved():
    # A Friedlander wave rises in zero time, so at any real sample rate the
    # measured rise spans under two samples and must be reported as an upper
    # bound rather than as a measurement.
    fs = 96000
    _, resolved = compute_rise_time(_friedlander_window(200.0, 0.001, fs), fs)
    assert resolved is False


# ---------------------------------------------------------------------------
# Sound exposure level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "level_dB,duration_s,expected_SEL",
    [
        (100.0, 0.1, 90.0),     # 10*log10(p^2 * 0.1 / (p_ref^2 * 1)) = 100 - 10
        (100.0, 1.0, 100.0),    # one second of a level IS its SEL
        (94.0, 10.0, 104.0),    # ten seconds adds exactly 10 dB
        (120.0, 0.001, 90.0),   # 120 - 30
    ],
)
def test_SEL_of_a_constant_rms_burst(level_dB, duration_s, expected_SEL):
    """
    A signal of constant RMS pressure p lasting D seconds has

        SEL = 10*log10( p^2 * D / (p_ref^2 * 1 s) ) = L + 10*log10(D)
    """
    fs = 48000
    # 1000 Hz * integral duration -> whole cycles -> exact RMS.
    x = make_sine(1000.0, level_dB, fs, duration_s)
    assert compute_exposure_level(x, fs) == pytest.approx(expected_SEL, abs=0.2)


def test_SEL_of_silence_is_minus_infinity_floor():
    assert compute_exposure_level(np.array([]), 48000) == float("-inf")


# ---------------------------------------------------------------------------
# B-duration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.003, 0.005, 0.010, 0.020])
def test_b_duration_of_a_decaying_sinusoid(tau):
    """
    For A*exp(-t/tau)*sin(wt) the analytic-signal ENVELOPE is A*exp(-t/tau),
    which falls 20 dB (a factor of 10) at exactly

        t = tau * ln(10) = 2.302585 * tau

    so B-duration should be tau*ln(10)*1000 ms.

    The superseded implementation counted samples whose |p| exceeded the
    threshold. |p| dips below any threshold twice per cycle, so it understated
    B-duration by roughly 25% - which is exactly the direction that flatters a
    suppressor.
    """
    fs = 96000
    x = make_decaying_sinusoid(100.0, tau, 1000.0, fs, int(0.5 * fs))
    expected_ms = tau * math.log(10.0) * 1000.0
    assert compute_b_duration(x, fs) == pytest.approx(expected_ms, rel=0.15)


def test_b_duration_uses_the_envelope_not_sample_counting():
    # Direct demonstration of the superseded behaviour: counting |p| samples
    # above the -20 dB threshold gives ~2/pi * arcsin-weighted duty cycle, well
    # under the true envelope duration.
    fs, tau = 96000, 0.005
    x = make_decaying_sinusoid(100.0, tau, 1000.0, fs, int(0.5 * fs))
    threshold = float(np.max(np.abs(x))) * 0.1

    naive_ms = float(np.count_nonzero(np.abs(x) >= threshold)) / fs * 1000.0
    envelope_ms = compute_b_duration(x, fs)
    truth_ms = tau * math.log(10.0) * 1000.0

    assert naive_ms < 0.85 * truth_ms          # the old code's ~25% shortfall
    assert envelope_ms == pytest.approx(truth_ms, rel=0.15)


def test_signal_envelope_of_a_decaying_sinusoid_is_the_exponential():
    fs, tau = 96000, 0.010
    n = int(0.1 * fs)
    x = make_decaying_sinusoid(100.0, tau, 2000.0, fs, n)
    env = signal_envelope(x)
    t = np.arange(n) / fs
    # Compare away from the array edges, where the Hilbert transform rings.
    mid = slice(int(0.005 * fs), int(0.030 * fs))
    assert np.allclose(env[mid], 100.0 * np.exp(-t[mid] / tau), rtol=0.02)


# ---------------------------------------------------------------------------
# Character metrics must describe the blast, not the window
# ---------------------------------------------------------------------------

def _windowed_blast(fs, pre_ms, total_ms):
    n, pre = int(total_ms / 1000.0 * fs), int(pre_ms / 1000.0 * fs)
    x = np.zeros(n)
    blast = make_decaying_sinusoid(200.0, 0.004, 900.0, fs, min(n - pre, int(0.05 * fs)))
    x[pre:pre + blast.size] += blast
    return x


def test_character_metrics_are_window_independent():
    """
    REGRESSION: kurtosis, crest factor and spectral centroid computed over the
    whole extraction window measure the window, not the shot - a window that is
    90% silence scores differently from one that is 50% silence, so the same
    round changes character when the operator changes the margins. Scoping them
    to the blast span fixes that.
    """
    fs = 96000
    short = compute_shot_metrics(_windowed_blast(fs, 20, 100), fs, compute_bands=False)
    long = compute_shot_metrics(_windowed_blast(fs, 100, 500), fs, compute_bands=False)

    assert short.duration_s == pytest.approx(0.100, abs=1e-6)
    assert long.duration_s == pytest.approx(0.500, abs=1e-6)

    assert short.kurtosis == pytest.approx(long.kurtosis, rel=0.03)
    assert short.crest_factor_dB == pytest.approx(long.crest_factor_dB, rel=0.03)
    assert short.spectral_centroid_Hz == pytest.approx(long.spectral_centroid_Hz, rel=0.03)

    # And the blast span itself is the same physical event in both.
    assert short.blast_duration_s == pytest.approx(long.blast_duration_s, rel=0.03)


def test_find_blast_span_brackets_the_event_not_the_window():
    fs = 96000
    x = _windowed_blast(fs, 100, 500)
    start, stop = find_blast_span(x, fs)
    # The blast starts at 100 ms; the span must begin within the 1 ms guard.
    assert abs(start - int(0.100 * fs)) <= int(0.002 * fs)
    # And it must be far shorter than the 500 ms window.
    assert (stop - start) / fs < 0.05


def test_crest_factor_of_a_sine_is_3_01_dB():
    # peak/rms = sqrt(2) -> 20*log10(sqrt(2)) = 3.0103 dB.
    fs = 48000
    x = make_sine(1000.0, 94.0, fs, 1.0)
    assert compute_crest_factor(x) == pytest.approx(20.0 * math.log10(math.sqrt(2.0)), abs=1e-6)


def test_kurtosis_of_gaussian_noise_is_zero():
    # Excess kurtosis (Fisher) of a Gaussian is 0 by definition.
    x = np.random.default_rng(11).normal(0.0, 1.0, 2_000_000)
    assert compute_kurtosis(x) == pytest.approx(0.0, abs=0.02)


def test_kurtosis_of_a_sine_is_minus_1_5():
    # A pure sine has m4/m2^2 = 1.5, so excess kurtosis = 1.5 - 3 = -1.5 exactly.
    fs = 48000
    x = make_sine(1000.0, 94.0, fs, 1.0)
    assert compute_kurtosis(x) == pytest.approx(-1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Hazard
# ---------------------------------------------------------------------------

def test_hazard_LAeq8h_closed_form():
    """
    LAeq8h = LAE_energy_mean + 10*log10(N) - 10*log10(28800)

    For N = 100 rounds each at LAE = 100 dB:
        100 + 20 - 44.5939 = 75.4061 dB
    """
    hazard = compute_hazard([100.0] * 100)
    expected = 100.0 + 10.0 * math.log10(100.0) - 10.0 * math.log10(EIGHT_HOURS_S)
    assert expected == pytest.approx(75.40602, abs=1e-4)

    assert hazard.n_rounds == 100
    assert hazard.LAE_mean == pytest.approx(100.0, abs=1e-9)
    assert hazard.LAeq8h_dB == pytest.approx(expected, abs=1e-9)
    assert hazard.exceeds_limit is False  # 75.4 dB is under the 85 dB criterion


def test_hazard_dose_is_the_3dB_exchange_rate():
    # dose = 100 * 10^((LAeq8h - criterion)/10); at exactly the criterion it is
    # 100%, and 3 dB above it is 200%.
    at_limit = compute_hazard([85.0 + 10.0 * math.log10(EIGHT_HOURS_S)])
    assert at_limit.LAeq8h_dB == pytest.approx(85.0, abs=1e-9)
    assert at_limit.dose_percent == pytest.approx(100.0, rel=1e-9)


def test_hazard_allowable_rounds_inverts_LAeq8h():
    """
    allowable N solves criterion = LAE + 10*log10(N) - 10*log10(28800), i.e.
        N = 10^((criterion - LAE + 10*log10(28800)) / 10)
    For LAE = 100 dB and an 85 dB criterion that is 10^2.9594 = 910.9 rounds.
    """
    hazard = compute_hazard([100.0] * 100)
    expected_N = 10.0 ** ((85.0 - 100.0 + 10.0 * math.log10(EIGHT_HOURS_S)) / 10.0)
    # 10^((85 - 100 + 44.593925)/10) = 10^2.9593925 = 910.736 rounds/day
    assert expected_N == pytest.approx(910.736, abs=0.01)
    assert hazard.allowable_rounds == pytest.approx(expected_N, rel=1e-9)

    # Feeding that many rounds back through must land exactly on the criterion.
    round_trip = (
        hazard.LAE_mean
        + 10.0 * math.log10(hazard.allowable_rounds)
        - 10.0 * math.log10(EIGHT_HOURS_S)
    )
    assert round_trip == pytest.approx(85.0, abs=1e-9)


def test_hazard_hearing_protection_subtracts_NRR():
    plain = compute_hazard([120.0] * 50)
    protected = compute_hazard([120.0] * 50, protection_NRR_dB=25.0)
    assert protected.LAeq8h_dB == pytest.approx(plain.LAeq8h_dB - 25.0, abs=1e-9)
    # 25 dB of protection multiplies the allowable round count by 10^2.5.
    assert protected.allowable_rounds == pytest.approx(
        plain.allowable_rounds * 10.0 ** 2.5, rel=1e-9
    )


def test_hazard_of_no_shots():
    hazard = compute_hazard([])
    assert hazard.n_rounds == 0
    assert math.isnan(hazard.LAeq8h_dB)
    assert hazard.allowable_rounds == float("inf")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _shot(level_dB: float, shot_number: int = 0) -> ShotMetrics:
    """A ShotMetrics whose every level is `level_dB`."""
    return ShotMetrics(
        Lpeak_Z=level_dB, Lpeak_A=level_dB, Lpeak_C=level_dB,
        LAE=level_dB, LZE=level_dB, LCE=level_dB,
        LAFmax=level_dB, LASmax=level_dB, LZFmax=level_dB, LZSmax=level_dB,
        LAImax=level_dB, LZImax=level_dB,
        shot_number=shot_number,
    )


def test_aggregate_levels_use_the_energy_mean_not_the_arithmetic_mean():
    """
    Two shots 10 dB apart energy-average to

        10*log10((10^10 + 10^11)/2) = 10*log10(5.5e10) = 107.4036 dB

    The arithmetic mean, 105 dB, understates the true energy by 2.4 dB.
    """
    agg = compute_aggregate_metrics([_shot(100.0, 1), _shot(110.0, 2)])
    expected = 10.0 * math.log10((1e10 + 1e11) / 2.0)
    assert expected == pytest.approx(107.4036, abs=1e-4)

    for key in ("Lpeak_Z", "Lpeak_C", "LAE", "LZE", "LAImax"):
        assert agg.stats[key].mean == pytest.approx(expected, abs=1e-9), key
        assert abs(agg.stats[key].mean - 105.0) > 2.0

    assert agg.stats["LAE"].mean == pytest.approx(energy_average_dB([100.0, 110.0]), abs=1e-12)
    assert agg.LAE_mean == pytest.approx(expected, abs=1e-9)


def test_aggregate_reports_sample_standard_deviation_ddof_1():
    """
    The shots are a SAMPLE of the weapon's behaviour, so dispersion is the
    sample standard deviation. For [100, 110]:
        ddof=1 -> sqrt(((100-105)^2 + (110-105)^2) / 1) = sqrt(50) = 7.0711
        ddof=0 -> 5.0
    """
    agg = compute_aggregate_metrics([_shot(100.0, 1), _shot(110.0, 2)])
    stats = agg.stats["Lpeak_Z"]
    assert stats.n == 2
    assert stats.std == pytest.approx(math.sqrt(50.0), abs=1e-12)
    assert stats.std == pytest.approx(7.0710678, abs=1e-6)
    assert abs(stats.std - 5.0) > 1.0  # not the population std

    # min/max/median are plain decibel order statistics.
    assert stats.minimum == 100.0
    assert stats.maximum == 110.0
    assert stats.median == pytest.approx(105.0, abs=1e-12)
    # 95% CI half width = 1.96 * s / sqrt(n)
    assert stats.ci95_half_width == pytest.approx(1.96 * math.sqrt(50.0) / math.sqrt(2), abs=1e-9)


def test_aggregate_arithmetic_mean_for_non_level_metrics():
    # Durations and kurtosis are not levels, so they use the arithmetic mean.
    a, b = _shot(100.0, 1), _shot(100.0, 2)
    a.a_duration_ms, b.a_duration_ms = 1.0, 3.0
    agg = compute_aggregate_metrics([a, b])
    assert agg.stats["a_duration_ms"].mean == pytest.approx(2.0, abs=1e-12)


def test_aggregate_excludes_invalid_shots_by_default():
    good, bad = _shot(100.0, 1), _shot(160.0, 2)
    bad.valid = False
    agg = compute_aggregate_metrics([good, bad])
    assert agg.n_shots == 2 and agg.n_valid == 1
    assert agg.stats["Lpeak_Z"].mean == pytest.approx(100.0, abs=1e-9)

    included = compute_aggregate_metrics([good, bad], include_invalid=True)
    assert included.n_valid == 2


# ---------------------------------------------------------------------------
# Insertion loss
# ---------------------------------------------------------------------------

def test_insertion_loss_of_a_known_20_dB_difference():
    reference = compute_aggregate_metrics([_shot(160.0, i) for i in range(1, 6)])
    test = compute_aggregate_metrics([_shot(140.0, i) for i in range(1, 6)])

    results = {il.metric: il for il in compute_insertion_loss(reference, test)}
    assert results, "no metrics were compared"
    for name, il in results.items():
        assert il.reference_dB == pytest.approx(160.0, abs=1e-9), name
        assert il.test_dB == pytest.approx(140.0, abs=1e-9), name
        assert il.reduction_dB == pytest.approx(20.0, abs=1e-9), name
        assert il.reference_n == 5 and il.test_n == 5
        # Identical shots -> zero dispersion -> zero confidence interval.
        assert il.combined_ci95 == pytest.approx(0.0, abs=1e-12)


def test_insertion_loss_is_zero_for_identical_strings():
    agg = compute_aggregate_metrics([_shot(150.0, i) for i in range(1, 4)])
    for il in compute_insertion_loss(agg, agg):
        assert il.reduction_dB == pytest.approx(0.0, abs=1e-12)


def test_insertion_loss_skips_metrics_absent_from_either_side():
    empty = AggregateMetrics(n_shots=0, n_valid=0)
    agg = compute_aggregate_metrics([_shot(150.0, 1)])
    assert compute_insertion_loss(agg, empty) == []
