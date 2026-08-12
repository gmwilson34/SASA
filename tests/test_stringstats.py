"""
test_stringstats.py - first-round pop and shot-string breakdown.

The prediction interval, the energy mean and the one-sided t probability all
have closed forms, so they are recomputed here from their definitions rather
than compared against what the module produced. The refusals are tested for
what they must decline to claim, since a first-round pop quoted from one shot
against three is the archetypal confident wrong number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from stringstats import (
    MIN_STRINGS_FOR_CLAIM,
    MIN_SUBSEQUENT_SHOTS,
    POP_ALPHA,
    compare_with_and_without_first_round,
    energy_average_dB,
    first_round_pop,
    first_round_pop_across_strings,
    string_summary,
)


# ---------------------------------------------------------------------------
# Energy averaging
# ---------------------------------------------------------------------------

def test_energy_mean_of_a_constant_is_that_constant():
    assert energy_average_dB([137.0] * 9) == pytest.approx(137.0, rel=1e-12)


def test_energy_mean_matches_its_definition_and_exceeds_the_arithmetic_mean():
    """
    10*log10(mean(10^(L/10))). For 100 and 110 dB that is 100 + 10*log10(5.5)
    = 107.404 dB, above the arithmetic mean of 105.
    """
    got = energy_average_dB([100.0, 110.0])
    assert got == pytest.approx(100.0 + 10.0 * math.log10(5.5), rel=1e-12)
    assert got > 105.0


def test_energy_mean_ignores_non_finite_levels():
    assert energy_average_dB([120.0, float("nan"), 120.0]) == pytest.approx(120.0)


def test_energy_mean_of_nothing_is_not_a_number():
    assert math.isnan(energy_average_dB([]))


# ---------------------------------------------------------------------------
# First-round pop, single string
# ---------------------------------------------------------------------------

def test_prediction_interval_is_the_closed_form_for_one_new_observation():
    """
    m +/- t(1-alpha, n-1) * s * sqrt(1 + 1/n), recomputed from scipy. This is
    the interval a single further shot should fall inside, and is deliberately
    wider than the confidence interval on the mean.
    """
    rest = [140.0, 140.6, 139.4, 140.2, 139.8, 140.4]
    result = first_round_pop([145.0, *rest])

    n = len(rest)
    mean = float(np.mean(rest))
    sd = float(np.std(rest, ddof=1))
    se_pred = sd * math.sqrt(1.0 + 1.0 / n)
    t_crit = float(stats.t.ppf(1.0 - POP_ALPHA, n - 1))

    assert result.prediction_upper_dB == pytest.approx(mean + t_crit * se_pred, rel=1e-12)
    assert result.prediction_lower_dB == pytest.approx(mean - t_crit * se_pred, rel=1e-12)


def test_the_prediction_interval_is_wider_than_the_confidence_interval():
    """
    Using the confidence interval on the mean would declare a pop on nearly
    every string. sqrt(1 + 1/n) against sqrt(1/n) is the whole difference.
    """
    rest = [140.0, 140.6, 139.4, 140.2, 139.8, 140.4]
    result = first_round_pop([141.0, *rest])
    n = len(rest)
    sd = float(np.std(rest, ddof=1))
    mean = float(np.mean(rest))
    t_crit = float(stats.t.ppf(1.0 - POP_ALPHA, n - 1))
    confidence_upper = mean + t_crit * sd / math.sqrt(n)
    assert result.prediction_upper_dB > confidence_upper


def test_the_p_value_is_the_one_sided_t_probability():
    rest = [140.0, 140.6, 139.4, 140.2, 139.8, 140.4]
    first = 142.5
    result = first_round_pop([first, *rest])

    n = len(rest)
    mean = float(np.mean(rest))
    sd = float(np.std(rest, ddof=1))
    se_pred = sd * math.sqrt(1.0 + 1.0 / n)
    expected = float(stats.t.sf((first - mean) / se_pred, n - 1))
    assert result.p_value == pytest.approx(expected, rel=1e-12)


def test_a_large_pop_against_a_tight_string_is_established():
    rest = [140.0, 140.1, 139.9, 140.05, 139.95, 140.0]
    result = first_round_pop([146.0, *rest])
    assert result.established
    assert result.p_value < POP_ALPHA
    assert result.observed_dB == pytest.approx(146.0 - energy_average_dB(rest), rel=1e-12)


def test_a_small_pop_against_a_loose_string_is_not_established():
    """
    A one-decibel first shot in a string with two decibels of scatter is the
    scatter, and must not be reported as a finding.
    """
    rng = np.random.default_rng(4)
    rest = list(140.0 + rng.normal(0.0, 2.0, 9))
    result = first_round_pop([141.0, *rest])
    assert not result.established
    assert math.isfinite(result.observed_dB)  # still reported, just not claimed


def test_a_quieter_first_shot_is_not_reported_as_negative_pop():
    """
    Pop is a mechanism that can only add energy. A first shot below the interval
    is a fault to investigate, not a negative pop.
    """
    rest = [140.0, 140.1, 139.9, 140.05, 139.95, 140.0]
    result = first_round_pop([132.0, *rest])
    assert not result.established
    assert result.first_shot_quieter
    assert "QUIETER" in result.summary()


def test_a_string_too_short_to_judge_is_refused():
    result = first_round_pop([145.0] + [140.0] * (MIN_SUBSEQUENT_SHOTS - 1))
    assert result.refusal
    assert not result.established
    assert "below the" in result.refusal


def test_the_minimum_string_length_is_accepted():
    rest = [140.0, 140.5, 139.5, 140.2][:MIN_SUBSEQUENT_SHOTS]
    result = first_round_pop([145.0, *rest])
    assert not result.refusal


def test_a_string_with_no_spread_is_refused_rather_than_divided_by_zero():
    result = first_round_pop([145.0] + [140.0] * 8)
    assert result.refusal
    assert "identical" in result.refusal
    assert not result.established


def test_an_empty_string_is_refused():
    assert first_round_pop([]).refusal


def test_a_single_string_result_says_it_is_a_single_observation():
    rest = [140.0, 140.6, 139.4, 140.2, 139.8, 140.4]
    result = first_round_pop([145.0, *rest])
    assert result.basis == "single-string"
    assert any("single observation" in note for note in result.notes)


def test_the_pop_result_serialises_without_non_finite_values():
    rest = [140.0, 140.6, 139.4, 140.2, 139.8, 140.4]
    data = first_round_pop([145.0, *rest]).to_dict()
    for key, value in data.items():
        if isinstance(value, float):
            assert math.isfinite(value)
    assert data["established"] is True


# ---------------------------------------------------------------------------
# First-round pop, across strings
# ---------------------------------------------------------------------------

def _string_with_pop(rng, pop_dB, n=8, sd=0.4):
    rest = list(140.0 + rng.normal(0.0, sd, n))
    return [140.0 + pop_dB + float(rng.normal(0.0, sd)), *rest]


def test_across_strings_estimates_the_planted_pop():
    rng = np.random.default_rng(9)
    strings = [_string_with_pop(rng, 3.0) for _ in range(8)]
    result = first_round_pop_across_strings(strings)

    assert result.basis == "across-strings"
    assert result.established
    assert result.observed_dB == pytest.approx(3.0, abs=0.6)
    lo, hi = result.ci95_dB
    assert lo < result.observed_dB < hi


def test_across_strings_declines_a_claim_when_there_is_no_pop():
    rng = np.random.default_rng(13)
    strings = [_string_with_pop(rng, 0.0) for _ in range(8)]
    result = first_round_pop_across_strings(strings)
    assert not result.established


def test_across_strings_needs_enough_strings():
    rng = np.random.default_rng(2)
    strings = [_string_with_pop(rng, 3.0) for _ in range(MIN_STRINGS_FOR_CLAIM - 1)]
    result = first_round_pop_across_strings(strings)
    assert result.refusal
    assert not result.established


def test_across_strings_reports_which_strings_it_could_not_use():
    rng = np.random.default_rng(6)
    strings = [_string_with_pop(rng, 3.0) for _ in range(4)] + [[140.0, 141.0]]
    result = first_round_pop_across_strings(strings)
    assert result.n_strings == 4
    assert any("too short" in note for note in result.notes)


def test_the_across_strings_probability_matches_a_one_sided_t_test():
    rng = np.random.default_rng(21)
    strings = [_string_with_pop(rng, 2.0) for _ in range(6)]
    result = first_round_pop_across_strings(strings)

    deltas = [
        s[0] - energy_average_dB(s[1:]) for s in strings
    ]
    expected = float(stats.ttest_1samp(deltas, 0.0, alternative="greater").pvalue)
    assert result.p_value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# String summary
# ---------------------------------------------------------------------------

def test_the_with_and_without_first_round_means_differ_by_the_stated_cost():
    levels = [146.0, 140.0, 140.2, 139.8, 140.1, 139.9]
    summary = string_summary(levels)
    assert summary.energy_mean_dB == pytest.approx(energy_average_dB(levels), rel=1e-12)
    assert summary.energy_mean_excluding_first_dB == pytest.approx(
        energy_average_dB(levels[1:]), rel=1e-12
    )
    assert summary.first_round_cost_dB == pytest.approx(
        summary.energy_mean_dB - summary.energy_mean_excluding_first_dB, rel=1e-12
    )
    assert summary.first_round_cost_dB > 0


def test_a_string_with_no_pop_costs_almost_nothing_to_include_the_first_round():
    levels = [140.0] * 10
    summary = string_summary(levels)
    assert summary.first_round_cost_dB == pytest.approx(0.0, abs=1e-9)


def test_percentiles_match_numpy():
    rng = np.random.default_rng(1)
    levels = list(140.0 + rng.normal(0, 1.0, 20))
    summary = string_summary(levels)
    for key, value in summary.percentiles_dB.items():
        assert value == pytest.approx(float(np.percentile(levels, float(key))), rel=1e-12)


def test_the_trend_excludes_the_first_round_so_pop_is_not_read_as_drift():
    """
    A flat string with a large first-round pop has no drift. Including the first
    shot in the regression would produce a strong negative slope out of nothing.
    """
    levels = [150.0] + [140.0, 140.1, 139.9, 140.0, 140.05, 139.95, 140.02, 140.01]
    summary = string_summary(levels)
    assert abs(summary.trend_dB_per_shot) < 0.05
    assert not summary.trend_established


def test_a_real_drift_is_detected():
    levels = [140.0 + 0.5 * i for i in range(12)]
    summary = string_summary(levels)
    assert summary.trend_dB_per_shot == pytest.approx(0.5, rel=1e-9)
    assert summary.trend_established


def test_the_range_is_the_max_minus_the_min():
    levels = [138.0, 141.0, 139.5, 140.0, 142.5]
    summary = string_summary(levels)
    assert summary.range_dB == pytest.approx(142.5 - 138.0, rel=1e-12)


def test_an_empty_string_summarises_without_error():
    summary = string_summary([])
    assert summary.n_shots == 0
    assert math.isnan(summary.energy_mean_dB)


def test_the_summary_serialises():
    levels = [146.0, 140.0, 140.2, 139.8, 140.1, 139.9]
    data = string_summary(levels).to_dict()
    for key in ("energy_mean_dB", "energy_mean_excluding_first_dB",
                "first_round_cost_dB", "percentiles_dB", "first_round_pop"):
        assert key in data


# ---------------------------------------------------------------------------
# Reduction with and without the first round
# ---------------------------------------------------------------------------

def test_both_reductions_are_reported_and_differ_by_the_first_round_penalty():
    """
    Quoting only the better of the two is the commonest way a suppressor test
    flatters its subject, so both must be produced and their difference stated.
    """
    reference = [165.0] * 10
    test = [148.0, 140.0, 140.0, 140.0, 140.0, 140.0]

    result = compare_with_and_without_first_round(reference, test)
    including = 165.0 - energy_average_dB(test)
    excluding = 165.0 - energy_average_dB(test[1:])

    assert result["reduction_including_first_dB"] == pytest.approx(including, abs=0.01)
    assert result["reduction_excluding_first_dB"] == pytest.approx(excluding, abs=0.01)
    assert result["first_round_penalty_dB"] == pytest.approx(excluding - including, abs=0.01)
    # Excluding the pop always flatters the suppressor.
    assert result["reduction_excluding_first_dB"] > result["reduction_including_first_dB"]


def test_with_no_pop_the_two_reductions_agree():
    reference = [165.0] * 10
    test = [140.0] * 10
    result = compare_with_and_without_first_round(reference, test)
    assert result["first_round_penalty_dB"] == pytest.approx(0.0, abs=1e-6)
