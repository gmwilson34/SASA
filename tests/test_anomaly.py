"""
test_anomaly.py - shot-string outlier detection.

The robust statistics here have exact hand-computable answers: the median
absolute deviation of a small integer set is an integer, and the modified
z-score is then a closed-form multiple of 0.6745. Every threshold assertion
below is against a value derived by hand, not against what the code returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from scipy import stats

from anomaly import (
    FAMILY_ALPHA,
    MIN_SHOTS_FOR_STATISTICS,
    MIN_SNR_dB,
    NORMAL_QUARTILE,
    SEVERITY_EXCLUDE,
    SEVERITY_REVIEW,
    esd_critical_value,
    generalised_esd,
    median_absolute_deviation,
    modified_z_scores,
    review_shot_string,
)


@dataclass
class FakeShot:
    """Minimal stand-in carrying the ShotMetrics attribute names anomaly.py reads."""
    shot_number: int
    Lpeak_Z: float = 160.0
    LAE: float = 130.0
    spectral_centroid_Hz: float = 1200.0
    a_duration_ms: float = 0.80
    rise_time_us: float = 30.0
    snr_dB: float = 40.0
    clipped: bool = False
    window_truncated: bool = False
    rise_time_resolved: bool = True
    valid: bool = True


def make_string(n: int, **overrides) -> List[FakeShot]:
    """A clean string of n shots with a little spread, so the MAD is non-zero."""
    rng = np.random.default_rng(12345)
    shots = []
    for i in range(n):
        shot = FakeShot(
            shot_number=i + 1,
            Lpeak_Z=160.0 + float(rng.normal(0, 0.5)),
            LAE=130.0 + float(rng.normal(0, 0.5)),
            spectral_centroid_Hz=1200.0 + float(rng.normal(0, 30.0)),
            a_duration_ms=0.80 + float(rng.normal(0, 0.02)),
            rise_time_us=30.0 + float(rng.normal(0, 2.0)),
        )
        for k, v in overrides.items():
            setattr(shot, k, v)
        shots.append(shot)
    return shots


# ---------------------------------------------------------------------------
# Robust statistics, against hand-computed values
# ---------------------------------------------------------------------------

def test_mad_of_a_hand_computed_set():
    """
    x = [1, 2, 3, 4, 100]; median = 3; |x - 3| = [2, 1, 0, 1, 97];
    median of that is 1.
    """
    assert median_absolute_deviation(np.array([1.0, 2, 3, 4, 100])) == 1.0


def test_mad_with_an_even_count_averages_the_middle_pair():
    """
    x = [1, 2, 3, 4]; median = 2.5; |x - 2.5| = [1.5, .5, .5, 1.5];
    median of that is 1.0.
    """
    assert median_absolute_deviation(np.array([1.0, 2, 3, 4])) == 1.0


def test_mad_of_identical_values_is_zero():
    assert median_absolute_deviation(np.full(7, 5.0)) == 0.0


def test_modified_z_is_the_closed_form_multiple_of_the_normal_quartile():
    """
    With median 3 and MAD 1, M_i = 0.6745*(x_i - 3). The planted 100 gives
    0.6745*97 = 65.4265 exactly.
    """
    z = modified_z_scores(np.array([1.0, 2, 3, 4, 100]))
    assert z[4] == pytest.approx(NORMAL_QUARTILE * 97.0, rel=1e-15)
    assert z[0] == pytest.approx(NORMAL_QUARTILE * -2.0, rel=1e-15)
    assert z[2] == 0.0


def test_modified_z_of_a_constant_string_is_all_zeros():
    """Zero dispersion means no deviation can be measured, not a division by zero."""
    z = modified_z_scores(np.full(9, 42.0))
    assert np.all(z == 0.0)
    assert np.all(np.isfinite(z))


def test_modified_z_ignores_non_finite_entries():
    z = modified_z_scores(np.array([1.0, 2.0, 3.0, np.nan, 4.0]))
    assert z[3] == 0.0
    assert np.all(np.isfinite(z))


def test_the_normal_quartile_constant_is_the_published_value():
    assert NORMAL_QUARTILE == 0.6745
    # 0.6745 is the 0.75 quantile of the standard normal, to four places.
    assert stats.norm.ppf(0.75) == pytest.approx(NORMAL_QUARTILE, abs=5e-5)


# ---------------------------------------------------------------------------
# Generalised ESD: exact critical values and a MEASURED error rate
# ---------------------------------------------------------------------------

def test_esd_critical_value_matches_rosners_closed_form():
    """
    lambda_i = (n-i)*t / sqrt((n-i-1 + t^2)*(n-i+1)), with t the (1-p) quantile
    of Student's t on n-i-1 df and p = alpha/(2*(n-i+1)). Recomputed here
    directly from scipy rather than taken from the implementation.
    """
    for n in (10, 25, 40):
        for i in (1, 2, 3):
            nu = n - i - 1
            p = 1.0 - 0.05 / (2.0 * (n - i + 1))
            t = stats.t.ppf(p, nu)
            expected = (n - i) * t / math.sqrt((nu + t * t) * (n - i + 1))
            assert esd_critical_value(n, i, 0.05) == pytest.approx(expected, rel=1e-12)


def test_esd_critical_value_is_infinite_without_degrees_of_freedom():
    """A step with no df cannot be failed, so nothing can be declared an outlier there."""
    assert esd_critical_value(5, 4, 0.05) == float("inf")
    assert esd_critical_value(3, 2, 0.05) == float("inf")


def test_esd_critical_value_tightens_as_the_sample_grows():
    """More data makes a given studentised deviation less easy to explain by chance."""
    values = [esd_critical_value(n, 1, 0.05) for n in (10, 20, 40, 80)]
    assert values == sorted(values)


def test_esd_holds_its_nominal_false_alarm_rate_on_clean_gaussian_strings():
    """
    The claim the module makes about itself, measured rather than assumed: on
    clean normal data the test flags a string at about alpha. This is what the
    modified z-score rule failed to do, at any sample size.

    2000 trials at alpha = 0.05 has a standard error of sqrt(.05*.95/2000) =
    0.0049, so a 4-sigma band is 0.05 +/- 0.02.
    """
    rng = np.random.default_rng(20260812)
    for n in (10, 20, 40):
        false_alarms = sum(
            1 for _ in range(2000)
            if generalised_esd(rng.normal(0.0, 1.0, n), alpha=0.05).outlier_indices
        )
        rate = false_alarms / 2000
        assert 0.03 <= rate <= 0.07, f"n={n} gave a false-alarm rate of {rate:.3f}"


def test_esd_detects_a_large_outlier_most_of_the_time():
    """
    Power, measured. At twenty shots a 5-sigma outlier must be caught in the
    large majority of trials, otherwise the feature is decorative.
    """
    rng = np.random.default_rng(31415)
    hits = 0
    for _ in range(1000):
        sample = rng.normal(0.0, 1.0, 20)
        sample[0] += 5.0
        if 0 in generalised_esd(sample, alpha=0.05).outlier_indices:
            hits += 1
    assert hits / 1000 > 0.80


def test_esd_resists_masking_by_a_second_outlier():
    """
    Two outliers together inflate the standard deviation enough that neither is
    extreme on its own. The iterative procedure must still find both, which a
    single-pass test cannot.
    """
    sample = np.concatenate([np.linspace(-1.0, 1.0, 30), np.array([9.0, 9.2])])
    found = set(generalised_esd(sample, alpha=0.05).outlier_indices)
    assert {30, 31} <= found


def test_search_depth_is_one_tenth_of_the_string():
    """
    The depth is what holds the realised false-alarm rate at its nominal value;
    a deeper search inflates it. Short strings still get one step.
    """
    assert len(generalised_esd(np.random.default_rng(1).normal(0, 1, 10),
                               alpha=0.05).test_statistics) == 1
    assert len(generalised_esd(np.random.default_rng(1).normal(0, 1, 30),
                               alpha=0.05).test_statistics) == 3


def test_esd_reports_nothing_on_a_constant_sample():
    result = generalised_esd(np.full(12, 7.0), alpha=0.05)
    assert result.outlier_indices == []


def test_esd_declines_on_a_sample_below_the_minimum():
    result = generalised_esd(np.arange(float(MIN_SHOTS_FOR_STATISTICS - 1)), alpha=0.05)
    assert not result.applied
    assert result.outlier_indices == []


def test_esd_never_reports_a_non_finite_entry_as_an_outlier():
    sample = np.array([1.0, 2.0, 3.0, 2.5, 1.5, np.nan, 2.2, np.inf, 1.9])
    result = generalised_esd(sample, alpha=0.05)
    for index in result.outlier_indices:
        assert math.isfinite(sample[index])


def test_family_alpha_is_split_across_the_metrics_tested():
    """
    Five metrics at the family rate would trip a clean string about a quarter of
    the time; the per-metric level must be the corrected one.
    """
    report = review_shot_string(make_string(12))
    assert report.alpha == pytest.approx(FAMILY_ALPHA / 5.0)


# ---------------------------------------------------------------------------
# Per-shot conditions
# ---------------------------------------------------------------------------

def test_a_clipped_shot_is_excluded_not_merely_flagged():
    shots = make_string(8)
    shots[3].clipped = True
    report = review_shot_string(shots)
    assert 4 in report.shots_to_exclude()
    codes = {f.code for f in report.for_shot(4)}
    assert "clipped" in codes
    assert any(f.severity == SEVERITY_EXCLUDE for f in report.for_shot(4))


def test_a_clipped_shot_is_caught_even_in_a_string_too_short_for_statistics():
    """Per-shot conditions do not need a population to compare against."""
    shots = make_string(2)
    shots[0].clipped = True
    report = review_shot_string(shots)
    assert not report.statistics_applied
    assert report.shots_to_exclude() == [1]


def test_a_truncated_window_is_flagged_for_review():
    shots = make_string(8)
    shots[2].window_truncated = True
    report = review_shot_string(shots)
    assert 3 in report.shots_to_review()
    assert "truncated" in {f.code for f in report.for_shot(3)}


def test_low_snr_is_flagged_with_its_quantified_energy_error():
    """
    At the boundary the noise contributes 10**(-snr/10) of the total energy, so
    the error in the exposure level is 10*log10(1 + 10**(-snr/10)) dB. At 8 dB
    that is 0.6389 dB, and the message must carry it.
    """
    shots = make_string(8)
    shots[5].snr_dB = 8.0
    report = review_shot_string(shots)
    flags = [f for f in report.for_shot(6) if f.code == "low_snr"]
    assert len(flags) == 1
    expected_error = 10.0 * math.log10(1.0 + 10.0 ** (-8.0 / 10.0))
    assert expected_error == pytest.approx(0.6389, abs=1e-4)
    assert f"{expected_error:.2f}" in flags[0].message


def test_snr_exactly_at_the_limit_is_not_flagged():
    shots = make_string(8)
    shots[1].snr_dB = MIN_SNR_dB
    report = review_shot_string(shots)
    assert "low_snr" not in {f.code for f in report.for_shot(2)}


def test_unresolved_rise_time_is_information_not_a_review_item():
    shots = make_string(8)
    shots[0].rise_time_resolved = False
    report = review_shot_string(shots)
    codes = {f.code for f in report.for_shot(1)}
    assert "rise_time_unresolved" in codes
    assert 1 not in report.shots_to_review()


# ---------------------------------------------------------------------------
# String statistics
# ---------------------------------------------------------------------------

def test_a_planted_level_outlier_is_the_only_shot_flagged():
    shots = make_string(12)
    shots[7].Lpeak_Z += 15.0
    report = review_shot_string(shots)
    assert report.statistics_applied
    assert report.shots_to_review() == [8]


def test_a_clean_string_raises_nothing():
    report = review_shot_string(make_string(12))
    assert report.statistics_applied
    assert report.is_clean
    assert report.shots_to_review() == []
    assert report.shots_to_exclude() == []


def test_a_character_outlier_is_caught_even_when_the_level_is_normal():
    """
    A reflection measured as if it were the muzzle blast has an ordinary level
    but the wrong spectral balance, which is exactly the case a level-only check
    misses.
    """
    shots = make_string(12)
    shots[4].spectral_centroid_Hz += 900.0
    report = review_shot_string(shots)
    assert report.shots_to_review() == [5]
    flags = [f for f in report.for_shot(5) if f.metric == "spectral_centroid_Hz"]
    assert flags and flags[0].severity == SEVERITY_REVIEW


def test_the_flag_reports_the_deviation_it_measured():
    shots = make_string(12)
    shots[2].Lpeak_Z += 20.0
    report = review_shot_string(shots)
    flag = next(f for f in report.for_shot(3) if f.metric == "Lpeak_Z")
    assert flag.esd_statistic > flag.esd_critical
    assert abs(flag.modified_z) > 3.0
    assert flag.value == pytest.approx(shots[2].Lpeak_Z)
    # The stated median must be the median of the shots actually compared.
    assert flag.string_median == pytest.approx(
        float(np.median([s.Lpeak_Z for s in shots]))
    )


def test_a_short_string_refuses_to_judge_outliers():
    """
    Below the minimum, a single bad shot inflates the MAD enough to mask itself,
    so no statistical flag is raised and the report says why.
    """
    shots = make_string(MIN_SHOTS_FOR_STATISTICS - 1)
    shots[0].Lpeak_Z += 40.0
    report = review_shot_string(shots)
    assert not report.statistics_applied
    assert report.shots_to_review() == []
    assert any("below the" in n for n in report.notes)


def test_the_minimum_string_length_does_permit_statistics():
    shots = make_string(MIN_SHOTS_FOR_STATISTICS)
    report = review_shot_string(shots)
    assert report.statistics_applied


def test_clipped_shots_do_not_participate_in_the_string_statistics():
    """
    A clipped shot's level is a lower bound, so letting it into the median would
    drag the reference the other shots are judged against.
    """
    shots = make_string(10)
    shots[0].clipped = True
    shots[0].Lpeak_Z = 300.0
    report = review_shot_string(shots)
    assert report.n_evaluated == 9
    assert report.shots_to_review() == []


def test_a_string_with_no_dispersion_says_so_rather_than_flagging_everything():
    shots = [FakeShot(shot_number=i + 1) for i in range(10)]
    report = review_shot_string(shots)
    assert report.is_clean
    assert any("carries no dispersion" in n for n in report.notes)


def test_an_empty_string_is_handled_without_error():
    report = review_shot_string([])
    assert report.n_shots == 0
    assert report.is_clean
    assert any("nothing to review" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

def test_excluded_shots_are_not_listed_again_as_review_items():
    shots = make_string(10)
    shots[0].clipped = True
    shots[0].window_truncated = True
    report = review_shot_string(shots)
    assert 1 in report.shots_to_exclude()
    assert 1 not in report.shots_to_review()


def test_report_serialises_without_non_finite_values():
    shots = make_string(10)
    shots[3].Lpeak_Z += 18.0
    shots[6].clipped = True
    data = review_shot_string(shots).to_dict()
    assert data["n_shots"] == 10
    assert data["shots_to_exclude"] == [7]
    for flag in data["flags"]:
        for key in ("value", "modified_z", "string_median"):
            assert flag[key] is None or math.isfinite(flag[key])


def test_summary_names_every_reviewable_shot():
    shots = make_string(12)
    shots[1].Lpeak_Z += 18.0
    shots[9].clipped = True
    text = review_shot_string(shots).summary()
    assert "EXCLUDE" in text and "10" in text
    assert "REVIEW" in text and "Shot 2" in text


# ---------------------------------------------------------------------------
# First-round pop is an explained outlier, not a fault
# ---------------------------------------------------------------------------

def test_an_established_first_round_pop_is_not_reported_as_a_suspect_shot():
    """
    A genuine pop makes shot one a statistical outlier BY DEFINITION. Reporting
    it as a possible squib sends the technician after the one thing that was
    expected, so the flag must be restated as explained.
    """
    shots = make_string(12)
    shots[0].Lpeak_Z += 6.0

    unaware = review_shot_string(shots)
    assert 1 in unaware.shots_to_review()

    aware = review_shot_string(shots, first_round_pop_established=True)
    assert 1 not in aware.shots_to_review()
    assert any("popped" in note for note in aware.notes)


def test_the_explained_pop_is_still_shown_with_its_size():
    """Explained is not hidden: the deviation must still be visible."""
    shots = make_string(12)
    shots[0].Lpeak_Z += 6.0
    report = review_shot_string(shots, first_round_pop_established=True)

    flags = [f for f in report.for_shot(1) if f.metric == "Lpeak_Z"]
    assert flags, "the deviation must still be reported"
    assert flags[0].severity == "info"
    assert "explained" in flags[0].message
    assert math.isfinite(flags[0].modified_z)


def test_a_clipped_first_round_is_still_excluded_even_when_pop_is_established():
    """
    Pop explains a LEVEL deviation. It does not make a clipped shot usable, and
    must not launder one.
    """
    shots = make_string(12)
    shots[0].Lpeak_Z += 6.0
    shots[0].clipped = True
    report = review_shot_string(shots, first_round_pop_established=True)
    assert report.shots_to_exclude() == [1]


def test_pop_does_not_explain_a_later_shot():
    """Only the FIRST round is explained by first-round pop."""
    shots = make_string(12)
    shots[6].Lpeak_Z += 12.0
    report = review_shot_string(shots, first_round_pop_established=True)
    assert 7 in report.shots_to_review()
