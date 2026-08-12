"""
test_plots.py - the statistics the new figures draw, and that they draw at all.

A figure is a claim. The claims made by the insertion-loss-with-confidence plot
are arithmetic - an energy mean, a standard error, and which bands fall inside
their own interval - so they are checked against closed forms here rather than
against a rendered image. The rendering itself is smoke-tested, because a figure
that raises is a figure the report silently loses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plots  # noqa: E402
from plots import (  # noqa: E402
    _band_stats_dB,
    plot_band_insertion_loss_ci,
    plot_session_trend,
    plot_shot_variability,
)

BANDS = np.array([
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
], dtype=float)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@dataclass
class FakeShot:
    shot_number: int
    Lpeak_Z: float = 160.0
    LAE: float = 130.0
    spectral_centroid_Hz: float = 1200.0
    a_duration_ms: float = 0.8


# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------

def test_energy_mean_of_identical_levels_is_that_level():
    """Averaging a constant on any basis returns the constant."""
    mean, se, n = _band_stats_dB(np.full((7, 5), 120.0))
    assert np.allclose(mean, 120.0)
    assert np.allclose(se, 0.0)
    assert n == 7


def test_energy_mean_matches_the_closed_form():
    """
    The energy mean of L and L+10 is 10*log10((10^(L/10) + 10^((L+10)/10))/2),
    which for L = 100 is 100 + 10*log10(5.5) = 107.404 dB. An arithmetic mean
    would give 105, so this distinguishes the two.
    """
    mean, _, _ = _band_stats_dB(np.array([[100.0], [110.0]]))
    expected = 10.0 * math.log10((10.0 ** 10.0 + 10.0 ** 11.0) / 2.0)
    assert mean[0] == pytest.approx(expected, rel=1e-12)
    assert mean[0] == pytest.approx(100.0 + 10.0 * math.log10(5.5), rel=1e-12)
    assert mean[0] != pytest.approx(105.0, abs=1.0)


def test_standard_error_is_the_sample_deviation_over_root_n():
    values = np.array([[100.0], [102.0], [104.0], [106.0]])
    _, se, n = _band_stats_dB(values)
    expected = float(np.std(values[:, 0], ddof=1)) / math.sqrt(4)
    assert n == 4
    assert se[0] == pytest.approx(expected, rel=1e-12)


def test_a_single_shot_has_no_measurable_spread():
    """One shot gives no estimate of dispersion, so the error must be zero, not NaN."""
    mean, se, n = _band_stats_dB(np.array([[118.0, 121.0]]))
    assert n == 1
    assert np.allclose(se, 0.0)
    assert np.allclose(mean, [118.0, 121.0])


def test_empty_input_produces_empty_statistics():
    mean, se, n = _band_stats_dB(np.array([]))
    assert mean.size == 0 and se.size == 0 and n == 0


# ---------------------------------------------------------------------------
# Insertion loss with confidence
# ---------------------------------------------------------------------------

def test_insertion_loss_figure_renders():
    rng = np.random.default_rng(0)
    ref = 140.0 + rng.normal(0, 0.4, (8, BANDS.size))
    test = 128.0 + rng.normal(0, 0.4, (8, BANDS.size))
    fig, ax = plot_band_insertion_loss_ci(ref, test, BANDS)
    assert fig is not None
    assert ax.get_ylabel().startswith("Insertion loss")


def test_a_band_with_no_real_reduction_falls_inside_its_own_interval():
    """
    Two strings drawn from the same population differ by nothing, so with
    realistic scatter almost every band must be reported as within measurement
    scatter rather than as an established reduction.
    """
    rng = np.random.default_rng(7)
    ref = 140.0 + rng.normal(0, 2.0, (4, BANDS.size))
    test = 140.0 + rng.normal(0, 2.0, (4, BANDS.size))

    ref_mean, ref_se, _ = _band_stats_dB(ref)
    test_mean, test_se, _ = _band_stats_dB(test)
    il = ref_mean - test_mean
    ci = 1.96 * np.sqrt(ref_se ** 2 + test_se ** 2)

    inconclusive = int(np.count_nonzero(np.abs(il) <= ci))
    assert inconclusive >= BANDS.size - 4, (
        f"only {inconclusive} of {BANDS.size} bands were inconclusive for two "
        f"strings from the same population"
    )


def test_a_large_real_reduction_is_established_in_every_band():
    rng = np.random.default_rng(11)
    ref = 140.0 + rng.normal(0, 0.3, (10, BANDS.size))
    test = 120.0 + rng.normal(0, 0.3, (10, BANDS.size))

    ref_mean, ref_se, _ = _band_stats_dB(ref)
    test_mean, test_se, _ = _band_stats_dB(test)
    il = ref_mean - test_mean
    ci = 1.96 * np.sqrt(ref_se ** 2 + test_se ** 2)
    assert np.all(il > ci)


def test_mismatched_band_counts_are_truncated_not_misaligned():
    """
    Drawing a 24-band reference against a 20-band test by index would silently
    compare 1 kHz against 2.5 kHz. The figure must use only the common bands.
    """
    ref = np.full((5, 24), 140.0)
    test = np.full((5, 20), 130.0)
    fig, ax = plot_band_insertion_loss_ci(ref, test, BANDS)
    # One bar per COMMON band, not per reference band.
    assert len(ax.patches) == 20
    # Every bar carries the same 10 dB reduction, so none was misaligned.
    assert all(patch.get_height() == pytest.approx(10.0) for patch in ax.patches)


def test_insertion_loss_figure_survives_a_single_shot_per_string():
    fig, _ = plot_band_insertion_loss_ci(
        np.full((1, BANDS.size), 140.0), np.full((1, BANDS.size), 130.0), BANDS
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Variability and trend
# ---------------------------------------------------------------------------

def test_variability_figure_has_one_row_per_metric():
    shots = [FakeShot(i + 1) for i in range(10)]
    fig, axes = plot_shot_variability(shots)
    assert len(axes) == len(plots._VARIABILITY_METRICS)
    assert axes[-1].get_xlabel() == "Shot number"


def test_variability_figure_marks_the_flagged_shots():
    rng = np.random.default_rng(2)
    shots = [FakeShot(i + 1, Lpeak_Z=160.0 + rng.normal(0, 0.4)) for i in range(12)]
    shots[5].Lpeak_Z += 12.0
    fig, axes = plot_shot_variability(shots, flagged_shots=[6])
    handles, labels = axes[0].get_legend_handles_labels()
    assert "flagged for review" in labels


def test_variability_figure_handles_an_empty_string():
    fig, axes = plot_shot_variability([])
    assert fig is not None


def test_session_trend_draws_a_trend_only_with_enough_strings():
    """A line through two points is not evidence of drift, so it is not drawn."""
    fig, ax = plot_session_trend([160.0, 161.0])
    assert ax.get_legend() is None

    fig2, ax2 = plot_session_trend([160.0, 161.0, 162.0, 163.0])
    _, labels = ax2.get_legend_handles_labels()
    assert any("trend" in text for text in labels)


def test_session_trend_reports_the_least_squares_slope():
    """
    Four strings rising by exactly 1 dB each have a slope of exactly
    +1.00 dB/string, which the legend must state.
    """
    fig, ax = plot_session_trend([160.0, 161.0, 162.0, 163.0])
    _, labels = ax.get_legend_handles_labels()
    trend = next(text for text in labels if "trend" in text)
    assert "+1.00" in trend


def test_session_trend_accepts_confidence_intervals_and_labels():
    fig, ax = plot_session_trend(
        [160.0, 160.4, 161.1],
        labels=["ref", "can A", "can B"],
        ci95=[0.2, 0.3, 0.25],
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ref", "can A", "can B"]


def test_every_new_figure_renders_in_both_themes():
    """
    plots.py carries the light/dark contract, so a figure that only works in one
    of them is broken in the report.
    """
    rng = np.random.default_rng(4)
    ref = 140.0 + rng.normal(0, 0.4, (6, BANDS.size))
    test = 130.0 + rng.normal(0, 0.4, (6, BANDS.size))
    shots = [FakeShot(i + 1) for i in range(6)]

    for theme in ("light", "dark"):
        plots.setup_plot_style(theme)
        assert plot_band_insertion_loss_ci(ref, test, BANDS)[0] is not None
        assert plot_shot_variability(shots)[0] is not None
        assert plot_session_trend([1.0, 2.0, 3.0])[0] is not None
    plots.setup_plot_style("light")
