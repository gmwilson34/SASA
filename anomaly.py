#!/usr/bin/env python3
"""
anomaly.py - Shot-String Anomaly Detection

A shot string is not automatically a measurement. One round in ten may be a
squib, a double-feed, a gust across the windscreen, a reflection off a vehicle
that parked downrange between strings, or simply the operator bumping the stand.
Averaging those in silently is how a suppressor gets credited with performance it
does not have, or blamed for performance it does.

This module reads a completed shot string and says which shots a technician
should look at, and WHY. It does not silently drop anything: every flag names the
metric, the value, how far from the string it sits, and what that usually means.
The decision to exclude stays with the operator, except where a shot is
physically incapable of carrying a number (a clipped one), which metrics.py
already refuses.

METHOD

The decision to flag is made by the generalised extreme studentised deviate test
(Rosner 1983), which detects up to r outliers in a sample without knowing in
advance how many there are. At step i it forms

    R_i = max_j |x_j - mean(x)| / sd(x)

on the sample with the previous i-1 extremes removed, and compares it against

    lambda_i = (n-i) * t / sqrt( (n-i-1 + t^2) * (n-i+1) )

where t is the (1 - p) quantile of Student's t with n-i-1 degrees of freedom and
p = alpha / (2*(n-i+1)). Those critical values are exact and closed-form, so the
false-alarm rate this module runs at is a stated property rather than a hope.

WHY NOT THE MODIFIED Z-SCORE

The obvious choice is the median absolute deviation with Iglewicz and Hoaglin's
threshold of 3.5. It was measured against its own null distribution and rejected:
on clean Gaussian strings that rule raises a false flag on 9% to 20% of strings
PER METRIC, and the rate does not fall with sample size, because the MAD's own
sampling variability dominates at these sample sizes. Held instead to a
defensible 1% false-alarm rate it detects only one genuine 5-sigma outlier in
five at ten shots. The ESD test holds its nominal rate at every n and catches
about four in five at the same string length. See tests/test_anomaly.py, which
measures both claims rather than asserting them.

The modified z-score is still computed and reported, because "3.2 robust
standard deviations above the string" is the right thing to SHOW a technician.
It is just not what decides.

MULTIPLE METRICS

Five metrics are tested per string, so the per-metric significance level is
Bonferroni-corrected to FAMILY_ALPHA / (number of metrics). Without that
correction a clean string would trip one metric or another about a quarter of
the time, and a flag that fires on a quarter of clean strings teaches the
operator to ignore flags.

SMALL STRINGS AND SENSITIVITY

Below MIN_SHOTS_FOR_STATISTICS the test cannot form the degrees of freedom it
needs. Above it the test is valid at every length, but its POWER still depends
strongly on n, and the report says so: at ten shots a five-sigma outlier is
caught about four times in five, and a three-sigma one only about one time in
four. A clean report from a short string is weak evidence, and is labelled as
weak evidence rather than presented as a clean bill of health.

Usage:
    from anomaly import review_shot_string

    report = review_shot_string(shot_metrics_list)
    print(report.summary())
    for flag in report.flags_for_review():
        print(flag.shot_number, flag.message)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# 0.75 quantile of the standard normal distribution. MAD/0.6745 estimates sigma.
# Used for the reported magnitude of a deviation, not for the decision to flag.
NORMAL_QUARTILE: float = 0.6745

# Family-wise false-alarm rate across all metrics for one clean shot string.
# This is the rate the module runs at by construction, and tests/test_anomaly.py
# measures it against simulated clean strings rather than assuming it.
FAMILY_ALPHA: float = 0.05

# Largest fraction of a string the outlier search will consider bad.
#
# This is not a matter of taste: the multi-step ESD procedure's actual size
# drifts above its nominal alpha as the search deepens, because each extra step
# is another chance to reject. Measured on clean Gaussian samples at alpha =
# 0.05, the realised family false-alarm rate is
#
#     search depth        n=5     n=10    n=20    n=50
#     r = 1              0.046   0.051   0.047   0.046
#     r = 2              0.094   0.072   0.057   0.054
#     r = n/5            0.050   0.064   0.061   0.053
#     r = n/10           0.051   0.048   0.058   0.053
#
# so a tenth of the string holds the stated rate at every length while still
# allowing two or three outliers to be found once the string is long enough to
# support looking for them. tests/test_anomaly.py re-measures this.
MAX_OUTLIER_FRACTION: float = 0.1

# The ESD test needs n - i - 1 >= 1 degrees of freedom at its first step, so
# n >= 3 is the hard floor. Five is used because below it the test is valid but
# so insensitive that a clean result carries essentially no information.
MIN_SHOTS_FOR_STATISTICS: int = 5

# Signal-to-noise below which a shot's integrated energy is contaminated by the
# noise floor. At 10 dB SNR the noise contributes about 10% of the measured
# energy, which is 0.41 dB of error in the exposure level:
#     10*log10(1 + 10**(-10/10)) = 0.414 dB
# That is the point at which SEL stops being a measurement of the shot alone.
MIN_SNR_dB: float = 10.0

# Severities, in increasing order of consequence.
SEVERITY_INFO = "info"
SEVERITY_REVIEW = "review"
SEVERITY_EXCLUDE = "exclude"

_SEVERITY_RANK = {SEVERITY_INFO: 0, SEVERITY_REVIEW: 1, SEVERITY_EXCLUDE: 2}

# Metrics tested for outliers, with the plain-language meaning of a deviation.
# (attribute, label, unit, interpretation)
_LEVEL_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    ("Lpeak_Z", "peak level", "dB",
     "a shot this far from the string is usually a different event: a squib, a "
     "double, or a reflection arriving louder than the muzzle blast"),
    ("LAE", "sound exposure level", "dB",
     "exposure differs from the string, which points at a contaminated "
     "integration window rather than a different muzzle blast"),
)

_CHARACTER_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    ("spectral_centroid_Hz", "spectral centroid", "Hz",
     "the shot has a different spectral balance from the rest of the string, "
     "which is what a ballistic crack, a reflection or a mechanical noise looks "
     "like when it is measured as if it were the muzzle blast"),
    ("a_duration_ms", "A-duration", "ms",
     "the positive-phase duration is inconsistent with the string, so the blast "
     "span was probably located on the wrong arrival"),
    ("rise_time_us", "rise time", "us",
     "the leading edge is unlike the rest of the string, which distinguishes a "
     "direct blast from something that arrived via a reflection"),
)


class AnomalyError(ValueError):
    """Raised when a shot string cannot be reviewed at all."""


# ---- Robust statistics ----

def median_absolute_deviation(values: np.ndarray) -> float:
    """
    Median absolute deviation about the median.

        MAD = median(|x_i - median(x)|)

    Returns 0.0 when more than half the values are identical, which is a real
    result and not an error; callers must handle it rather than dividing by it.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(np.abs(arr - np.median(arr))))


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    """
    Modified z-scores (Iglewicz & Hoaglin 1993).

        M_i = 0.6745 * (x_i - median(x)) / MAD(x)

    Returns an array of zeros when the MAD is zero, because in that case the
    values carry no dispersion to measure a deviation against. Non-finite inputs
    map to zero so they cannot be flagged as outliers by accident; they are
    caught separately as missing data.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out

    mad = median_absolute_deviation(arr)
    if not math.isfinite(mad) or mad <= 0.0:
        return out

    median = float(np.median(arr[finite]))
    out[finite] = NORMAL_QUARTILE * (arr[finite] - median) / mad
    return out


def esd_critical_value(n: int, i: int, alpha: float) -> float:
    """
    Rosner's critical value for step i of the generalised ESD test.

        lambda_i = (n-i) * t / sqrt( (n-i-1 + t^2) * (n-i+1) )

    with t the (1 - p) quantile of Student's t on n-i-1 degrees of freedom and
    p = alpha / (2*(n-i+1)).

    Args:
        n: Size of the original sample.
        i: Step number, 1-based.
        alpha: Family-wise significance level for this metric.

    Returns:
        The critical value, or infinity when the step has no degrees of freedom
        left, which makes that step impossible to fail.
    """
    nu = n - i - 1
    if nu < 1 or i < 1 or i > n:
        return float("inf")
    p = 1.0 - alpha / (2.0 * (n - i + 1))
    t = float(stats.t.ppf(p, nu))
    if not math.isfinite(t):
        return float("inf")
    return float((n - i) * t / math.sqrt((nu + t * t) * (n - i + 1)))


@dataclass
class ESDResult:
    """Outcome of a generalised extreme studentised deviate test."""
    outlier_indices: List[int] = field(default_factory=list)
    test_statistics: List[float] = field(default_factory=list)
    critical_values: List[float] = field(default_factory=list)
    candidate_indices: List[int] = field(default_factory=list)
    n: int = 0
    alpha: float = FAMILY_ALPHA
    applied: bool = False

    def statistic_for(self, index: int) -> float:
        """The R statistic at the step that removed this index, or NaN."""
        if index in self.candidate_indices:
            return self.test_statistics[self.candidate_indices.index(index)]
        return float("nan")

    def critical_for(self, index: int) -> float:
        """The critical value at the step that removed this index, or NaN."""
        if index in self.candidate_indices:
            return self.critical_values[self.candidate_indices.index(index)]
        return float("nan")


def generalised_esd(
    values: np.ndarray,
    *,
    max_outliers: Optional[int] = None,
    alpha: float = FAMILY_ALPHA,
) -> ESDResult:
    """
    Generalised extreme studentised deviate test (Rosner 1983).

    Repeatedly removes the most extreme studentised residual, recording its test
    statistic and the exact critical value for that step. The number of outliers
    is the LARGEST step index whose statistic exceeded its critical value, which
    is what protects the procedure against masking: two outliers together inflate
    the standard deviation enough that neither looks extreme on its own, but the
    second step, computed after the first is removed, still sees the second.

    Args:
        values: Sample to test. Non-finite entries are excluded from the test and
                can never be reported as outliers.
        max_outliers: Upper bound on how many outliers to look for. Defaults to
                      MAX_OUTLIER_FRACTION of the sample, at least one.
        alpha: Significance level for this metric.

    Returns:
        ESDResult. `.applied` is False when the sample was too small or carried
        no dispersion, in which case no index is reported as an outlier.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(arr))
    n = int(finite_positions.size)

    result = ESDResult(n=n, alpha=alpha)
    if n < MIN_SHOTS_FOR_STATISTICS:
        return result

    if max_outliers is None:
        max_outliers = max(1, int(n * MAX_OUTLIER_FRACTION))
    # Every step must retain at least one degree of freedom.
    max_outliers = max(1, min(max_outliers, n - 2))

    work = arr[finite_positions].copy()
    work_positions = list(finite_positions)

    for step in range(1, max_outliers + 1):
        if work.size < 3:
            break
        sd = float(np.std(work, ddof=1))
        if sd <= 0.0:
            break
        deviations = np.abs(work - float(np.mean(work)))
        j = int(np.argmax(deviations))

        result.test_statistics.append(float(deviations[j] / sd))
        result.critical_values.append(esd_critical_value(n, step, alpha))
        result.candidate_indices.append(int(work_positions[j]))

        work = np.delete(work, j)
        work_positions.pop(j)

    result.applied = bool(result.test_statistics)

    n_outliers = 0
    for step in range(1, len(result.test_statistics) + 1):
        if result.test_statistics[step - 1] > result.critical_values[step - 1]:
            n_outliers = step
    result.outlier_indices = result.candidate_indices[:n_outliers]
    return result


# ---- Flags ----

@dataclass
class ShotFlag:
    """One reason one shot is worth a technician's attention."""
    shot_number: int
    code: str
    severity: str
    message: str
    metric: str = ""
    value: float = float("nan")
    modified_z: float = float("nan")
    string_median: float = float("nan")
    esd_statistic: float = float("nan")
    esd_critical: float = float("nan")

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK.get(self.severity, 0)

    def to_dict(self) -> Dict[str, Any]:
        def num(x):
            return round(float(x), 3) if math.isfinite(x) else None
        return {
            "shot_number": self.shot_number,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "value": num(self.value),
            "modified_z": num(self.modified_z),
            "string_median": num(self.string_median),
            "esd_statistic": num(self.esd_statistic),
            "esd_critical": num(self.esd_critical),
        }


@dataclass
class AnomalyReport:
    """What a shot string looks like, and which shots do not belong to it."""
    n_shots: int
    n_evaluated: int
    flags: List[ShotFlag] = field(default_factory=list)
    statistics_applied: bool = False
    notes: List[str] = field(default_factory=list)
    alpha: float = FAMILY_ALPHA

    @property
    def sensitivity_statement(self) -> str:
        """
        What a clean result from THIS string length is worth.

        Power is a strong function of string length, so a clean report from six
        shots and a clean report from thirty are not the same evidence, and must
        not read the same way.
        """
        n = self.n_evaluated
        if not self.statistics_applied:
            return (
                "Sensitivity: none. The string is too short to test for "
                "shot-to-shot outliers at all."
            )
        if n < 10:
            strength = "roughly half of large (5-sigma) outliers and few small ones"
        elif n < 20:
            strength = "about four in five large (5-sigma) outliers, but under a third of 3-sigma ones"
        elif n < 30:
            strength = "about nine in ten large (5-sigma) outliers, and about a third of 3-sigma ones"
        else:
            strength = "over nine in ten large (5-sigma) outliers, and about a third of 3-sigma ones"
        return (
            f"Sensitivity: at {n} shots this test catches {strength}. A clean "
            f"result is not proof that the string is uniform."
        )

    # -- queries --

    def for_shot(self, shot_number: int) -> List[ShotFlag]:
        """Every flag raised against one shot, worst first."""
        return sorted(
            (f for f in self.flags if f.shot_number == shot_number),
            key=lambda f: -f.rank,
        )

    def flagged_shot_numbers(self) -> List[int]:
        """Shot numbers carrying any flag, in order."""
        return sorted({f.shot_number for f in self.flags})

    def flags_for_review(self) -> List[ShotFlag]:
        """Flags at review severity or worse."""
        return [f for f in self.flags if f.rank >= _SEVERITY_RANK[SEVERITY_REVIEW]]

    def shots_to_exclude(self) -> List[int]:
        """Shots that cannot carry a number at all."""
        return sorted({f.shot_number for f in self.flags if f.severity == SEVERITY_EXCLUDE})

    def shots_to_review(self) -> List[int]:
        """
        Shots a technician should look at before the string is reported.

        Excluded shots are not listed again: they are already out.
        """
        excluded = set(self.shots_to_exclude())
        return sorted(
            {f.shot_number for f in self.flags_for_review()} - excluded
        )

    @property
    def n_flagged(self) -> int:
        return len(self.flagged_shot_numbers())

    @property
    def is_clean(self) -> bool:
        return not self.flags_for_review()

    # -- output --

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_shots": self.n_shots,
            "n_evaluated": self.n_evaluated,
            "statistics_applied": self.statistics_applied,
            "alpha_per_metric": round(self.alpha, 4),
            "sensitivity": self.sensitivity_statement,
            "n_flagged": self.n_flagged,
            "shots_to_exclude": self.shots_to_exclude(),
            "shots_to_review": self.shots_to_review(),
            "flags": [f.to_dict() for f in self.flags],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [f"  Shot string review: {self.n_shots} shot(s)"]
        if not self.statistics_applied:
            lines.append(
                f"    Outlier statistics NOT applied: fewer than "
                f"{MIN_SHOTS_FOR_STATISTICS} shots to compare against."
            )
        for note in self.notes:
            lines.append(f"    {note}")

        excluded = self.shots_to_exclude()
        review = self.shots_to_review()
        if excluded:
            lines.append(f"    EXCLUDE: shot(s) {', '.join(str(n) for n in excluded)}")
        if review:
            lines.append(f"    REVIEW:  shot(s) {', '.join(str(n) for n in review)}")
        if not excluded and not review:
            lines.append("    No shot departs from the string.")
        lines.append(f"    {self.sensitivity_statement}")

        for shot in self.flagged_shot_numbers():
            for flag in self.for_shot(shot):
                if flag.rank >= _SEVERITY_RANK[SEVERITY_REVIEW]:
                    lines.append(f"      Shot {shot}: {flag.message}")
        return "\n".join(lines)


# ---- Review ----

def _get(shot: Any, name: str, default=None):
    return getattr(shot, name, default)


def _describe_deviation(
    label: str,
    value: float,
    median: float,
    unit: str,
    z: float,
    statistic: float,
    critical: float,
    interpretation: str,
) -> str:
    delta = value - median
    unit_txt = f" {unit}" if unit else ""
    magnitude = (
        f"{abs(z):.1f} robust standard deviations from the string"
        if math.isfinite(z) and z != 0.0
        else "clear of the string"
    )
    return (
        f"{label} is {value:.4g}{unit_txt}, {delta:+.3g}{unit_txt} from the string "
        f"median of {median:.4g}{unit_txt} and {magnitude} "
        f"(ESD statistic {statistic:.2f} against a critical value of {critical:.2f}); "
        f"{interpretation}"
    )


def _statistical_flags(
    shots: Sequence[Any],
    spec: Sequence[Tuple[str, str, str, str]],
    code_prefix: str,
    alpha: float,
) -> List[ShotFlag]:
    """Outlier flags for one family of metrics across the string."""
    out: List[ShotFlag] = []
    for attr, label, unit, interpretation in spec:
        values = np.array(
            [float(_get(s, attr, float("nan")) or float("nan")) for s in shots],
            dtype=np.float64,
        )
        if not np.any(np.isfinite(values)):
            continue

        esd = generalised_esd(values, alpha=alpha)
        if not esd.outlier_indices:
            continue

        z = modified_z_scores(values)
        median = float(np.median(values[np.isfinite(values)]))
        for position in esd.outlier_indices:
            shot = shots[position]
            out.append(ShotFlag(
                shot_number=int(_get(shot, "shot_number", 0) or 0),
                code=f"{code_prefix}_{attr}",
                severity=SEVERITY_REVIEW,
                message=_describe_deviation(
                    label, float(values[position]), median, unit,
                    float(z[position]),
                    esd.statistic_for(position), esd.critical_for(position),
                    interpretation,
                ),
                metric=attr,
                value=float(values[position]),
                modified_z=float(z[position]),
                string_median=median,
                esd_statistic=esd.statistic_for(position),
                esd_critical=esd.critical_for(position),
            ))
    return out


def review_shot_string(
    shot_metrics: Sequence[Any],
    *,
    min_snr_dB: float = MIN_SNR_dB,
    first_round_pop_established: bool = False,
) -> AnomalyReport:
    """
    Review a completed shot string and flag shots that do not belong to it.

    Args:
        shot_metrics: Per-shot metrics, in shot order. Anything exposing the
                      ShotMetrics attribute names is accepted.
        min_snr_dB: Signal-to-noise below which a shot's exposure level is
                    contaminated by the noise floor.
        first_round_pop_established: True when stringstats has already shown the
                    first round to be louder than the string explains. A genuine
                    first-round pop makes shot one a statistical outlier BY
                    DEFINITION, so without this the review reports it as a
                    possible squib or double - telling the technician to
                    investigate the one thing that was expected. When set, the
                    first shot's level outliers are restated as explained rather
                    than suppressed: the deviation is still shown, with its
                    cause.

    Returns:
        AnomalyReport. Statistical flags are only present when the string is long
        enough for the outlier test to mean anything; check `.statistics_applied`.
    """
    shots = list(shot_metrics)
    report = AnomalyReport(n_shots=len(shots), n_evaluated=0)

    if not shots:
        report.notes.append("No shots were detected, so there is nothing to review.")
        return report

    flags: List[ShotFlag] = []

    # -- per-shot conditions, independent of the rest of the string --
    for shot in shots:
        number = int(_get(shot, "shot_number", 0) or 0)

        if _get(shot, "clipped", False):
            flags.append(ShotFlag(
                shot_number=number,
                code="clipped",
                severity=SEVERITY_EXCLUDE,
                message=(
                    "samples reached digital full scale, so the true peak is unknown "
                    "and every level derived from this shot is a lower bound only"
                ),
            ))

        if _get(shot, "window_truncated", False):
            flags.append(ShotFlag(
                shot_number=number,
                code="truncated",
                severity=SEVERITY_REVIEW,
                message=(
                    "the extraction window hit a file or chunk boundary, so the "
                    "integrated exposure level is missing part of the event"
                ),
            ))

        snr = _get(shot, "snr_dB", float("nan"))
        if snr is not None and math.isfinite(snr) and snr < min_snr_dB:
            error_dB = 10.0 * math.log10(1.0 + 10.0 ** (-snr / 10.0))
            flags.append(ShotFlag(
                shot_number=number,
                code="low_snr",
                severity=SEVERITY_REVIEW,
                message=(
                    f"signal-to-noise is {snr:.1f} dB, below {min_snr_dB:g} dB; the "
                    f"noise floor contributes about {error_dB:.2f} dB to this shot's "
                    f"exposure level, which is no longer a measurement of the shot alone"
                ),
                metric="snr_dB",
                value=float(snr),
            ))

        if not _get(shot, "rise_time_resolved", True):
            flags.append(ShotFlag(
                shot_number=number,
                code="rise_time_unresolved",
                severity=SEVERITY_INFO,
                message=(
                    "the sample rate cannot resolve this shot's rise time, so the "
                    "reported value is an upper bound set by the sample interval"
                ),
                metric="rise_time_us",
                value=float(_get(shot, "rise_time_us", float("nan")) or float("nan")),
            ))

    # -- conditions that only exist relative to the rest of the string --
    usable = [s for s in shots if _get(s, "valid", True) and not _get(s, "clipped", False)]
    report.n_evaluated = len(usable)

    n_metrics = len(_LEVEL_METRICS) + len(_CHARACTER_METRICS)
    per_metric_alpha = FAMILY_ALPHA / n_metrics
    report.alpha = per_metric_alpha

    if len(usable) >= MIN_SHOTS_FOR_STATISTICS:
        report.statistics_applied = True
        flags.extend(
            _statistical_flags(usable, _LEVEL_METRICS, "level_outlier", per_metric_alpha)
        )
        flags.extend(
            _statistical_flags(usable, _CHARACTER_METRICS, "character_outlier", per_metric_alpha)
        )

        for attr, label, _unit, _interp in (*_LEVEL_METRICS, *_CHARACTER_METRICS):
            values = np.array(
                [float(_get(s, attr, float("nan")) or float("nan")) for s in usable],
                dtype=np.float64,
            )
            if np.any(np.isfinite(values)) and float(np.std(values[np.isfinite(values)])) == 0.0:
                report.notes.append(
                    f"{label} is identical across the string, so it carries no "
                    f"dispersion and no outlier could be detected in it."
                )
    else:
        report.notes.append(
            f"Only {len(usable)} shot(s) can be compared, below the "
            f"{MIN_SHOTS_FOR_STATISTICS} at which the outlier test carries useful "
            f"sensitivity. Shot-to-shot outliers were NOT tested for; per-shot "
            f"conditions such as clipping were."
        )

    # A first-round pop that stringstats has already established is not an
    # unexplained outlier. Restate it rather than send the technician looking
    # for a squib that is not there.
    if first_round_pop_established and shots:
        first_number = min(
            int(_get(s, "shot_number", 0) or 0) for s in shots
        )
        for flag in flags:
            if (
                flag.shot_number == first_number
                and flag.code.startswith("level_outlier")
                and flag.severity == SEVERITY_REVIEW
            ):
                flag.severity = SEVERITY_INFO
                flag.code = f"first_round_pop_{flag.metric}"
                flag.message = (
                    f"{flag.message.split(';')[0]}; this is the first round of the "
                    f"string and first-round pop has been established for it, so the "
                    f"deviation is explained. It is reported here so the size of the "
                    f"pop is visible, not because the shot is suspect."
                )
        report.notes.append(
            "The first round is a statistical outlier because it popped. Its level "
            "flags are marked as explained rather than as faults."
        )

    flags.sort(key=lambda f: (f.shot_number, -f.rank, f.code))
    report.flags = flags
    return report


# ---- CLI for testing ----

def main() -> int:
    """Demonstrate the review on a synthetic string with one planted outlier."""
    import argparse

    parser = argparse.ArgumentParser(description="Shot-string anomaly review")
    parser.add_argument("--n", type=int, default=10, help="Number of shots")
    parser.add_argument("--outlier-dB", type=float, default=12.0,
                        help="How far the planted outlier sits from the string")
    args = parser.parse_args()

    @dataclass
    class Fake:
        shot_number: int
        Lpeak_Z: float
        LAE: float
        spectral_centroid_Hz: float = 1200.0
        a_duration_ms: float = 0.8
        rise_time_us: float = 30.0
        snr_dB: float = 40.0
        clipped: bool = False
        window_truncated: bool = False
        rise_time_resolved: bool = True
        valid: bool = True

    rng = np.random.default_rng(0)
    shots = [
        Fake(shot_number=i + 1,
             Lpeak_Z=160.0 + rng.normal(0, 0.4),
             LAE=130.0 + rng.normal(0, 0.4))
        for i in range(args.n)
    ]
    if shots:
        shots[-1].Lpeak_Z += args.outlier_dB

    print(review_shot_string(shots).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
