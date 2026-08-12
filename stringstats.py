#!/usr/bin/env python3
"""
stringstats.py - Shot-String Statistics and First-Round Pop

A suppressor's behaviour is not one number. Over a string it changes: the first
round out of a cold can is louder than the rest, the can heats, the bore fouls,
and the operator wants to know all of that rather than a single average that
hides it. This module breaks a string down into the things that actually matter
when judging a suppressor.

FIRST-ROUND POP

The first round fired through a suppressor that has been sitting is louder than
those that follow. The volume inside the can is full of ordinary air; the first
shot's propellant gases are fuel-rich and deflagrate with that oxygen, adding an
audible report that later shots - fired into a can already purged to inert
combustion products - do not produce. It is a real, repeatable, and commercially
important property, and it is the single most-quoted suppressor number after
overall reduction.

It is also very easy to report dishonestly. One first round against nine
subsequent rounds is ONE observation. If the string's own shot-to-shot spread is
2 dB, a 2 dB "pop" is not a finding, it is the spread. So:

  * From a single string, first-round pop is tested against the PREDICTION
    interval for one new observation drawn from the remaining shots,

        m +/- t(1-alpha, n-2) * s * sqrt(1 + 1/n_rest)

    which is the exact interval a single further shot should fall inside. Not
    the confidence interval on the mean, which is narrower and would declare a
    pop on nearly every string.

  * From several strings, each string contributes one first-round observation
    and the pop is estimated properly, with a confidence interval, from those.
    This is the form that supports a published claim, and the module says so.

  * Below MIN_SUBSEQUENT_SHOTS it refuses outright rather than producing a
    number with no meaning.

The test is ONE-SIDED. First-round pop is a physical mechanism that can only add
energy; a first shot that is quieter than the rest is not negative pop, it is a
different problem, and it is reported as such rather than as a signed pop.

Usage:
    from stringstats import first_round_pop, string_summary

    pop = first_round_pop([m.Lpeak_Z for m in shot_metrics])
    print(pop.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# Significance level for the one-sided first-round-pop test.
POP_ALPHA: float = 0.05

# A standard deviation needs at least two values, and a prediction interval
# built on two values is so wide that nothing can fall outside it. Four
# subsequent shots is the point at which the interval starts to discriminate.
MIN_SUBSEQUENT_SHOTS: int = 4

# Strings needed before a first-round pop is estimated across strings, which is
# the only form that supports a claim about the suppressor rather than about one
# particular first shot.
MIN_STRINGS_FOR_CLAIM: int = 3


class StringStatsError(ValueError):
    """Raised when a string cannot be summarised at all."""


def energy_average_dB(levels_dB: Sequence[float]) -> float:
    """
    Energy (ISO) mean of a set of levels.

        L = 10 * log10( mean( 10^(L_i/10) ) )

    This is the correct average for levels, and differs from the arithmetic mean
    whenever the levels differ: it is always the larger of the two.
    """
    arr = np.asarray([v for v in levels_dB if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(10.0 ** (arr / 10.0))))


# ---- First-round pop ----

@dataclass
class FirstRoundPop:
    """
    Whether the first round of a string was louder than those that followed.

    `established` is the only field that should drive a claim. `observed_dB` is
    always populated when it can be computed at all, precisely so that a
    not-established result still shows the operator what was seen.
    """
    metric: str = "Lpeak_Z"
    observed_dB: float = float("nan")
    first_shot_dB: float = float("nan")
    subsequent_mean_dB: float = float("nan")
    subsequent_sd_dB: float = float("nan")
    n_subsequent: int = 0
    prediction_upper_dB: float = float("nan")
    prediction_lower_dB: float = float("nan")
    p_value: float = float("nan")
    established: bool = False
    basis: str = "none"          # "single-string" | "across-strings" | "none"
    alpha: float = POP_ALPHA
    n_strings: int = 1
    ci95_dB: Tuple[float, float] = (float("nan"), float("nan"))
    refusal: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def measurable(self) -> bool:
        return not self.refusal and math.isfinite(self.observed_dB)

    @property
    def first_shot_quieter(self) -> bool:
        """A first shot BELOW the prediction interval is not pop; it is a fault."""
        return (
            math.isfinite(self.first_shot_dB)
            and math.isfinite(self.prediction_lower_dB)
            and self.first_shot_dB < self.prediction_lower_dB
        )

    def to_dict(self) -> Dict[str, Any]:
        def num(x):
            return round(float(x), 3) if isinstance(x, (int, float)) and math.isfinite(x) else None
        return {
            "metric": self.metric,
            "observed_dB": num(self.observed_dB),
            "first_shot_dB": num(self.first_shot_dB),
            "subsequent_mean_dB": num(self.subsequent_mean_dB),
            "subsequent_sd_dB": num(self.subsequent_sd_dB),
            "n_subsequent": self.n_subsequent,
            "n_strings": self.n_strings,
            "prediction_upper_dB": num(self.prediction_upper_dB),
            "prediction_lower_dB": num(self.prediction_lower_dB),
            "p_value": num(self.p_value),
            "established": self.established,
            "first_shot_quieter": self.first_shot_quieter,
            "basis": self.basis,
            "alpha": self.alpha,
            "ci95_dB": [num(self.ci95_dB[0]), num(self.ci95_dB[1])],
            "refusal": self.refusal,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        if self.refusal:
            return f"  First-round pop: NOT MEASURED - {self.refusal}"

        lines = []
        if self.basis == "across-strings":
            lo, hi = self.ci95_dB
            lines.append(
                f"  First-round pop ({self.metric}): {self.observed_dB:+.2f} dB "
                f"across {self.n_strings} strings [95% CI {lo:+.2f} to {hi:+.2f} dB]"
            )
        else:
            lines.append(
                f"  First-round pop ({self.metric}): {self.observed_dB:+.2f} dB observed "
                f"from one string"
            )
            lines.append(
                f"    First shot {self.first_shot_dB:.1f} dB against "
                f"{self.subsequent_mean_dB:.1f} dB for the following "
                f"{self.n_subsequent} (sd {self.subsequent_sd_dB:.2f} dB)"
            )
            lines.append(
                f"    A further shot from this string would be expected between "
                f"{self.prediction_lower_dB:.1f} and {self.prediction_upper_dB:.1f} dB"
            )

        if self.established:
            lines.append(
                f"    ESTABLISHED: the first shot is louder than the string explains "
                f"(p = {self.p_value:.4f})"
            )
        elif self.first_shot_quieter:
            lines.append(
                "    The first shot was QUIETER than the string explains. That is not "
                "first-round pop; check for a squib, a misfire or a detection error."
            )
        else:
            lines.append(
                f"    NOT ESTABLISHED: the first shot sits inside the spread of the "
                f"string (p = {self.p_value:.4f}). Any pop is smaller than this "
                f"measurement can resolve."
            )
        for note in self.notes:
            lines.append(f"    {note}")
        return "\n".join(lines)


def first_round_pop(
    levels_dB: Sequence[float],
    *,
    metric: str = "Lpeak_Z",
    alpha: float = POP_ALPHA,
) -> FirstRoundPop:
    """
    Test whether the first shot of one string was louder than the rest.

    The comparison is against the PREDICTION interval for a single further
    observation, not the confidence interval on the mean. The question is "would
    one more shot land here", and a confidence interval answers a different and
    much easier question, which is why using it would declare a pop on almost
    every string.

    Args:
        levels_dB: Per-shot levels in shot order. The first entry is the first
                   round.
        metric: Name of the level being tested, for the report.
        alpha: One-sided significance level.

    Returns:
        FirstRoundPop. Check `.refusal` first, then `.established`.
    """
    values = np.asarray(list(levels_dB), dtype=np.float64)
    result = FirstRoundPop(metric=metric, alpha=alpha)

    if values.size == 0:
        result.refusal = "the string has no shots"
        return result
    if not math.isfinite(values[0]):
        result.refusal = "the first shot has no valid level"
        return result

    rest = values[1:]
    rest = rest[np.isfinite(rest)]
    result.n_subsequent = int(rest.size)
    result.first_shot_dB = float(values[0])

    if rest.size < MIN_SUBSEQUENT_SHOTS:
        result.refusal = (
            f"only {rest.size} shot(s) followed the first, below the "
            f"{MIN_SUBSEQUENT_SHOTS} needed to say what a normal shot from this "
            f"string looks like"
        )
        return result

    mean = float(np.mean(rest))
    sd = float(np.std(rest, ddof=1))
    result.subsequent_mean_dB = mean
    result.subsequent_sd_dB = sd
    # Reported against the energy mean of the rest, matching how the string mean
    # is reported everywhere else in SASA.
    result.observed_dB = float(values[0]) - energy_average_dB(rest)

    if sd <= 0.0:
        result.refusal = (
            "every subsequent shot has an identical level, so the string carries no "
            "spread to judge the first shot against"
        )
        return result

    # Prediction interval for ONE new observation from the same population.
    n = rest.size
    se_pred = sd * math.sqrt(1.0 + 1.0 / n)
    t_crit = float(stats.t.ppf(1.0 - alpha, n - 1))
    result.prediction_upper_dB = mean + t_crit * se_pred
    result.prediction_lower_dB = mean - t_crit * se_pred

    t_stat = (float(values[0]) - mean) / se_pred
    result.p_value = float(stats.t.sf(t_stat, n - 1))
    result.established = bool(float(values[0]) > result.prediction_upper_dB)
    result.basis = "single-string"

    result.notes.append(
        "From one string this is a single observation. A first-round pop that "
        "supports a published claim needs the first shot of several strings; see "
        "first_round_pop_across_strings()."
    )
    return result


def first_round_pop_across_strings(
    strings: Sequence[Sequence[float]],
    *,
    metric: str = "Lpeak_Z",
    alpha: float = POP_ALPHA,
) -> FirstRoundPop:
    """
    Estimate first-round pop from the first shot of each of several strings.

    Each string contributes one observation

        d_i = L_first,i - energy_mean(L_rest,i)

    and the pop is the mean of those with a confidence interval, tested
    one-sided against zero. This is the form that supports a claim about the
    suppressor rather than about one particular first shot.

    Args:
        strings: One sequence of per-shot levels per string, in shot order.
        metric: Name of the level being tested.
        alpha: One-sided significance level.

    Returns:
        FirstRoundPop with `basis` set to "across-strings" when it succeeded.
    """
    result = FirstRoundPop(metric=metric, alpha=alpha)
    deltas: List[float] = []
    skipped = 0

    for levels in strings:
        values = np.asarray(list(levels), dtype=np.float64)
        if values.size < MIN_SUBSEQUENT_SHOTS + 1 or not math.isfinite(values[0]):
            skipped += 1
            continue
        rest = values[1:][np.isfinite(values[1:])]
        if rest.size < MIN_SUBSEQUENT_SHOTS:
            skipped += 1
            continue
        deltas.append(float(values[0]) - energy_average_dB(rest))

    result.n_strings = len(deltas)
    if skipped:
        result.notes.append(
            f"{skipped} string(s) were too short to contribute a first-round "
            f"observation and were left out."
        )

    if len(deltas) < MIN_STRINGS_FOR_CLAIM:
        result.refusal = (
            f"only {len(deltas)} usable string(s), below the "
            f"{MIN_STRINGS_FOR_CLAIM} needed to estimate first-round pop across "
            f"strings rather than within one"
        )
        return result

    arr = np.asarray(deltas, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))

    result.observed_dB = mean
    result.subsequent_sd_dB = sd
    result.basis = "across-strings"

    if sd <= 0.0:
        # Identical deltas: real, but a t-test has no scale to work with.
        result.ci95_dB = (mean, mean)
        result.p_value = 0.0 if mean > 0 else 1.0
        result.established = mean > 0.0
        result.notes.append(
            "Every string produced exactly the same first-round difference, so no "
            "confidence interval could be formed from their spread."
        )
        return result

    se = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    result.ci95_dB = (mean - t_crit * se, mean + t_crit * se)
    result.p_value = float(stats.t.sf(mean / se, n - 1))
    result.established = bool(result.p_value < alpha)
    return result


# ---- String breakdown ----

@dataclass
class StringSummary:
    """A shot string described the way a technician needs to read it."""
    metric: str = "Lpeak_Z"
    n_shots: int = 0
    energy_mean_dB: float = float("nan")
    energy_mean_excluding_first_dB: float = float("nan")
    arithmetic_mean_dB: float = float("nan")
    median_dB: float = float("nan")
    sd_dB: float = float("nan")
    minimum_dB: float = float("nan")
    maximum_dB: float = float("nan")
    range_dB: float = float("nan")
    percentiles_dB: Dict[str, float] = field(default_factory=dict)
    ci95_half_width_dB: float = float("nan")
    trend_dB_per_shot: float = float("nan")
    trend_established: bool = False
    trend_p_value: float = float("nan")
    first_round_pop: Optional[FirstRoundPop] = None

    @property
    def first_round_cost_dB(self) -> float:
        """
        How much the first round adds to the string mean.

        This is the number that matters commercially: the difference between the
        average a customer hears including the first round and the average
        excluding it.
        """
        if not (math.isfinite(self.energy_mean_dB)
                and math.isfinite(self.energy_mean_excluding_first_dB)):
            return float("nan")
        return self.energy_mean_dB - self.energy_mean_excluding_first_dB

    def to_dict(self) -> Dict[str, Any]:
        def num(x):
            return round(float(x), 3) if isinstance(x, (int, float)) and math.isfinite(x) else None
        return {
            "metric": self.metric,
            "n_shots": self.n_shots,
            "energy_mean_dB": num(self.energy_mean_dB),
            "energy_mean_excluding_first_dB": num(self.energy_mean_excluding_first_dB),
            "first_round_cost_dB": num(self.first_round_cost_dB),
            "arithmetic_mean_dB": num(self.arithmetic_mean_dB),
            "median_dB": num(self.median_dB),
            "sd_dB": num(self.sd_dB),
            "min_dB": num(self.minimum_dB),
            "max_dB": num(self.maximum_dB),
            "range_dB": num(self.range_dB),
            "percentiles_dB": {k: num(v) for k, v in self.percentiles_dB.items()},
            "ci95_half_width_dB": num(self.ci95_half_width_dB),
            "trend_dB_per_shot": num(self.trend_dB_per_shot),
            "trend_established": self.trend_established,
            "trend_p_value": num(self.trend_p_value),
            "first_round_pop": self.first_round_pop.to_dict() if self.first_round_pop else None,
        }

    def summary(self) -> str:
        lines = [
            f"  {self.metric} across {self.n_shots} shot(s)",
            f"    Energy mean       {self.energy_mean_dB:8.2f} dB "
            f"(+/-{self.ci95_half_width_dB:.2f} at 95%)",
            f"    Excluding shot 1  {self.energy_mean_excluding_first_dB:8.2f} dB "
            f"(first round costs {self.first_round_cost_dB:+.2f} dB)",
            f"    Median            {self.median_dB:8.2f} dB",
            f"    Spread            {self.minimum_dB:.2f} to {self.maximum_dB:.2f} dB "
            f"(range {self.range_dB:.2f}, sd {self.sd_dB:.2f})",
        ]
        if self.percentiles_dB:
            parts = "  ".join(
                f"p{k}={v:.1f}" for k, v in sorted(
                    self.percentiles_dB.items(), key=lambda kv: float(kv[0])
                )
            )
            lines.append(f"    Percentiles       {parts}")
        if math.isfinite(self.trend_dB_per_shot):
            verdict = "established" if self.trend_established else "not established"
            lines.append(
                f"    Trend             {self.trend_dB_per_shot:+.3f} dB/shot "
                f"({verdict}, p = {self.trend_p_value:.3f})"
            )
        if self.first_round_pop is not None:
            lines.append(self.first_round_pop.summary())
        return "\n".join(lines)


_PERCENTILES: Tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)


def string_summary(
    levels_dB: Sequence[float],
    *,
    metric: str = "Lpeak_Z",
    alpha: float = POP_ALPHA,
) -> StringSummary:
    """
    Break one string down into the statistics that describe how it behaved.

    Includes the with/without-first-round averages, the distribution, and a test
    for drift ACROSS the string excluding the first round, so that first-round
    pop cannot masquerade as a heating trend.

    Args:
        levels_dB: Per-shot levels in shot order.
        metric: Name of the level, for the report.
        alpha: Significance level for the pop and trend tests.

    Returns:
        StringSummary.
    """
    values = np.asarray([v for v in levels_dB], dtype=np.float64)
    finite = values[np.isfinite(values)]

    out = StringSummary(metric=metric, n_shots=int(values.size))
    if finite.size == 0:
        return out

    out.energy_mean_dB = energy_average_dB(finite)
    out.arithmetic_mean_dB = float(np.mean(finite))
    out.median_dB = float(np.median(finite))
    out.minimum_dB = float(np.min(finite))
    out.maximum_dB = float(np.max(finite))
    out.range_dB = out.maximum_dB - out.minimum_dB
    out.sd_dB = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    out.ci95_half_width_dB = (
        1.96 * out.sd_dB / math.sqrt(finite.size) if finite.size > 1 else 0.0
    )
    out.percentiles_dB = {
        f"{p:g}": float(np.percentile(finite, p)) for p in _PERCENTILES
    }

    if values.size > 1:
        rest = values[1:][np.isfinite(values[1:])]
        if rest.size:
            out.energy_mean_excluding_first_dB = energy_average_dB(rest)

    # Drift across the string, excluding the first round so that pop is not read
    # as a trend. Needs three points for a slope with any residual degrees of
    # freedom to test it against.
    if values.size >= 4:
        rest = values[1:]
        mask = np.isfinite(rest)
        if int(np.count_nonzero(mask)) >= 3:
            index = np.arange(rest.size, dtype=np.float64)[mask]
            regression = stats.linregress(index, rest[mask])
            out.trend_dB_per_shot = float(regression.slope)
            out.trend_p_value = float(regression.pvalue)
            out.trend_established = bool(regression.pvalue < alpha)

    out.first_round_pop = first_round_pop(values, metric=metric, alpha=alpha)
    return out


def compare_with_and_without_first_round(
    reference_levels_dB: Sequence[float],
    test_levels_dB: Sequence[float],
    *,
    metric: str = "Lpeak_Z",
) -> Dict[str, Any]:
    """
    Insertion loss computed both including and excluding the first round.

    A suppressor whose first round pops has two honest reduction figures: the one
    a shooter hears on the first round of the day, and the one they hear
    thereafter. Quoting only the better of the two is the most common way a
    suppressor test flatters its subject, so both are produced together.

    Args:
        reference_levels_dB: Per-shot levels of the unsuppressed reference.
        test_levels_dB: Per-shot levels of the suppressed test.
        metric: Name of the level, for the report.

    Returns:
        Dict carrying both reductions and the difference between them.
    """
    ref = [v for v in reference_levels_dB if math.isfinite(v)]
    test = list(test_levels_dB)

    ref_all = energy_average_dB(ref)
    test_all = energy_average_dB([v for v in test if math.isfinite(v)])
    test_rest = energy_average_dB([v for v in test[1:] if math.isfinite(v)]) if len(test) > 1 else float("nan")

    including = ref_all - test_all
    excluding = ref_all - test_rest

    def num(x):
        return round(float(x), 2) if math.isfinite(x) else None

    return {
        "metric": metric,
        "reference_energy_mean_dB": num(ref_all),
        "test_energy_mean_dB": num(test_all),
        "test_energy_mean_excluding_first_dB": num(test_rest),
        "reduction_including_first_dB": num(including),
        "reduction_excluding_first_dB": num(excluding),
        "first_round_penalty_dB": num(excluding - including),
        "n_reference": len(ref),
        "n_test": len([v for v in test if math.isfinite(v)]),
    }


# ---- CLI for testing ----

def main() -> int:
    """Demonstrate the string breakdown on a synthetic string with a planted pop."""
    import argparse

    parser = argparse.ArgumentParser(description="Shot-string statistics")
    parser.add_argument("--n", type=int, default=10, help="Shots in the string")
    parser.add_argument("--pop-dB", type=float, default=3.0,
                        help="First-round pop to plant")
    parser.add_argument("--sd-dB", type=float, default=0.6,
                        help="Shot-to-shot standard deviation")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    levels = list(140.0 + rng.normal(0.0, args.sd_dB, args.n))
    levels[0] += args.pop_dB

    print(string_summary(levels).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
