#!/usr/bin/env python3
"""
pairing.py - Reference Pairing and Comparability

Insertion loss is a DIFFERENCE between two recordings, so it inherits every way
in which those two recordings were not the same experiment. Move the microphone
half a metre between the reference string and the test string and the arithmetic
still produces a confident number; it is just no longer a measurement of the
suppressor.

This module does two jobs:

  assess_comparability()  Decide whether a reference/test pair can support an
                          insertion-loss claim, and QUANTIFY each objection in
                          decibels wherever the physics allows. "The microphone
                          moved" is a note; "the microphone moved, and that
                          accounts for 3.5 dB of your 12 dB reduction" is a
                          finding.

  auto_pair()             Given a range session's worth of recordings, match each
                          suppressed string to its unsuppressed reference. It
                          refuses to guess when two candidates are equally good,
                          because a silently mis-paired reference produces a
                          plausible number that is wrong.

Nothing here corrects anything. Corrections live in atmosphere.py and are applied
explicitly; this module decides whether a comparison should be made at all, and
tells the technician what it would cost to make it anyway.

Usage:
    from pairing import assess_comparability, auto_pair

    report = assess_comparability(reference_meta, test_meta)
    if not report.comparable:
        print(report.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from atmosphere import Atmosphere, geometric_spreading_dB

# ---- Severities ----

# The comparison is not an insertion loss. No correction can rescue it.
BLOCKING = "blocking"
# The comparison is possible but a stated number of decibels is not the suppressor.
MATERIAL = "material"
# Worth recording; does not by itself change the number.
ADVISORY = "advisory"

_SEVERITY_RANK = {ADVISORY: 0, MATERIAL: 1, BLOCKING: 2}

# ---- Reporting thresholds ----
#
# These decide what gets SAID, not what gets computed. Every quantified objection
# below reports its own magnitude regardless; these only set the point at which
# an objection is promoted from a note to a finding.

# Microphone distance agreement. Below this the geometric difference is under
# 0.1 dB at a 1 m standoff, which is inside ordinary stand-placement repeatability.
DISTANCE_TOLERANCE_M: float = 0.01

# Microphone angle agreement. Muzzle-blast directivity is steep and there is no
# general correction for it, so any real difference in angle is a difference in
# what was measured. ISO 17201 and MIL-STD-1474E both fix the angle for this
# reason. Two degrees is ordinary protractor repeatability on a range.
ANGLE_TOLERANCE_DEG: float = 2.0

# Temperature/humidity agreement, above which absorption is quantified.
TEMPERATURE_TOLERANCE_C: float = 5.0
HUMIDITY_TOLERANCE_PCT: float = 20.0

# Frequencies at which a weather difference is quantified for the operator. Low
# enough to matter for blast energy, high enough for absorption to be visible.
QUANTIFY_FREQUENCIES_HZ: Tuple[float, ...] = (1000.0, 4000.0)

# Calibration drift limit, matching TestMetadata.calibration_drift_dB.
CALIBRATION_DRIFT_LIMIT_dB: float = 0.5

# Minimum shots for a string mean to be worth differencing.
MIN_SHOTS_PER_STRING: int = 5


class PairingError(ValueError):
    """Raised when pairing inputs are structurally unusable."""


# ---- Objections ----

@dataclass
class Objection:
    """One reason a reference/test pair is not a clean comparison."""
    code: str
    severity: str
    message: str
    quantified_dB: Optional[float] = None
    correctable: bool = False

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK.get(self.severity, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "quantified_dB": (
                round(self.quantified_dB, 2) if self.quantified_dB is not None else None
            ),
            "correctable": self.correctable,
        }


@dataclass
class ComparabilityReport:
    """Whether a reference/test pair can carry an insertion-loss claim."""
    objections: List[Objection] = field(default_factory=list)
    reference_label: str = "reference"
    test_label: str = "test"

    @property
    def comparable(self) -> bool:
        """False when any objection makes this not an insertion loss at all."""
        return not any(o.severity == BLOCKING for o in self.objections)

    @property
    def blocking(self) -> List[Objection]:
        return [o for o in self.objections if o.severity == BLOCKING]

    @property
    def material(self) -> List[Objection]:
        return [o for o in self.objections if o.severity == MATERIAL]

    @property
    def advisory(self) -> List[Objection]:
        return [o for o in self.objections if o.severity == ADVISORY]

    @property
    def unexplained_dB(self) -> float:
        """
        Total decibels of the reported reduction that are attributable to
        differences between the two setups rather than to the suppressor.

        Contributions are summed in decibels because each is an independent
        additive offset on a level; this is a magnitude for the operator to weigh
        against the reduction, not a correction to apply.
        """
        return float(sum(
            abs(o.quantified_dB) for o in self.objections
            if o.quantified_dB is not None
        ))

    @property
    def all_correctable(self) -> bool:
        """Whether every quantified objection has a correction available."""
        quantified = [o for o in self.objections if o.quantified_dB is not None]
        return bool(quantified) and all(o.correctable for o in quantified)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparable": self.comparable,
            "reference": self.reference_label,
            "test": self.test_label,
            "unexplained_dB": round(self.unexplained_dB, 2),
            "all_correctable": self.all_correctable,
            "objections": [o.to_dict() for o in self.objections],
        }

    def summary(self) -> str:
        head = "COMPARABLE" if self.comparable else "NOT COMPARABLE"
        lines = [f"  {self.reference_label} vs {self.test_label}: {head}"]
        if not self.objections:
            lines.append("    No objection: the two strings describe the same experiment.")
            return "\n".join(lines)

        for group, title in ((self.blocking, "BLOCKING"),
                             (self.material, "MATERIAL"),
                             (self.advisory, "note")):
            for o in group:
                amount = (
                    f" [{o.quantified_dB:+.2f} dB]" if o.quantified_dB is not None else ""
                )
                lines.append(f"    {title}: {o.message}{amount}")

        if self.unexplained_dB > 0:
            lines.append(
                f"    {self.unexplained_dB:.2f} dB of any reported reduction is "
                f"attributable to differences between the setups, not to the suppressor."
            )
        return "\n".join(lines)


# ---- Metadata access ----

def _get(meta: Any, name: str, default=None):
    if meta is None:
        return default
    if isinstance(meta, dict):
        value = meta.get(name, default)
    else:
        value = getattr(meta, name, default)
    return default if value is None else value


def _text(meta: Any, name: str) -> str:
    return str(_get(meta, name, "") or "").strip()


def _num(meta: Any, name: str) -> Optional[float]:
    value = _get(meta, name, None)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_date(text: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# ---- Individual checks ----

def _check_configuration(ref: Any, test: Any, out: List[Objection]) -> None:
    ref_cfg, test_cfg = _text(ref, "configuration").lower(), _text(test, "configuration").lower()

    if not ref_cfg or not test_cfg:
        out.append(Objection(
            code="configuration_missing",
            severity=ADVISORY,
            message=(
                "configuration was not recorded on one or both strings, so nothing "
                "confirms which is the unsuppressed reference"
            ),
        ))
        return

    if ref_cfg == test_cfg:
        out.append(Objection(
            code="configuration_identical",
            severity=BLOCKING,
            message=(
                f"both strings are marked '{ref_cfg}'. A difference between two "
                f"recordings of the same configuration is repeatability, not "
                f"insertion loss."
            ),
        ))
        return

    if ref_cfg != "unsuppressed":
        out.append(Objection(
            code="reference_not_unsuppressed",
            severity=BLOCKING,
            message=(
                f"the reference string is marked '{ref_cfg}', not 'unsuppressed'. "
                f"Insertion loss is defined against the unsuppressed weapon."
            ),
        ))


def _check_source(ref: Any, test: Any, out: List[Objection]) -> None:
    for field_name, label in (("weapon", "weapon"), ("ammunition", "ammunition")):
        r, t = _text(ref, field_name), _text(test, field_name)
        if not r or not t:
            out.append(Objection(
                code=f"{field_name}_missing",
                severity=ADVISORY,
                message=f"{label} was not recorded on one or both strings",
            ))
        elif r.lower() != t.lower():
            out.append(Objection(
                code=f"{field_name}_differs",
                severity=BLOCKING,
                message=(
                    f"{label} differs: reference used '{r}', test used '{t}'. The "
                    f"difference between them includes the change of {label}, so it "
                    f"is not the suppressor's insertion loss."
                ),
            ))


def _check_geometry(ref: Any, test: Any, out: List[Objection]) -> None:
    ref_d, test_d = _num(ref, "mic_distance_m"), _num(test, "mic_distance_m")
    if ref_d is None or test_d is None:
        out.append(Objection(
            code="distance_missing",
            severity=MATERIAL,
            message=(
                "mic_distance_m was not recorded on one or both strings, so it cannot "
                "be shown that the two were measured from the same place, and no "
                "distance correction is possible"
            ),
        ))
    elif ref_d <= 0 or test_d <= 0:
        out.append(Objection(
            code="distance_invalid",
            severity=BLOCKING,
            message=f"mic_distance_m must be positive; got {ref_d} m and {test_d} m",
        ))
    elif abs(ref_d - test_d) > DISTANCE_TOLERANCE_M:
        # The test string was measured at test_d; referring it to ref_d changes it
        # by this much, and that amount lands directly in the reported reduction.
        spreading = geometric_spreading_dB(test_d, ref_d)
        out.append(Objection(
            code="distance_differs",
            severity=MATERIAL,
            message=(
                f"microphone distance differs: reference at {ref_d:g} m, test at "
                f"{test_d:g} m. Spherical spreading alone accounts for "
                f"{spreading:+.2f} dB of the difference between the two strings. "
                f"This is correctable: normalise both to a common distance."
            ),
            quantified_dB=spreading,
            correctable=True,
        ))

    ref_a, test_a = _num(ref, "mic_angle_deg"), _num(test, "mic_angle_deg")
    if ref_a is None or test_a is None:
        out.append(Objection(
            code="angle_missing",
            severity=MATERIAL,
            message=(
                "mic_angle_deg was not recorded on one or both strings. Muzzle-blast "
                "radiation is strongly directional, so an unrecorded angle leaves an "
                "unbounded difference between the two measurements"
            ),
        ))
    elif abs(ref_a - test_a) > ANGLE_TOLERANCE_DEG:
        out.append(Objection(
            code="angle_differs",
            severity=BLOCKING,
            message=(
                f"microphone angle differs: reference at {ref_a:g} deg, test at "
                f"{test_a:g} deg, a difference of {abs(ref_a - test_a):g} deg. Muzzle "
                f"blast is strongly directional and SASA has no directivity model, so "
                f"this difference cannot be corrected or even bounded. The two strings "
                f"measured different things."
            ),
        ))

    ref_h, test_h = _num(ref, "mic_height_m"), _num(test, "mic_height_m")
    if ref_h is not None and test_h is not None and abs(ref_h - test_h) > DISTANCE_TOLERANCE_M:
        out.append(Objection(
            code="height_differs",
            severity=MATERIAL,
            message=(
                f"microphone height differs: {ref_h:g} m vs {test_h:g} m. The ground "
                f"reflection arrives at a different delay, which changes the "
                f"interference pattern across the spectrum"
            ),
        ))

    ref_g, test_g = _text(ref, "ground_surface"), _text(test, "ground_surface")
    if ref_g and test_g and ref_g.lower() != test_g.lower():
        out.append(Objection(
            code="ground_differs",
            severity=MATERIAL,
            message=(
                f"ground surface differs: '{ref_g}' vs '{test_g}'. Surface impedance "
                f"sets the strength of the ground reflection"
            ),
        ))


def _check_atmosphere(ref: Any, test: Any, out: List[Objection]) -> None:
    ref_t, test_t = _num(ref, "temperature_C"), _num(test, "temperature_C")
    ref_h, test_h = _num(ref, "humidity_pct"), _num(test, "humidity_pct")

    if ref_t is None or test_t is None or ref_h is None or test_h is None:
        out.append(Objection(
            code="weather_missing",
            severity=ADVISORY,
            message=(
                "temperature or humidity was not recorded on one or both strings, so "
                "the difference in atmospheric absorption between them is unknown"
            ),
        ))
        return

    differs = (
        abs(ref_t - test_t) > TEMPERATURE_TOLERANCE_C
        or abs(ref_h - test_h) > HUMIDITY_TOLERANCE_PCT
    )
    if not differs:
        return

    distance = _num(test, "mic_distance_m") or _num(ref, "mic_distance_m")
    if distance is None or distance <= 0:
        out.append(Objection(
            code="weather_differs",
            severity=ADVISORY,
            message=(
                f"weather differs between strings ({ref_t:g} C / {ref_h:g}% RH versus "
                f"{test_t:g} C / {test_h:g}% RH), but without mic_distance_m the "
                f"absorption difference cannot be quantified"
            ),
        ))
        return

    ref_air = Atmosphere.from_metadata(ref)
    test_air = Atmosphere.from_metadata(test)
    freqs = np.array(QUANTIFY_FREQUENCIES_HZ, dtype=np.float64)
    delta = np.atleast_1d(
        test_air.absorption_coefficient_dB_per_m(freqs)
        - ref_air.absorption_coefficient_dB_per_m(freqs)
    ) * distance

    worst = float(delta[int(np.argmax(np.abs(delta)))])
    detail = ", ".join(
        f"{f:.0f} Hz: {d:+.2f} dB" for f, d in zip(freqs, delta)
    )
    out.append(Objection(
        code="weather_differs",
        severity=MATERIAL,
        message=(
            f"weather differs between strings ({ref_t:g} C / {ref_h:g}% RH versus "
            f"{test_t:g} C / {test_h:g}% RH). Over the {distance:g} m path this "
            f"changes atmospheric absorption by {detail}. This is correctable per "
            f"band: normalise both spectra to a common atmosphere."
        ),
        quantified_dB=worst,
        correctable=True,
    ))


def _check_chain(ref: Any, test: Any, out: List[Objection]) -> None:
    ref_mic, test_mic = _text(ref, "mic_model"), _text(test, "mic_model")
    if ref_mic and test_mic and ref_mic.lower() != test_mic.lower():
        out.append(Objection(
            code="mic_model_differs",
            severity=MATERIAL,
            message=(
                f"microphone model differs: '{ref_mic}' vs '{test_mic}'. Two "
                f"microphones have different free-field responses, so part of the "
                f"measured difference is the instrument"
            ),
        ))

    ref_serial, test_serial = _text(ref, "mic_serial"), _text(test, "mic_serial")
    if ref_serial and test_serial and ref_serial != test_serial:
        out.append(Objection(
            code="mic_serial_differs",
            severity=ADVISORY,
            message=(
                f"a different microphone was used: serial {ref_serial} vs {test_serial}"
            ),
        ))

    for meta, label in ((ref, "reference"), (test, "test")):
        drift = _get(meta, "calibration_drift_dB", None)
        if drift is None and hasattr(meta, "calibration_drift_dB"):
            drift = meta.calibration_drift_dB
        if drift is not None and math.isfinite(float(drift)):
            if abs(float(drift)) > CALIBRATION_DRIFT_LIMIT_dB:
                out.append(Objection(
                    code=f"{label}_calibration_drift",
                    severity=MATERIAL,
                    message=(
                        f"the {label} string's calibration drifted {float(drift):+.2f} dB "
                        f"across the session, beyond the {CALIBRATION_DRIFT_LIMIT_dB:g} dB "
                        f"limit; the chain was not stable while it was recorded"
                    ),
                    quantified_dB=float(drift),
                ))

    ref_windscreen, test_windscreen = _text(ref, "windscreen"), _text(test, "windscreen")
    if ref_windscreen and test_windscreen and ref_windscreen.lower() != test_windscreen.lower():
        out.append(Objection(
            code="windscreen_differs",
            severity=MATERIAL,
            message=(
                f"windscreen differs: '{ref_windscreen}' vs '{test_windscreen}'. A "
                f"windscreen has its own insertion loss, mostly above 2 kHz"
            ),
        ))


def _check_session(ref: Any, test: Any, out: List[Objection]) -> None:
    ref_date, test_date = _parse_date(_text(ref, "date")), _parse_date(_text(test, "date"))
    if ref_date and test_date and ref_date != test_date:
        gap = abs((test_date - ref_date).days)
        out.append(Objection(
            code="different_day",
            severity=ADVISORY if gap <= 1 else MATERIAL,
            message=(
                f"the two strings were recorded {gap} day(s) apart "
                f"({ref_date.isoformat()} and {test_date.isoformat()}). The chain was "
                f"re-rigged in between, so the calibration must be shown to hold "
                f"across both"
            ),
        ))

    ref_loc, test_loc = _text(ref, "location"), _text(test, "location")
    if ref_loc and test_loc and ref_loc.lower() != test_loc.lower():
        out.append(Objection(
            code="location_differs",
            severity=BLOCKING,
            message=(
                f"the strings were recorded at different locations: '{ref_loc}' and "
                f"'{test_loc}'. The acoustic environment, not just the suppressor, "
                f"differs between them."
            ),
        ))


def _check_strings(
    reference_aggregate: Any,
    test_aggregate: Any,
    out: List[Objection],
) -> None:
    for agg, label in ((reference_aggregate, "reference"), (test_aggregate, "test")):
        if agg is None:
            continue
        n_valid = _get(agg, "n_valid", None)
        if n_valid is None:
            continue
        n_valid = int(n_valid)
        if n_valid == 0:
            out.append(Objection(
                code=f"{label}_empty",
                severity=BLOCKING,
                message=f"the {label} string has no valid shots to average",
            ))
        elif n_valid < MIN_SHOTS_PER_STRING:
            out.append(Objection(
                code=f"{label}_short",
                severity=MATERIAL,
                message=(
                    f"the {label} string has only {n_valid} valid shot(s), below the "
                    f"{MIN_SHOTS_PER_STRING} needed for a string mean whose confidence "
                    f"interval is narrower than ordinary shot-to-shot spread"
                ),
            ))

    ref_bands = _get(reference_aggregate, "band_frequencies", None)
    test_bands = _get(test_aggregate, "band_frequencies", None)
    if ref_bands is not None and test_bands is not None:
        ref_arr, test_arr = np.asarray(ref_bands), np.asarray(test_bands)
        if ref_arr.size and test_arr.size and ref_arr.shape != test_arr.shape:
            out.append(Objection(
                code="band_layout_differs",
                severity=BLOCKING,
                message=(
                    f"the two strings were analysed over different filter banks "
                    f"({ref_arr.size} bands versus {test_arr.size}), which usually "
                    f"means different sample rates. Per-band insertion loss cannot be "
                    f"computed across them."
                ),
            ))


def assess_comparability(
    reference_metadata: Any,
    test_metadata: Any,
    reference_aggregate: Any = None,
    test_aggregate: Any = None,
    *,
    reference_label: str = "reference",
    test_label: str = "test",
) -> ComparabilityReport:
    """
    Decide whether a reference/test pair can support an insertion-loss claim.

    Args:
        reference_metadata: TestMetadata (or dict) for the UNSUPPRESSED string.
        test_metadata: TestMetadata (or dict) for the suppressed string.
        reference_aggregate: Optional AggregateMetrics for the reference string.
        test_aggregate: Optional AggregateMetrics for the test string.
        reference_label: Name for the reference in messages.
        test_label: Name for the test in messages.

    Returns:
        ComparabilityReport. Check `.comparable` before reporting insertion loss,
        and `.unexplained_dB` for how much of the difference is not the suppressor.
    """
    objections: List[Objection] = []

    _check_configuration(reference_metadata, test_metadata, objections)
    _check_source(reference_metadata, test_metadata, objections)
    _check_geometry(reference_metadata, test_metadata, objections)
    _check_atmosphere(reference_metadata, test_metadata, objections)
    _check_chain(reference_metadata, test_metadata, objections)
    _check_session(reference_metadata, test_metadata, objections)
    _check_strings(reference_aggregate, test_aggregate, objections)

    objections.sort(key=lambda o: (-o.rank, o.code))
    return ComparabilityReport(
        objections=objections,
        reference_label=reference_label,
        test_label=test_label,
    )


# ---- Auto-pairing ----

# Fields that must agree for two recordings to be the same experiment, and the
# score awarded when they do. Weapon and ammunition are handled as gates rather
# than scores: differing on them is disqualifying, not merely a lower score.
_SCORE_FIELDS: Tuple[Tuple[str, float], ...] = (
    ("mic_distance_m", 4.0),
    ("mic_angle_deg", 4.0),
    ("mic_model", 2.0),
    ("mic_serial", 1.0),
    ("location", 2.0),
    ("date", 2.0),
    ("operator", 1.0),
    ("windscreen", 1.0),
    ("ground_surface", 1.0),
    ("mic_height_m", 1.0),
)


@dataclass
class PairCandidate:
    """One possible reference for a test string, with the case for and against it."""
    reference_index: int
    reference_label: str
    score: float
    report: ComparabilityReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "reference_label": self.reference_label,
            "score": round(self.score, 2),
            "comparable": self.report.comparable,
            "unexplained_dB": round(self.report.unexplained_dB, 2),
        }


@dataclass
class PairingResult:
    """The outcome of trying to find a reference for one test string."""
    test_index: int
    test_label: str
    matched: Optional[PairCandidate] = None
    candidates: List[PairCandidate] = field(default_factory=list)
    refusal: str = ""

    @property
    def paired(self) -> bool:
        return self.matched is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_index": self.test_index,
            "test_label": self.test_label,
            "paired": self.paired,
            "matched": self.matched.to_dict() if self.matched else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "refusal": self.refusal,
        }

    def summary(self) -> str:
        if not self.paired:
            return f"  {self.test_label}: NOT PAIRED - {self.refusal}"
        lines = [
            f"  {self.test_label} -> {self.matched.reference_label} "
            f"(match score {self.matched.score:.1f})"
        ]
        lines.append(self.matched.report.summary())
        return "\n".join(lines)


def _fields_agree(ref: Any, test: Any, name: str) -> bool:
    """Whether one field agrees between two records, ignoring unrecorded values."""
    r_num, t_num = _num(ref, name), _num(test, name)
    if r_num is not None and t_num is not None:
        tolerance = {
            "mic_distance_m": DISTANCE_TOLERANCE_M,
            "mic_height_m": DISTANCE_TOLERANCE_M,
            "mic_angle_deg": ANGLE_TOLERANCE_DEG,
        }.get(name, 0.0)
        return abs(r_num - t_num) <= tolerance
    r_txt, t_txt = _text(ref, name), _text(test, name)
    if r_txt and t_txt:
        return r_txt.lower() == t_txt.lower()
    return False


def match_score(reference_metadata: Any, test_metadata: Any) -> float:
    """
    How strongly two records look like the same experiment.

    Score is the sum of the weights of the fields that agree. Fields that were
    not recorded score nothing: an absent field is not evidence of a match.
    """
    return float(sum(
        weight for name, weight in _SCORE_FIELDS
        if _fields_agree(reference_metadata, test_metadata, name)
    ))


def _is_disqualified(ref: Any, test: Any) -> str:
    """Reason this reference cannot serve this test at all, or ""."""
    for name, label in (("weapon", "weapon"), ("ammunition", "ammunition")):
        r, t = _text(ref, name), _text(test, name)
        if r and t and r.lower() != t.lower():
            return f"{label} differs ('{r}' vs '{t}')"
    ref_cfg = _text(ref, "configuration").lower()
    if ref_cfg and ref_cfg != "unsuppressed":
        return f"candidate is marked '{ref_cfg}', not 'unsuppressed'"
    return ""


def auto_pair(
    recordings: Sequence[Any],
    *,
    labels: Optional[Sequence[str]] = None,
    aggregates: Optional[Sequence[Any]] = None,
) -> List[PairingResult]:
    """
    Match every suppressed string in a session to its unsuppressed reference.

    A test string is paired only when exactly one reference scores highest. When
    two references tie, the pairing is refused and both are reported: a silently
    mis-paired reference produces a plausible insertion loss that is wrong, which
    is worse than producing none.

    Args:
        recordings: TestMetadata records (or dicts) for the whole session.
        labels: Display names, defaulting to test_id or an index.
        aggregates: Optional AggregateMetrics per recording, same order, used to
                    check string lengths and band layouts.

    Returns:
        One PairingResult per suppressed string, in input order.
    """
    metas = list(recordings)
    if labels is not None and len(labels) != len(metas):
        raise PairingError(
            f"labels has {len(labels)} entries but recordings has {len(metas)}"
        )
    if aggregates is not None and len(aggregates) != len(metas):
        raise PairingError(
            f"aggregates has {len(aggregates)} entries but recordings has {len(metas)}"
        )

    def label_of(i: int) -> str:
        if labels is not None:
            return str(labels[i])
        return _text(metas[i], "test_id") or f"recording {i}"

    def aggregate_of(i: int):
        return aggregates[i] if aggregates is not None else None

    reference_indices = [
        i for i, m in enumerate(metas)
        if _text(m, "configuration").lower() == "unsuppressed"
    ]
    test_indices = [
        i for i, m in enumerate(metas)
        if _text(m, "configuration").lower() == "suppressed"
    ]

    results: List[PairingResult] = []
    for ti in test_indices:
        result = PairingResult(test_index=ti, test_label=label_of(ti))

        if not reference_indices:
            result.refusal = (
                "no recording in this session is marked 'unsuppressed', so there is "
                "no reference to measure insertion loss against"
            )
            results.append(result)
            continue

        candidates: List[PairCandidate] = []
        disqualified: List[str] = []
        for ri in reference_indices:
            reason = _is_disqualified(metas[ri], metas[ti])
            if reason:
                disqualified.append(f"{label_of(ri)} ({reason})")
                continue
            report = assess_comparability(
                metas[ri], metas[ti],
                aggregate_of(ri), aggregate_of(ti),
                reference_label=label_of(ri), test_label=label_of(ti),
            )
            candidates.append(PairCandidate(
                reference_index=ri,
                reference_label=label_of(ri),
                score=match_score(metas[ri], metas[ti]),
                report=report,
            ))

        candidates.sort(key=lambda c: (-c.score, c.reference_index))
        result.candidates = candidates

        if not candidates:
            result.refusal = (
                "every unsuppressed recording is disqualified as a reference: "
                + "; ".join(disqualified)
            )
        elif len(candidates) > 1 and candidates[0].score == candidates[1].score:
            tied = [c.reference_label for c in candidates if c.score == candidates[0].score]
            result.refusal = (
                f"{len(tied)} references match equally well ({', '.join(tied)}). "
                f"Pairing was not guessed; choose one explicitly."
            )
        else:
            result.matched = candidates[0]

        results.append(result)

    return results


# ---- CLI for testing ----

def main() -> int:
    """Demonstrate comparability on two synthetic metadata records."""
    import argparse

    parser = argparse.ArgumentParser(description="Reference pairing and comparability")
    parser.add_argument("--test-distance", type=float, default=1.5,
                        help="Test string microphone distance (m)")
    args = parser.parse_args()

    reference = {
        "configuration": "unsuppressed", "weapon": "AR-15 10.5in",
        "ammunition": "55gr FMJ", "mic_distance_m": 1.0, "mic_angle_deg": 90.0,
        "temperature_C": 20.0, "humidity_pct": 50.0, "pressure_kPa": 101.3,
        "location": "Bay 3", "date": "2026-08-12", "test_id": "REF-01",
    }
    test = dict(reference)
    test.update({
        "configuration": "suppressed", "mic_distance_m": args.test_distance,
        "test_id": "SUP-01",
    })

    print(assess_comparability(reference, test, reference_label="REF-01",
                               test_label="SUP-01").summary())
    print()
    for result in auto_pair([reference, test]):
        print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
