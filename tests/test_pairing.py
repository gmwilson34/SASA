"""
test_pairing.py - reference pairing and comparability.

The quantified objections are checked against the same closed forms the physics
gives: a distance difference costs exactly 20*log10(d_test/d_ref) dB, and a
weather difference costs exactly the ISO 9613-1 absorption difference over the
path. The structural objections are checked for what they must REFUSE, since the
cost of a wrong pairing is a confident insertion-loss number for an experiment
that was never run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atmosphere import absorption_coefficient_dB_per_m
from pairing import (
    ADVISORY,
    ANGLE_TOLERANCE_DEG,
    BLOCKING,
    MATERIAL,
    MIN_SHOTS_PER_STRING,
    QUANTIFY_FREQUENCIES_HZ,
    PairingError,
    assess_comparability,
    auto_pair,
    match_score,
)


def reference_record(**overrides):
    """A fully specified unsuppressed reference; every field the checks read."""
    record = {
        "configuration": "unsuppressed",
        "weapon": "AR-15 10.5in",
        "ammunition": "55gr FMJ",
        "mic_model": "GRAS 46BE",
        "mic_serial": "12345",
        "mic_distance_m": 1.0,
        "mic_angle_deg": 90.0,
        "mic_height_m": 1.6,
        "ground_surface": "short grass",
        "windscreen": "foam 90mm",
        "temperature_C": 20.0,
        "humidity_pct": 50.0,
        "pressure_kPa": 101.325,
        "location": "Bay 3",
        "date": "2026-08-12",
        "operator": "G. Wilson",
        "test_id": "REF-01",
    }
    record.update(overrides)
    return record


def suppressed_record(**overrides):
    """The suppressed counterpart, identical but for the configuration."""
    fields = {"configuration": "suppressed", "test_id": "SUP-01"}
    fields.update(overrides)
    return reference_record(**fields)


class FakeAggregate:
    def __init__(self, n_valid, band_frequencies=None):
        self.n_valid = n_valid
        self.band_frequencies = (
            np.array([]) if band_frequencies is None else np.asarray(band_frequencies)
        )


def codes(report):
    return {o.code for o in report.objections}


# ---------------------------------------------------------------------------
# The clean case
# ---------------------------------------------------------------------------

def test_two_identical_setups_raise_no_objection():
    report = assess_comparability(reference_record(), suppressed_record())
    assert report.comparable
    assert report.objections == []
    assert report.unexplained_dB == 0.0


def test_the_clean_case_reads_as_clean():
    report = assess_comparability(reference_record(), suppressed_record())
    assert "No objection" in report.summary()


# ---------------------------------------------------------------------------
# Objections that make the comparison meaningless
# ---------------------------------------------------------------------------

def test_two_strings_of_the_same_configuration_cannot_be_an_insertion_loss():
    report = assess_comparability(
        reference_record(), suppressed_record(configuration="unsuppressed")
    )
    assert not report.comparable
    assert "configuration_identical" in codes(report)


def test_a_suppressed_reference_is_refused():
    report = assess_comparability(
        reference_record(configuration="suppressed"),
        suppressed_record(configuration="unsuppressed"),
    )
    assert not report.comparable
    assert "reference_not_unsuppressed" in codes(report)


def test_a_different_weapon_blocks_the_comparison():
    report = assess_comparability(reference_record(), suppressed_record(weapon="AR-15 16in"))
    assert not report.comparable
    assert "weapon_differs" in codes(report)


def test_a_different_ammunition_blocks_the_comparison():
    report = assess_comparability(
        reference_record(), suppressed_record(ammunition="77gr OTM")
    )
    assert not report.comparable
    assert "ammunition_differs" in codes(report)


def test_a_different_location_blocks_the_comparison():
    report = assess_comparability(reference_record(), suppressed_record(location="Bay 7"))
    assert not report.comparable
    assert "location_differs" in codes(report)


def test_a_different_microphone_angle_blocks_and_says_it_cannot_be_corrected():
    """
    Muzzle blast is directional and SASA has no directivity model, so an angle
    difference is not a correctable offset: it means the two strings measured
    different things.
    """
    report = assess_comparability(
        reference_record(), suppressed_record(mic_angle_deg=90.0 + ANGLE_TOLERANCE_DEG + 1.0)
    )
    assert not report.comparable
    objection = next(o for o in report.objections if o.code == "angle_differs")
    assert objection.severity == BLOCKING
    assert not objection.correctable
    assert objection.quantified_dB is None


def test_an_angle_difference_within_setup_repeatability_is_accepted():
    report = assess_comparability(
        reference_record(), suppressed_record(mic_angle_deg=90.0 + ANGLE_TOLERANCE_DEG)
    )
    assert report.comparable
    assert "angle_differs" not in codes(report)


def test_mismatched_filter_banks_block_per_band_insertion_loss():
    report = assess_comparability(
        reference_record(), suppressed_record(),
        FakeAggregate(10, np.arange(24)), FakeAggregate(10, np.arange(28)),
    )
    assert not report.comparable
    assert "band_layout_differs" in codes(report)


def test_an_empty_string_blocks_the_comparison():
    report = assess_comparability(
        reference_record(), suppressed_record(), FakeAggregate(0), FakeAggregate(10)
    )
    assert not report.comparable
    assert "reference_empty" in codes(report)


# ---------------------------------------------------------------------------
# Objections that are quantified in decibels
# ---------------------------------------------------------------------------

def test_a_distance_difference_is_quantified_as_spherical_spreading():
    """
    Moving the microphone from 1.0 m to 1.5 m changes the level by exactly
    20*log10(1.5/1.0) = 3.5218 dB, and that lands directly in the reported
    reduction.
    """
    report = assess_comparability(reference_record(), suppressed_record(mic_distance_m=1.5))
    objection = next(o for o in report.objections if o.code == "distance_differs")
    assert objection.quantified_dB == pytest.approx(20.0 * math.log10(1.5), rel=1e-12)
    assert objection.severity == MATERIAL
    assert objection.correctable
    assert report.comparable  # correctable, so not blocking
    assert report.unexplained_dB == pytest.approx(20.0 * math.log10(1.5), rel=1e-12)


def test_a_closer_test_microphone_gives_a_negative_correction():
    report = assess_comparability(reference_record(), suppressed_record(mic_distance_m=0.5))
    objection = next(o for o in report.objections if o.code == "distance_differs")
    assert objection.quantified_dB == pytest.approx(20.0 * math.log10(0.5), rel=1e-12)


def test_a_weather_difference_is_quantified_as_absorption_over_the_path():
    """
    The stated cost must equal the ISO 9613-1 absorption difference between the
    two atmospheres over the measurement path, at the frequency where it is
    largest.
    """
    ref = reference_record(temperature_C=5.0, humidity_pct=90.0, mic_distance_m=10.0)
    test = suppressed_record(temperature_C=35.0, humidity_pct=15.0, mic_distance_m=10.0)

    report = assess_comparability(ref, test)
    objection = next(o for o in report.objections if o.code == "weather_differs")

    freqs = np.array(QUANTIFY_FREQUENCIES_HZ)
    delta = (
        np.atleast_1d(absorption_coefficient_dB_per_m(freqs, 35.0, 15.0, 101.325))
        - np.atleast_1d(absorption_coefficient_dB_per_m(freqs, 5.0, 90.0, 101.325))
    ) * 10.0
    expected = float(delta[int(np.argmax(np.abs(delta)))])
    assert objection.quantified_dB == pytest.approx(expected, rel=1e-9)
    assert objection.correctable


def test_identical_weather_raises_no_absorption_objection():
    report = assess_comparability(reference_record(), suppressed_record())
    assert "weather_differs" not in codes(report)


def test_calibration_drift_beyond_the_limit_is_material():
    class Drifting(dict):
        calibration_drift_dB = 0.9

    ref = Drifting(reference_record())
    report = assess_comparability(ref, suppressed_record())
    assert "reference_calibration_drift" in codes(report)


def test_a_short_string_is_material_but_not_blocking():
    report = assess_comparability(
        reference_record(), suppressed_record(),
        FakeAggregate(MIN_SHOTS_PER_STRING - 1), FakeAggregate(10),
    )
    assert report.comparable
    objection = next(o for o in report.objections if o.code == "reference_short")
    assert objection.severity == MATERIAL


def test_a_different_windscreen_is_material():
    report = assess_comparability(reference_record(), suppressed_record(windscreen="none"))
    assert "windscreen_differs" in codes(report)
    assert report.comparable


def test_a_different_microphone_model_is_material():
    report = assess_comparability(reference_record(), suppressed_record(mic_model="PCB 378B02"))
    assert "mic_model_differs" in codes(report)


def test_unexplained_decibels_accumulate_across_objections():
    """Two independent offsets both land in the reported reduction."""
    ref = reference_record(temperature_C=5.0, humidity_pct=90.0)
    test = suppressed_record(temperature_C=35.0, humidity_pct=15.0, mic_distance_m=2.0)
    report = assess_comparability(ref, test)
    quantified = [abs(o.quantified_dB) for o in report.objections if o.quantified_dB is not None]
    assert len(quantified) >= 2
    assert report.unexplained_dB == pytest.approx(sum(quantified))


# ---------------------------------------------------------------------------
# Missing metadata
# ---------------------------------------------------------------------------

def test_an_unrecorded_distance_is_material_and_uncorrectable():
    record = suppressed_record()
    record["mic_distance_m"] = None
    report = assess_comparability(reference_record(), record)
    objection = next(o for o in report.objections if o.code == "distance_missing")
    assert objection.severity == MATERIAL
    assert not objection.correctable


def test_an_unrecorded_angle_is_material():
    record = suppressed_record()
    record["mic_angle_deg"] = None
    report = assess_comparability(reference_record(), record)
    assert "angle_missing" in codes(report)


def test_unrecorded_weather_is_advisory_not_a_silent_pass():
    record = suppressed_record()
    record["temperature_C"] = None
    report = assess_comparability(reference_record(), record)
    objection = next(o for o in report.objections if o.code == "weather_missing")
    assert objection.severity == ADVISORY


def test_an_unrecorded_configuration_cannot_confirm_the_reference():
    report = assess_comparability(
        reference_record(configuration=""), suppressed_record(configuration="")
    )
    assert "configuration_missing" in codes(report)


# ---------------------------------------------------------------------------
# Match scoring
# ---------------------------------------------------------------------------

def test_identical_records_score_every_field():
    """The score is the sum of the weights of the fields that agree."""
    score = match_score(reference_record(), suppressed_record())
    assert score == pytest.approx(4 + 4 + 2 + 1 + 2 + 2 + 1 + 1 + 1 + 1)


def test_an_unrecorded_field_scores_nothing_rather_than_matching():
    """Absence of a field is not evidence that two setups agree."""
    blank = reference_record(mic_model="", mic_serial="")
    assert match_score(blank, suppressed_record()) == pytest.approx(
        match_score(reference_record(), suppressed_record()) - 3.0
    )


def test_a_differing_distance_costs_its_weight():
    assert match_score(reference_record(), suppressed_record(mic_distance_m=2.0)) == pytest.approx(
        match_score(reference_record(), suppressed_record()) - 4.0
    )


# ---------------------------------------------------------------------------
# Auto-pairing
# ---------------------------------------------------------------------------

def test_a_single_obvious_reference_is_paired():
    results = auto_pair([reference_record(), suppressed_record()])
    assert len(results) == 1
    assert results[0].paired
    assert results[0].matched.reference_label == "REF-01"


def test_the_better_matching_reference_wins():
    near = reference_record(test_id="REF-NEAR")
    far = reference_record(test_id="REF-FAR", mic_distance_m=5.0, location="Bay 3")
    results = auto_pair([far, near, suppressed_record()])
    assert results[0].paired
    assert results[0].matched.reference_label == "REF-NEAR"


def test_two_equally_good_references_are_refused_rather_than_guessed():
    """
    A silently mis-paired reference produces a plausible insertion loss for an
    experiment nobody ran, which is the exact failure this codebase exists to
    prevent.
    """
    a = reference_record(test_id="REF-A")
    b = reference_record(test_id="REF-B")
    results = auto_pair([a, b, suppressed_record()])
    assert not results[0].paired
    assert "match equally well" in results[0].refusal
    assert len(results[0].candidates) == 2


def test_a_session_with_no_unsuppressed_recording_is_refused():
    results = auto_pair([suppressed_record(test_id="SUP-01"), suppressed_record(test_id="SUP-02")])
    assert len(results) == 2
    assert all(not r.paired for r in results)
    assert all("no recording" in r.refusal for r in results)


def test_a_reference_for_a_different_weapon_is_disqualified_not_ranked():
    wrong = reference_record(test_id="REF-WRONG", weapon="Glock 19")
    results = auto_pair([wrong, suppressed_record()])
    assert not results[0].paired
    assert "disqualified" in results[0].refusal
    assert results[0].candidates == []


def test_each_suppressed_string_gets_its_own_result():
    session = [
        reference_record(test_id="REF-A"),
        suppressed_record(test_id="SUP-1"),
        suppressed_record(test_id="SUP-2", mic_serial="12345"),
    ]
    results = auto_pair(session)
    assert [r.test_label for r in results] == ["SUP-1", "SUP-2"]
    assert all(r.paired for r in results)


def test_the_pairing_carries_the_comparability_report_with_it():
    results = auto_pair([reference_record(), suppressed_record(mic_distance_m=1.5)])
    report = results[0].matched.report
    assert "distance_differs" in {o.code for o in report.objections}
    assert report.unexplained_dB == pytest.approx(20.0 * math.log10(1.5), rel=1e-12)


def test_explicit_labels_are_used_when_supplied():
    results = auto_pair(
        [reference_record(), suppressed_record()], labels=["baseline.wav", "can.wav"]
    )
    assert results[0].test_label == "can.wav"
    assert results[0].matched.reference_label == "baseline.wav"


def test_mismatched_label_count_is_an_error():
    with pytest.raises(PairingError):
        auto_pair([reference_record(), suppressed_record()], labels=["only-one"])


def test_aggregates_are_consulted_during_pairing():
    results = auto_pair(
        [reference_record(), suppressed_record()],
        aggregates=[FakeAggregate(2), FakeAggregate(10)],
    )
    assert results[0].paired
    assert "reference_short" in {o.code for o in results[0].matched.report.objections}
