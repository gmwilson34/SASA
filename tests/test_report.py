"""
test_report.py - the customer-facing report.

The report is where a wrong number does the most damage, because it is the
artefact that leaves the building. These tests hold it to three things: it states
what the record says, it states plainly when the record establishes nothing, and
it cannot be made to emit markup from a metadata field.
"""

from __future__ import annotations

import json
import re

import pytest

from report import build_report_from_directory, check_self_contained, generate_report

BANDS = [50.0, 125.0, 500.0, 1000.0, 4000.0, 10000.0]


def base_record(**overrides):
    """A minimal but complete schema-2.0 analysis record."""
    record = {
        "schema_version": "2.0",
        "software": {"name": "SASA", "version": "2.0.0"},
        "analysis": {"timestamp": "2026-08-12T10:00:00-04:00", "input_file": "test.wav"},
        "source": {"sample_rate": 96000, "channels": 1, "duration_s": 12.0},
        "calibration": {"calibrated": True, "level_unit": "dB SPL", "Pa_per_FS": 200.0},
        "quality": {"errors": [], "warnings": []},
        "detection": {"n_detected": 10},
        "test_metadata": {
            "configuration": "suppressed", "weapon": "AR-15", "ammunition": "55gr",
            "mic_distance_m": 1.0, "mic_angle_deg": 90.0, "temperature_C": 18.0,
            "humidity_pct": 62.0, "pressure_kPa": 100.4, "missing_required": [],
            "test_id": "SUP-01",
        },
        "atmosphere": {
            "temperature_C": 18.0, "humidity_pct": 62.0, "pressure_kPa": 100.4,
            "speed_of_sound_m_per_s": 342.0, "density_kg_per_m3": 1.2,
            "defaulted": [], "out_of_standard_range": [],
        },
        "aggregate": {"n_shots": 10, "n_valid": 10, "statistics": {}},
        "per_shot_metrics": [],
        "validity": {"measurement_valid": True, "calibrated": True,
                     "level_unit": "dB SPL", "reasons": []},
        "warnings": [],
        "insertion_loss": None,
    }
    record.update(overrides)
    return record


def string_stats(*, established, **pop_overrides):
    pop = {
        "metric": "Lpeak_Z", "observed_dB": 4.2, "first_shot_dB": 144.2,
        "subsequent_mean_dB": 140.0, "subsequent_sd_dB": 0.3, "n_subsequent": 9,
        "prediction_upper_dB": 140.9, "prediction_lower_dB": 139.1,
        "p_value": 0.0001 if established else 0.42, "established": established,
        "first_shot_quieter": False, "basis": "single-string", "n_strings": 1,
        "ci95_dB": [None, None], "refusal": "", "notes": [],
    }
    pop.update(pop_overrides)
    return {
        "Lpeak_Z": {
            "metric": "Lpeak_Z", "n_shots": 10,
            "energy_mean_dB": 140.7, "energy_mean_excluding_first_dB": 140.0,
            "first_round_cost_dB": 0.7, "median_dB": 140.0, "sd_dB": 1.2,
            "min_dB": 139.5, "max_dB": 144.2, "range_dB": 4.7,
            "percentiles_dB": {"5": 139.6, "50": 140.0, "95": 143.0},
            "ci95_half_width_dB": 0.4, "trend_dB_per_shot": -0.02,
            "trend_established": False, "trend_p_value": 0.5,
            "first_round_pop": pop,
        }
    }


def write(tmp_path, record):
    directory = tmp_path / "run"
    directory.mkdir(exist_ok=True)
    (directory / "analysis_metadata.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def render(tmp_path, record):
    path = generate_report(record, tmp_path / "report.html")
    return path.read_text(encoding="utf-8")


def flat(html):
    """Collapse whitespace so an assertion is not defeated by source wrapping."""
    return re.sub(r"\s+", " ", html)


# ---------------------------------------------------------------------------
# The report exists and is self-contained
# ---------------------------------------------------------------------------

def test_a_minimal_record_produces_a_self_contained_report(tmp_path):
    path = generate_report(base_record(), tmp_path / "report.html")
    assert path.exists()
    assert check_self_contained(path) == []


def test_build_from_directory_finds_the_record(tmp_path):
    directory = write(tmp_path, base_record())
    path = build_report_from_directory(directory)
    assert path.exists()


def test_a_record_that_is_not_a_dict_is_refused(tmp_path):
    with pytest.raises(TypeError):
        generate_report(["not", "a", "record"], tmp_path / "report.html")


# ---------------------------------------------------------------------------
# First-round pop
# ---------------------------------------------------------------------------

def test_an_established_pop_is_stated_with_its_evidence(tmp_path):
    html = render(tmp_path, base_record(string_statistics=string_stats(established=True)))
    assert "First-round pop established" in html
    # The prediction interval it was judged against must be shown, not just the verdict.
    assert "139.1" in html and "140.9" in html
    assert "0.0001" in html


def test_an_unestablished_pop_says_what_it_does_not_prove(tmp_path):
    """
    "No pop detected" and "no pop exists" are different claims, and the report
    must make only the first.
    """
    html = render(tmp_path, base_record(string_statistics=string_stats(established=False)))
    assert "No first-round pop this measurement can resolve" in html
    assert "not proof that the suppressor does not pop" in flat(html)


def test_a_refused_pop_is_reported_as_not_measured(tmp_path):
    stats = string_stats(established=False, refusal="only 2 shots followed the first")
    html = render(tmp_path, base_record(string_statistics=stats))
    assert "not measured" in html
    assert "only 2 shots followed the first" in html


def test_a_quieter_first_round_is_not_called_pop(tmp_path):
    stats = string_stats(established=False, first_shot_quieter=True)
    html = render(tmp_path, base_record(string_statistics=stats))
    assert "QUIETER" in html
    assert "not first-round pop" in html


def test_a_single_string_result_carries_its_caveat(tmp_path):
    html = render(tmp_path, base_record(string_statistics=string_stats(established=True)))
    assert "one first round, from one string" in html


def test_both_averages_are_given_together(tmp_path):
    """Quoting only the flattering average is the failure this table exists to prevent."""
    html = render(tmp_path, base_record(string_statistics=string_stats(established=True)))
    assert "Excluding first" in html
    assert "flatters its subject" in flat(html)
    assert "140.7" in html and "140.0" in html


def test_the_string_section_is_absent_when_the_record_has_no_string_statistics(tmp_path):
    html = render(tmp_path, base_record())
    assert "String behaviour" not in html


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def insertion_loss_block(objections, *, normalised=None):
    return {
        "reference_dir": "/tmp/ref", "reference_input": "ref.wav",
        "reference_n_shots": 10, "test_n_shots": 10, "level_unit": "dB SPL",
        "metrics": [{"metric": "Lpeak_Z", "reference_dB": 152.0, "test_dB": 140.0,
                     "reduction_dB": 12.0, "ci95_dB": 0.2,
                     "reference_n": 10, "test_n": 10}],
        "bands": {"frequencies_Hz": BANDS, "reference_dB": [150.0] * 6,
                  "test_dB": [138.0] * 6, "insertion_loss_dB": [12.0] * 6},
        "bands_normalised": normalised or {"valid": False, "refusal": "not attempted"},
        "comparability": {
            "comparable": not any(o["severity"] == "blocking" for o in objections),
            "reference": "REF-01", "test": "SUP-01",
            "unexplained_dB": sum(abs(o.get("quantified_dB") or 0.0) for o in objections),
            "all_correctable": all(o.get("correctable") for o in objections) if objections else False,
            "objections": objections,
        },
        "warnings": [],
    }


def test_a_blocking_objection_says_the_result_is_not_an_insertion_loss(tmp_path):
    objections = [{
        "code": "angle_differs", "severity": "blocking", "correctable": False,
        "quantified_dB": None,
        "message": "microphone angle differs: reference at 90 deg, test at 45 deg",
    }]
    html = render(tmp_path, base_record(insertion_loss=insertion_loss_block(objections)))
    assert "This is not a valid insertion loss" in html
    assert "microphone angle differs" in html


def test_a_priced_objection_shows_what_it_is_worth(tmp_path):
    objections = [{
        "code": "distance_differs", "severity": "material", "correctable": True,
        "quantified_dB": 3.52,
        "message": "microphone distance differs: reference at 1 m, test at 1.5 m",
    }]
    html = render(tmp_path, base_record(insertion_loss=insertion_loss_block(objections)))
    assert "3.52" in html
    assert "attributable to the two setups not matching" in html


def test_a_clean_comparison_says_so(tmp_path):
    html = render(tmp_path, base_record(insertion_loss=insertion_loss_block([])))
    assert "describe the same experiment" in html


# ---------------------------------------------------------------------------
# Normalised insertion loss
# ---------------------------------------------------------------------------

def test_the_normalised_table_shows_measured_and_corrected_side_by_side(tmp_path):
    """
    A corrected number that cannot be checked against the measurement it came
    from is not a measurement record, so both columns must be present.
    """
    normalised = {
        "valid": True, "normalisation_distance_m": 1.0,
        "reference_distance_m": 1.0, "test_distance_m": 1.5,
        "frequencies_Hz": BANDS,
        "raw_insertion_loss_dB": [12.0] * 6,
        "insertion_loss_dB": [8.5] * 6,
        "shift_dB": [-3.52] * 6,
        "largest_shift_dB": -3.52,
        "refusal": "", "warnings": [], "assumptions": ["free-field spherical spreading"],
    }
    record = base_record(insertion_loss=insertion_loss_block([], normalised=normalised))
    html = render(tmp_path, record)
    assert "referred to a common distance" in html
    assert "As measured" in html
    assert "12.0" in html and "8.5" in html and "−3.52" in html or "-3.52" in html
    assert "free-field spherical spreading" in html


def test_a_refused_normalisation_says_the_figures_are_uncorrected(tmp_path):
    normalised = {"valid": False, "refusal": "mic_distance_m was not recorded."}
    record = base_record(insertion_loss=insertion_loss_block([], normalised=normalised))
    html = render(tmp_path, record)
    assert "was not normalised to a common distance" in html
    assert "still in them" in html


# ---------------------------------------------------------------------------
# Flagged shots
# ---------------------------------------------------------------------------

def test_flagged_shots_are_listed_with_their_reasons(tmp_path):
    review = {
        "n_shots": 10, "n_evaluated": 10, "statistics_applied": True,
        "sensitivity": "Sensitivity: at 10 shots this test catches about four in five",
        "shots_to_exclude": [3], "shots_to_review": [7],
        "flags": [
            {"shot_number": 3, "severity": "exclude", "code": "clipped",
             "message": "samples reached digital full scale"},
            {"shot_number": 7, "severity": "review", "code": "low_snr",
             "message": "signal-to-noise is 8.0 dB"},
        ],
        "notes": [],
    }
    html = render(tmp_path, base_record(
        shot_review=review, string_statistics=string_stats(established=False)))
    assert "Shot 3" in html and "digital full scale" in html
    assert "Shot 7" in html and "signal-to-noise" in html
    assert "four in five" in html


def test_a_clean_string_says_no_shot_departs(tmp_path):
    review = {"n_shots": 10, "flags": [], "shots_to_exclude": [], "shots_to_review": [],
              "sensitivity": "Sensitivity: at 10 shots", "statistics_applied": True,
              "notes": []}
    html = render(tmp_path, base_record(
        shot_review=review, string_statistics=string_stats(established=False)))
    assert "No shot departs from the string" in html


# ---------------------------------------------------------------------------
# The report cannot be made to emit markup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["weapon", "ammunition", "test_id", "notes", "operator"])
def test_metadata_cannot_inject_markup(tmp_path, field):
    """
    Every metadata field reaches the document as text. A record is operator
    input; a report that executes it is a report that can be forged.
    """
    payload = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    record = base_record()
    record["test_metadata"][field] = payload
    html = render(tmp_path, record)
    # No LIVE markup: the payload's angle brackets must have been escaped. The
    # escaped text still contains the literal characters, so the check has to be
    # for an actual tag, not for the substring.
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html


def test_a_flag_message_cannot_inject_markup(tmp_path):
    review = {
        "n_shots": 5, "flags": [{"shot_number": 1, "severity": "review", "code": "x",
                                 "message": "<script>bad()</script>"}],
        "shots_to_exclude": [], "shots_to_review": [1], "sensitivity": "",
        "statistics_applied": True, "notes": [],
    }
    html = render(tmp_path, base_record(
        shot_review=review, string_statistics=string_stats(established=False)))
    assert "<script" not in html
    assert "&lt;script&gt;bad()" in html


def test_an_objection_message_cannot_inject_markup(tmp_path):
    objections = [{"code": "x", "severity": "material", "correctable": False,
                   "quantified_dB": None, "message": "<img src=x onerror=alert(1)>"}]
    html = render(tmp_path, base_record(insertion_loss=insertion_loss_block(objections)))
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


# ---------------------------------------------------------------------------
# Uncalibrated output must never read as sound pressure
# ---------------------------------------------------------------------------

def test_an_uncalibrated_record_never_claims_dB_SPL(tmp_path):
    record = base_record()
    record["calibration"] = {"calibrated": False, "level_unit": "dB re FS", "Pa_per_FS": 1.0}
    record["validity"] = {"measurement_valid": True, "calibrated": False,
                          "level_unit": "dB re FS", "reasons": []}
    record["string_statistics"] = string_stats(established=True)
    html = render(tmp_path, record)
    assert "dB re FS" in html
    # The string section states its unit from the record, so it must not say SPL.
    section = re.search(r"<h2>String behaviour</h2>.*?</section>", html, re.S)
    assert section is not None
    assert "dB SPL" not in section.group(0)
