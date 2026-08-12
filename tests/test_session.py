"""
test_session.py - whole-range-session batching.

The session runner is injected with the analysis step, so these tests exercise
the batching, pairing and drift logic against synthetic records without touching
audio. What matters is what it refuses: a session that silently mis-pairs a
reference produces a plausible insertion loss for an experiment nobody ran.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from session import (
    MIN_BASELINES_FOR_TREND,
    SessionError,
    compare_session_pairs,
    discover_recordings,
    run_session,
)


def record(configuration, *, test_id, peaks, n_valid=None, **meta):
    """A synthetic analysis record with the fields session.py actually reads."""
    metadata = {
        "configuration": configuration,
        "weapon": "AR-15 10.5in",
        "ammunition": "55gr FMJ",
        "mic_model": "GRAS 46BE",
        "mic_distance_m": 1.0,
        "mic_angle_deg": 90.0,
        "temperature_C": 18.0,
        "humidity_pct": 62.0,
        "pressure_kPa": 100.4,
        "location": "Bay 3",
        "date": "2026-08-12",
        "test_id": test_id,
    }
    metadata.update(meta)
    return {
        "test_metadata": metadata,
        "analysis": {"output_dir": f"/tmp/{test_id}"},
        "aggregate": {
            "n_valid": len(peaks) if n_valid is None else n_valid,
            "band_frequencies_Hz": [125.0, 250.0, 500.0, 1000.0],
        },
        "per_shot_metrics": [
            {"shot_number": i + 1, "valid": True, "Lpeak_Z_dB": p}
            for i, p in enumerate(peaks)
        ],
    }


def session(records):
    """Build (paths, analyse) for a list of records, in order."""
    paths = [Path(f"/tmp/{r['test_metadata']['test_id']}.wav") for r in records]
    lookup = {p: r for p, r in zip(paths, records)}
    return paths, lambda p: lookup[p]


BASE = [140.0, 140.2, 139.8, 140.1, 139.9, 140.0, 140.3, 139.7]
QUIET = [128.0, 128.2, 127.8, 128.1, 127.9, 128.0, 128.3, 127.7]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_recordings_are_discovered_in_a_stable_order(tmp_path):
    """A session must run the same way twice, or "the third string" means nothing."""
    for name in ("c.wav", "a.wav", "b.flac", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    found = [p.name for p in discover_recordings(tmp_path)]
    assert found == ["a.wav", "b.flac", "c.wav"]


def test_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(SessionError):
        discover_recordings(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_every_recording_is_analysed_and_paired():
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)

    assert len(result.analysed) == 2
    assert len(result.failed) == 0
    assert len(result.paired) == 1
    assert result.paired[0].matched.reference_label == "REF-01"


def test_one_bad_recording_does_not_end_the_session():
    """A session is a morning's work; one unreadable file must not discard it."""
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET),
    ]
    paths, analyse = session(records)
    bad = Path("/tmp/broken.wav")

    def flaky(path):
        if path == bad:
            raise RuntimeError("could not read the file")
        return analyse(path)

    result = run_session([paths[0], bad, paths[1]], analyse=flaky)
    assert len(result.analysed) == 2
    assert len(result.failed) == 1
    assert "could not read the file" in result.failed[0].error
    assert len(result.paired) == 1


def test_an_analysis_that_returns_the_wrong_type_is_recorded_as_a_failure():
    result = run_session([Path("/tmp/x.wav")], analyse=lambda p: "not a record")
    assert len(result.failed) == 1
    assert len(result.analysed) == 0


def test_a_session_where_nothing_analysed_says_so():
    def always_fails(path):
        raise RuntimeError("boom")

    result = run_session([Path("/tmp/a.wav")], analyse=always_fails)
    assert not result.analysed
    assert any("nothing could be paired" in w for w in result.warnings)


def test_recordings_without_a_configuration_are_named():
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("", test_id="MYSTERY", peaks=QUIET),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)
    assert any("MYSTERY" in w for w in result.warnings)


def test_progress_is_reported_for_each_recording():
    records = [record("unsuppressed", test_id=f"R{i}", peaks=BASE) for i in range(3)]
    paths, analyse = session(records)
    seen = []
    run_session(paths, analyse=analyse, progress=lambda pct, msg: seen.append((pct, msg)))
    assert len(seen) == 3
    assert all(0 <= pct <= 100 for pct, _ in seen)


# ---------------------------------------------------------------------------
# Pairing refusals carry through the session
# ---------------------------------------------------------------------------

def test_two_equally_good_references_are_refused_not_guessed():
    records = [
        record("unsuppressed", test_id="REF-A", peaks=BASE),
        record("unsuppressed", test_id="REF-B", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)

    assert len(result.paired) == 0
    assert len(result.unpaired) == 1
    assert "match equally well" in result.unpaired[0].refusal


def test_a_session_with_no_reference_pairs_nothing():
    records = [
        record("suppressed", test_id="SUP-01", peaks=QUIET),
        record("suppressed", test_id="SUP-02", peaks=QUIET),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)
    assert len(result.paired) == 0
    assert all("no recording" in p.refusal for p in result.unpaired)


def test_the_comparability_objection_travels_with_the_pairing():
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET, mic_distance_m=1.5),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)

    rows = compare_session_pairs(result)
    assert len(rows) == 1
    assert rows[0]["unexplained_dB"] == pytest.approx(20.0 * math.log10(1.5), abs=0.01)
    assert rows[0]["comparable"] is True


def test_a_blocking_objection_is_surfaced_in_the_comparison_rows():
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET, mic_angle_deg=45.0),
    ]
    paths, analyse = session(records)
    rows = compare_session_pairs(run_session(paths, analyse=analyse))
    assert rows[0]["comparable"] is False
    assert rows[0]["blocking"]


# ---------------------------------------------------------------------------
# Baseline drift across the session
# ---------------------------------------------------------------------------

def test_baseline_drift_is_measured_on_the_unsuppressed_strings_only():
    """
    The baselines are the same experiment repeated, so a change across them is
    the rig rather than the suppressor. A suppressed string must not enter it.
    """
    records = [
        record("unsuppressed", test_id="REF-1", peaks=[140.0] * 8),
        record("suppressed", test_id="SUP-1", peaks=[128.0] * 8),
        record("unsuppressed", test_id="REF-2", peaks=[141.0] * 8),
        record("suppressed", test_id="SUP-2", peaks=[128.0] * 8),
        record("unsuppressed", test_id="REF-3", peaks=[142.0] * 8),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)

    assert result.baseline_labels == ["REF-1", "REF-2", "REF-3"]
    assert result.baseline_trend_dB_per_string == pytest.approx(1.0, abs=1e-9)


def test_two_baselines_report_their_difference_but_no_trend():
    records = [
        record("unsuppressed", test_id="REF-1", peaks=[140.0] * 8),
        record("unsuppressed", test_id="REF-2", peaks=[143.0] * 8),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)

    assert math.isnan(result.baseline_trend_dB_per_string)
    assert any("+3.00 dB" in note for note in result.notes)


def test_a_single_baseline_cannot_show_drift():
    records = [
        record("unsuppressed", test_id="REF-1", peaks=BASE),
        record("suppressed", test_id="SUP-1", peaks=QUIET),
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)
    assert math.isnan(result.baseline_trend_dB_per_string)
    assert len(result.baseline_levels_dB) == 1


def test_the_trend_needs_the_stated_minimum():
    records = [
        record("unsuppressed", test_id=f"REF-{i}", peaks=[140.0 + i] * 8)
        for i in range(MIN_BASELINES_FOR_TREND)
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)
    assert math.isfinite(result.baseline_trend_dB_per_string)


# ---------------------------------------------------------------------------
# First-round pop across the session
# ---------------------------------------------------------------------------

def test_first_round_pop_is_estimated_across_every_suppressed_string():
    """
    Several strings is the form that supports a claim, and a session is where
    several strings come from.
    """
    popped = [132.0, 128.0, 128.2, 127.8, 128.1, 127.9, 128.0, 128.1]
    records = [record("unsuppressed", test_id="REF-1", peaks=BASE)] + [
        record("suppressed", test_id=f"SUP-{i}", peaks=popped) for i in range(4)
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)

    assert result.first_round_pop is not None
    assert result.first_round_pop.basis == "across-strings"
    assert result.first_round_pop.established
    assert result.first_round_pop.observed_dB > 3.0


def test_a_session_without_pop_does_not_claim_one():
    records = [record("unsuppressed", test_id="REF-1", peaks=BASE)] + [
        record("suppressed", test_id=f"SUP-{i}", peaks=QUIET) for i in range(4)
    ]
    paths, analyse = session(records)
    result = run_session(paths, analyse=analyse)
    assert result.first_round_pop is not None
    assert not result.first_round_pop.established


def test_a_session_with_no_suppressed_string_reports_no_pop():
    records = [record("unsuppressed", test_id="REF-1", peaks=BASE)]
    paths, analyse = session(records)
    assert run_session(paths, analyse=analyse).first_round_pop is None


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_the_session_serialises():
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET),
    ]
    paths, analyse = session(records)
    data = run_session(paths, analyse=analyse).to_dict()
    for key in ("n_recordings", "n_analysed", "n_failed", "n_paired",
                "entries", "pairings", "baseline", "first_round_pop"):
        assert key in data
    assert data["n_paired"] == 1


def test_the_summary_names_failures_and_pairings():
    records = [
        record("unsuppressed", test_id="REF-01", peaks=BASE),
        record("suppressed", test_id="SUP-01", peaks=QUIET),
    ]
    paths, analyse = session(records)
    bad = Path("/tmp/broken.wav")

    def flaky(path):
        if path == bad:
            raise OSError("unreadable")
        return analyse(path)

    text = run_session([*paths, bad], analyse=flaky).summary()
    assert "FAILED" in text and "broken.wav" in text
    assert "SUP-01" in text and "REF-01" in text
