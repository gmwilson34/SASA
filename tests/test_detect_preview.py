"""
Tests for main.detect_preview, the --detect-only CLI mode, and the resolution
that decides which detection settings a run actually uses.

The preview exists so a detection setting can be moved against the answer
instead of against a guess. Two properties matter and are checked here: that it
answers rather than raises for anything a person could point it at, and that
the answer it gives is the answer a real run would give with those settings. A
preview that disagrees with the run it previews is worse than none.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from main import (
    FALLBACK_MIN_SNR_DB,
    FALLBACK_POST_SHOT_MS,
    FALLBACK_PRE_SHOT_MS,
    FALLBACK_REFRACTORY_MS,
    FALLBACK_THRESHOLD_RELATIVE_DB,
    AnalysisConfig,
    ConfigurationError,
    detect_preview,
    resolve_detection,
    resolve_min_snr,
)
from shot_detect import autotune_detection, detect_shots

from test_autotune import make_string   # the same known-answer signals

REPO_ROOT = Path(__file__).resolve().parent.parent
FS = 48000


@pytest.fixture(scope="module")
def string_wav(tmp_path_factory) -> Path:
    """Four shots 0.8 s apart, written once for the whole module."""
    path = tmp_path_factory.mktemp("preview") / "string.wav"
    sf.write(str(path), make_string(4, spacing_s=0.8, fs=FS), FS, subtype="PCM_24")
    return path


# ---------------------------------------------------------------------------
# What it reports
# ---------------------------------------------------------------------------

def test_preview_finds_the_shots_that_are_there(string_wav):
    preview = detect_preview(string_wav)

    assert preview["readable"] is True
    assert preview["problem"] is None
    assert preview["n_shots"] == 4
    assert len(preview["shots"]) == 4
    assert preview["sample_rate"] == FS


def test_preview_agrees_with_the_run_it_previews(string_wav):
    """
    The whole point. Whatever settings the preview resolved, running detection
    with them must produce the count the preview showed.
    """
    preview = detect_preview(string_wav)
    settings = preview["settings"]

    samples, rate = sf.read(str(string_wav), always_2d=True)
    shots = detect_shots(
        samples[:, 0], rate,
        threshold_relative_dB=settings["threshold_relative_dB"],
        refractory_ms=settings["refractory_ms"],
        pre_ms=settings["pre_ms"],
        post_ms=settings["post_ms"],
        min_snr_dB=settings["min_snr_dB"],
    )
    assert len(shots) == preview["n_shots"]
    assert [round(s.time_s, 4) for s in shots] == \
           [round(s["time_s"], 4) for s in preview["shots"]]


def test_preview_levels_are_relative_to_full_scale(string_wav):
    """
    There is no calibration here, so there is no sound pressure level. Every
    level must be dB re FS and be labelled as such - a peak at full scale is
    0 dBFS, never the 94 dB that the uncalibrated pass-through would give.
    """
    preview = detect_preview(string_wav)

    assert preview["level_unit"] == "dB re FS"
    for shot in preview["shots"]:
        assert -200.0 < shot["peak_dBFS"] <= 0.5
    assert preview["detection"]["peak_dBFS"] <= 0.5
    # The ambiguous absolute spellings must not survive into the payload.
    assert "peak_dB" not in preview["tuning"]
    assert "noise_floor_dB" not in preview["tuning"]
    assert preview["tuning"]["peak_dBFS"] <= 0.5


def test_preview_carries_the_tuning_and_its_basis(string_wav):
    tuning = detect_preview(string_wav)["tuning"]
    assert tuning["applied"] is True
    for key in ("threshold", "refractory", "post", "pre"):
        assert tuning["basis"][key]


def test_preview_reports_clipping_without_refusing_to_look(tmp_path):
    """
    A limited recording is inadmissible as a measurement, but detection still
    works on it and the operator still needs to see where the shots are. The
    preview says so rather than refusing.
    """
    # Driven hard into a ceiling at -3 dBFS. The gain has to be large enough
    # that the waveform DWELLS above the ceiling: clipping broadband noise by a
    # hair produces isolated clipped samples, not the flat top that is the
    # thing being detected.
    x = make_string(3, spacing_s=0.9, fs=FS)
    x = np.clip(x / np.max(np.abs(x)) * 8.0, -0.7079, 0.7079)
    path = tmp_path / "limited.wav"
    sf.write(str(path), x, FS, subtype="PCM_24")

    preview = detect_preview(path)
    assert preview["readable"] is True
    assert preview["n_shots"] > 0
    assert preview["quality"]["ceiling_clipped"] is True
    assert preview["quality"]["ceiling_dBFS"] == pytest.approx(-3.0, abs=0.3)
    assert preview["quality"]["errors"]


def test_preview_draws_an_envelope(string_wav):
    envelope = detect_preview(string_wav)["envelope"]
    assert envelope["columns"] > 100
    assert len(envelope["lo"]) == len(envelope["hi"]) == envelope["columns"]
    assert all(lo <= hi for lo, hi in zip(envelope["lo"], envelope["hi"]))


# ---------------------------------------------------------------------------
# Settings it is given
# ---------------------------------------------------------------------------

def test_manual_settings_are_obeyed_exactly(string_wav):
    preview = detect_preview(
        string_wav, auto_detect=False,
        threshold_relative_dB=12.0, refractory_ms=100.0, pre_ms=30.0, post_ms=300.0,
    )
    settings = preview["settings"]
    assert settings["threshold_relative_dB"] == 12.0
    assert settings["refractory_ms"] == 100.0
    assert settings["pre_ms"] == 30.0
    assert settings["post_ms"] == 300.0
    assert preview["tuning"] is None
    assert set(settings["source"].values()) == {"operator"}


def test_a_named_setting_wins_and_the_rest_are_still_measured(string_wav):
    preview = detect_preview(string_wav, refractory_ms=90.0)
    settings = preview["settings"]

    assert settings["refractory_ms"] == 90.0
    assert settings["source"]["refractory"] == "operator"
    assert settings["source"]["threshold"] == "measured"
    assert settings["source"]["post"] == "measured"


def test_a_tighter_threshold_finds_no_more_shots(string_wav):
    """Monotonic in the direction that matters: tightening cannot add events."""
    wide = detect_preview(string_wav, auto_detect=False, threshold_relative_dB=35.0)
    tight = detect_preview(string_wav, auto_detect=False, threshold_relative_dB=8.0)
    assert tight["n_shots"] <= wide["n_shots"]


# ---------------------------------------------------------------------------
# The impulsiveness gate
# ---------------------------------------------------------------------------

def test_the_gate_defaults_to_the_shipped_value():
    assert resolve_min_snr(AnalysisConfig()) == FALLBACK_MIN_SNR_DB
    assert resolve_min_snr(AnalysisConfig(min_snr_dB=25.0)) == 25.0


def test_the_gate_reaches_both_the_tuner_and_the_detector(string_wav):
    """
    It bounds the tuner's search as well as rejecting candidates, so a raised
    gate has to change the reported settings and not only the shot list.
    """
    low = detect_preview(string_wav, min_snr_dB=15.0)
    high = detect_preview(string_wav, min_snr_dB=40.0)

    assert low["settings"]["min_snr_dB"] == 15.0
    assert high["settings"]["min_snr_dB"] == 40.0
    assert high["tuning"]["stable_to_dB"] is None \
        or high["tuning"]["stable_to_dB"] <= low["tuning"]["stable_to_dB"]


def test_a_gate_with_no_room_left_refuses_to_tune_rather_than_inventing_a_span():
    x = make_string(3, spacing_s=1.0, fs=FS)
    tuning = autotune_detection(x, FS, min_snr_dB=79.0)
    assert tuning.applied is False
    assert "gate" in tuning.reason


def test_an_absurd_gate_is_refused_at_configuration_time():
    """
    Above 80 dB nothing real survives. Rejected when the config is built, not
    silently applied and then reported as "no shots detected".
    """
    with pytest.raises(ConfigurationError):
        AnalysisConfig(min_snr_dB=120.0)
    with pytest.raises(ConfigurationError):
        AnalysisConfig(min_snr_dB=-1.0)


# ---------------------------------------------------------------------------
# resolve_detection, on its own
# ---------------------------------------------------------------------------

def test_resolution_falls_back_to_the_shipped_defaults_with_no_tuning():
    resolved = resolve_detection(AnalysisConfig(auto_detect=False), None)

    assert resolved.threshold_relative_dB == FALLBACK_THRESHOLD_RELATIVE_DB
    assert resolved.refractory_ms == FALLBACK_REFRACTORY_MS
    assert resolved.pre_ms == FALLBACK_PRE_SHOT_MS
    assert resolved.post_ms == FALLBACK_POST_SHOT_MS
    assert resolved.min_snr_dB == FALLBACK_MIN_SNR_DB
    assert set(resolved.source.values()) == {"default"}
    assert resolved.tuned is False


def test_resolution_never_overrides_a_value_the_operator_gave():
    x = make_string(4, spacing_s=0.8, fs=FS)
    tuning = autotune_detection(x, FS)
    assert tuning.applied

    config = AnalysisConfig(refractory_ms=77.0, post_shot_ms=333.0)
    resolved = resolve_detection(config, tuning)

    assert resolved.refractory_ms == 77.0
    assert resolved.post_ms == 333.0
    assert resolved.threshold_relative_dB == tuning.threshold_relative_dB
    assert resolved.source["refractory"] == "operator"
    assert resolved.source["threshold"] == "measured"


def test_an_absolute_threshold_suppresses_the_relative_one_only():
    x = make_string(4, spacing_s=0.8, fs=FS)
    tuning = autotune_detection(x, FS)
    resolved = resolve_detection(AnalysisConfig(detection_threshold_dB=140.0), tuning)

    assert resolved.threshold_dB == 140.0
    assert resolved.threshold_relative_dB is None
    # The window and the spacing are measurements of the recording, not of the
    # threshold, so they stay measured.
    assert resolved.source["post"] == "measured"
    assert resolved.source["refractory"] == "measured"


def test_a_refused_tuning_leaves_the_defaults_in_place():
    rng = np.random.default_rng(5)
    tuning = autotune_detection(rng.normal(0.0, 0.01, FS * 3), FS)
    assert tuning.applied is False

    resolved = resolve_detection(AnalysisConfig(), tuning)
    assert resolved.refractory_ms == FALLBACK_REFRACTORY_MS
    assert set(resolved.source.values()) == {"default"}
    assert resolved.tuned is False


# ---------------------------------------------------------------------------
# Files it cannot use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda d: d / "missing.wav",
    lambda d: d,
    lambda d: d / "empty.wav",
    lambda d: d / "junk.wav",
])
def test_preview_explains_rather_than_raises(tmp_path, make):
    (tmp_path / "empty.wav").write_bytes(b"")
    (tmp_path / "junk.wav").write_bytes(b"not a wave file at all")

    result = detect_preview(make(tmp_path))          # must not raise
    assert isinstance(result, dict)
    assert result["readable"] is False
    assert result["problem"]
    assert result["shots"] == []


def test_preview_of_a_channel_that_does_not_exist_explains(string_wav):
    result = detect_preview(string_wav, channel=7)
    assert result["readable"] is False
    assert result["problem"]


# ---------------------------------------------------------------------------
# The CLI contract the servers depend on
# ---------------------------------------------------------------------------

def _run_detect(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--detect-only", *map(str, args)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )


def test_detect_only_prints_one_json_object_on_stdout(string_wav):
    result = _run_detect(string_wav)
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip())      # exactly one object, nothing else
    assert payload["readable"] is True
    assert payload["n_shots"] == 4


def test_detect_only_still_prints_json_for_a_file_it_cannot_read(tmp_path):
    result = _run_detect(tmp_path / "gone.wav")
    assert result.returncode != 0
    payload = json.loads(result.stdout.strip())
    assert payload["readable"] is False
    assert payload["problem"]


def test_detect_only_needs_no_calibration_and_no_output_directory(string_wav):
    result = _run_detect(string_wav)
    assert result.returncode == 0
    assert "calibration" not in result.stderr.lower()


def test_detect_only_accepts_the_settings_the_servers_pass(string_wav):
    result = _run_detect(
        string_wav, "--no-auto-detect", "--threshold-relative-dB", "18",
        "--refractory-ms", "120", "--pre-ms", "25", "--post-ms", "260",
        "--min-snr-dB", "20", "--expected-shots", "4",
    )
    assert result.returncode == 0
    settings = json.loads(result.stdout.strip())["settings"]
    assert settings["threshold_relative_dB"] == 18.0
    assert settings["refractory_ms"] == 120.0
    assert settings["pre_ms"] == 25.0
    assert settings["post_ms"] == 260.0
    assert settings["min_snr_dB"] == 20.0
