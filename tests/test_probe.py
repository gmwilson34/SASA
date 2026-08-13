"""
Tests for main.probe_input and the --probe CLI mode.

The probe exists so the interface can say true things about a recording before
it is analysed: bin spacing, the top of the usable band range, which channel
will be measured, and whether the sample rate can resolve a rise time at all.
Everything here therefore checks two properties: that the numbers match the
file, and that a file it cannot read produces an explanation rather than an
exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from main import MIN_USEFUL_SAMPLE_RATE_HZ, probe_input


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_wav(path: Path, *, fs: int, channels: int, seconds: float, subtype: str = "PCM_24"):
    n = int(round(fs * seconds))
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.05, (n, channels)) if channels > 1 else rng.normal(0.0, 0.05, n)
    sf.write(str(path), data, fs, subtype=subtype)
    return path


# ---------------------------------------------------------------------------
# What it reads off a real file
# ---------------------------------------------------------------------------

def test_probe_reports_the_files_own_properties(tmp_path):
    wav = _write_wav(tmp_path / "shot.wav", fs=96000, channels=1, seconds=0.5)
    probe = probe_input(wav)

    assert probe["readable"] is True
    assert probe["problem"] is None
    assert probe["needs_extraction"] is False
    assert probe["sample_rate"] == 96000
    assert probe["channels"] == 1
    assert probe["frames"] == 48000
    assert probe["duration_s"] == pytest.approx(0.5, abs=1e-4)
    assert probe["subtype"] == "PCM_24"
    assert probe["size_bytes"] == wav.stat().st_size


def test_probe_derives_nyquist_and_adequacy(tmp_path):
    wav = _write_wav(tmp_path / "fast.wav", fs=96000, channels=1, seconds=0.1)
    probe = probe_input(wav)
    assert probe["nyquist_Hz"] == pytest.approx(48000.0)
    assert probe["sample_rate_adequate"] is True


def test_probe_flags_a_rate_that_cannot_resolve_a_rise(tmp_path):
    # 44.1 kHz is what a phone records at, and it is below the rate at which a
    # muzzle blast's rise time means anything.
    assert 44100 < MIN_USEFUL_SAMPLE_RATE_HZ
    wav = _write_wav(tmp_path / "phone.wav", fs=44100, channels=1, seconds=0.1)
    probe = probe_input(wav)

    assert probe["sample_rate_adequate"] is False
    assert any("resolve" in note for note in probe["notes"])
    # It must say what the consequence is, not just that the rate is low.
    assert any("upper bound" in note for note in probe["notes"])


def test_probe_says_which_channel_will_be_measured(tmp_path):
    wav = _write_wav(tmp_path / "stereo.wav", fs=48000, channels=2, seconds=0.1)
    probe = probe_input(wav)
    assert probe["channels"] == 2
    note = " ".join(probe["notes"])
    assert "channel 0" in note
    # And that the channels are NOT mixed, because mixing changes the level.
    assert "not mixed" in note


def test_probe_of_a_mono_file_says_nothing_about_channels(tmp_path):
    wav = _write_wav(tmp_path / "mono.wav", fs=48000, channels=1, seconds=0.1)
    assert probe_input(wav)["notes"] == []


# ---------------------------------------------------------------------------
# Files it cannot use
# ---------------------------------------------------------------------------

def test_probe_of_a_missing_file_explains_rather_than_raises(tmp_path):
    probe = probe_input(tmp_path / "nope.wav")
    assert probe["readable"] is False
    assert "not found" in probe["problem"].lower()


def test_probe_of_a_directory_explains_rather_than_raises(tmp_path):
    probe = probe_input(tmp_path)
    assert probe["readable"] is False
    assert "not a file" in probe["problem"].lower()


def test_probe_of_a_file_that_is_not_audio_explains_rather_than_raises(tmp_path):
    junk = tmp_path / "notes.wav"
    junk.write_bytes(b"this is not a wave file at all, not even close")
    probe = probe_input(junk)
    assert probe["readable"] is False
    assert probe["problem"]


def test_probe_never_raises_on_anything_a_person_could_pick(tmp_path):
    candidates = [
        tmp_path / "missing.wav",
        tmp_path,
        tmp_path / "empty.wav",
    ]
    (tmp_path / "empty.wav").write_bytes(b"")
    for path in candidates:
        result = probe_input(path)          # must not raise
        assert isinstance(result, dict)
        assert "readable" in result


# ---------------------------------------------------------------------------
# The CLI contract the UI server depends on
# ---------------------------------------------------------------------------

def _run_probe(path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--probe", str(path)],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )


def test_probe_cli_prints_one_json_object_on_stdout(tmp_path):
    wav = _write_wav(tmp_path / "cli.wav", fs=48000, channels=1, seconds=0.1)
    result = _run_probe(wav)

    assert result.returncode == 0
    payload = json.loads(result.stdout.strip())     # exactly one object, nothing else
    assert payload["readable"] is True
    assert payload["sample_rate"] == 48000


def test_probe_cli_still_prints_json_for_an_unreadable_file(tmp_path):
    # The server parses stdout whatever the exit code is, because the JSON is
    # where the reason lives.
    missing = tmp_path / "gone.wav"
    result = _run_probe(missing)

    assert result.returncode != 0
    payload = json.loads(result.stdout.strip())
    assert payload["readable"] is False
    assert payload["problem"]


def test_probe_cli_needs_no_calibration_and_no_output_directory(tmp_path):
    """
    A probe answers before any of the run's own preconditions are resolved.

    If it did not, choosing a file would fail with "no calibration was
    supplied" — which is the message a run gives, on a page where the operator
    has not reached calibration yet.
    """
    wav = _write_wav(tmp_path / "bare.wav", fs=48000, channels=1, seconds=0.1)
    result = _run_probe(wav)
    assert result.returncode == 0
    assert "calibration" not in result.stderr.lower()
