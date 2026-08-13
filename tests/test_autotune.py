"""
Tests for shot_detect.autotune_detection and the repeated-delay scan.

The tuner's claim is not "these settings are good"; it is "the recording's own
answer does not change over this span of settings". So every test here builds a
signal whose shot count, spacing and decay are known by construction, and then
asserts that the tuner recovers them AND that detect_shots() run with the tuned
settings returns the count the tuner promised. A tuner whose numbers do not
survive being used is worse than no tuner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shot_detect import (
    AUTOTUNE_MAX_RELATIVE_DB,
    AUTOTUNE_REFRACTORY_LADDER,
    AUTOTUNE_MIN_RELATIVE_DB,
    AUTOTUNE_POST_MAX_MS,
    AUTOTUNE_POST_MIN_MS,
    AUTOTUNE_PRE_MS,
    AUTOTUNE_REFRACTORY_MAX_MS,
    AUTOTUNE_REFRACTORY_MIN_MS,
    DEFAULT_REFRACTORY_MS,
    ReflectionPattern,
    autotune_detection,
    detect_shots,
    find_reflection_pattern,
)

FS = 48000


# ---------------------------------------------------------------------------
# Signals whose answer is known by construction
# ---------------------------------------------------------------------------

def make_string(
    n_shots: int,
    *,
    spacing_s: float = 1.0,
    decay_ms: float = 200.0,
    fs: int = FS,
    noise: float = 1e-4,
    lead_s: float = 0.5,
    tail_s: float = 1.0,
    amplitudes=None,
    echo_delay_s: float = 0.0,
    echo_drop_dB: float = 12.0,
    seed: int = 0,
) -> np.ndarray:
    """
    A string of broadband impulsive events with a known count, spacing and decay.

    Each event is white noise under a decaying exponential, which is a far
    better stand-in for muzzle blast than a tone burst: a tone's RMS envelope
    ripples at twice its frequency and invents local maxima that no real blast
    has.

    The noise is normalised to unit RMS in 1 ms blocks before the exponential is
    applied, so the short-term envelope IS the exponential and nothing else.
    Without that, the ~10 % scatter of a 1 ms RMS estimate puts local maxima
    part-way down the decay, and the signal no longer has one right answer for
    the tuner to be tested against.
    """
    rng = np.random.default_rng(seed)
    total = lead_s + max(0, n_shots - 1) * spacing_s + tail_s
    n = int(round(total * fs))
    x = rng.normal(0.0, noise, n)

    tau = (decay_ms / 1000.0) / (60.0 / 8.6858896)     # 60 dB over decay_ms
    burst_n = min(n, int(round(6.0 * tau * fs)))
    envelope = np.exp(-np.arange(burst_n) / (tau * fs))

    block = max(1, fs // 1000)

    def flat_noise(size: int) -> np.ndarray:
        """White noise whose RMS is 1.0 in every 1 ms block."""
        pad = (-size) % block
        v = rng.normal(0.0, 1.0, size + pad).reshape(-1, block)
        v /= np.sqrt(np.mean(v ** 2, axis=1, keepdims=True))
        return v.ravel()[:size]

    if amplitudes is None:
        amplitudes = [1.0] * n_shots

    def place(at_s: float, amplitude: float) -> None:
        i0 = int(round(at_s * fs))
        if i0 >= n:
            return
        span = min(burst_n, n - i0)
        x[i0:i0 + span] += amplitude * envelope[:span] * flat_noise(span)

    for k in range(n_shots):
        at = lead_s + k * spacing_s
        place(at, float(amplitudes[k]))
        if echo_delay_s > 0:
            place(at + echo_delay_s, float(amplitudes[k]) * 10.0 ** (-echo_drop_dB / 20.0))

    return x


def tuned_count(x: np.ndarray, tuning, fs: int = FS) -> int:
    """How many shots detect_shots() returns when driven by this tuning."""
    return len(detect_shots(
        x, fs,
        threshold_relative_dB=tuning.threshold_relative_dB,
        refractory_ms=tuning.refractory_ms,
        pre_ms=tuning.pre_ms,
        post_ms=tuning.post_ms,
    ))


# ---------------------------------------------------------------------------
# The count it finds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_shots", [1, 3, 5, 10])
def test_tuner_recovers_the_shot_count(n_shots):
    x = make_string(n_shots, spacing_s=0.8)
    tuning = autotune_detection(x, FS)

    assert tuning.applied, tuning.reason
    assert tuning.n_shots == n_shots
    assert tuned_count(x, tuning) == n_shots


def test_the_settings_it_reports_are_the_settings_that_produce_its_count():
    """
    The count, the threshold and the refractory period must come from one round.

    Reporting a count found at one refractory period beside a refractory period
    chosen at another describes a run that never happened.
    """
    x = make_string(6, spacing_s=0.45, decay_ms=150.0)
    tuning = autotune_detection(x, FS)
    assert tuning.applied
    assert tuned_count(x, tuning) == tuning.n_shots


@pytest.mark.parametrize("spacing_s,expected_rpm", [(0.100, 600), (0.200, 300)])
def test_tuner_follows_a_fast_string(spacing_s, expected_rpm):
    x = make_string(5, spacing_s=spacing_s, decay_ms=60.0)
    tuning = autotune_detection(x, FS)

    assert tuning.applied, tuning.reason
    assert tuning.n_shots == 5
    assert tuning.tightest_spacing_ms == pytest.approx(spacing_s * 1000.0, rel=0.05)
    assert tuning.implied_rate_rpm == pytest.approx(expected_rpm, rel=0.05)
    # The refractory period must be short enough not to merge them.
    assert tuning.refractory_ms < spacing_s * 1000.0
    assert tuned_count(x, tuning) == 5


def test_tuner_finds_quiet_shots_beside_loud_ones():
    """A 20 dB spread between shots must not cost the quiet ones."""
    x = make_string(4, spacing_s=0.9, amplitudes=[1.0, 0.1, 1.0, 0.1])
    tuning = autotune_detection(x, FS)

    assert tuning.applied, tuning.reason
    assert tuning.n_shots == 4
    assert tuned_count(x, tuning) == 4


def test_a_reverberant_tail_is_not_counted_as_a_string_of_shots():
    """
    The failure this guards against is the confident kind.

    A long decay with structure in it offers dozens of local maxima that are all
    well above the noise floor. A threshold deep enough to reach them finds them
    in a count that barely changes from one decibel to the next - reverberation
    is smooth - so a tuner that only asked "which count is most stable" would
    report a hundred shots in a three-second recording and be very sure of it.
    """
    rng = np.random.default_rng(3)
    x = make_string(3, spacing_s=1.0, decay_ms=250.0, tail_s=1.5)
    # A dense reverberant tail after each shot: many maxima, none impulsive.
    for k in range(3):
        i0 = int((0.5 + k) * FS) + int(0.05 * FS)
        span = int(0.7 * FS)
        decay = np.exp(-np.arange(span) / (0.15 * FS))
        x[i0:i0 + span] += 0.05 * decay * rng.normal(0.0, 1.0, span)

    tuning = autotune_detection(x, FS)
    assert tuning.applied, tuning.reason
    assert tuning.n_shots <= 6, f"chased the reverberation: {tuning.n_shots} events"
    assert tuned_count(x, tuning) == tuning.n_shots


def test_the_sweep_never_reaches_below_the_declared_floor():
    """
    40 dB below the loudest event is one hundredth of its pressure. Nothing
    that quiet is another discharge at the same station, so no chosen threshold
    may sit past it however stable the count looks down there.
    """
    x = make_string(4, spacing_s=0.7, decay_ms=300.0)
    tuning = autotune_detection(x, FS)
    assert tuning.applied
    assert tuning.threshold_relative_dB <= AUTOTUNE_MAX_RELATIVE_DB
    assert tuning.stable_to_dB <= AUTOTUNE_MAX_RELATIVE_DB


def test_a_refractory_period_that_sets_the_spacing_is_rejected():
    """
    Events exactly one refractory period apart are an artefact of the setting.

    If the closest gap sits at the minimum the setting allowed, the count
    describes the setting rather than the recording, and must not be chosen.
    """
    x = make_string(5, spacing_s=0.15, decay_ms=50.0)
    tuning = autotune_detection(x, FS)
    assert tuning.applied, tuning.reason
    if math.isfinite(tuning.tightest_spacing_ms):
        assert tuning.tightest_spacing_ms >= tuning.refractory_ms


# ---------------------------------------------------------------------------
# The stability claim itself
# ---------------------------------------------------------------------------

def test_the_stable_span_is_real():
    """
    Every threshold inside the reported span must give the reported count.

    This is the tuner's whole claim; if it does not hold, the number it prints
    is a decoration.
    """
    x = make_string(5, spacing_s=0.7)
    tuning = autotune_detection(x, FS)
    assert tuning.applied

    for rel in np.arange(tuning.stable_from_dB, tuning.stable_to_dB + 0.001, 1.0):
        shots = detect_shots(
            x, FS, threshold_relative_dB=float(rel),
            refractory_ms=tuning.refractory_ms,
            pre_ms=tuning.pre_ms, post_ms=tuning.post_ms,
        )
        assert len(shots) == tuning.n_shots, f"count changed at {rel:.0f} dB below peak"


def test_chosen_threshold_sits_inside_the_stable_span():
    x = make_string(4, spacing_s=0.8)
    tuning = autotune_detection(x, FS)
    assert tuning.applied
    assert tuning.stable_from_dB <= tuning.threshold_relative_dB <= tuning.stable_to_dB


# ---------------------------------------------------------------------------
# The decay it measures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decay_ms", [120.0, 400.0])
def test_post_window_follows_the_measured_decay(decay_ms):
    """
    The post-trigger window has to outlast a 20 dB fall, since that is what
    B-duration is measured over. It is derived from a 40 dB fall, so it must
    land above the 20 dB point and below the full 60 dB decay.
    """
    x = make_string(3, spacing_s=2.0, decay_ms=decay_ms, tail_s=2.0)
    tuning = autotune_detection(x, FS)
    assert tuning.applied, tuning.reason

    twenty_dB_point = decay_ms / 3.0
    assert tuning.post_ms > twenty_dB_point
    assert tuning.post_ms <= decay_ms + 10.0


def test_a_short_decay_does_not_produce_an_unusably_short_window():
    x = make_string(3, spacing_s=1.0, decay_ms=20.0)
    tuning = autotune_detection(x, FS)
    assert tuning.applied
    assert tuning.post_ms >= AUTOTUNE_POST_MIN_MS


# ---------------------------------------------------------------------------
# What it refuses to do
# ---------------------------------------------------------------------------

def test_pure_noise_is_refused_rather_than_tuned():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.01, FS * 3)
    tuning = autotune_detection(x, FS)

    assert tuning.applied is False
    assert tuning.reason
    # And the defaults it carries are the module's, untouched.
    assert tuning.refractory_ms == DEFAULT_REFRACTORY_MS


def test_an_empty_recording_is_refused():
    tuning = autotune_detection(np.array([]), FS)
    assert tuning.applied is False
    assert "empty" in tuning.reason


def test_a_recording_shorter_than_an_envelope_is_refused():
    tuning = autotune_detection(np.zeros(3), FS)
    assert tuning.applied is False
    assert tuning.reason


def test_a_negative_sample_rate_raises():
    with pytest.raises(ValueError):
        autotune_detection(np.zeros(1000), 0)


def test_every_choice_stays_inside_its_declared_bounds():
    for n, spacing, decay in ((1, 1.0, 300.0), (4, 0.1, 40.0), (8, 0.6, 800.0)):
        x = make_string(n, spacing_s=spacing, decay_ms=decay, tail_s=2.0)
        tuning = autotune_detection(x, FS)
        if not tuning.applied:
            continue
        assert AUTOTUNE_MIN_RELATIVE_DB <= tuning.threshold_relative_dB <= AUTOTUNE_MAX_RELATIVE_DB
        assert AUTOTUNE_REFRACTORY_MIN_MS <= tuning.refractory_ms <= AUTOTUNE_REFRACTORY_MAX_MS
        assert AUTOTUNE_POST_MIN_MS <= tuning.post_ms <= AUTOTUNE_POST_MAX_MS
        assert tuning.pre_ms == AUTOTUNE_PRE_MS


# ---------------------------------------------------------------------------
# The expected count
# ---------------------------------------------------------------------------

def test_an_expectation_that_the_recording_supports_is_met():
    x = make_string(5, spacing_s=0.8)
    tuning = autotune_detection(x, FS, expected_shots=5)
    assert tuning.applied
    assert tuning.expectation_met is True
    assert tuning.n_shots == 5


def test_an_expectation_the_recording_does_not_support_is_reported_not_forced():
    """
    A count the recording cannot produce must not bend the settings.

    Widening the threshold until the expected number appears is choosing the
    answer before measuring it, which is the failure this whole codebase exists
    to avoid.
    """
    x = make_string(3, spacing_s=1.0)
    tuning = autotune_detection(x, FS, expected_shots=9)

    assert tuning.applied
    assert tuning.expectation_met is False
    assert tuning.n_shots == 3
    assert any("expected" in note for note in tuning.notes)
    assert any("before measuring" in note for note in tuning.notes)


# ---------------------------------------------------------------------------
# Repeated delays
# ---------------------------------------------------------------------------

def test_an_echo_at_a_constant_delay_is_reported():
    x = make_string(3, spacing_s=1.2, decay_ms=120.0, echo_delay_s=0.33, echo_drop_dB=12.0)
    tuning = autotune_detection(x, FS)

    assert tuning.applied, tuning.reason
    assert tuning.reflection is not None
    assert tuning.reflection.detected
    assert tuning.reflection.delay_ms == pytest.approx(330.0, abs=20.0)
    assert tuning.reflection.n_pairs == 3
    assert tuning.reflection.drop_dB == pytest.approx(12.0, abs=4.0)
    # It flags; it does not decide.
    assert tuning.n_shots == 6
    assert any("reject them" in note for note in tuning.notes)


def test_a_plain_string_has_no_repeated_delay():
    x = make_string(5, spacing_s=0.83)
    tuning = autotune_detection(x, FS)
    assert tuning.applied
    assert tuning.reflection is not None
    assert not tuning.reflection.detected


def test_evenly_spaced_shots_of_equal_level_are_not_called_reflections():
    """
    A metronomic string has a constant delay too. What it does not have is a
    quieter follower, which is the whole discriminant.
    """
    events = np.array([0, 48000, 96000, 144000], dtype=np.int64)
    levels = np.array([1.0, 1.0, 1.0, 1.0])
    assert not find_reflection_pattern(events, levels, FS).detected


def test_a_single_quieter_follower_is_not_a_pattern():
    events = np.array([0, 20000, 96000, 200000], dtype=np.int64)
    levels = np.array([1.0, 0.1, 1.0, 1.0])
    assert not find_reflection_pattern(events, levels, FS).detected


def test_a_follower_too_close_to_be_a_return_is_ignored():
    """Under 20 ms the second arrival is part of the same blast."""
    delay = int(0.005 * FS)
    events = np.array([0, delay, 96000, 96000 + delay], dtype=np.int64)
    levels = np.array([1.0, 0.2, 1.0, 0.2])
    assert not find_reflection_pattern(events, levels, FS).detected


def test_each_event_is_counted_once():
    delay = int(0.30 * FS)
    events = np.array([0, delay, 2 * delay, 96000, 96000 + delay], dtype=np.int64)
    levels = np.array([1.0, 0.2, 0.05, 1.0, 0.2])
    found = find_reflection_pattern(events, levels, FS)
    assert found.detected
    assert len(found.followers) == len(set(found.followers))
    assert not (set(found.followers) & set(found.sources))


def test_reflection_pattern_reports_the_geometry_it_implies():
    delay = int(0.20 * FS)
    events = np.array([0, delay, 96000, 96000 + delay], dtype=np.int64)
    levels = np.array([1.0, 0.25, 1.0, 0.25])
    found = find_reflection_pattern(events, levels, FS)

    assert found.detected
    assert found.path_difference_m == pytest.approx(0.200 * 343.0, rel=0.1)
    assert "reflection" in found.describe()


def test_reflection_scan_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        find_reflection_pattern(np.array([0, 1, 2, 3]), np.array([1.0, 1.0]), FS)


def test_an_unscanned_pattern_is_distinguishable_from_an_empty_one():
    assert ReflectionPattern().detected is False
    assert autotune_detection(np.array([]), FS).reflection is None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_every_number_carries_its_basis():
    x = make_string(4, spacing_s=0.9)
    tuning = autotune_detection(x, FS)
    assert tuning.applied

    for key in ("threshold", "refractory", "post", "pre"):
        assert key in tuning.basis
        assert tuning.basis[key].endswith(".")


def test_the_dictionary_is_serialisable_and_complete():
    import json

    x = make_string(3, spacing_s=1.0)
    payload = autotune_detection(x, FS, expected_shots=3).to_dict()
    json.dumps(payload)          # must not raise

    for key in ("applied", "threshold_relative_dB", "refractory_ms", "pre_ms",
                "post_ms", "n_shots", "stable_from_dB", "stable_to_dB",
                "reflection", "basis", "notes"):
        assert key in payload


def test_a_refused_tuning_serialises_without_nan():
    import json

    payload = autotune_detection(np.array([]), FS).to_dict()
    text = json.dumps(payload)
    assert "NaN" not in text and "Infinity" not in text


def test_summary_says_what_was_chosen_and_over_what_span():
    x = make_string(5, spacing_s=0.8)
    tuning = autotune_detection(x, FS)
    line = tuning.summary()
    assert "below peak" in line
    assert "refractory" in line
    assert f"{tuning.n_shots} shots" in line


def test_summary_of_a_refusal_gives_the_reason():
    line = autotune_detection(np.array([]), FS).summary()
    assert "not tuned" in line
    assert "empty" in line


def test_a_narrow_stable_span_is_called_out():
    """
    A span barely wide enough to qualify is weak evidence and must say so,
    rather than presenting the same confident sentence as a 20 dB span.
    """
    x = make_string(3, spacing_s=1.0, echo_delay_s=0.4, echo_drop_dB=4.0)
    tuning = autotune_detection(x, FS)
    if tuning.applied and tuning.stable_width_dB < 6.0:
        assert any("sensitive to the threshold" in note for note in tuning.notes)


def test_implied_rate_is_nan_for_a_single_shot():
    x = make_string(1)
    tuning = autotune_detection(x, FS)
    assert tuning.applied
    assert math.isnan(tuning.implied_rate_rpm)
    assert math.isnan(tuning.tightest_spacing_ms)
    assert "one event" in tuning.basis["refractory"]


def test_a_single_shot_keeps_the_refractory_its_count_was_found_at():
    """
    Substituting a default here would print a pair of numbers that never ran
    together: the count comes from one refractory period, the report from
    another. The reported period must be a ladder value, and running with it
    must reproduce the count.
    """
    x = make_string(1)
    tuning = autotune_detection(x, FS)
    assert tuning.refractory_ms in AUTOTUNE_REFRACTORY_LADDER
    assert tuned_count(x, tuning) == tuning.n_shots == 1
