"""
test_shot_detect.py - shot detection, windowing and arrival separation.

The oracles here are the injected shot times and counts: the generator in
conftest.py places events at exact sample offsets, so a detector can be held to
the number of events and their timing rather than to its own past behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration import amplitude_to_dB_SPL
from shot_detect import (
    DEFAULT_REFRACTORY_MS,
    DetectionReport,
    detect_shots,
    detect_shots_adaptive,
    classify_arrivals,
    find_arrivals,
    find_shot_string_span,
    refine_peak_location,
    summarize_shots,
)
from tests.conftest import make_decaying_sinusoid, make_shot_train


FS = 96000


# ---------------------------------------------------------------------------
# Regression: uncalibrated recordings must still be analysable
# ---------------------------------------------------------------------------

def test_uncalibrated_recording_detects_all_shots():
    """
    CRITICAL REGRESSION.

    With Pa_per_FS = 1.0 the samples are full-scale units, so the loudest value
    the recording can even represent is

        20*log10(1.0 / 20e-6) = 93.98 dB re 20 uPa.

    The shipped default detection threshold was an ABSOLUTE 120 dB SPL, which is
    26 dB above that ceiling, so an uncalibrated analysis found zero shots and
    reported no reason. Detection must default to a RELATIVE threshold.
    """
    times = [0.30, 0.60, 0.90, 1.20, 1.50]
    x = make_shot_train(FS, times, duration=2.0, amplitude=0.95, noise_rms=0.001)

    assert amplitude_to_dB_SPL(np.abs(x).max()) < 94.0, "fixture should be at full scale"

    shots = detect_shots(x, FS)
    assert len(shots) == len(times)


def test_absolute_threshold_above_full_scale_warns_and_still_detects():
    """
    An absolute threshold that the recording cannot reach can never trigger. The
    detector must say so and fall back, not silently return nothing.
    """
    times = [0.30, 0.60, 0.90]
    x = make_shot_train(FS, times, duration=1.5, amplitude=0.95, noise_rms=0.001)

    report: list[DetectionReport] = []
    shots = detect_shots(
        x, FS,
        threshold_dB=120.0,
        full_scale_dB=float(amplitude_to_dB_SPL(1.0)),
        report=report,
    )

    assert len(shots) == len(times)
    assert report and any("can never trigger" in w for w in report[0].warnings)
    assert report[0].threshold_mode == "relative"


# ---------------------------------------------------------------------------
# Regression: the refractory period must not silently discard shots
# ---------------------------------------------------------------------------

def test_rapid_fire_string_fully_detected_at_default_refractory():
    """
    Five rounds 120 ms apart is roughly 500 rpm - an ordinary semi-automatic
    string, and the normal case for this product. The old 200 ms default
    refractory silently reported three of them.
    """
    times = [0.50, 0.62, 0.74, 0.86, 0.98]
    x = make_shot_train(FS, times, duration=2.0, amplitude=0.9, noise_rms=0.001)

    assert DEFAULT_REFRACTORY_MS < 120.0, "default must admit ordinary semi-auto fire"

    shots = detect_shots(x, FS)
    assert len(shots) == len(times)


def test_oversized_refractory_reports_its_suppressions():
    """
    A refractory longer than the shot spacing legitimately merges events, but it
    must ACCOUNT for what it removed rather than dropping it without trace.
    """
    times = [0.50, 0.62, 0.74, 0.86, 0.98]
    x = make_shot_train(FS, times, duration=2.0, amplitude=0.9, noise_rms=0.001)

    report: list[DetectionReport] = []
    shots = detect_shots(x, FS, refractory_ms=200.0, report=report)

    assert len(shots) < len(times)
    assert report[0].n_suppressed_by_refractory > 0
    assert report[0].n_candidates >= len(times)
    assert any("refractory" in w for w in report[0].warnings)


# ---------------------------------------------------------------------------
# Timing accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", [48000, 96000, 192000])
def test_detected_times_match_injected_times(fs):
    """Detected peaks must land within 2 ms of where the events were placed."""
    times = [0.25, 0.55, 0.85]
    x = make_shot_train(fs, times, duration=1.2, amplitude=0.9, noise_rms=0.001)

    shots = detect_shots(x, fs)
    assert len(shots) == len(times)

    for shot, expected in zip(shots, times):
        assert abs(shot.time_s - expected) < 2e-3, (
            f"shot {shot.shot_number} at {shot.time_s:.4f}s, expected {expected:.4f}s"
        )


def test_shot_numbers_are_sequential_and_time_ordered():
    times = [0.20, 0.50, 0.80, 1.10]
    x = make_shot_train(FS, times, duration=1.5, amplitude=0.9, noise_rms=0.001)

    shots = detect_shots(x, FS)
    assert [s.shot_number for s in shots] == list(range(1, len(shots) + 1))
    assert all(b.time_s > a.time_s for a, b in zip(shots, shots[1:]))


# ---------------------------------------------------------------------------
# Peak refinement must be defined in time, not in samples
# ---------------------------------------------------------------------------

def test_peak_refinement_spans_the_same_time_at_any_sample_rate():
    """
    The old implementation searched a hard-coded +/-500 SAMPLES, which is 10.4 ms
    at 48 kHz but only 2.6 ms at 192 kHz - the same recording analysed at two
    rates refined over different physical windows.

    Here a decoy larger peak is placed 3 ms after the nominal position. A
    time-based search of +/-5 ms must find it at BOTH sample rates.
    """
    found = {}
    for fs in (48000, 192000):
        n = int(fs * 0.2)
        x = np.zeros(n)
        nominal = int(fs * 0.10)
        x[nominal] = 0.5
        x[nominal + int(fs * 0.003)] = 1.0          # decoy, 3 ms later

        refined = refine_peak_location(x, nominal, fs, search_ms=5.0)
        found[fs] = (refined - nominal) / fs

    assert found[48000] == pytest.approx(0.003, abs=1e-4)
    assert found[192000] == pytest.approx(0.003, abs=1e-4)
    assert found[48000] == pytest.approx(found[192000], abs=1e-4)


# ---------------------------------------------------------------------------
# Arrival separation: crack vs blast
# ---------------------------------------------------------------------------

def test_single_blast_reports_exactly_one_arrival():
    """
    REGRESSION: arrivals were found by peak-picking the raw waveform, so a
    900 Hz ringdown - which stays within 20 dB of its peak for about eight
    cycles, each a local maximum - was reported as eight separate arrivals.
    Arrivals must be located on the envelope.
    """
    window = make_decaying_sinusoid(150.0, 0.004, 900.0, FS, int(FS * 0.3))
    arrivals = find_arrivals(window, FS)
    assert len(arrivals) == 1


def test_supersonic_round_separates_crack_from_blast():
    """
    A supersonic round produces a ballistic crack and a muzzle blast. A
    suppressor acts only on the blast, so both must be surfaced.
    """
    n = int(FS * 0.4)
    x = np.zeros(n)
    i0 = int(FS * 0.05)

    crack = make_decaying_sinusoid(120.0, 0.0004, 3000.0, FS, int(FS * 0.02))
    blast = make_decaying_sinusoid(150.0, 0.004, 900.0, FS, int(FS * 0.3))
    x[i0:i0 + len(crack)] += crack
    i1 = i0 + int(FS * 0.003)
    x[i1:i1 + len(blast)] += blast

    arrivals = find_arrivals(x, FS)
    assert len(arrivals) == 2

    separation = arrivals[1].offset_s - arrivals[0].offset_s
    assert separation == pytest.approx(0.003, abs=6e-4)

    assert arrivals[0].label == "crack"
    assert arrivals[1].label == "blast"
    assert arrivals[1].peak_Pa > arrivals[0].peak_Pa


# ---------------------------------------------------------------------------
# Window handling
# ---------------------------------------------------------------------------

def test_window_truncated_at_the_file_boundary_is_flagged():
    """
    A shot too close to the start of the recording cannot carry its full
    pre-trigger context. Energy metrics under-report, so it must be flagged
    rather than silently shortened.
    """
    n = int(FS * 0.6)
    x = np.zeros(n)
    blast = make_decaying_sinusoid(0.9, 0.004, 900.0, FS, int(FS * 0.25))
    x[:len(blast)] += blast                      # begins at sample 0

    shots = detect_shots(x, FS, pre_ms=50.0, post_ms=200.0)
    assert len(shots) >= 1
    assert shots[0].truncated is True
    assert shots[0].window_start == 0


def test_clipping_is_flagged_per_shot_when_full_scale_samples_supplied():
    times = [0.30, 0.70]
    x = make_shot_train(FS, times, duration=1.2, amplitude=0.9, noise_rms=0.001)

    clipped = np.clip(x * 4.0, -1.0, 1.0)        # drives the first shot into the rail
    shots = detect_shots(clipped, FS, samples_FS=clipped)

    assert len(shots) >= 1
    assert any(s.clipped for s in shots)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_refractory_raises(bad):
    x = make_shot_train(FS, [0.3], duration=0.6, amplitude=0.9)
    with pytest.raises(ValueError):
        detect_shots(x, FS, refractory_ms=bad)


@pytest.mark.parametrize("bad", [-5.0, float("nan"), float("inf")])
def test_invalid_pre_ms_raises(bad):
    x = make_shot_train(FS, [0.3], duration=0.6, amplitude=0.9)
    with pytest.raises(ValueError):
        detect_shots(x, FS, pre_ms=bad)


def test_zero_sample_rate_raises():
    with pytest.raises(ValueError):
        detect_shots(np.zeros(100), 0)


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError):
        detect_shots(np.zeros((10, 2)), FS)


def test_empty_signal_returns_no_shots():
    report: list[DetectionReport] = []
    assert detect_shots(np.array([]), FS, report=report) == []
    assert report and report[0].n_detected == 0


# ---------------------------------------------------------------------------
# Adaptive detection
# ---------------------------------------------------------------------------

def test_adaptive_rejects_threshold_kwargs_it_owns():
    """
    detect_shots_adaptive sets the threshold itself; forwarding one through
    **kwargs would collide and raise an opaque TypeError from detect_shots.
    It must reject the argument with a message naming the alternative.
    """
    x = make_shot_train(FS, [0.3, 0.6], duration=1.0, amplitude=0.9)

    with pytest.raises(TypeError, match="start_relative_dB"):
        detect_shots_adaptive(x, FS, threshold_dB=100.0)

    with pytest.raises(TypeError, match="start_relative_dB"):
        detect_shots_adaptive(x, FS, threshold_relative_dB=20.0)


def test_adaptive_reaches_the_requested_count():
    times = [0.30, 0.55, 0.80, 1.05, 1.30]
    x = make_shot_train(
        FS, times, duration=1.8, amplitude=0.9,
        amplitude_jitter=0.5, noise_rms=0.002,
    )
    shots = detect_shots_adaptive(x, FS, target_count=len(times))
    assert len(shots) >= len(times)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def test_summary_reports_interval_and_cyclic_rate():
    """Five rounds 120 ms apart is exactly 60000/120 = 500 rpm."""
    times = [0.50, 0.62, 0.74, 0.86, 0.98]
    x = make_shot_train(FS, times, duration=2.0, amplitude=0.9, noise_rms=0.001)

    shots = detect_shots(x, FS)
    summary = summarize_shots(shots, FS)

    assert summary["count"] == len(times)
    assert summary["mean_interval_ms"] == pytest.approx(120.0, abs=2.0)
    assert summary["cyclic_rate_rpm"] == pytest.approx(500.0, rel=0.03)


def test_noise_only_recording_detects_nothing():
    """
    CRITICAL REGRESSION.

    The default threshold is RELATIVE to the recording's own peak, so on a file
    containing no gunfire it simply selects the loudest thing present. The peak of
    n Gaussian samples sits about sqrt(2*ln(n)) sigma above the RMS - roughly
    4.8 sigma, or 13 dB, for a second at 96 kHz. With the old 6 dB SNR gate that
    passed, and a recording of pure noise produced one confident "shot" with a full
    metrics record and measurement_valid: true.

    Real muzzle blast clears its noise floor by 30-60 dB, so nothing legitimate is
    lost by requiring substantially more than a noise peak can supply.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-5, int(FS * 1.0))

    report: list[DetectionReport] = []
    shots = detect_shots(noise, FS, report=report)

    assert shots == [], f"noise produced {len(shots)} spurious shot(s)"
    assert report[0].n_detected == 0

    # Two paths reach the same correct outcome: the SNR floor can lift the threshold
    # above anything the noise reaches, or a candidate can clear the threshold and
    # then be rejected for insufficient impulsiveness. Either way the report must
    # SAY why nothing was found rather than presenting an empty result bare.
    explanation = " ".join(report[0].warnings)
    assert (
        "No events found above the detection threshold" in explanation
        or "no gunfire" in explanation
    ), f"absence was not explained: {report[0].warnings}"


def test_genuine_shots_clear_the_snr_gate_by_a_wide_margin():
    """The gate must not cost real detections; blast SNR should be far above it."""
    times = [0.30, 0.60, 0.90, 1.20, 1.50]
    x = make_shot_train(FS, times, duration=2.0, amplitude=0.3, noise_rms=1e-4)

    shots = detect_shots(x, FS)
    assert len(shots) == len(times)
    assert min(s.snr_dB for s in shots) > 30.0


def test_summary_of_no_shots_is_empty_not_zero():
    """
    An empty result must not be reported as a measurement of zero - a 0.0 dB
    level is a fabricated number, not an absence.
    """
    summary = summarize_shots([], FS)
    assert summary["count"] == 0
    assert summary["peak_dB_max"] is None
    assert summary["mean_interval_ms"] is None


# ---------------------------------------------------------------------------
# Detection report
# ---------------------------------------------------------------------------

def test_report_is_populated_and_serialisable():
    times = [0.30, 0.70]
    x = make_shot_train(FS, times, duration=1.2, amplitude=0.9, noise_rms=0.001)

    report: list[DetectionReport] = []
    detect_shots(x, FS, report=report)

    assert len(report) == 1
    data = report[0].to_dict()
    for key in ("n_detected", "n_candidates", "n_suppressed_by_refractory",
                "threshold_dB", "threshold_mode", "peak_level_dB",
                "noise_floor_dB", "warnings"):
        assert key in data
    assert data["n_detected"] == len(times)
    assert data["peak_level_dB"] > data["noise_floor_dB"]


def test_shot_event_serialises_for_the_output_record():
    x = make_shot_train(FS, [0.3], duration=0.8, amplitude=0.9, noise_rms=0.001)
    shot = detect_shots(x, FS)[0]

    data = shot.to_dict()
    for key in ("shot_number", "time_s", "peak_Pa", "peak_dB", "window_start",
                "window_end", "truncated", "clipped", "arrivals"):
        assert key in data
    assert data["window_end"] > data["window_start"]


# ---------------------------------------------------------------------------
# Auto-trim to the shot string
# ---------------------------------------------------------------------------

def test_auto_trim_spans_the_string_plus_its_margin_exactly():
    """
    The span runs from the first shot's window start to the last shot's window
    end, plus the margin, clamped to the recording. That is exact arithmetic on
    the detected windows, so it is checked exactly.
    """
    x = make_shot_train(FS, [4.0, 4.5, 5.0], duration=12.0, amplitude=0.9,
                        noise_rms=0.001)
    shots = detect_shots(x, FS)
    span = find_shot_string_span(shots, x.size, FS, margin_s=1.0)

    first = min(s.window_start for s in shots)
    last = max(s.window_end for s in shots)
    assert span.applied
    assert span.start == max(0, first - FS)
    assert span.end == min(x.size, last + FS)


def test_auto_trim_never_cuts_into_a_shot_window():
    x = make_shot_train(FS, [3.0, 3.6, 4.2], duration=10.0, amplitude=0.9,
                        noise_rms=0.001)
    shots = detect_shots(x, FS)
    span = find_shot_string_span(shots, x.size, FS, margin_s=0.0)
    for shot in shots:
        assert span.start <= shot.window_start
        assert span.end >= shot.window_end


def test_auto_trim_removes_the_silence_it_claims_to():
    x = make_shot_train(FS, [8.0, 8.4], duration=20.0, amplitude=0.9, noise_rms=0.001)
    shots = detect_shots(x, FS)
    span = find_shot_string_span(shots, x.size, FS, margin_s=0.5)

    trimmed = span.apply(x)
    assert trimmed.size == span.end - span.start
    assert span.removed_s == pytest.approx((x.size - trimmed.size) / FS, abs=1e-9)
    assert trimmed.size < x.size


def test_auto_trim_keeps_every_shot_detectable_after_trimming():
    """The point of the trim is that it changes what is analysed, not the result."""
    x = make_shot_train(FS, [6.0, 6.5, 7.0, 7.5], duration=18.0, amplitude=0.9,
                        noise_rms=0.001)
    shots = detect_shots(x, FS)
    span = find_shot_string_span(shots, x.size, FS)
    assert len(detect_shots(span.apply(x), FS)) == len(shots)


def test_auto_trim_declines_when_no_shots_were_detected():
    span = find_shot_string_span([], 96000, FS)
    assert not span.applied
    assert "no shots were detected" in span.reason
    assert span.apply(np.zeros(10)).size == 10


def test_auto_trim_declines_when_the_string_already_fills_the_recording():
    x = make_shot_train(FS, [0.2, 0.6], duration=1.0, amplitude=0.9, noise_rms=0.001)
    shots = detect_shots(x, FS)
    span = find_shot_string_span(shots, x.size, FS, margin_s=5.0)
    assert not span.applied
    assert "nothing to remove" in span.reason


def test_auto_trim_serialises_for_the_output_record():
    x = make_shot_train(FS, [5.0], duration=15.0, amplitude=0.9, noise_rms=0.001)
    shots = detect_shots(x, FS)
    data = find_shot_string_span(shots, x.size, FS).to_dict()
    for key in ("applied", "start_sample", "end_sample", "start_s", "end_s",
                "duration_s", "removed_s", "n_shots", "reason"):
        assert key in data


def test_auto_trim_rejects_an_impossible_sample_rate():
    with pytest.raises(ValueError):
        find_shot_string_span([], 100, 0)


# ---------------------------------------------------------------------------
# Arrival classification: the basis must match the evidence
# ---------------------------------------------------------------------------

def _two_arrivals(separation_s, fs=FS):
    """A short high-frequency burst followed by a longer low-frequency one."""
    n = int(0.05 * fs)
    x = np.zeros(n)
    crack = make_decaying_sinusoid(0.35, 0.00012, 9000.0, fs, int(0.004 * fs))
    blast = make_decaying_sinusoid(0.95, 0.004, 700.0, fs, int(0.03 * fs))
    i0 = int(0.005 * fs)
    i1 = i0 + int(round(separation_s * fs))
    x[i0:i0 + crack.size] += crack
    x[i1:i1 + blast.size] += blast
    return x


def test_geometry_identifies_the_crack_outright():
    """
    When the observed separation matches the Mach-cone delay computed from the
    recorded geometry, the crack is identified rather than guessed, and the
    basis says so.
    """
    separation = 0.0020
    arrivals = find_arrivals(_two_arrivals(separation), FS)
    assert len(arrivals) == 2

    observed = arrivals[1].offset_s - arrivals[0].offset_s
    classify_arrivals(arrivals, expected_delay_s=observed)
    assert [a.label for a in arrivals] == ["crack", "blast"]
    assert all(a.basis == "geometric" for a in arrivals)


def test_a_separation_that_contradicts_the_geometry_is_not_labelled_geometrically():
    """
    A reflection does not arrive at the Mach-cone delay. When the geometry says
    the crack should lead by 2 ms and the second arrival is 15 ms later, the
    geometric identification must not fire.
    """
    arrivals = find_arrivals(_two_arrivals(0.0020), FS)
    classify_arrivals(arrivals, expected_delay_s=0.015)
    assert all(a.basis != "geometric" for a in arrivals)


def test_a_subsonic_round_has_no_crack_to_find():
    arrivals = find_arrivals(_two_arrivals(0.0020), FS)
    classify_arrivals(arrivals, projectile_supersonic=False)
    labels = [a.label for a in arrivals]
    assert "crack" not in labels
    assert "blast" in labels
    blast = next(a for a in arrivals if a.label == "blast")
    assert blast.basis == "subsonic"


def test_a_single_arrival_is_not_claimed_to_be_the_muzzle_blast():
    """
    One arrival cannot be shown to be blast rather than crack, and a suppressor
    credited against the crack has been credited with nothing.
    """
    window = make_decaying_sinusoid(0.9, 0.004, 900.0, FS, int(0.05 * FS))
    arrivals = find_arrivals(window, FS)
    assert len(arrivals) == 1
    assert arrivals[0].label == "unclassified"
    assert not arrivals[0].classified
    assert "single arrival" in arrivals[0].ambiguity


def test_spectral_character_separates_an_n_wave_from_a_blast():
    """
    The crack is both higher in centre frequency and faster to decay. Both
    indicators must agree before the label is applied on character alone.
    """
    arrivals = find_arrivals(_two_arrivals(0.0025), FS)
    assert len(arrivals) == 2
    assert arrivals[0].centroid_Hz > arrivals[1].centroid_Hz
    assert arrivals[0].decay_ms < arrivals[1].decay_ms
    assert [a.label for a in arrivals] == ["crack", "blast"]
    assert arrivals[0].basis == "spectral"


def test_an_ordering_only_label_admits_it_is_ordering_only():
    """
    Two arrivals of the same character cannot be told apart on character, so the
    label falls back to ordering and must carry that admission.
    """
    n = int(0.05 * FS)
    x = np.zeros(n)
    quiet = make_decaying_sinusoid(0.4, 0.004, 800.0, FS, int(0.03 * FS))
    loud = make_decaying_sinusoid(0.95, 0.004, 800.0, FS, int(0.03 * FS))
    x[int(0.005 * FS):int(0.005 * FS) + quiet.size] += quiet
    x[int(0.012 * FS):int(0.012 * FS) + loud.size] += loud

    arrivals = find_arrivals(x, FS)
    if len(arrivals) >= 2:
        crack = next((a for a in arrivals if a.label == "crack"), None)
        assert crack is not None
        assert crack.basis == "ordering"
        assert "order alone" in crack.ambiguity


def test_arrival_serialises_its_basis_and_character():
    arrivals = find_arrivals(_two_arrivals(0.0025), FS)
    data = arrivals[0].to_dict()
    for key in ("offset_s", "peak_Pa", "peak_dB", "label", "basis", "ambiguity",
                "centroid_Hz", "decay_ms"):
        assert key in data


def test_classifying_an_empty_arrival_list_is_safe():
    assert classify_arrivals([]) == []


# ---------------------------------------------------------------------------
# Overlapping shot windows
#
# The warning used to be raised from the SETTINGS (post_ms > refractory_ms),
# which is a statement about what could happen rather than what did. With the
# shipped defaults that made it fire on every run, including runs whose shots
# were a second apart.
# ---------------------------------------------------------------------------

def _detect_with_report(x, fs, **kwargs):
    report = []
    shots = detect_shots(x, fs, report=report, **kwargs)
    return shots, report[0]


def test_no_overlap_warning_when_the_windows_do_not_touch():
    fs = 48000
    # 700 ms apart, 200 ms of post-trigger window: nothing overlaps.
    x = make_shot_train(fs, [0.4, 1.1, 1.8], duration=2.6, amplitude=0.5, noise_rms=1e-4)
    shots, report = _detect_with_report(
        x, fs, threshold_relative_dB=25.0, pre_ms=20.0, post_ms=200.0, refractory_ms=50.0,
    )
    assert len(shots) == 3
    assert not any("overlap" in w for w in report.warnings), report.warnings


def test_no_overlap_warning_from_the_settings_alone():
    """post_ms above refractory_ms is not, by itself, an overlap."""
    fs = 48000
    x = make_shot_train(fs, [0.4, 1.4, 2.4], duration=3.4, amplitude=0.5, noise_rms=1e-4)
    shots, report = _detect_with_report(
        # 500 ms window, 200 ms refractory: the shipped defaults. The shots are
        # a second apart, so no window reaches the next one.
        x, fs, threshold_relative_dB=25.0, pre_ms=20.0, post_ms=500.0, refractory_ms=200.0,
    )
    assert len(shots) == 3
    assert not any("overlap" in w for w in report.warnings), report.warnings


def test_overlap_warning_when_the_windows_really_do_overlap():
    fs = 48000
    # 150 ms apart with a 500 ms window: every pair overlaps.
    x = make_shot_train(fs, [0.4, 0.55, 0.70], duration=1.6, amplitude=0.5, noise_rms=1e-4)
    shots, report = _detect_with_report(
        x, fs, threshold_relative_dB=25.0, pre_ms=20.0, post_ms=500.0, refractory_ms=50.0,
    )
    assert len(shots) == 3
    overlap = [w for w in report.warnings if "overlap" in w]
    assert len(overlap) == 1, report.warnings
    # It states how many pairs and how close the closest two are.
    assert "2 pairs" in overlap[0]
    assert "150 ms apart" in overlap[0]
