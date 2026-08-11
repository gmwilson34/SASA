"""
test_ahaah.py - pins ahaah.py to the ARL release, and refuses to let it lie.

The tests fall into three groups.

  1. Things the ARL release states or that are exactly derivable from it: the band
     geometry, the max-not-sum reduction, N = 500/ARU, the CTS relation, the .AHA
     reader. These must pass, always. They are cheap and they are the parts of the
     answer that are actually solid.

  2. Behavioural guarantees that exist to stop a wrong number reaching a customer:
     the model must REFUSE on uncalibrated, undersampled, clipped and sub-130 dB
     input; a louder impulse must never yield a lower index; the warned case must
     come out below the unwarned case; the integration must RAISE rather than
     return a large finite number when it diverges; and, above all,
     `compute_ahaah` must emit no number at all while VALIDATION_STATUS is
     "not_validated" (test_compute_ahaah_emits_no_number_at_all).

     Note that every test needing a number goes through the `_research` helper.
     That is deliberate. `compute_ahaah` gives none, so a test cannot accidentally
     assert something about an "ARU" that this module does not produce.

  3. THE VALIDATION TEST (test_VALIDATION_against_ARL_160F_reference). It runs the
     one reference case in the release and compares against 391.0 ARU warned /
     2237 ARU unwarned and the 23-band table. It is written to REPORT the achieved
     agreement in its failure message.

     IT IS EXPECTED TO FAIL, AND IT MUST NOT BE LOOSENED TO MAKE IT PASS.
     Four of the choices ahaah.py has to make are undocumented in the public
     release and each can move the answer by an order of magnitude; the module
     says so, its `status` field says "not_validated", and this test is the thing
     that keeps that honest. Loosening the tolerance would convert a known-unknown
     into a shipped wrong number, which is the specific failure mode this codebase
     was rebuilt to remove. Fix the model or get the missing ARL material; do not
     touch the numbers below.

     Beware in particular of the warned total, which comes out within 0.8 % of the
     reference. It is a coincidence -- the band curve it is the maximum of misses
     20 of 23 bands by up to 3.4x and peaks in the wrong band, and 4 of 96
     configurations of the exposed switches hit 391 just as well. T2.3 (argmax
     band) and T2.4 (per-band) exist precisely so that this one number cannot be
     mistaken for a pass.

     It is marked `@pytest.mark.validation` so a build can deselect it
     (`-m "not validation"`) while remaining honest about why.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ahaah import (
    AHAAH_WORKING_RATE_HZ,
    MAX_PLAUSIBLE_STAPES_DISPLACEMENT_UM,
    ModelDivergedError,
    NOT_AN_ARU_LABEL,
    VALIDATION_STATUS,
    ARU_LIMIT_OCCASIONAL,
    ARU_LIMIT_OCCUPATIONAL,
    CATEGORY_C_COUNT,
    COCHLEAR_GAIN_FACTOR,
    CTS_INTERCEPT,
    CTS_SLOPE,
    DC,
    DECLARED_ASSUMPTIONS,
    FO,
    MAN_COE_MD5,
    MIN_PEAK_dB,
    MIN_SAMPLE_RATE_HZ,
    SHAW_CANAL_TO_EARDRUM_dB,
    SHAW_FF_TO_CANAL_dB_AZ0,
    SHAW_FF_TO_CANAL_dB_AZ90,
    SHAW_FREQ_kHz,
    SUSCEPTIBILITY_95_GAIN,
    XBM_FROM,
    XBM_NO,
    XBM_TO,
    AhaahResult,
    accumulate_hazard,
    allowed_exposures,
    band_frequencies_Hz,
    band_positions_cm,
    compute_ahaah,
    compute_ahaah_both,
    load_aha,
    load_haz,
    run_unvalidated_model,
    run_unvalidated_model_both,
    simulate_middle_ear,
    threshold_shift_dB,
    total_aru,
)
from calibration import P_REF

DATA = Path(__file__).parent / "data" / "ahaah"
AHA_160F = DATA / "160F.AHA"
HAZ_160F = DATA / "160F.HAZ"
FFEDM90 = DATA / "FFEDM90.DAT"

# ARL's own results for 160F, from the .AHA header.
REF_ARU_WARNED = 391.0
REF_ARU_UNWARNED = 2237.0
REF_EXPOSURES_WARNED = 1.3
REF_EXPOSURES_UNWARNED = 0.2
REF_PEAK_dB = 186.936
REF_ARGMAX_BAND = 9          # 160F.HAZ maximum, 1-based
REF_RATIO = REF_ARU_UNWARNED / REF_ARU_WARNED   # 5.72


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def waveform_160F():
    return load_aha(AHA_160F)


@pytest.fixture(scope="module")
def haz_160F():
    return load_haz(HAZ_160F)


@pytest.fixture(scope="module")
def result_160F(waveform_160F):
    """
    The full model run on the reference case, both reflex conditions.

    Note the entry point: `compute_ahaah` deliberately emits no number, so the
    validation gate has to drive the research path. The numbers it returns are
    microns squared, not ARU -- that is the whole point of the gate.
    """
    return run_unvalidated_model_both(
        waveform_160F.pressure_Pa,
        waveform_160F.sample_rate,
        acknowledge_not_validated=True,
        input_location=waveform_160F.input_location,
    )


def _blast(peak_Pa: float, fs: float = 125000.0, T: float = 5e-4,
           duration: float = 0.02, pre_ms: float = 1.0) -> np.ndarray:
    """Friedlander blast preceded by silence: p(t) = P0 (1 - t/T) exp(-t/T)."""
    t = np.arange(int(duration * fs)) / fs
    p = peak_Pa * (1.0 - t / T) * np.exp(-t / T)
    return np.concatenate([np.zeros(int(pre_ms / 1000.0 * fs)), p])


def _research(pressure_Pa, fs, **kwargs):
    """
    Drive the model itself.

    Every test that needs a NUMBER has to come through here, because
    `compute_ahaah` emits none. The `acknowledge_not_validated=True` is required by
    the API and is the point: it makes each call site state, in the source, that
    what comes back is microns squared and not an AHAAH ARU.
    """
    return run_unvalidated_model(pressure_Pa, fs, acknowledge_not_validated=True, **kwargs)


# ---------------------------------------------------------------------------
# 1. Band geometry -- documented, and checkable against 160F.HAZ exactly
# ---------------------------------------------------------------------------

def test_band_positions_are_the_23_man_coe_locations():
    x = band_positions_cm()
    assert len(x) == XBM_NO == 23
    assert x[0] == pytest.approx(XBM_FROM)
    assert x[-1] == pytest.approx(XBM_TO)
    # uniform 0.125 cm spacing
    assert np.allclose(np.diff(x), 0.125)


def test_band_frequencies_follow_Fo_exp_minus_x_over_Dc():
    x = band_positions_cm()
    assert np.allclose(band_frequencies_Hz(), FO * np.exp(-x / DC), rtol=1e-12)


def test_band_frequencies_match_160F_HAZ_to_file_precision(haz_160F):
    """The oracle: all 23 Freq(kHz) values in ARL's own hazard table, to its 2 dp."""
    ref_kHz, _, ref_x = haz_160F
    assert len(ref_kHz) == 23
    model_kHz = band_frequencies_Hz() / 1000.0
    assert np.allclose(np.round(model_kHz, 2), ref_kHz, atol=1e-9), (
        f"band frequencies differ: model {np.round(model_kHz, 2)} vs file {ref_kHz}"
    )
    assert np.allclose(band_positions_cm(), ref_x, atol=1e-9)


def test_band_spacing_is_0_2254_octave_not_one_third():
    """The ARL prose says 'roughly one-third octave'. It is not. Use the formula."""
    f = band_frequencies_Hz()
    octaves = np.log2(f[:-1] / f[1:])
    assert np.allclose(octaves, 0.125 / DC / math.log2(math.e) * math.log2(math.e), atol=1e-9) \
        or np.allclose(octaves, octaves[0])
    assert octaves[0] == pytest.approx(0.2254, abs=1e-4)
    assert 1.0 / octaves[0] == pytest.approx(4.436, abs=1e-3)


# ---------------------------------------------------------------------------
# 1. Hazard reduction rules -- verified against the reference numbers
# ---------------------------------------------------------------------------

def test_total_aru_is_the_max_of_the_bands_not_the_sum(haz_160F):
    _, ref_ahu, _ = haz_160F
    assert total_aru(ref_ahu) == pytest.approx(ref_ahu.max())
    # The ARL header for this exact table reports 391.0 A.R.U. (warned).
    assert total_aru(ref_ahu) == pytest.approx(REF_ARU_WARNED, abs=0.05)
    # The sum matches nothing in the release, and is a full order of magnitude out.
    assert ref_ahu.sum() == pytest.approx(4031.24, abs=0.01)
    assert total_aru(ref_ahu) < 0.11 * ref_ahu.sum()


def test_total_aru_argmax_is_band_9(haz_160F):
    _, ref_ahu, _ = haz_160F
    assert int(np.argmax(ref_ahu)) + 1 == REF_ARGMAX_BAND


def test_allowed_exposures_is_500_over_aru():
    for aru in (1.0, 50.0, 391.0, 500.0, 2237.0):
        assert allowed_exposures(aru) == pytest.approx(ARU_LIMIT_OCCASIONAL / aru)
        assert allowed_exposures(aru, occupational=True) == pytest.approx(
            ARU_LIMIT_OCCUPATIONAL / aru
        )
    # Reproduces both header values after rounding to 1 dp.
    assert round(allowed_exposures(REF_ARU_WARNED), 1) == REF_EXPOSURES_WARNED
    assert round(allowed_exposures(REF_ARU_UNWARNED), 1) == REF_EXPOSURES_UNWARNED
    assert allowed_exposures(0.0) == math.inf


def test_threshold_shift_at_500_aru_is_25_2_dB():
    """ARL: '500 ARUs is about 25 dB of shift'. CTS = 26.6 ln(ARU) - 140.1."""
    assert threshold_shift_dB(500.0) == pytest.approx(25.2, abs=0.1)
    assert threshold_shift_dB(1000.0) == pytest.approx(
        CTS_SLOPE * math.log(1000.0) - CTS_INTERCEPT
    )
    # The relation goes negative at low ARU; that is clamped, not reported.
    assert threshold_shift_dB(1.0) == 0.0
    assert threshold_shift_dB(0.0) == 0.0


def test_threshold_shift_is_monotone_in_aru():
    values = [threshold_shift_dB(a) for a in (200, 500, 1000, 2237, 5000)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# 1. Hazard accumulation mechanics
# ---------------------------------------------------------------------------

def test_accumulate_hazard_sums_squared_upward_peaks():
    # Two clean positive excursions of 3 and 4 microns -> 9 + 16 = 25 AHU.
    d = np.array([[0.0, 1.0, 3.0, 1.0, 0.0, -5.0, 0.0, 2.0, 4.0, 1.0, 0.0]])
    assert accumulate_hazard(d, rule="excursion")[0] == pytest.approx(25.0)


def test_accumulate_hazard_ignores_downward_displacement():
    d = np.array([[0.0, -10.0, -20.0, -10.0, 0.0]])
    assert accumulate_hazard(d)[0] == 0.0


def test_excursion_rule_counts_a_rippled_excursion_once():
    """One positive excursion with a ripple: 'as each peak passes' -> one count."""
    d = np.array([[0.0, 2.0, 1.5, 3.0, 1.0, 0.0]])
    assert accumulate_hazard(d, rule="excursion")[0] == pytest.approx(9.0)
    # The local-maximum reading counts both, which is why the rule is declared.
    assert accumulate_hazard(d, rule="local_max")[0] == pytest.approx(4.0 + 9.0)


# ---------------------------------------------------------------------------
# 1. The .AHA / .HAZ readers
# ---------------------------------------------------------------------------

def test_aha_reader_parses_the_reference_header(waveform_160F):
    wf = waveform_160F
    assert len(wf.pressure_Pa) == 2048
    assert wf.sample_rate == 125000.0
    assert wf.calc_code == 1
    assert wf.input_location == "free_field_normal"
    assert wf.reference_ARU_warned == REF_ARU_WARNED
    assert wf.reference_ARU_unwarned == REF_ARU_UNWARNED
    assert wf.reference_exposures_warned == REF_EXPOSURES_WARNED
    assert wf.reference_exposures_unwarned == REF_EXPOSURES_UNWARNED
    assert wf.reference_peak_dB == pytest.approx(REF_PEAK_dB)
    assert wf.reference_Leq_dB == pytest.approx(165.848)
    assert wf.reference_LAeq_dB == pytest.approx(165.123)
    assert wf.reference_LAeq8hr_dB == pytest.approx(101.846)
    assert wf.reference_A_weighted_energy_J_m2 == pytest.approx(422.641)
    assert wf.duration_s == pytest.approx(2048 / 125000.0)


def test_aha_samples_are_pascals_and_reproduce_the_header_peak(waveform_160F):
    p = waveform_160F.pressure_Pa
    assert p.max() == pytest.approx(44340.0)
    assert p.min() == pytest.approx(-15020.0)
    assert int(np.argmax(np.abs(p))) == 628                  # 5.024 ms
    peak_dB = 20 * math.log10(np.abs(p).max() / P_REF)
    # 186.915 from the raw samples vs 186.936 in the header. The 0.021 dB is
    # unexplained (parabolic interpolation gives 187.042, also not a match) and is
    # recorded rather than papered over -- see docs/AHAAH-SPEC.md section 11.1.
    assert peak_dB == pytest.approx(186.915, abs=0.002)
    assert abs(peak_dB - REF_PEAK_dB) < 0.05


def test_haz_reader(haz_160F):
    f_kHz, ahu, x_cm = haz_160F
    assert len(f_kHz) == len(ahu) == len(x_cm) == 23
    assert ahu.max() == pytest.approx(390.99399)
    assert f_kHz[0] == pytest.approx(11.76)
    assert x_cm[-1] == pytest.approx(3.175)


def test_aha_reader_rejects_a_sample_count_that_disagrees_with_the_header(tmp_path):
    bad = tmp_path / "bad.AHA"
    bad.write_text(
        "Sampling rate\t125000\t\n"
        "Microphone relative to ear\t1\t\n"
        "Number of Samples\t5\t\n"
        "1.0\n2.0\n3.0\n"
    )
    with pytest.raises(ValueError, match="declares 5 samples"):
        load_aha(bad)


def test_aha_reader_rejects_a_file_with_no_payload(tmp_path):
    bad = tmp_path / "empty.AHA"
    bad.write_text("Sampling rate\t125000\t\n")
    with pytest.raises(ValueError, match="no numeric sample payload"):
        load_aha(bad)


# ---------------------------------------------------------------------------
# 1. Embedded coefficients
# ---------------------------------------------------------------------------

def test_aha_reader_refuses_to_guess_the_measurement_geometry(tmp_path):
    """
    The calculation code is the microphone geometry. Free field carries the head
    diffraction and ear-canal gain that the eardrum route must not have -- about
    19 dB at 2.7 kHz -- so defaulting it silently would put a geometry error
    straight into the answer with nothing on screen to show for it.
    """
    src = AHA_160F.read_text()

    unknown = src.replace("Microphone relative to ear     \t1 ",
                          "Microphone relative to ear     \t9 ")
    assert unknown != src
    bad = tmp_path / "unknown.AHA"
    bad.write_text(unknown)
    with pytest.raises(ValueError, match="calculation code 9"):
        load_aha(bad)

    absent = "\n".join(ln for ln in src.splitlines()
                       if not ln.startswith("Microphone relative"))
    missing = tmp_path / "absent.AHA"
    missing.write_text(absent)
    with pytest.raises(ValueError, match="calculation code"):
        load_aha(missing)


def test_man_coe_fixture_matches_the_md5_the_constants_were_taken_from():
    import hashlib
    digest = hashlib.md5((DATA / "man.coe").read_bytes()).hexdigest()
    assert digest == MAN_COE_MD5, (
        "the checked-in man.coe is not the file ahaah.py's constants were transcribed "
        "from; re-check every embedded value before trusting any output"
    )


def test_shaw_tables_are_all_the_same_length_as_the_frequency_axis():
    n = len(SHAW_FREQ_kHz)
    assert n == 76
    for table in (SHAW_FF_TO_CANAL_dB_AZ90, SHAW_FF_TO_CANAL_dB_AZ0,
                  SHAW_CANAL_TO_EARDRUM_dB):
        assert len(table) == n
    assert SHAW_FREQ_kHz[0] == 0.2
    assert SHAW_FREQ_kHz[-1] == 15.0
    assert list(SHAW_FREQ_kHz) == sorted(SHAW_FREQ_kHz)


def test_declared_assumptions_are_well_formed():
    keys = [a.key for a in DECLARED_ASSUMPTIONS]
    assert len(keys) == len(set(keys)), "duplicate assumption key"
    for a in DECLARED_ASSUMPTIONS:
        assert a.category in ("a", "b", "c")
        assert a.choice and a.rationale
    assert CATEGORY_C_COUNT == sum(1 for a in DECLARED_ASSUMPTIONS if a.category == "c")
    assert CATEGORY_C_COUNT > 0, (
        "if there are genuinely no inferences left, the validation gate must be "
        "reopened deliberately, not by an empty list"
    )


# ---------------------------------------------------------------------------
# 2. Refusal behaviour -- no number when the input cannot support one
# ---------------------------------------------------------------------------

def _assert_refused(result: AhaahResult, needle: str) -> None:
    assert result.valid is False
    assert result.status == "refused"
    assert math.isnan(result.total_ARU)
    assert np.all(np.isnan(result.band_ahu))
    assert math.isnan(result.allowed_exposures)
    assert math.isnan(result.threshold_shift_dB)
    joined = " ".join(result.notes).lower()
    assert "refused" in joined
    assert needle.lower() in joined, f"note did not explain {needle!r}: {result.notes}"
    assert "unavailable" in result.headline_label.lower()


def test_refuses_uncalibrated_input():
    p = _blast(50000.0)
    _assert_refused(compute_ahaah(p, 125000.0, calibrated=False), "uncalibrated")


def test_refuses_sample_rate_below_96_kHz():
    p = _blast(50000.0, fs=48000.0)
    r = compute_ahaah(p, 48000.0)
    _assert_refused(r, "sample rate")
    assert "96000" in " ".join(r.notes)
    assert MIN_SAMPLE_RATE_HZ == 96000.0


def test_accepts_96_kHz_and_records_the_resampling():
    p = _blast(50000.0, fs=96000.0)
    # The gate passes: compute_ahaah refuses for the MODEL's reason, not the input's.
    assert compute_ahaah(p, 96000.0).status == "not_validated"
    r = _research(p, 96000.0)
    assert r.resampled
    assert r.resample_ratio == "125/96"
    assert r.working_rate_Hz == pytest.approx(AHAAH_WORKING_RATE_HZ)
    assert any("resampled" in n for n in r.notes)


def test_low_rate_can_be_forced_but_stays_flagged():
    p = _blast(50000.0, fs=64000.0)
    r = _research(p, 64000.0, allow_low_rate=True)
    assert any("below the 96000 Hz minimum" in n for n in r.notes)
    # Far below half the minimum there is nothing worth computing at all.
    _assert_refused(compute_ahaah(_blast(50000.0, fs=44100.0), 44100.0,
                                  allow_low_rate=True), "sample rate")
    with pytest.raises(ValueError, match="Sample rate 44100"):
        _research(_blast(50000.0, fs=44100.0), 44100.0, allow_low_rate=True)


def test_refuses_clipped_input():
    p = _blast(50000.0)
    p = np.clip(p, -1e9, 20000.0)      # flat-topped peak
    _assert_refused(compute_ahaah(p, 125000.0), "clipped")


def test_refuses_peaks_below_130_dB():
    # 130 dB re 20 uPa = 63.2 Pa.
    quiet = _blast(40.0)                # ~126 dB
    r = compute_ahaah(quiet, 125000.0)
    _assert_refused(r, "below 130 dB")
    assert MIN_PEAK_dB == 130.0
    # And just above the threshold the LEVEL gate passes -- what stops the number
    # then is the model's own standing, not the recording.
    loud = _blast(200.0)                # ~140 dB
    assert compute_ahaah(loud, 125000.0).status == "not_validated"
    assert _research(loud, 125000.0).max_band_sum_sq_um2 > 0


def test_refuses_non_finite_and_degenerate_input():
    p = _blast(50000.0)
    p[100] = np.nan
    _assert_refused(compute_ahaah(p, 125000.0), "non-finite")
    _assert_refused(compute_ahaah(np.zeros(4096), 125000.0), "zeros")
    _assert_refused(compute_ahaah(np.zeros(4), 125000.0), "too short")


def test_rejects_an_unknown_input_location():
    with pytest.raises(ValueError, match="input_location"):
        compute_ahaah(_blast(50000.0), 125000.0, input_location="behind_the_ear")


# ---------------------------------------------------------------------------
# 2. Behavioural guarantees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("location", ["free_field_normal", "free_field_grazing",
                                      "ear_canal_entrance", "eardrum"])
def test_all_input_locations_produce_a_result(location):
    r = _research(_blast(30000.0), 125000.0, input_location=location)
    assert r.max_band_sum_sq_um2 > 0


def test_grazing_needs_the_shaw_table_and_says_so():
    _assert_refused(
        compute_ahaah(_blast(30000.0), 125000.0,
                      input_location="free_field_grazing", outer_ear="circuit"),
        "grazing",
    )


def test_a_louder_impulse_never_yields_a_lower_index():
    """
    Monotonicity, taken far enough up to meet the integrator's stability limit.

    Above roughly 200 dB the stiffening annular ligament makes the trapezoidal
    solve diverge. It used to return a large finite number (6.7e36 at 214 dB, from
    a stapes displacement of 7e14 metres) and call itself valid. Now it raises, so
    this test can assert monotonicity across the whole admissible range instead of
    stopping at 120 kPa and never seeing the runaway.
    """
    fs = 125000.0
    previous = -1.0
    diverged_at = None
    for peak in (200.0, 2000.0, 10000.0, 30000.0, 60000.0, 120000.0,
                 3.0e5, 1.0e6, 3.0e6, 1.0e7):
        try:
            r = _research(_blast(peak, fs=fs), fs)
        except ModelDivergedError:
            diverged_at = peak
            break
        value = r.max_band_sum_sq_um2
        assert math.isfinite(value), f"non-finite index at {peak:g} Pa"
        # The annular ligament is a peak limiter, so growth saturates; it must
        # never reverse. 1% of slack absorbs integrator noise, nothing more.
        assert value >= previous * 0.99, (
            f"the index fell from {previous:.4g} to {value:.4g} when the input got louder"
        )
        previous = value
    assert diverged_at is not None, (
        "the integrator no longer diverges at extreme level -- good, but then this "
        "test's upper bound should be raised rather than the guard trusted blindly"
    )
    assert diverged_at >= 1.2e5, (
        f"divergence at {diverged_at:g} Pa is inside the range of real muzzle blast; "
        "the nonlinear solve needs fixing, not just guarding"
    )


def test_a_diverging_run_raises_instead_of_returning_a_number():
    with pytest.raises(ModelDivergedError, match="diverged"):
        _research(_blast(1.0e7), 125000.0)
    assert MAX_PLAUSIBLE_STAPES_DISPLACEMENT_UM == 1.0e6


def test_warned_index_is_below_unwarned_at_a_level_that_evokes_the_reflex():
    unwarned, warned = run_unvalidated_model_both(
        _blast(40000.0), 125000.0, acknowledge_not_validated=True)
    assert warned.warned is True and unwarned.warned is False
    assert warned.max_band_sum_sq_um2 < unwarned.max_band_sum_sq_um2, (
        "a pre-established middle-ear-muscle contraction must reduce the hazard; "
        f"got warned {warned.max_band_sum_sq_um2:.4g} vs "
        f"unwarned {unwarned.max_band_sum_sq_um2:.4g}"
    )


def test_compute_ahaah_both_returns_unwarned_first():
    unwarned, warned = compute_ahaah_both(_blast(40000.0), 125000.0)
    assert unwarned.warned is False
    assert warned.warned is True


def test_the_95th_percentile_adjustment_raises_the_result():
    p = _blast(40000.0)
    with_95 = _research(p, 125000.0, percentile_95=True)
    median = _research(p, 125000.0, percentile_95=False)
    assert SUSCEPTIBILITY_95_GAIN == pytest.approx(3.16228, abs=1e-5)
    assert with_95.max_band_sum_sq_um2 > median.max_band_sum_sq_um2
    assert any("NOT applied" in n for n in median.notes)


# ---------------------------------------------------------------------------
# 2b. The gate that decides what SASA may print
# ---------------------------------------------------------------------------

def test_compute_ahaah_emits_no_number_at_all():
    """
    The whole point of the module in its current state.

    The reference case is not reproduced (see the validation test), so under
    docs/AHAAH-SPEC.md section 11.5 no ARU may be emitted. Every hazard field must
    be NaN on a perfectly good recording, and the reason must say so.
    """
    r = compute_ahaah(_blast(40000.0), 125000.0)
    assert r.valid is False
    assert r.status == "not_validated" == VALIDATION_STATUS
    assert math.isnan(r.total_ARU)
    assert np.all(np.isnan(r.band_ahu))
    assert math.isnan(r.allowed_exposures)
    assert math.isnan(r.allowed_exposures_occupational)
    assert math.isnan(r.threshold_shift_dB)
    assert math.isnan(r.predicted_PTS_dB)
    joined = " ".join(r.notes)
    assert "NO AHAAH RESULT" in joined
    assert "does not reproduce the ARL reference case" in joined
    # It must not blame the operator's recording for the model's problem.
    assert "passed every admissibility check" in joined
    # And it must point at the metric that IS exact.
    assert "A-weighted" in joined


def test_compute_ahaah_reports_the_recording_problem_first():
    """A bad recording must be diagnosed as a bad recording, not as 'not validated'."""
    r = compute_ahaah(_blast(40000.0), 125000.0, calibrated=False)
    assert r.status == "refused"
    assert "uncalibrated" in r.notes[0].lower()


def test_the_research_path_cannot_be_called_by_accident():
    """
    The only route to a number requires the caller to say, in the source, that they
    know it is not an ARU.
    """
    with pytest.raises(ValueError, match="acknowledge_not_validated"):
        run_unvalidated_model(_blast(40000.0), 125000.0)
    with pytest.raises(ValueError, match="acknowledge_not_validated"):
        run_unvalidated_model(_blast(40000.0), 125000.0, acknowledge_not_validated=1)


def test_no_field_of_the_research_result_is_called_ARU():
    """
    Names are the interface. A field called `total_ARU` on an unvalidated model is
    how a wrong number ends up in a customer report.
    """
    r = _research(_blast(40000.0), 125000.0)
    for name in vars(r):
        assert "aru" not in name.lower(), f"research result exposes a field named {name!r}"
    assert r.label == NOT_AN_ARU_LABEL
    assert r.label.startswith("NOT AN ARU")
    assert r.status == "not_validated"


def test_results_are_marked_not_validated_and_carry_the_standing_note():
    r = compute_ahaah(_blast(40000.0), 125000.0)
    assert r.status == "not_validated"
    assert "unavailable" in r.headline_label.lower()
    joined = " ".join(r.notes)
    assert "NO AHAAH RESULT" in joined
    assert "NATO" in joined and "unwarned" in joined
    assert r.category_c_count == CATEGORY_C_COUNT
    assert r.man_coe_md5 == MAN_COE_MD5


def test_result_serialises_to_json():
    r = compute_ahaah(_blast(40000.0), 125000.0)
    blob = json.dumps(r.to_dict())
    back = json.loads(blob)
    assert back["status"] == "not_validated"
    assert back["valid"] is False
    assert back["total_ARU"] is None
    assert back["band_AHU"] == [None] * 23
    assert len(back["assumptions"]) == len(DECLARED_ASSUMPTIONS)
    assert back["man_coe_md5"] == MAN_COE_MD5


def test_invalid_result_also_serialises():
    r = compute_ahaah(_blast(40000.0), 125000.0, calibrated=False)
    back = json.loads(json.dumps(r.to_dict()))
    assert back["valid"] is False
    assert back["total_ARU"] is None
    assert back["band_AHU"] == [None] * 23


def test_the_integrator_step_does_not_change_the_answer(waveform_160F):
    """
    docs/AHAAH-SPEC.md section 6 asks for this: if oversampling 4x -> 8x moves the
    result, the trapezoidal-integrator assumption is load-bearing and the number
    depends on an undocumented choice. It does not.
    """
    wf = waveform_160F
    a = _research(wf.pressure_Pa, wf.sample_rate,
                  input_location=wf.input_location, oversample=4)
    b = _research(wf.pressure_Pa, wf.sample_rate,
                  input_location=wf.input_location, oversample=8)
    assert b.max_band_sum_sq_um2 == pytest.approx(a.max_band_sum_sq_um2, rel=0.02)


def test_peak_detection_rule_is_a_real_but_bounded_uncertainty(waveform_160F):
    """
    Spec section 10.6 warns that "each upward displacement" is ambiguous and could
    be worth "tens of percent". Measured on the reference case it is +9 %: the
    every-local-maximum reading must be >= the one-per-excursion reading (it counts
    a superset of peaks) and here it is not the dominant error. Pinning the size of
    the gap keeps the claim in the assumption list honest.
    """
    wf = waveform_160F
    a = _research(wf.pressure_Pa, wf.sample_rate,
                  input_location=wf.input_location, peak_rule="excursion")
    b = _research(wf.pressure_Pa, wf.sample_rate,
                  input_location=wf.input_location, peak_rule="local_max")
    assert b.max_band_sum_sq_um2 >= a.max_band_sum_sq_um2
    assert b.max_band_sum_sq_um2 < 1.20 * a.max_band_sum_sq_um2, (
        "peak rule changes the answer by "
        f"{100 * (b.max_band_sum_sq_um2 / a.max_band_sum_sq_um2 - 1):.0f} %, "
        "more than spec section 10.6 allows for; it would then dominate the result"
    )


# ---------------------------------------------------------------------------
# Tier 1 -- the linear transfer functions, against ARL's own validation data
# ---------------------------------------------------------------------------

def _transfer_dB(input_location: str, output: str = "eardrum_pressure",
                 n: int = 16384, fs: float = 125000.0):
    """Impulse response of the linearised network, as (frequencies, dB)."""
    p = np.zeros(n)
    p[1024] = 1.0
    me = simulate_middle_ear(p, fs, linear=True, mem_enabled=False,
                             oversample=4, input_location=input_location)
    x = getattr(me, output)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    h = np.fft.rfft(x) / np.fft.rfft(p)
    return f, 20.0 * np.log10(np.abs(h) + 1e-300)


def _read_dat_curve(path: Path):
    """ARL .DAT plot files: rows of `value frequency_kHz` after some header lines."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            value, freq_kHz = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        out.append((freq_kHz * 1000.0, value))
    return np.array([f for f, _ in out]), np.array([v for _, v in out])


def test_circuit_canal_reproduces_the_tabulated_canal_to_eardrum_transfer():
    """
    Tier 1. The chain relies on the circuit for canal -> eardrum, so it has to
    agree with the release's own tabulated row for that step. Checked to 8 kHz at
    5 dB (measured: 4.7 dB worst, 2.0 dB rms). Above 8 kHz the lumped ladder and
    the tabulated curve diverge to 7.7 dB and neither is trustworthy there -- which
    is itself worth knowing, because band 1 sits at 11.76 kHz.
    """
    f, h_dB = _transfer_dB("ear_canal_entrance")
    table_f = np.asarray(SHAW_FREQ_kHz) * 1000.0
    errors = [h_dB[int(np.argmin(np.abs(f - fq)))] - ref
              for fq, ref in zip(table_f, SHAW_CANAL_TO_EARDRUM_dB) if fq <= 8000.0]
    worst = max(abs(e) for e in errors)
    assert worst < 5.0, f"circuit canal deviates from Shaw fig 12 by up to {worst:.1f} dB"


def test_free_field_to_eardrum_matches_FFEDM90():
    """
    Tier 1, the check that settled the outer-ear ambiguity of spec section 10.1.
    Shaw azimuth transfer + the circuit's canal, against Mehrgardt & Mellert as
    ARL themselves plot it.
    """
    from ahaah import SHAW_FF_TO_CANAL_dB_AZ90, apply_outer_ear_hrtf

    n, fs = 16384, 125000.0
    p = np.zeros(n)
    p[1024] = 1.0
    pe = apply_outer_ear_hrtf(p, fs, SHAW_FF_TO_CANAL_dB_AZ90)
    me = simulate_middle_ear(pe, fs, linear=True, mem_enabled=False,
                             oversample=4, input_location="ear_canal_entrance")
    f = np.fft.rfftfreq(n, 1.0 / fs)
    h_dB = 20.0 * np.log10(np.abs(np.fft.rfft(me.eardrum_pressure) / np.fft.rfft(p)) + 1e-300)

    ref_f, ref_dB = _read_dat_curve(FFEDM90)
    band = (ref_f >= 200.0) & (ref_f <= 8000.0)
    errors = []
    for fq, ref in zip(ref_f[band], ref_dB[band]):
        i = int(np.argmin(np.abs(f - fq)))
        errors.append(h_dB[i] - ref)
    errors = np.asarray(errors)
    # Spec section 11.2 sets this tolerance at +/-3 dB. Measured: 3.00 dB worst,
    # 1.42 dB rms over 0.2-8 kHz.
    assert np.abs(errors).max() <= 3.05, (
        f"free-field -> eardrum deviates from FFEDM90.DAT by up to "
        f"{np.abs(errors).max():.1f} dB (rms {np.sqrt(np.mean(errors**2)):.1f} dB)"
    )
    # And the main peak must land near 2.66 kHz, within a third of an octave.
    search = (f >= 500.0) & (f <= 8000.0)
    peak_f = f[search][int(np.argmax(h_dB[search]))]
    assert 2660 / 2 ** (1 / 3) < peak_f < 2660 * 2 ** (1 / 3), (
        f"outer-ear resonance at {peak_f:.0f} Hz, expected within 1/3 octave of 2660 Hz"
    )


def test_mem_contraction_attenuates_20_dB_below_1_kHz_and_less_above():
    """
    ARL: 'they attenuate on the order of 20 dB at frequencies below about 1.0 kHz
    and progressively less at higher frequencies'. This is the only quantitative
    statement about the MEM in the whole release, and it is what fixes the element
    mapping (which man.coe does not state).
    """
    n, fs = 16384, 125000.0
    p = np.zeros(n)
    p[1024] = 1.0

    def stapes(mem: bool):
        me = simulate_middle_ear(p, fs, linear=True, mem_enabled=mem, warned=True,
                                 oversample=4, input_location="ear_canal_entrance")
        return np.fft.rfft(me.stapes_volume_velocity)

    f = np.fft.rfftfreq(n, 1.0 / fs)
    att = 20.0 * np.log10(np.abs(stapes(True)) / np.abs(stapes(False)) + 1e-300)

    def at(fq):
        return att[int(np.argmin(np.abs(f - fq)))]

    for fq in (125.0, 250.0, 500.0, 1000.0):
        assert -25.0 < at(fq) < -15.0, f"MEM attenuation at {fq:.0f} Hz is {at(fq):.1f} dB"
    # Progressively less above 1 kHz, and never a gain.
    highs = [at(fq) for fq in (2000.0, 3150.0, 5000.0, 8000.0, 11760.0)]
    assert all(b >= a - 0.5 for a, b in zip(highs[:-1], highs[1:])), (
        f"MEM attenuation is not monotonically decreasing above 1 kHz: {highs}"
    )
    assert max(highs) < 0.0, f"MEM produces GAIN at high frequency: {highs}"


def test_annular_ligament_limits_stapes_displacement():
    """
    ARL: the ligament 'stops the stapes from displacing more than a few tens of
    microns' where a linear middle ear 'would try to displace 1000 microns or
    more'. Both halves of that are asserted here.
    """
    fs = 125000.0
    p_dyne = _blast(60000.0, fs=fs) * 10.0 * SUSCEPTIBILITY_95_GAIN
    nl = simulate_middle_ear(p_dyne, fs, linear=False, mem_enabled=False)
    lin = simulate_middle_ear(p_dyne, fs, linear=True, mem_enabled=False)
    nl_um = np.abs(nl.stapes_displacement_cm).max() * 1e4
    lin_um = np.abs(lin.stapes_displacement_cm).max() * 1e4
    assert 10.0 < nl_um < 60.0, f"peak stapes displacement {nl_um:.1f} um"
    assert lin_um > 10.0 * nl_um, (
        f"the nonlinearity barely limits: linear {lin_um:.1f} um vs {nl_um:.1f} um"
    )


# ---------------------------------------------------------------------------
# 3. THE VALIDATION TEST
# ---------------------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.xfail(
    strict=True,
    reason=(
        "AHAAH is NOT reproduced. This is the known, documented state, not a flaky "
        "test. The public ARL release does not specify the annular-ligament "
        "nonlinearity (Weapon_noise_AHAAH.pdf defers it to an Appendix A that is not "
        "in the released PDF), leaves CochlearGainFactor ambiguous between 0.0724 and "
        "0.025 (a factor of 8.4 in ARU), and does not define the WKB taper or the "
        "MemMagK/MemMagR triples. See docs/AHAAH-SPEC.md section 12.\n\n"
        "strict=True is deliberate: if this test ever PASSES, pytest fails the run. "
        "That is the alarm. It means either the missing specification arrived and the "
        "model is now genuinely validated - in which case update VALIDATION_STATUS, "
        "remove this marker, and let the assertions stand - or someone tuned a free "
        "parameter until the number matched. Sweeping only the switches ahaah.py "
        "already exposes yields 96 configurations spanning 10.3-2617 ARU on the warned "
        "figure, 4 of which land near 391 by chance, so hitting the reference is NOT "
        "by itself evidence of correctness. Check the per-band table too: a correct "
        "model matches the shape across all 23 bands, not just their maximum."
    ),
)
def test_VALIDATION_against_ARL_160F_reference(result_160F, haz_160F):
    """
    Spec section 11.3, Tier 2. The gate that decides whether SASA may print an ARU.

    DO NOT LOOSEN THESE TOLERANCES. If this fails, the model does not reproduce
    AHAAH and `ahaah.VALIDATION_STATUS` must stay "not_validated", which is exactly
    what it says. Widening the bounds here would ship a wrong number wearing a
    MIL-STD label -- the one outcome this project treats as a failure.

    T2.1 (the warned total) passing on its own means nothing: see the module
    docstring for why 387.7 vs 391.0 is a coincidence. T2.3 and T2.4 are the tests
    that look at the SHAPE of the answer, and they are the ones to trust.
    """
    unwarned, warned = result_160F
    _, ref_ahu, _ = haz_160F

    band_ratio = warned.band_sum_sq_displacement_um2 / ref_ahu
    # 500/index, i.e. what the exposure count WOULD be if the index were an ARU.
    exposures_warned = allowed_exposures(warned.max_band_sum_sq_um2)
    exposures_unwarned = allowed_exposures(unwarned.max_band_sum_sq_um2)
    report = [
        "",
        "AHAAH validation against the ARL 160F reference case",
        "=" * 72,
        f"  T2.1 warned total         {warned.max_band_sum_sq_um2:12.3f}   ref {REF_ARU_WARNED:8.1f}"
        f"   {100 * (warned.max_band_sum_sq_um2 - REF_ARU_WARNED) / REF_ARU_WARNED:+8.1f} %"
        f"   (need +/-5 %)",
        f"  T2.2 unwarned total       {unwarned.max_band_sum_sq_um2:12.3f}   ref {REF_ARU_UNWARNED:8.1f}"
        f"   {100 * (unwarned.max_band_sum_sq_um2 - REF_ARU_UNWARNED) / REF_ARU_UNWARNED:+8.1f} %"
        f"   (need +/-5 %)",
        f"  T2.3 warned argmax band   {warned.peak_band_index:12d}   ref {REF_ARGMAX_BAND:8d}"
        f"                (need exact)",
        f"  T2.6 unwarned/warned      {unwarned.max_band_sum_sq_um2 / warned.max_band_sum_sq_um2:12.3f}"
        f"   ref {REF_RATIO:8.2f}   "
        f"{100 * (unwarned.max_band_sum_sq_um2 / warned.max_band_sum_sq_um2 - REF_RATIO) / REF_RATIO:+8.1f} %"
        f"   (need +/-10 %)",
        f"  T2.7 exposures warned     {round(exposures_warned, 1):12.1f}"
        f"   ref {REF_EXPOSURES_WARNED:8.1f}",
        f"       exposures unwarned   {round(exposures_unwarned, 1):12.1f}"
        f"   ref {REF_EXPOSURES_UNWARNED:8.1f}",
        f"  T2.8 peak stapes displ.   {unwarned.peak_stapes_displacement_um:12.2f} um"
        f"              (need 10-60 um)",
        "",
        "  T2.4 per-band (warned), need every ratio within +/-10 %:",
        f"    {'band':>4} {'f (kHz)':>9} {'model AHU':>13} {'ref AHU':>12} {'ratio':>8}",
    ]
    for i in range(XBM_NO):
        report.append(
            f"    {i + 1:4d} {warned.band_frequencies_Hz[i] / 1000:9.2f} "
            f"{warned.band_sum_sq_displacement_um2[i]:13.5f} {ref_ahu[i]:12.5f} {band_ratio[i]:8.3f}"
        )
    n_in_band = int(np.sum(np.abs(band_ratio - 1.0) <= 0.10))
    report += [
        f"    bands within +/-10 %: {n_in_band} of {XBM_NO}",
        "",
        f"  category-(c) inferences carried by this result: {unwarned.category_c_count}",
        f"  cochlear gain factor in use: {COCHLEAR_GAIN_FACTOR}",
        "=" * 72,
    ]
    detail = "\n".join(report)

    failures = []
    if abs(warned.max_band_sum_sq_um2 - REF_ARU_WARNED) / REF_ARU_WARNED > 0.05:
        failures.append("T2.1 warned total ARU outside +/-5 %")
    if abs(unwarned.max_band_sum_sq_um2 - REF_ARU_UNWARNED) / REF_ARU_UNWARNED > 0.05:
        failures.append("T2.2 unwarned total ARU outside +/-5 %")
    if warned.peak_band_index != REF_ARGMAX_BAND:
        failures.append(
            f"T2.3 warned argmax is band {warned.peak_band_index}, not {REF_ARGMAX_BAND}"
        )
    if n_in_band < XBM_NO:
        failures.append(f"T2.4 only {n_in_band}/{XBM_NO} bands within +/-10 %")
    ratio = unwarned.max_band_sum_sq_um2 / warned.max_band_sum_sq_um2
    if abs(ratio - REF_RATIO) / REF_RATIO > 0.10:
        failures.append(f"T2.6 unwarned/warned ratio {ratio:.2f} vs {REF_RATIO:.2f}")
    if round(exposures_warned, 1) != REF_EXPOSURES_WARNED:
        failures.append("T2.7 warned allowed exposures do not round to 1.3")
    if round(exposures_unwarned, 1) != REF_EXPOSURES_UNWARNED:
        failures.append("T2.7 unwarned allowed exposures do not round to 0.2")
    if not 10.0 <= unwarned.peak_stapes_displacement_um <= 60.0:
        failures.append("T2.8 peak stapes displacement outside 10-60 um")

    if failures:
        pytest.fail(
            detail
            + "\n\nFAILED:\n  - "
            + "\n  - ".join(failures)
            + "\n\nThis is the expected state. ahaah.VALIDATION_STATUS is 'not_validated',\n"
              "compute_ahaah() emits no number at all, and SASA presents no ARU.\n"
              "\n"
              "DO NOT widen these tolerances, and DO NOT read T2.1 as encouraging. The\n"
              "warned total lands within 1 % of 391 while T2.3 and T2.4 show the band\n"
              "curve it is the maximum of is wrong in 20 of 23 bands and peaks in the\n"
              "wrong band. Sweeping only the switches ahaah.py already exposes gives 96\n"
              "configurations spanning 10.3-2617 on the warned figure; 4 of them land\n"
              "within 5 % of 391 and none of those 4 is also within 5 % of 2237.\n"
              "\n"
              "The diagnostic lead is T2.6, not T2.1. Across all 96 of those\n"
              "configurations the unwarned/warned ratio stays between 2.59 and 3.17\n"
              "against a reference of 5.72, while the LEVEL moves by a factor of 255.\n"
              "Scaling the annular ligament (Lgap 10-300) moves the level by 33000x and\n"
              "the ratio only 2.78-3.16, so the ligament is not what is wrong with the\n"
              "ratio -- the middle-ear-muscle attenuator is. MemMagK/MemMagR near 40 on a\n"
              "single element reproduces 5.54, which points at the ELEMENT MAPPING of\n"
              "man.coe's three-element triples '12 1 6' / '12 1 12', of which ahaah.py\n"
              "applies only the leading 12, to one element. Ask ARL about that mapping\n"
              "first, then about the annular-ligament functional form; both are in\n"
              "docs/AHAAH-SPEC.md section 12.",
            pytrace=False,
        )


@pytest.mark.validation
def test_validation_status_is_consistent_with_the_reference_run(result_160F):
    """
    The status field must never claim more than the reference run supports. If the
    Tier-2 test above starts passing, this one fails until VALIDATION_STATUS is
    deliberately changed -- so the gate cannot be opened by accident.
    """
    import ahaah

    unwarned, warned = result_160F
    passes = (
        abs(warned.max_band_sum_sq_um2 - REF_ARU_WARNED) / REF_ARU_WARNED <= 0.05
        and abs(unwarned.max_band_sum_sq_um2 - REF_ARU_UNWARNED) / REF_ARU_UNWARNED <= 0.05
        and warned.peak_band_index == REF_ARGMAX_BAND
    )
    if passes:
        assert ahaah.VALIDATION_STATUS in ("validated", "fitted"), (
            "the reference case now reproduces; review the category-(c) choices that "
            "were varied to get there and set VALIDATION_STATUS deliberately"
        )
    else:
        assert ahaah.VALIDATION_STATUS == "not_validated"
