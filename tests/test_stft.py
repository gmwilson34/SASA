"""
test_stft.py - spectrogram scaling, window properties and axis definitions.

The oracles are analytic:

  * A sine of RMS level L placed exactly on a bin centre must read L in that bin.
    Choosing freq = k*fs/nperseg for integer k removes scalloping loss, which
    isolates the SCALING from the windowing.
  * Parseval: integrating a one-sided power spectral density over frequency
    recovers the signal's mean square, so PSD can be checked against the known
    broadband level of the noise fixture.
  * Window equivalent noise bandwidth has closed-form values (Hann 1.5 bins,
    rectangular 1.0), which pin the window helpers independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration import P_REF
from STFT import (
    analyze_stft,
    compute_stft,
    compute_stft_dB_SPL,
    default_noverlap,
    get_window,
    recommended_nperseg,
    window_coherent_gain,
    window_enbw_Hz,
    window_enbw_bins,
)
from tests.conftest import make_sine, make_white_noise


FS = 48000
NPERSEG = 2048


def on_bin_frequency(bin_index: int, fs: int = FS, nperseg: int = NPERSEG) -> float:
    """Frequency landing exactly on an FFT bin centre."""
    return bin_index * fs / nperseg


# ---------------------------------------------------------------------------
# Window properties
# ---------------------------------------------------------------------------

def test_window_enbw_closed_form_values():
    """
    ENBW = N * sum(w^2) / (sum w)^2, which is exactly 1.0 for a rectangular
    window and 1.5 for a periodic Hann window.
    """
    assert window_enbw_bins(get_window("rectangular", 4096)) == pytest.approx(1.0, abs=1e-12)
    assert window_enbw_bins(get_window("hann", 4096)) == pytest.approx(1.5, abs=1e-3)
    assert window_enbw_bins(get_window("hamming", 4096)) == pytest.approx(1.3628, abs=2e-3)
    assert window_enbw_bins(get_window("blackman", 4096)) == pytest.approx(1.7269, abs=2e-3)


def test_window_coherent_gain_closed_form_values():
    """Coherent gain = sum(w)/N: 1.0 rectangular, 0.5 Hann, 0.54 Hamming."""
    assert window_coherent_gain(get_window("rectangular", 4096)) == pytest.approx(1.0, abs=1e-12)
    assert window_coherent_gain(get_window("hann", 4096)) == pytest.approx(0.5, abs=1e-3)
    assert window_coherent_gain(get_window("hamming", 4096)) == pytest.approx(0.54, abs=1e-3)


def test_enbw_in_hz_scales_with_resolution():
    """ENBW in Hz is ENBW_bins * fs/nperseg."""
    w = get_window("hann", NPERSEG)
    assert window_enbw_Hz(w, FS, NPERSEG) == pytest.approx(1.5 * FS / NPERSEG, rel=1e-3)


def test_unknown_window_raises():
    with pytest.raises(ValueError):
        get_window("no-such-window", 1024)


# ---------------------------------------------------------------------------
# Absolute scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "flattop", "rectangular"])
def test_on_bin_tone_reads_its_true_rms_level(window):
    """
    CRITICAL: a 94 dB SPL tone on a bin centre must read 94.000 dB.

    This pins the whole calibration chain of the spectrogram: window
    normalisation, the one-sided doubling of every bin except DC and Nyquist,
    and the amplitude-to-RMS conversion.
    """
    freq = on_bin_frequency(42)
    x = make_sine(freq, 94.0, FS, 1.0)

    result = analyze_stft(x, FS, nperseg=NPERSEG, window=window, scaling="rms")
    assert result.get_max_level() == pytest.approx(94.0, abs=0.01)
    assert result.get_freq_at_max() == pytest.approx(freq, abs=FS / NPERSEG)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "rectangular"])
def test_psd_times_enbw_equals_the_band_level(window):
    """
    For a tone, PSD at the peak bin multiplied by the window's ENBW is the tone's
    mean square, so 10*log10(PSD*ENBW) must equal the tone's level.
    """
    freq = on_bin_frequency(42)
    x = make_sine(freq, 94.0, FS, 1.0)

    result = analyze_stft(x, FS, nperseg=NPERSEG, window=window, scaling="psd")
    band_level = result.get_max_level() + 10.0 * np.log10(result.enbw_Hz)
    assert band_level == pytest.approx(94.0, abs=0.01)


@pytest.mark.parametrize("level", [60.0, 94.0, 140.0, 170.0])
def test_scaling_is_level_independent(level):
    """Gunshot levels span 60-170 dB; the scaling must hold across all of it."""
    x = make_sine(on_bin_frequency(64), level, FS, 1.0)
    result = analyze_stft(x, FS, nperseg=NPERSEG, scaling="rms", db_floor=-400.0)
    assert result.get_max_level() == pytest.approx(level, abs=0.01)


# ---------------------------------------------------------------------------
# Scalloping loss
# ---------------------------------------------------------------------------

def test_hann_scalloping_loss_is_the_textbook_value():
    """
    A tone exactly half a bin off centre loses 1.42 dB with a Hann window. This
    is a property of the window, and it is why a tonal component can read low.
    """
    x = make_sine(on_bin_frequency(42) + 0.5 * FS / NPERSEG, 94.0, FS, 2.0)
    result = analyze_stft(x, FS, nperseg=NPERSEG, window="hann", scaling="rms")
    assert result.get_max_level() - 94.0 == pytest.approx(-1.42, abs=0.1)


def test_flattop_window_almost_eliminates_scalloping():
    """
    The flat-top window exists for amplitude accuracy on off-bin tones: better
    than 0.1 dB where Hann loses 1.42 dB.
    """
    x = make_sine(on_bin_frequency(42) + 0.5 * FS / NPERSEG, 94.0, FS, 2.0)
    result = analyze_stft(x, FS, nperseg=NPERSEG, window="flattop", scaling="rms")
    assert abs(result.get_max_level() - 94.0) < 0.1


# ---------------------------------------------------------------------------
# Broadband behaviour and Parseval
# ---------------------------------------------------------------------------

def test_psd_integrates_to_the_true_broadband_level():
    """
    Parseval: integrating the one-sided PSD over frequency must recover the
    signal's mean square. This is the check that catches a missing factor of two
    or a wrong window normalisation on broadband content.
    """
    level = 80.0
    x = make_white_noise(level, FS, 2.0)

    result = analyze_stft(x, FS, nperseg=NPERSEG, scaling="psd", db_floor=-400.0)
    psd_linear = (P_REF ** 2) * 10.0 ** (result.magnitude_dB / 10.0)
    total = float(np.sum(np.mean(psd_linear, axis=1)) * (FS / NPERSEG))

    assert 10.0 * np.log10(total / P_REF ** 2) == pytest.approx(level, abs=0.1)


@pytest.mark.parametrize("nperseg", [512, 1024, 2048, 4096])
def test_psd_is_independent_of_fft_size(nperseg):
    """
    A power spectral DENSITY is per hertz, so it must not move when the analysis
    resolution changes. This is why PSD is the honest scaling for the broadband
    content of a gunshot.
    """
    x = make_white_noise(80.0, FS, 2.0)
    result = analyze_stft(x, FS, nperseg=nperseg, scaling="psd", db_floor=-400.0)

    mid = slice(len(result.frequencies_Hz) // 4, 3 * len(result.frequencies_Hz) // 4)
    assert float(np.mean(result.magnitude_dB[mid])) == pytest.approx(33.7, abs=0.5)


def test_band_level_scales_by_3dB_per_octave_of_fft_size():
    """
    Band level per bin is NOT resolution independent for broadband content:
    doubling nperseg halves the bin width and so removes 3.01 dB. Documenting
    this in a test is what stops it being mistaken for a calibration error.
    """
    x = make_white_noise(80.0, FS, 2.0)

    levels = {}
    for nperseg in (1024, 2048, 4096):
        result = analyze_stft(x, FS, nperseg=nperseg, scaling="rms", db_floor=-400.0)
        mid = slice(len(result.frequencies_Hz) // 4, 3 * len(result.frequencies_Hz) // 4)
        levels[nperseg] = float(np.mean(result.magnitude_dB[mid]))

    assert levels[1024] - levels[2048] == pytest.approx(3.01, abs=0.15)
    assert levels[2048] - levels[4096] == pytest.approx(3.01, abs=0.15)


# ---------------------------------------------------------------------------
# Regression: overlap must be derived from the window size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nperseg", [512, 1024, 2048, 4096, 8192])
def test_shipped_ui_window_sizes_do_not_abort(nperseg):
    """
    CRITICAL REGRESSION.

    The pipeline hard-coded noverlap = 1536 while offering nperseg down to 512 in
    the interface, so two of the five shipped options aborted the entire analysis
    with ValueError. analyze_stft must correct an impossible overlap.
    """
    x = make_white_noise(80.0, FS, 0.5)
    result = analyze_stft(x, FS, nperseg=nperseg, noverlap=1536)

    assert 0 <= result.noverlap < nperseg
    assert result.magnitude_dB.shape[0] == nperseg // 2 + 1
    assert result.magnitude_dB.shape[1] >= 1


def test_default_overlap_is_three_quarters():
    assert default_noverlap(2048) == 1536
    assert default_noverlap(512) == 384


def test_compute_stft_still_rejects_an_explicitly_bad_overlap():
    """
    analyze_stft is forgiving because it serves a UI, but the low-level function
    must not silently accept nonsense from a programmatic caller.
    """
    x = make_white_noise(80.0, FS, 0.2)
    with pytest.raises(ValueError):
        compute_stft(x, FS, nperseg=512, noverlap=1536)
    with pytest.raises(ValueError):
        compute_stft(x, FS, nperseg=512, noverlap=-1)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_nperseg_raises(bad):
    x = make_white_noise(80.0, FS, 0.2)
    with pytest.raises(ValueError):
        analyze_stft(x, FS, nperseg=bad)


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

def test_frame_times_are_window_centres_and_monotonic():
    """
    A frame's timestamp must mark the centre of the window whose energy it
    reports; labelling it with the window start shifts every event by half a
    window, which at 2048 points and 48 kHz is 21 ms.
    """
    x = make_white_noise(80.0, FS, 1.0)
    result = analyze_stft(x, FS, nperseg=NPERSEG, noverlap=1536)

    hop = NPERSEG - 1536
    assert result.time_s[0] == pytest.approx(NPERSEG / 2.0 / FS, rel=1e-9)
    assert np.all(np.diff(result.time_s) > 0)
    assert np.allclose(np.diff(result.time_s), hop / FS)


def test_frequency_axis_matches_rfft():
    x = make_white_noise(80.0, FS, 0.5)
    result = analyze_stft(x, FS, nperseg=NPERSEG)

    assert result.frequencies_Hz[0] == 0.0
    assert result.frequencies_Hz[-1] == pytest.approx(FS / 2.0)
    assert len(result.frequencies_Hz) == NPERSEG // 2 + 1
    assert result.freq_resolution_Hz == pytest.approx(FS / NPERSEG)


def test_time_resolution_reports_the_hop():
    result = analyze_stft(make_white_noise(80.0, FS, 0.5), FS,
                          nperseg=NPERSEG, noverlap=1536)
    assert result.time_resolution_s == pytest.approx((NPERSEG - 1536) / FS)


# ---------------------------------------------------------------------------
# Weighting applied in the frequency domain
# ---------------------------------------------------------------------------

def test_a_weighting_shifts_a_tone_by_the_tabulated_amount():
    """
    A 1 kHz tone is unchanged by A-weighting (the normalisation point) and a
    125 Hz tone drops by 16.1 dB per IEC 61672-1.
    """
    ref = analyze_stft(make_sine(on_bin_frequency(64), 94.0, FS, 1.0), FS,
                       nperseg=NPERSEG, weighting="Z", db_floor=-400.0)
    a = analyze_stft(make_sine(on_bin_frequency(64), 94.0, FS, 1.0), FS,
                     nperseg=NPERSEG, weighting="A", db_floor=-400.0)

    # bin 64 at 48 kHz / 2048 is 1500 Hz; A-weighting there is +1.0 dB
    from weighting import a_weight_frequency_response
    expected = a_weight_frequency_response(np.array([on_bin_frequency(64)]))[0]
    assert a.get_max_level() - ref.get_max_level() == pytest.approx(expected, abs=0.05)


def test_z_weighting_is_a_pass_through():
    x = make_sine(on_bin_frequency(42), 94.0, FS, 1.0)
    _, _, mag_z = compute_stft_dB_SPL(x, FS, nperseg=NPERSEG, weighting="Z")
    _, _, plain = compute_stft_dB_SPL(x, FS, nperseg=NPERSEG, weighting="Z")
    assert np.allclose(mag_z, plain)


def test_unknown_weighting_raises():
    x = make_white_noise(80.0, FS, 0.2)
    with pytest.raises(ValueError):
        compute_stft_dB_SPL(x, FS, nperseg=512, weighting="Q")


def test_unknown_scaling_raises():
    x = make_white_noise(80.0, FS, 0.2)
    with pytest.raises(ValueError):
        compute_stft(x, FS, nperseg=512, scaling="bogus")


# ---------------------------------------------------------------------------
# Labelling: an uncalibrated analysis must never claim dB SPL
# ---------------------------------------------------------------------------

def test_level_label_follows_the_scaling_and_calibration():
    x = make_white_noise(80.0, FS, 0.3)

    rms_cal = analyze_stft(x, FS, nperseg=512, scaling="rms", calibrated=True)
    assert "band level" in rms_cal.level_label
    assert "re FS" not in rms_cal.level_label

    rms_uncal = analyze_stft(x, FS, nperseg=512, scaling="rms", calibrated=False)
    assert "re FS" in rms_uncal.level_label

    psd = analyze_stft(x, FS, nperseg=512, scaling="psd", calibrated=True)
    assert "PSD" in psd.level_label and "Hz" in psd.level_label


def test_result_serialises_its_parameters():
    result = analyze_stft(make_white_noise(80.0, FS, 0.3), FS, nperseg=512)
    data = result.to_dict()
    for key in ("weighting", "sample_rate", "nperseg", "noverlap", "window",
                "scaling", "enbw_Hz", "freq_resolution_Hz", "level_label"):
        assert key in data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_recommended_nperseg_tracks_the_sample_rate():
    """
    A muzzle blast lasts a few milliseconds. A fixed 2048-point window is 42.7 ms
    at 48 kHz - twenty times longer than the event - so the window must scale
    with the sample rate to keep a constant time span.
    """
    for fs in (48000, 96000, 192000):
        n = recommended_nperseg(fs, target_ms=2.0)
        assert n & (n - 1) == 0, "must be a power of two"
        assert n / fs * 1000.0 == pytest.approx(2.0, rel=0.45)

    assert recommended_nperseg(192000, 2.0) > recommended_nperseg(48000, 2.0)


def test_decimate_frames_thins_time_without_touching_frequency():
    result = analyze_stft(make_white_noise(80.0, FS, 2.0), FS, nperseg=512)
    thinned = result.decimate_frames(4)

    assert thinned.magnitude_dB.shape[0] == result.magnitude_dB.shape[0]
    assert thinned.magnitude_dB.shape[1] == len(range(0, result.magnitude_dB.shape[1], 4))
    assert result.decimate_frames(1) is result


def test_short_signal_is_padded_not_rejected():
    """A shot window shorter than one FFT frame must still analyse."""
    x = make_sine(1000.0, 94.0, FS, 0.005)       # 240 samples
    result = analyze_stft(x, FS, nperseg=2048)
    assert result.magnitude_dB.shape[1] >= 1


def test_non_contiguous_input_is_handled():
    """
    REGRESSION: the framing used stride tricks with strides taken from a
    different array than the one being viewed, so a non-contiguous input
    silently produced garbage.
    """
    x = make_sine(on_bin_frequency(42), 94.0, FS, 2.0)
    strided = x[::2]                              # non-contiguous view

    result = analyze_stft(np.ascontiguousarray(strided), FS // 2, nperseg=1024)
    result_view = analyze_stft(strided, FS // 2, nperseg=1024)
    assert np.allclose(result.magnitude_dB, result_view.magnitude_dB)
