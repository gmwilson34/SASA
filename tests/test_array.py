"""
test_array.py - multi-microphone array geometry and consistency.

The oracles are geometric and exact: a microphone three times further away
hears exactly 20*log10(3) dB less, and hears it exactly 2/c seconds later. The
consistency check is held to detecting a planted fault, and to admitting what it
cannot attribute.
"""

from __future__ import annotations

import math

import pytest

from array import (
    CONSISTENCY_TOLERANCE_dB,
    MIN_CHANNELS,
    ArrayError,
    ArrayGeometry,
    MicPosition,
    check_array_consistency,
    expected_arrival_delays,
)
from atmosphere import Atmosphere, speed_of_sound


def pair(near_m=1.0, far_m=3.0):
    return ArrayGeometry([
        MicPosition(channel=0, distance_m=near_m, angle_deg=90.0, label="near"),
        MicPosition(channel=1, distance_m=far_m, angle_deg=90.0, label="far"),
    ])


def spreading(from_m, to_m):
    return 20.0 * math.log10(from_m / to_m)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_a_position_must_be_physically_possible():
    with pytest.raises(ArrayError):
        MicPosition(channel=0, distance_m=0.0, angle_deg=90.0)
    with pytest.raises(ArrayError):
        MicPosition(channel=0, distance_m=-1.0, angle_deg=90.0)
    with pytest.raises(ArrayError):
        MicPosition(channel=0, distance_m=1.0, angle_deg=float("nan"))


def test_a_channel_cannot_appear_twice():
    with pytest.raises(ArrayError):
        ArrayGeometry([
            MicPosition(channel=0, distance_m=1.0, angle_deg=0.0),
            MicPosition(channel=0, distance_m=2.0, angle_deg=0.0),
        ])


def test_cartesian_position_places_zero_degrees_downrange():
    """The convention matches TestMetadata: 0 is downrange, 90 is abeam."""
    downrange = MicPosition(channel=0, distance_m=5.0, angle_deg=0.0)
    abeam = MicPosition(channel=1, distance_m=5.0, angle_deg=90.0)
    assert downrange.position_m[0] == pytest.approx(5.0)
    assert downrange.position_m[1] == pytest.approx(0.0, abs=1e-12)
    assert abeam.position_m[0] == pytest.approx(0.0, abs=1e-12)
    assert abeam.position_m[1] == pytest.approx(5.0)


def test_one_microphone_is_not_an_array():
    assert not ArrayGeometry([
        MicPosition(channel=0, distance_m=1.0, angle_deg=90.0)]).is_array
    assert pair().is_array


def test_metadata_with_one_position_does_not_describe_an_array():
    """
    A record carrying a single mic position cannot describe several microphones.
    Every channel gets that position, and the report says the comparison is
    therefore not evidence about separate microphones.
    """
    geometry = ArrayGeometry.from_metadata(
        {"mic_distance_m": 2.0, "mic_angle_deg": 90.0}, channels=[0, 1])
    assert len(geometry.positions) == 2
    assert all(p.distance_m == 2.0 for p in geometry.positions)

    report = check_array_consistency({0: 150.0, 1: 150.0}, geometry=geometry)
    assert any("not a described array" in note for note in report.notes)


def test_metadata_without_a_position_yields_no_geometry():
    assert ArrayGeometry.from_metadata({}, channels=[0, 1]).positions == []
    assert ArrayGeometry.from_metadata(
        {"mic_distance_m": None, "mic_angle_deg": 90.0}, channels=[0]).positions == []


# ---------------------------------------------------------------------------
# Arrival delays
# ---------------------------------------------------------------------------

def test_arrival_delay_is_the_path_difference_over_the_speed_of_sound():
    """
    Two metres of extra path at 18 C is 2/342.03 = 5.847 ms. This is a direct,
    checkable prediction of the recorded geometry.
    """
    delays = expected_arrival_delays(pair(1.0, 3.0), temperature_C=18.0)
    expected = 2.0 / speed_of_sound(18.0)
    assert delays[0] == pytest.approx(0.0, abs=1e-15)
    assert delays[1] == pytest.approx(expected, rel=1e-12)
    assert delays[1] == pytest.approx(0.005847, abs=1e-6)


def test_arrival_delays_are_relative_to_the_nearest_microphone():
    delays = expected_arrival_delays(pair(4.0, 2.0), temperature_C=20.0)
    assert min(delays.values()) == pytest.approx(0.0, abs=1e-15)
    assert delays[0] > delays[1]  # channel 0 is the far one here


def test_a_colder_day_delays_every_arrival():
    cold = expected_arrival_delays(pair(1.0, 5.0), temperature_C=-10.0)
    hot = expected_arrival_delays(pair(1.0, 5.0), temperature_C=40.0)
    assert cold[1] > hot[1]


def test_an_empty_array_has_no_delays():
    assert expected_arrival_delays(ArrayGeometry([])) == {}


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

def test_channels_measuring_one_blast_agree_exactly():
    """
    A microphone three times further away hears exactly 20*log10(3) dB less.
    Referred back to a common distance, the two must give the same number.
    """
    geometry = pair(1.0, 3.0)
    near = 150.0
    far = near - spreading(3.0, 1.0)

    report = check_array_consistency({0: near, 1: far}, geometry=geometry)
    assert report.consistent
    assert report.spread_dB == pytest.approx(0.0, abs=1e-12)
    assert all(c.normalised_dB == pytest.approx(150.0, abs=1e-12) for c in report.channels)


def test_a_planted_fault_is_detected():
    geometry = pair(1.0, 3.0)
    near = 150.0
    far = near - spreading(3.0, 1.0) - 2.0   # far mic reads 2 dB low

    report = check_array_consistency({0: near, 1: far}, geometry=geometry)
    assert not report.consistent
    assert report.spread_dB == pytest.approx(2.0, rel=1e-12)
    assert len(report.disagreeing) == 2


def test_a_fault_inside_the_tolerance_is_accepted():
    geometry = pair(1.0, 3.0)
    near = 150.0
    far = near - spreading(3.0, 1.0) - (CONSISTENCY_TOLERANCE_dB * 0.5)
    report = check_array_consistency({0: near, 1: far}, geometry=geometry)
    assert report.consistent


def test_two_microphones_cannot_say_which_one_is_wrong():
    """
    The median of two sits between them, so both are equally far from it. The
    report must admit that rather than implying it has identified a culprit.
    """
    geometry = pair(1.0, 3.0)
    report = check_array_consistency(
        {0: 150.0, 1: 150.0 - spreading(3.0, 1.0) - 3.0}, geometry=geometry)
    assert not report.consistent
    assert any("not attributable" in note for note in report.notes)


def test_three_microphones_identify_the_odd_one_out():
    """With three, the median is a real vote and the outlier is named alone."""
    geometry = ArrayGeometry([
        MicPosition(channel=0, distance_m=1.0, angle_deg=90.0),
        MicPosition(channel=1, distance_m=2.0, angle_deg=90.0),
        MicPosition(channel=2, distance_m=4.0, angle_deg=90.0),
    ])
    levels = {
        0: 150.0,
        1: 150.0 - spreading(2.0, 1.0),
        2: 150.0 - spreading(4.0, 1.0) - 3.0,   # only channel 2 is wrong
    }
    report = check_array_consistency(levels, geometry=geometry)
    assert not report.consistent
    assert [c.channel for c in report.disagreeing] == [2]


def test_a_single_channel_cannot_be_checked():
    report = check_array_consistency({0: 150.0}, geometry=pair())
    assert report.refusal
    assert str(MIN_CHANNELS) in report.refusal


def test_a_channel_without_a_position_is_left_out():
    geometry = ArrayGeometry([MicPosition(channel=0, distance_m=1.0, angle_deg=90.0)])
    report = check_array_consistency({0: 150.0, 7: 148.0}, geometry=geometry)
    assert report.refusal  # only one usable channel remains


def test_no_geometry_refuses_rather_than_assuming():
    report = check_array_consistency({0: 150.0, 1: 149.0}, geometry=ArrayGeometry([]))
    assert report.refusal
    assert "no microphone positions" in report.refusal


def test_non_finite_levels_are_excluded():
    report = check_array_consistency(
        {0: 150.0, 1: float("nan")}, geometry=pair())
    assert report.refusal


def test_a_long_baseline_notes_the_absorption_it_did_not_apply():
    """
    Absorption is not applied to broadband levels because a broadband level has
    no single frequency. Over a long baseline that omission is material and must
    be stated rather than hidden.
    """
    geometry = pair(1.0, 60.0)
    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0)
    report = check_array_consistency(
        {0: 150.0, 1: 150.0 - spreading(60.0, 1.0)}, geometry=geometry, atmosphere=air)
    assert any("absorbs about" in note for note in report.notes)


def test_a_short_baseline_does_not_raise_the_absorption_note():
    geometry = pair(1.0, 1.2)
    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0)
    report = check_array_consistency(
        {0: 150.0, 1: 150.0 - spreading(1.2, 1.0)}, geometry=geometry, atmosphere=air)
    assert not any("absorbs about" in note for note in report.notes)


def test_the_report_serialises():
    report = check_array_consistency(
        {0: 150.0, 1: 140.0}, geometry=pair(1.0, 3.0))
    data = report.to_dict()
    for key in ("consistent", "describes_an_array", "spread_dB", "tolerance_dB",
                "channels", "disagreeing_channels"):
        assert key in data
    assert len(data["channels"]) == 2


def test_two_microphones_at_the_same_point_are_not_an_array():
    """
    Comparing two channels fed from the same position shows only that a splitter
    works. An array requires microphones at DIFFERENT places.
    """
    same_spot = ArrayGeometry([
        MicPosition(channel=0, distance_m=2.0, angle_deg=90.0),
        MicPosition(channel=1, distance_m=2.0, angle_deg=90.0),
    ])
    assert not same_spot.is_array
    assert pair(1.0, 3.0).is_array


def test_microphones_at_the_same_distance_but_different_angles_are_an_array():
    """Two mics abeam on opposite sides is a real array, and a useful one."""
    opposite = ArrayGeometry([
        MicPosition(channel=0, distance_m=1.0, angle_deg=90.0),
        MicPosition(channel=1, distance_m=1.0, angle_deg=270.0),
    ])
    assert opposite.is_array
