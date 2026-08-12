"""
test_atmosphere.py - ISO 9613-1 absorption, distance normalisation, Mach geometry.

Every assertion here is against a value that is derivable without running the
code under test: an exact reference condition stated by the standard, an
analytic limit of the absorption formula, or a closed-form solution of the
Mach-cone arrival time obtained by a different algebraic route than the
implementation uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atmosphere import (
    C_REF_M_PER_S,
    GAMMA_AIR,
    NONLINEAR_MACH_LIMIT,
    NONLINEAR_PEAK_dB,
    P_ATM_PA,
    P_REF_kPa,
    T_REF_K,
    T_TRIPLE_K,
    T_ZERO_C_K,
    Atmosphere,
    AtmosphereError,
    absorption_coefficient_dB_per_m,
    absorption_dB,
    air_density,
    crack_blast_delay_s,
    describe_atmospheric_effect,
    geometric_spreading_dB,
    molar_water_vapour_pct,
    normalise_insertion_loss_bands,
    relaxation_frequency_nitrogen,
    relaxation_frequency_oxygen,
    saturation_vapour_pressure_ratio,
    speed_of_sound,
)

# The 1/3-octave nominal band centres SASA reports over.
BAND_CENTRES = np.array([
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Reference conditions stated by the standard
# ---------------------------------------------------------------------------

def test_speed_of_sound_at_reference_temperature_is_exactly_343_2():
    """ISO 9613-1 defines c = 343.2 m/s at 20 C, so the formula must return it exactly."""
    assert speed_of_sound(20.0) == pytest.approx(C_REF_M_PER_S, abs=1e-12)


def test_speed_of_sound_scales_as_square_root_of_absolute_temperature():
    """c(T)/c(T0) = sqrt(T/T0) is the definition, not an approximation."""
    for t_C in (-20.0, 0.0, 15.0, 35.0, 50.0):
        expected = C_REF_M_PER_S * math.sqrt((t_C + T_ZERO_C_K) / T_REF_K)
        assert speed_of_sound(t_C) == pytest.approx(expected, rel=1e-15)


def test_saturation_vapour_pressure_at_the_triple_point_is_closed_form():
    """
    At T = T_01 the exponent collapses to -6.8346 + 4.6151 = -2.2195 exactly,
    because (T_01/T)**1.261 is 1.
    """
    assert saturation_vapour_pressure_ratio(T_TRIPLE_K) == pytest.approx(
        10.0 ** (-2.2195), rel=1e-15
    )


def test_air_density_matches_the_ideal_gas_law():
    """rho = p/(R T) with R = 287.058 J/(kg K)."""
    rho = air_density(20.0, P_REF_kPa)
    assert rho == pytest.approx(101325.0 / (287.058 * 293.15), rel=1e-12)
    # Sanity anchor: dry air at 20 C, one atmosphere is about 1.2 kg/m^3.
    assert 1.19 < rho < 1.21


def test_temperature_below_absolute_zero_is_refused():
    for fn in (speed_of_sound,):
        with pytest.raises(AtmosphereError):
            fn(-300.0)
    with pytest.raises(AtmosphereError):
        air_density(-300.0, P_REF_kPa)
    with pytest.raises(AtmosphereError):
        air_density(20.0, 0.0)


# ---------------------------------------------------------------------------
# Water vapour and relaxation frequencies
# ---------------------------------------------------------------------------

def test_molar_water_vapour_is_inversely_proportional_to_pressure():
    """
    h = h_rel * (p_sat/p_r) / (p_a/p_r). At fixed temperature and relative
    humidity, halving the ambient pressure exactly doubles the molar
    concentration.
    """
    t_K = 293.15
    h_full = molar_water_vapour_pct(t_K, 50.0, P_REF_kPa)
    h_half = molar_water_vapour_pct(t_K, 50.0, P_REF_kPa / 2.0)
    assert h_half == pytest.approx(2.0 * h_full, rel=1e-14)


def test_molar_water_vapour_is_proportional_to_relative_humidity():
    t_K = 293.15
    h_50 = molar_water_vapour_pct(t_K, 50.0, P_REF_kPa)
    h_100 = molar_water_vapour_pct(t_K, 100.0, P_REF_kPa)
    assert h_100 == pytest.approx(2.0 * h_50, rel=1e-14)


def test_dry_air_relaxation_frequencies_take_their_zero_humidity_values():
    """
    With h = 0 the ISO expressions collapse to f_rO = 24*(p/p_r) and
    f_rN = 9*(p/p_r)*(T/T0)^-0.5.
    """
    t_K = T_REF_K
    assert relaxation_frequency_oxygen(t_K, 0.0, P_REF_kPa) == pytest.approx(24.0, rel=1e-14)
    assert relaxation_frequency_nitrogen(t_K, 0.0, P_REF_kPa) == pytest.approx(9.0, rel=1e-14)


def test_relaxation_frequencies_scale_with_pressure_at_fixed_molar_concentration():
    """
    Both relaxation frequencies carry an explicit (p_a/p_r) factor. Holding the
    molar water concentration fixed (by scaling relative humidity with pressure)
    isolates it.
    """
    t_K = T_REF_K
    p_half = P_REF_kPa / 2.0
    # Halving pressure doubles h at fixed RH, so halve RH to hold h constant.
    h_ref = molar_water_vapour_pct(t_K, 40.0, P_REF_kPa)
    h_alt = molar_water_vapour_pct(t_K, 20.0, p_half)
    assert h_alt == pytest.approx(h_ref, rel=1e-14)

    f_ref = relaxation_frequency_oxygen(t_K, 40.0, P_REF_kPa)
    f_alt = relaxation_frequency_oxygen(t_K, 20.0, p_half)
    assert f_alt == pytest.approx(f_ref / 2.0, rel=1e-14)


# ---------------------------------------------------------------------------
# Absorption: analytic limits of the ISO 9613-1 formula
# ---------------------------------------------------------------------------

def _classical_alpha(f, t_C, rh, p_kPa):
    """
    The classical-plus-rotational term alone, which the standard gives in closed
    form as 8.686 * f^2 * 1.84e-11 * (p_a/p_r)^-1 * (T/T0)^0.5.
    """
    t_K = t_C + T_ZERO_C_K
    return 8.686 * f ** 2 * 1.84e-11 / (p_kPa / P_REF_kPa) * math.sqrt(t_K / T_REF_K)


def test_absorption_is_quadratic_in_frequency_at_low_frequency():
    """
    As f -> 0 both relaxation denominators tend to their constant f_r terms, so
    alpha becomes exactly proportional to f^2. The ratio alpha(2f)/alpha(f) must
    therefore approach 4.
    """
    f = 0.05  # far below f_rN ~ 9-500 Hz
    ratio = absorption_coefficient_dB_per_m(2 * f) / absorption_coefficient_dB_per_m(f)
    assert ratio == pytest.approx(4.0, rel=1e-6)


def test_relaxation_contribution_saturates_to_a_closed_form_constant():
    """
    The oxygen term contributes

        8.686 * t^-2.5 * 0.01275 * exp(-2239.1/T) * f^2 * f_rO/(f_rO^2 + f^2)

    which tends to 8.686 * t^-2.5 * 0.01275 * exp(-2239.1/T) * f_rO as f -> inf,
    and likewise for nitrogen. So alpha minus the classical term approaches an
    exactly computable constant.
    """
    t_C, rh, p = 20.0, 50.0, P_REF_kPa
    t_K = t_C + T_ZERO_C_K
    t_ratio = t_K / T_REF_K
    f_rO = relaxation_frequency_oxygen(t_K, rh, p)
    f_rN = relaxation_frequency_nitrogen(t_K, rh, p)

    expected = 8.686 * t_ratio ** -2.5 * (
        0.01275 * math.exp(-2239.1 / t_K) * f_rO
        + 0.1068 * math.exp(-3352.0 / t_K) * f_rN
    )

    f = 1e9  # far above both relaxation frequencies
    residual = absorption_coefficient_dB_per_m(f, t_C, rh, p) - _classical_alpha(f, t_C, rh, p)
    assert residual == pytest.approx(expected, rel=1e-6)


def test_relaxation_term_is_exactly_half_saturated_at_its_relaxation_frequency():
    """
    f^2 * f_r/(f_r^2 + f^2) equals f_r/2 exactly at f = f_r. Choosing an
    atmosphere where the two relaxation frequencies are far apart lets the
    oxygen term be checked at its own half-power point.
    """
    t_C, rh, p = 20.0, 50.0, P_REF_kPa
    t_K = t_C + T_ZERO_C_K
    t_ratio = t_K / T_REF_K
    f_rO = relaxation_frequency_oxygen(t_K, rh, p)
    f_rN = relaxation_frequency_nitrogen(t_K, rh, p)

    def relaxation_only(f):
        return absorption_coefficient_dB_per_m(f, t_C, rh, p) - _classical_alpha(f, t_C, rh, p)

    coeff_O = 8.686 * t_ratio ** -2.5 * 0.01275 * math.exp(-2239.1 / t_K)
    coeff_N = 8.686 * t_ratio ** -2.5 * 0.1068 * math.exp(-3352.0 / t_K)

    f = f_rO
    expected = (
        coeff_O * f ** 2 * f_rO / (f_rO ** 2 + f ** 2)
        + coeff_N * f ** 2 * f_rN / (f_rN ** 2 + f ** 2)
    )
    assert relaxation_only(f) == pytest.approx(expected, rel=1e-9)
    # The oxygen part of that is exactly half its own saturated value.
    assert coeff_O * f ** 2 * f_rO / (f_rO ** 2 + f ** 2) == pytest.approx(
        coeff_O * f_rO / 2.0, rel=1e-14
    )


def test_reciprocal_form_of_the_relaxation_denominator_matches():
    """
    (f_r + f^2/f_r)^-1 and f_r/(f_r^2 + f^2) are the same quantity written two
    ways. Evaluating the whole formula through the second form is an independent
    transcription of the standard.
    """
    t_C, rh, p = 12.0, 73.0, 98.4
    t_K = t_C + T_ZERO_C_K
    t_ratio = t_K / T_REF_K
    f_rO = relaxation_frequency_oxygen(t_K, rh, p)
    f_rN = relaxation_frequency_nitrogen(t_K, rh, p)

    f = BAND_CENTRES
    expected = 8.686 * f ** 2 * (
        1.84e-11 / (p / P_REF_kPa) * math.sqrt(t_ratio)
        + t_ratio ** -2.5 * (
            0.01275 * math.exp(-2239.1 / t_K) * f_rO / (f_rO ** 2 + f ** 2)
            + 0.1068 * math.exp(-3352.0 / t_K) * f_rN / (f_rN ** 2 + f ** 2)
        )
    )
    got = absorption_coefficient_dB_per_m(f, t_C, rh, p)
    assert np.allclose(got, expected, rtol=1e-12)


def test_absorption_is_positive_and_increases_with_frequency_across_the_audio_band():
    """
    Absorption has no mechanism that decreases with frequency in this range, so
    the tabulated 1/3-octave sequence must be strictly increasing.
    """
    alpha = np.atleast_1d(absorption_coefficient_dB_per_m(BAND_CENTRES, 20.0, 50.0))
    assert np.all(alpha > 0)
    assert np.all(np.diff(alpha) > 0)


def test_absorption_at_zero_frequency_is_zero():
    assert absorption_coefficient_dB_per_m(0.0) == 0.0


def test_absorption_over_a_path_is_linear_in_distance():
    """A = alpha * d, so doubling the path exactly doubles the loss."""
    a1 = absorption_dB(4000.0, 100.0)
    a2 = absorption_dB(4000.0, 200.0)
    assert a2 == pytest.approx(2.0 * a1, rel=1e-14)


def test_high_frequencies_are_absorbed_far_more_than_low():
    """
    The frequency dependence is the reason a broadband peak has no single
    absorption coefficient. Over 100 m at 20 C / 50% RH the 10 kHz band must lose
    more than an order of magnitude more than the 100 Hz band.
    """
    low = absorption_dB(100.0, 100.0)
    high = absorption_dB(10000.0, 100.0)
    assert high > 10.0 * low


def test_negative_frequency_and_distance_are_refused():
    with pytest.raises(AtmosphereError):
        absorption_coefficient_dB_per_m(-1.0)
    with pytest.raises(AtmosphereError):
        absorption_dB(1000.0, -1.0)


# ---------------------------------------------------------------------------
# Geometric spreading
# ---------------------------------------------------------------------------

def test_doubling_distance_costs_exactly_six_decibels():
    """20*log10(2) = 6.0205999... dB, the definition of spherical spreading."""
    assert geometric_spreading_dB(2.0, 1.0) == pytest.approx(20.0 * math.log10(2.0), rel=1e-15)


def test_spreading_correction_is_zero_when_distances_match():
    assert geometric_spreading_dB(7.5, 7.5) == 0.0


def test_spreading_correction_is_antisymmetric():
    assert geometric_spreading_dB(5.0, 1.0) == pytest.approx(-geometric_spreading_dB(1.0, 5.0))


def test_non_positive_distances_are_refused():
    with pytest.raises(AtmosphereError):
        geometric_spreading_dB(0.0, 1.0)
    with pytest.raises(AtmosphereError):
        geometric_spreading_dB(1.0, -2.0)


# ---------------------------------------------------------------------------
# Atmosphere construction and metadata
# ---------------------------------------------------------------------------

def test_from_metadata_records_which_fields_it_had_to_default():
    class Meta:
        temperature_C = 12.0
        humidity_pct = None
        pressure_kPa = None

    air = Atmosphere.from_metadata(Meta())
    assert air.temperature_C == 12.0
    assert set(air.defaulted) == {"humidity_pct", "pressure_kPa"}
    assert not air.is_reference_conditions


def test_from_metadata_with_nothing_recorded_is_flagged_as_all_assumed():
    air = Atmosphere.from_metadata({})
    assert air.is_reference_conditions
    assert len(air.defaulted) == 3


def test_from_metadata_accepts_a_plain_dict():
    air = Atmosphere.from_metadata(
        {"temperature_C": 5.0, "humidity_pct": 80.0, "pressure_kPa": 97.0}
    )
    assert (air.temperature_C, air.humidity_pct, air.pressure_kPa) == (5.0, 80.0, 97.0)
    assert air.defaulted == ()


def test_impossible_humidity_is_refused():
    with pytest.raises(AtmosphereError):
        Atmosphere(humidity_pct=120.0)
    with pytest.raises(AtmosphereError):
        Atmosphere(pressure_kPa=-1.0)


def test_conditions_outside_the_standards_stated_range_are_reported():
    air = Atmosphere(temperature_C=60.0, humidity_pct=5.0)
    problems = air.out_of_standard_range()
    assert len(problems) == 2
    assert any("temperature" in p for p in problems)
    assert any("humidity" in p for p in problems)


# ---------------------------------------------------------------------------
# Band normalisation
# ---------------------------------------------------------------------------

def test_band_normalisation_applies_spreading_plus_per_band_absorption():
    """
    L(d_ref, f) = L(d, f) + 20*log10(d/d_ref) + alpha(f)*(d - d_ref), evaluated
    band by band. Nothing here is an approximation, so it is checked exactly.
    """
    air = Atmosphere(temperature_C=15.0, humidity_pct=60.0, pressure_kPa=99.0)
    levels = np.full(BAND_CENTRES.shape, 120.0)
    d, d_ref = 5.0, 1.0

    result = air.normalise_band_levels(
        levels, BAND_CENTRES, measured_distance_m=d, reference_distance_m=d_ref
    )
    assert result.valid
    expected = (
        levels
        + 20.0 * math.log10(d / d_ref)
        + np.atleast_1d(air.absorption_coefficient_dB_per_m(BAND_CENTRES)) * (d - d_ref)
    )
    assert np.allclose(result.levels_dB, expected, rtol=1e-14)
    assert result.absorption_applied


def test_band_normalisation_round_trips_exactly():
    """
    Normalising 5 m -> 1 m and then 1 m -> 5 m must return the original spectrum:
    the two corrections are exact negatives of each other.
    """
    air = Atmosphere(temperature_C=25.0, humidity_pct=40.0)
    levels = np.linspace(130.0, 100.0, BAND_CENTRES.size)

    inward = air.normalise_band_levels(
        levels, BAND_CENTRES, measured_distance_m=5.0, reference_distance_m=1.0
    )
    outward = air.normalise_band_levels(
        inward.levels_dB, BAND_CENTRES, measured_distance_m=1.0, reference_distance_m=5.0
    )
    assert np.allclose(outward.levels_dB, levels, atol=1e-12)


def test_normalising_to_the_same_distance_changes_nothing():
    air = Atmosphere()
    levels = np.full(BAND_CENTRES.shape, 115.0)
    result = air.normalise_band_levels(
        levels, BAND_CENTRES, measured_distance_m=3.0, reference_distance_m=3.0
    )
    assert result.valid
    assert np.allclose(result.levels_dB, levels, atol=1e-15)
    assert result.geometric_dB == 0.0


def test_normalisation_without_a_recorded_distance_is_refused():
    """The governing case: no distance means no correction, not a guessed one."""
    air = Atmosphere()
    result = air.normalise_band_levels(
        np.full(BAND_CENTRES.shape, 120.0), BAND_CENTRES, measured_distance_m=None
    )
    assert not result.valid
    assert result.levels_dB.size == 0
    assert "mic_distance_m" in result.refusal


def test_mismatched_band_vectors_are_refused():
    air = Atmosphere()
    result = air.normalise_band_levels(
        np.zeros(5), BAND_CENTRES, measured_distance_m=2.0
    )
    assert not result.valid
    assert "filter bank" in result.refusal


def test_defaulted_weather_is_carried_into_the_assumptions():
    air = Atmosphere.from_metadata({})
    result = air.normalise_band_levels(
        np.full(BAND_CENTRES.shape, 120.0), BAND_CENTRES, measured_distance_m=2.0
    )
    assert result.valid
    assert any("weather not recorded" in a for a in result.assumptions)


# ---------------------------------------------------------------------------
# The nonlinearity guard
# ---------------------------------------------------------------------------

def test_nonlinear_limit_corresponds_to_the_documented_level():
    """
    dp = M * gamma * p0 = 0.01 * 1.4 * 101325 = 1418.55 Pa, which is
    20*log10(1418.55/20e-6) = 157.0 dB re 20 uPa.
    """
    dp = NONLINEAR_MACH_LIMIT * GAMMA_AIR * P_ATM_PA
    assert dp == pytest.approx(1418.55, rel=1e-6)
    assert NONLINEAR_PEAK_dB == pytest.approx(20.0 * math.log10(dp / 20e-6), rel=1e-12)
    assert 156.9 < NONLINEAR_PEAK_dB < 157.1


def test_peak_below_the_linear_limit_normalises():
    air = Atmosphere()
    result = air.normalise_band_levels(
        np.full(BAND_CENTRES.shape, 120.0), BAND_CENTRES,
        measured_distance_m=3.0, peak_Pa=1400.0,
    )
    assert result.valid


def test_peak_above_the_linear_limit_is_refused_rather_than_extrapolated():
    air = Atmosphere()
    result = air.normalise_band_levels(
        np.full(BAND_CENTRES.shape, 120.0), BAND_CENTRES,
        measured_distance_m=3.0, peak_Pa=1500.0,
    )
    assert not result.valid
    assert "Mach" in result.refusal
    assert result.levels_dB.size == 0


def test_nonlinear_refusal_can_be_overridden_but_stays_labelled():
    air = Atmosphere()
    result = air.normalise_band_levels(
        np.full(BAND_CENTRES.shape, 120.0), BAND_CENTRES,
        measured_distance_m=3.0, peak_Pa=5000.0, allow_nonlinear=True,
    )
    assert result.valid
    assert any("Mach" in w for w in result.warnings)


def test_unknown_peak_does_not_trip_the_guard():
    """An unrecorded peak is not evidence of nonlinearity."""
    air = Atmosphere()
    result = air.normalise_band_levels(
        np.full(BAND_CENTRES.shape, 120.0), BAND_CENTRES,
        measured_distance_m=3.0, peak_Pa=None,
    )
    assert result.valid


# ---------------------------------------------------------------------------
# Peak normalisation refuses to invent a broadband absorption coefficient
# ---------------------------------------------------------------------------

def test_peak_normalisation_applies_spreading_only():
    air = Atmosphere(temperature_C=30.0, humidity_pct=20.0)
    result = air.normalise_peak_level(
        150.0, measured_distance_m=10.0, reference_distance_m=1.0
    )
    assert result.valid
    assert not result.absorption_applied
    assert result.levels_dB[0] == pytest.approx(150.0 + 20.0 * math.log10(10.0), rel=1e-14)
    assert any("NOT applied" in w for w in result.warnings)


def test_peak_normalisation_refuses_without_a_distance():
    air = Atmosphere()
    result = air.normalise_peak_level(150.0, measured_distance_m=None)
    assert not result.valid


# ---------------------------------------------------------------------------
# Mach-cone crack/blast geometry
# ---------------------------------------------------------------------------

def _delay_closed_form(d, angle_deg, v, t_C=20.0):
    """
    Independent solution of the same problem, obtained by minimising the
    arrival time over the shedding instant rather than by constructing the cone:

        u = c*y / sqrt(v^2 - c^2)                (offset behind the mic)
        t_crack = (x - u)/v + v*y/(c*sqrt(v^2 - c^2))
        t_blast = d/c

    This shares no expression with the implementation beyond the inputs.
    """
    c = speed_of_sound(t_C)
    theta = math.radians(angle_deg)
    x, y = d * math.cos(theta), abs(d * math.sin(theta))
    root = math.sqrt(v * v - c * c)
    u = c * y / root
    if x - u < 0:
        return None
    t_crack = (x - u) / v + v * y / (c * root)
    return d / c - t_crack


def test_crack_blast_delay_matches_the_closed_form_solution():
    v = 2.0 * C_REF_M_PER_S  # Mach 2, so the cone half-angle is exactly 30 degrees
    for angle in (10.0, 20.0, 30.0, 45.0):
        got = crack_blast_delay_s(10.0, angle, v)
        expected = _delay_closed_form(10.0, angle, v)
        assert expected is not None
        assert got == pytest.approx(expected, rel=1e-12)


def test_crack_precedes_blast_inside_the_mach_cone():
    """Wherever a crack exists at all, it arrives first, so the delay is positive."""
    v = 2.0 * C_REF_M_PER_S
    for angle in (5.0, 15.0, 30.0, 45.0):
        delay = crack_blast_delay_s(10.0, angle, v)
        assert delay is not None and delay > 0.0


def test_subsonic_projectile_produces_no_crack():
    """Below the speed of sound there is no Mach cone, so there is nothing to label."""
    assert crack_blast_delay_s(10.0, 45.0, 300.0) is None
    assert crack_blast_delay_s(10.0, 45.0, C_REF_M_PER_S) is None


def test_microphone_outside_the_mach_cone_hears_no_crack():
    """
    At Mach 2 the cone half-angle is 30 degrees, so x_shed = x - y*tan(30) goes
    negative once the mic is far enough abeam. At 70 degrees off the line of fire
    the cone never reaches it.
    """
    v = 2.0 * C_REF_M_PER_S
    assert crack_blast_delay_s(10.0, 70.0, v) is None


def test_microphone_on_the_line_of_fire_has_no_crack_geometry():
    v = 2.0 * C_REF_M_PER_S
    assert crack_blast_delay_s(10.0, 0.0, v) is None


def test_crack_blast_delay_grows_with_distance_abeam():
    """Farther out, the path difference between cone and blast is larger."""
    v = 2.5 * C_REF_M_PER_S
    d5 = crack_blast_delay_s(5.0, 30.0, v)
    d20 = crack_blast_delay_s(20.0, 30.0, v)
    assert d5 is not None and d20 is not None
    assert d20 > d5


def test_crack_blast_delay_uses_the_recorded_temperature():
    """
    The speed of sound sets both arrival times, so a cold day and a hot day give
    measurably different delays for identical geometry.
    """
    v = 800.0
    cold = crack_blast_delay_s(10.0, 30.0, v, temperature_C=-10.0)
    hot = crack_blast_delay_s(10.0, 30.0, v, temperature_C=40.0)
    assert cold is not None and hot is not None
    assert cold != pytest.approx(hot, rel=1e-6)


# ---------------------------------------------------------------------------
# Normalising a comparison before differencing it
# ---------------------------------------------------------------------------

def _pair(n=None):
    """A reference and test spectrum with a known 12 dB separation."""
    size = BAND_CENTRES.size if n is None else n
    reference = np.full(size, 140.0)
    test = np.full(size, 128.0)
    return reference, test


def test_identical_setups_cancel_exactly_in_the_difference():
    """
    THE correctness property. When both strings were shot at the same distance
    in the same air, both corrections are identical and must cancel to the last
    bit in the difference. A normalisation that moves the insertion loss here is
    inventing a correction out of nothing.
    """
    reference, test = _pair()
    air = Atmosphere(temperature_C=17.0, humidity_pct=64.0, pressure_kPa=99.8)

    result = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=3.0, test_distance_m=3.0,
    )
    assert result.valid
    assert np.allclose(result.insertion_loss_dB, result.raw_insertion_loss_dB, atol=1e-12)
    assert np.allclose(result.shift_dB, 0.0, atol=1e-12)


def test_the_normalisation_distance_does_not_matter_when_the_air_matches():
    """
    With one atmosphere the absorption terms are common to both strings and
    cancel, so the insertion loss is the same wherever it is quoted.
    """
    reference, test = _pair()
    air = Atmosphere(temperature_C=20.0, humidity_pct=50.0)

    at_one = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=5.0, test_distance_m=5.0,
        normalisation_distance_m=1.0,
    )
    at_ten = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=5.0, test_distance_m=5.0,
        normalisation_distance_m=10.0,
    )
    assert np.allclose(at_one.insertion_loss_dB, at_ten.insertion_loss_dB, atol=1e-12)


def test_a_moved_microphone_is_removed_from_the_insertion_loss():
    """
    The reference at 1 m and the test at 1.5 m in the same air: spherical
    spreading alone accounts for 20*log10(1.5) = 3.522 dB of the apparent
    reduction, and normalising must remove exactly that.
    """
    reference, test = _pair()
    air = Atmosphere(temperature_C=20.0, humidity_pct=50.0)

    result = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=1.0, test_distance_m=1.5,
        normalisation_distance_m=1.0,
    )
    assert result.valid
    # Same air, so absorption over the differing path is the only other term.
    alpha = np.atleast_1d(air.absorption_coefficient_dB_per_m(BAND_CENTRES))
    expected_shift = -(20.0 * math.log10(1.5) + alpha * 0.5)
    assert np.allclose(result.shift_dB, expected_shift, atol=1e-12)
    # The apparent reduction shrinks, because part of it was the microphone.
    assert np.all(result.insertion_loss_dB < result.raw_insertion_loss_dB)


def test_the_raw_insertion_loss_is_always_retained():
    """A corrected number that cannot be checked against the measurement is not a record."""
    reference, test = _pair()
    air = Atmosphere()
    result = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=2.0, test_distance_m=2.0,
    )
    assert np.allclose(result.raw_insertion_loss_dB, reference - test)


def test_different_air_makes_the_quoted_distance_matter():
    """
    When the two strings were shot in different air the absorption over the
    shared path differs, so the insertion loss genuinely depends on the distance
    it is quoted at. That is physics, and the distance is therefore reported.
    """
    reference, test = _pair()
    cold = Atmosphere(temperature_C=0.0, humidity_pct=90.0)
    hot = Atmosphere(temperature_C=35.0, humidity_pct=15.0)

    at_one = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=cold, test_atmosphere=hot,
        reference_distance_m=10.0, test_distance_m=10.0,
        normalisation_distance_m=1.0,
    )
    at_five = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=cold, test_atmosphere=hot,
        reference_distance_m=10.0, test_distance_m=10.0,
        normalisation_distance_m=5.0,
    )
    assert not np.allclose(at_one.insertion_loss_dB, at_five.insertion_loss_dB, atol=1e-6)
    assert at_one.normalisation_distance_m == 1.0
    assert at_five.normalisation_distance_m == 5.0


def test_an_unrecorded_distance_refuses_the_whole_comparison():
    reference, test = _pair()
    air = Atmosphere()
    result = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=None, test_distance_m=2.0,
    )
    assert not result.valid
    assert "reference string" in result.refusal
    # The raw difference survives the refusal, so nothing is lost.
    assert result.raw_insertion_loss_dB.size == BAND_CENTRES.size


def test_a_nonlinear_peak_refuses_the_normalisation():
    reference, test = _pair()
    air = Atmosphere()
    result = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=1.0, test_distance_m=1.0,
        reference_peak_Pa=5000.0,
    )
    assert not result.valid
    assert "Mach" in result.refusal


def test_mismatched_filter_banks_are_refused():
    air = Atmosphere()
    result = normalise_insertion_loss_bands(
        np.zeros(24), np.zeros(20), BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=1.0, test_distance_m=1.0,
    )
    assert not result.valid
    assert "filter bank" in result.refusal


def test_assumed_weather_is_carried_into_the_comparison_warnings():
    reference, test = _pair()
    recorded = Atmosphere(temperature_C=12.0, humidity_pct=70.0, pressure_kPa=100.0)
    assumed = Atmosphere.from_metadata({})
    result = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=recorded, test_atmosphere=assumed,
        reference_distance_m=2.0, test_distance_m=2.0,
    )
    assert result.valid
    assert any("not recorded" in w for w in result.warnings)


def test_the_normalised_comparison_serialises():
    reference, test = _pair()
    air = Atmosphere()
    data = normalise_insertion_loss_bands(
        reference, test, BAND_CENTRES,
        reference_atmosphere=air, test_atmosphere=air,
        reference_distance_m=1.0, test_distance_m=1.5,
    ).to_dict()
    for key in ("valid", "normalisation_distance_m", "insertion_loss_dB",
                "raw_insertion_loss_dB", "shift_dB", "largest_shift_dB"):
        assert key in data
    assert len(data["insertion_loss_dB"]) == BAND_CENTRES.size


# ---------------------------------------------------------------------------
# What the atmosphere did to a measurement, and how much it matters
# ---------------------------------------------------------------------------

def test_the_reported_absorption_is_alpha_times_the_path():
    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0, pressure_kPa=100.4)
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=25.0)

    expected = np.atleast_1d(air.absorption_coefficient_dB_per_m(BAND_CENTRES)) * 25.0
    assert np.allclose(effect.absorption_dB, expected, rtol=1e-14)
    assert np.allclose(effect.alpha_dB_per_m, expected / 25.0, rtol=1e-14)


def test_the_geometric_term_is_the_spreading_from_the_reference_distance():
    air = Atmosphere()
    effect = describe_atmospheric_effect(
        BAND_CENTRES, atmosphere=air, distance_m=8.0, reference_distance_m=2.0)
    assert effect.geometric_dB == pytest.approx(20.0 * math.log10(4.0), rel=1e-14)
    assert np.allclose(effect.total_dB, effect.geometric_dB + effect.absorption_dB)


def test_absorption_is_worst_at_the_highest_band():
    """Absorption rises monotonically with frequency across the audio range."""
    air = Atmosphere()
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=50.0)
    assert effect.worst_band_Hz == BAND_CENTRES[-1]
    assert effect.worst_absorption_dB == pytest.approx(float(effect.absorption_dB[-1]))


def test_absorption_scales_exactly_with_the_path_length():
    air = Atmosphere()
    near = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=10.0)
    far = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=100.0)
    assert np.allclose(far.absorption_dB, 10.0 * near.absorption_dB, rtol=1e-13)


def test_the_atmosphere_does_not_matter_at_bench_distance():
    """
    At one metre the air takes a fraction of a decibel from even the top band,
    which is below the 0.5 dB a session is invalidated at. Saying so stops an
    operator worrying about the wrong thing.
    """
    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0)
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=1.0)
    assert not effect.matters
    assert effect.worst_absorption_dB < 0.5
    assert "not what limits this measurement" in effect.summary()


def test_the_atmosphere_matters_downrange():
    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0)
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=100.0)
    assert effect.matters
    # Over 100 m the top band loses more than 10 dB - the atmosphere is now the
    # dominant term in that band.
    assert effect.worst_absorption_dB > 10.0


def test_sensitivity_grows_with_the_path_just_as_absorption_does():
    air = Atmosphere()
    near = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=10.0)
    far = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=100.0)
    assert far.largest_sensitivity_dB == pytest.approx(
        10.0 * near.largest_sensitivity_dB, rel=1e-9)


def test_sensitivity_is_the_change_a_mis_recorded_condition_would_cause():
    """
    The humidity term must equal the difference between the absorption at the
    recorded humidity and at the perturbed one, recomputed independently.
    """
    air = Atmosphere(temperature_C=20.0, humidity_pct=50.0, pressure_kPa=P_REF_kPa)
    distance = 50.0
    effect = describe_atmospheric_effect(
        BAND_CENTRES, atmosphere=air, distance_m=distance)

    base = np.atleast_1d(absorption_coefficient_dB_per_m(
        BAND_CENTRES, 20.0, 50.0, P_REF_kPa)) * distance
    low = np.atleast_1d(absorption_coefficient_dB_per_m(
        BAND_CENTRES, 20.0, 30.0, P_REF_kPa)) * distance
    high = np.atleast_1d(absorption_coefficient_dB_per_m(
        BAND_CENTRES, 20.0, 70.0, P_REF_kPa)) * distance
    expected = np.where(np.abs(low - base) >= np.abs(high - base), low - base, high - base)

    assert np.allclose(effect.humidity_sensitivity_dB, expected, rtol=1e-12)


def test_humidity_is_the_condition_worth_recording_most_carefully():
    """
    At ordinary temperatures the oxygen relaxation frequency moves strongly with
    water vapour, so humidity dominates the uncertainty. This is the practical
    reason the operator is prompted for it.
    """
    air = Atmosphere(temperature_C=20.0, humidity_pct=50.0)
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=100.0)
    assert np.max(np.abs(effect.humidity_sensitivity_dB)) > np.max(
        np.abs(effect.pressure_sensitivity_dB))


def test_assumed_conditions_are_reported_as_a_risk():
    air = Atmosphere.from_metadata({})
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=50.0)
    assert any("assumed, not recorded" in note for note in effect.notes)


def test_an_unrecorded_distance_produces_no_absorption_claim():
    air = Atmosphere()
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=None)
    assert effect.absorption_dB.size == 0
    assert math.isnan(effect.worst_absorption_dB)
    assert not effect.matters
    assert any("was not recorded" in note for note in effect.notes)


def test_the_effect_serialises_with_alpha_in_dB_per_km():
    """dB/km is the unit ISO 9613-1 tabulates, so that is what a reader expects."""
    air = Atmosphere()
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=10.0)
    data = effect.to_dict()
    assert data["alpha_dB_per_km"][-1] == pytest.approx(
        float(effect.alpha_dB_per_m[-1]) * 1000.0, abs=0.01)
    for key in ("absorption_dB", "total_dB", "sensitivity", "matters", "worst_band_Hz"):
        assert key in data


def test_bands_beyond_the_tabulated_range_are_called_extrapolation():
    """
    A 96 kHz recording produces a 40 kHz band, four times beyond where ISO 9613-1
    states accuracy, and that is exactly where the formula grows fastest. A large
    number there must not read as a measured one.
    """
    wide = np.array([125.0, 1000.0, 8000.0, 20000.0, 40000.0])
    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0)
    effect = describe_atmospheric_effect(wide, atmosphere=air, distance_m=25.0)

    assert any("extrapolation of the formula" in n for n in effect.notes)
    assert any("40000 Hz" in n for n in effect.notes)
    # The in-range worst is stated separately so a reader has a defensible figure.
    assert any("Within the tabulated range" in n for n in effect.notes)


def test_bands_entirely_inside_the_range_raise_no_extrapolation_note():
    inside = np.array([125.0, 1000.0, 8000.0])
    air = Atmosphere()
    effect = describe_atmospheric_effect(inside, atmosphere=air, distance_m=25.0)
    assert not any("extrapolation" in n for n in effect.notes)


def test_out_of_range_conditions_are_carried_into_the_effect_notes():
    air = Atmosphere(temperature_C=60.0, humidity_pct=5.0)
    effect = describe_atmospheric_effect(BAND_CENTRES, atmosphere=air, distance_m=10.0)
    assert any("temperature" in n.lower() for n in effect.notes)
    assert any("humidity" in n.lower() for n in effect.notes)
