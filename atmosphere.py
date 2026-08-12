#!/usr/bin/env python3
"""
atmosphere.py - Atmospheric Absorption and Distance Normalisation

SASA records microphone distance, angle, temperature, humidity and barometric
pressure in TestMetadata. This module is what turns those from documentation into
correction: two sessions shot at different distances, on different days, in
different air, cannot be compared until their levels are referred to a common
geometry and a common atmosphere.

Two physical effects separate a level measured at one place from the same source
measured at another:

  Geometric spreading   A free-field spherical wave loses 20*log10(d2/d1) dB
                        between two distances. This depends only on geometry.

  Atmospheric absorption Air itself dissipates sound, strongly and very unevenly
                        with frequency: at 20 C and 50% RH a 10 kHz tone loses
                        roughly a hundred times more per metre than a 100 Hz tone.
                        Absorption is computed here from ISO 9613-1:1993.

WHAT THIS MODULE WILL AND WILL NOT DO

Absorption is a function of frequency. A broadband impulse has no single
frequency, so there is no honest single absorption coefficient for it. Therefore:

  * normalise_band_levels() corrects a 1/3-octave spectrum, applying each band's
    own absorption at its own midband frequency. This is correct and is the
    supported path.

  * normalise_peak_level() applies GEOMETRIC SPREADING ONLY and says so in the
    result. It does not invent a broadband absorption coefficient.

  * Above a documented overpressure, propagation is not linear and 1/r is simply
    the wrong law. Rather than return a confident wrong number, the normalisation
    refuses and explains. See NONLINEAR_MACH_LIMIT below.

Every result carries what was assumed, what was applied, and what was refused, so
a corrected level can never be mistaken for a measured one.

Reference:
    ISO 9613-1:1993, Acoustics - Attenuation of sound during propagation
    outdoors - Part 1: Calculation of the absorption of sound by the atmosphere.

Usage:
    from atmosphere import Atmosphere, absorption_coefficient_dB_per_m

    air = Atmosphere(temperature_C=15.0, humidity_pct=60.0, pressure_kPa=99.5)
    alpha = absorption_coefficient_dB_per_m(np.array([1000.0, 8000.0]), air)

    result = air.normalise_band_levels(
        band_levels_dB, band_frequencies_Hz,
        measured_distance_m=3.0, reference_distance_m=1.0,
    )
    if result.valid:
        print(result.levels_dB, result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---- Reference conditions (ISO 9613-1 clause 3) ----

# Reference ambient atmospheric pressure.
P_REF_kPa: float = 101.325

# Reference air temperature: 20 C expressed in kelvin.
T_REF_K: float = 293.15

# Triple-point isotherm temperature, used for saturation vapour pressure.
T_TRIPLE_K: float = 273.16

# 0 C in kelvin, for the Celsius/kelvin conversion.
T_ZERO_C_K: float = 273.15

# Speed of sound at the reference temperature, ISO 9613-1 clause 3.
C_REF_M_PER_S: float = 343.2

# Standard atmospheric pressure in pascals, for the overpressure ratio.
P_ATM_PA: float = 101325.0

# Ratio of specific heats for air.
GAMMA_AIR: float = 1.4

# Specific gas constant for dry air, J/(kg K).
R_SPECIFIC_AIR: float = 287.058

# ---- Validity limits ----

# ISO 9613-1 states its accuracy over these ranges. Outside them the formula is
# an extrapolation and is reported as such rather than silently trusted.
ISO_9613_TEMPERATURE_RANGE_C = (-20.0, 50.0)
ISO_9613_HUMIDITY_RANGE_PCT = (10.0, 100.0)
ISO_9613_FREQUENCY_RANGE_HZ = (50.0, 10000.0)

# Acoustic Mach number above which propagation is not linear.
#
# The acoustic Mach number of a wave of peak overpressure dp is
#
#     M = dp / (gamma * p0)
#
# because the excess wave speed is dp/(rho*c) and rho*c^2 = gamma*p0. Weak-shock
# propagation loses amplitude to the shock itself at a rate that grows with M, so
# the 1/r law of linear acoustics understates the true decay. At M = 0.01 the
# wave carries roughly a percent of ambient pressure; that is the conventional
# edge of "acoustically small", and it corresponds to
#
#     dp = 0.01 * 1.4 * 101325 Pa = 1418.6 Pa   ->   about 157 dB re 20 uPa.
#
# This is a physical criterion, not a tuned parameter: it is fixed by gamma and
# by ambient pressure, and nothing in this codebase is fitted to it.
NONLINEAR_MACH_LIMIT: float = 0.01


def _peak_dB_for_mach(mach: float) -> float:
    """Peak level in dB re 20 uPa corresponding to an acoustic Mach number."""
    return 20.0 * math.log10(mach * GAMMA_AIR * P_ATM_PA / 20e-6)


# Peak level at which the nonlinear guard trips, for messages. ~157.0 dB.
NONLINEAR_PEAK_dB: float = _peak_dB_for_mach(NONLINEAR_MACH_LIMIT)


class AtmosphereError(ValueError):
    """Raised when atmospheric inputs are physically impossible."""


# ---- Saturation vapour pressure and water vapour concentration ----

def saturation_vapour_pressure_ratio(temperature_K: float) -> float:
    """
    Saturation water vapour pressure as a fraction of the reference pressure.

    ISO 9613-1 Annex B:

        p_sat / p_r = 10 ** ( -6.8346 * (T_01/T)**1.261 + 4.6151 )

    with T_01 the triple-point isotherm temperature, 273.16 K.

    Args:
        temperature_K: Ambient temperature in kelvin.

    Returns:
        p_sat / p_r, dimensionless.
    """
    if temperature_K <= 0:
        raise AtmosphereError(f"temperature must be above absolute zero, got {temperature_K} K")
    exponent = -6.8346 * (T_TRIPLE_K / temperature_K) ** 1.261 + 4.6151
    return float(10.0 ** exponent)


def molar_water_vapour_pct(
    temperature_K: float,
    humidity_pct: float,
    pressure_kPa: float,
) -> float:
    """
    Molar concentration of water vapour, as a percentage.

    ISO 9613-1 equation (B.1):

        h = h_rel * (p_sat / p_r) / (p_a / p_r)

    This is the quantity the relaxation frequencies actually depend on. Relative
    humidity alone is not enough: the same relative humidity at a different
    temperature or pressure is a different amount of water.

    Args:
        temperature_K: Ambient temperature in kelvin.
        humidity_pct: Relative humidity, percent.
        pressure_kPa: Ambient atmospheric pressure in kilopascals.

    Returns:
        Molar concentration of water vapour in percent.
    """
    if pressure_kPa <= 0:
        raise AtmosphereError(f"pressure must be positive, got {pressure_kPa} kPa")
    p_sat_ratio = saturation_vapour_pressure_ratio(temperature_K)
    return float(humidity_pct * p_sat_ratio / (pressure_kPa / P_REF_kPa))


def relaxation_frequency_oxygen(
    temperature_K: float,
    humidity_pct: float,
    pressure_kPa: float,
) -> float:
    """
    Oxygen relaxation frequency in Hz, ISO 9613-1 equation (3).

        f_rO = (p_a/p_r) * ( 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h) )
    """
    h = molar_water_vapour_pct(temperature_K, humidity_pct, pressure_kPa)
    p_ratio = pressure_kPa / P_REF_kPa
    return float(p_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)))


def relaxation_frequency_nitrogen(
    temperature_K: float,
    humidity_pct: float,
    pressure_kPa: float,
) -> float:
    """
    Nitrogen relaxation frequency in Hz, ISO 9613-1 equation (4).

        f_rN = (p_a/p_r) * (T/T_0)**-0.5
               * ( 9 + 280 * h * exp( -4.170 * ((T/T_0)**(-1/3) - 1) ) )
    """
    h = molar_water_vapour_pct(temperature_K, humidity_pct, pressure_kPa)
    p_ratio = pressure_kPa / P_REF_kPa
    t_ratio = temperature_K / T_REF_K
    return float(
        p_ratio
        * t_ratio ** -0.5
        * (9.0 + 280.0 * h * math.exp(-4.170 * (t_ratio ** (-1.0 / 3.0) - 1.0)))
    )


def speed_of_sound(temperature_C: float) -> float:
    """
    Speed of sound in air, ISO 9613-1 clause 3.

        c = 343.2 * sqrt(T / T_0)

    At the reference temperature of 20 C this returns exactly 343.2 m/s.

    Args:
        temperature_C: Ambient temperature in degrees Celsius.

    Returns:
        Speed of sound in m/s.
    """
    t_K = temperature_C + T_ZERO_C_K
    if t_K <= 0:
        raise AtmosphereError(f"temperature must be above absolute zero, got {temperature_C} C")
    return float(C_REF_M_PER_S * math.sqrt(t_K / T_REF_K))


def air_density(temperature_C: float, pressure_kPa: float) -> float:
    """
    Density of dry air from the ideal gas law, kg/m^3.

        rho = p / (R_specific * T)

    Humidity lowers density by well under 1% at ordinary conditions, which is
    below the uncertainty of everything else in the chain, so dry air is used.

    Args:
        temperature_C: Ambient temperature in degrees Celsius.
        pressure_kPa: Ambient atmospheric pressure in kilopascals.

    Returns:
        Air density in kg/m^3.
    """
    t_K = temperature_C + T_ZERO_C_K
    if t_K <= 0:
        raise AtmosphereError(f"temperature must be above absolute zero, got {temperature_C} C")
    if pressure_kPa <= 0:
        raise AtmosphereError(f"pressure must be positive, got {pressure_kPa} kPa")
    return float(pressure_kPa * 1000.0 / (R_SPECIFIC_AIR * t_K))


# ---- Absorption ----

def absorption_coefficient_dB_per_m(
    frequency_Hz: np.ndarray | float,
    temperature_C: float = 20.0,
    humidity_pct: float = 50.0,
    pressure_kPa: float = P_REF_kPa,
) -> np.ndarray | float:
    """
    Pure-tone atmospheric attenuation coefficient, ISO 9613-1 equation (3).

        alpha = 8.686 * f^2 * [
                    1.84e-11 * (p_a/p_r)^-1 * (T/T_0)^0.5
                  + (T/T_0)^-2.5 * (
                        0.01275 * exp(-2239.1/T) * (f_rO + f^2/f_rO)^-1
                      + 0.1068 * exp(-3352.0/T) * (f_rN + f^2/f_rN)^-1
                    )
                ]

    The first term is the classical-plus-rotational contribution, which rises as
    f^2 without limit. The other two are the vibrational relaxation of oxygen and
    of nitrogen; each saturates once f is well above its relaxation frequency.

    Args:
        frequency_Hz: Frequency or array of frequencies in Hz.
        temperature_C: Ambient temperature in degrees Celsius.
        humidity_pct: Relative humidity, percent.
        pressure_kPa: Ambient atmospheric pressure in kilopascals.

    Returns:
        Attenuation coefficient in dB/m, same shape as frequency_Hz.
    """
    f = np.asarray(frequency_Hz, dtype=np.float64)
    scalar_input = f.ndim == 0

    t_K = temperature_C + T_ZERO_C_K
    if t_K <= 0:
        raise AtmosphereError(f"temperature must be above absolute zero, got {temperature_C} C")
    if pressure_kPa <= 0:
        raise AtmosphereError(f"pressure must be positive, got {pressure_kPa} kPa")
    if np.any(f < 0):
        raise AtmosphereError("frequency must be non-negative")

    p_ratio = pressure_kPa / P_REF_kPa
    t_ratio = t_K / T_REF_K

    f_rO = relaxation_frequency_oxygen(t_K, humidity_pct, pressure_kPa)
    f_rN = relaxation_frequency_nitrogen(t_K, humidity_pct, pressure_kPa)

    classical = 1.84e-11 / p_ratio * math.sqrt(t_ratio)
    oxygen = 0.01275 * math.exp(-2239.1 / t_K) / (f_rO + f ** 2 / f_rO)
    nitrogen = 0.1068 * math.exp(-3352.0 / t_K) / (f_rN + f ** 2 / f_rN)

    alpha = 8.686 * f ** 2 * (classical + t_ratio ** -2.5 * (oxygen + nitrogen))
    return float(alpha) if scalar_input else alpha


def absorption_dB(
    frequency_Hz: np.ndarray | float,
    distance_m: float,
    temperature_C: float = 20.0,
    humidity_pct: float = 50.0,
    pressure_kPa: float = P_REF_kPa,
) -> np.ndarray | float:
    """
    Total atmospheric absorption over a path, in dB.

        A = alpha(f) * d

    Args:
        frequency_Hz: Frequency or array of frequencies in Hz.
        distance_m: Path length in metres.
        temperature_C: Ambient temperature in degrees Celsius.
        humidity_pct: Relative humidity, percent.
        pressure_kPa: Ambient atmospheric pressure in kilopascals.

    Returns:
        Absorption in dB over the path.
    """
    if distance_m < 0:
        raise AtmosphereError(f"distance must be non-negative, got {distance_m} m")
    alpha = absorption_coefficient_dB_per_m(
        frequency_Hz, temperature_C, humidity_pct, pressure_kPa
    )
    return alpha * distance_m


def geometric_spreading_dB(measured_distance_m: float, reference_distance_m: float) -> float:
    """
    Free-field spherical spreading correction between two distances, in dB.

        dL = 20 * log10(d_measured / d_reference)

    Add this to a level measured at d_measured to obtain the level at
    d_reference. It is positive when the measurement was made farther away than
    the reference, because the reference point is louder.

    Args:
        measured_distance_m: Distance the level was measured at.
        reference_distance_m: Distance to refer the level to.

    Returns:
        Correction in dB, to be ADDED to the measured level.
    """
    if measured_distance_m <= 0 or reference_distance_m <= 0:
        raise AtmosphereError(
            f"distances must be positive, got measured={measured_distance_m} m, "
            f"reference={reference_distance_m} m"
        )
    return float(20.0 * math.log10(measured_distance_m / reference_distance_m))


# ---- Results ----

@dataclass
class NormalisationResult:
    """
    Outcome of a distance normalisation.

    A corrected level is only usable alongside the record of what was done to it,
    so the correction terms, the assumptions and any refusal travel with the
    numbers rather than being discarded at the call site.
    """
    valid: bool
    levels_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    geometric_dB: float = 0.0
    absorption_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    measured_distance_m: float = 0.0
    reference_distance_m: float = 0.0
    absorption_applied: bool = False
    refusal: str = ""
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    @property
    def total_correction_dB(self) -> np.ndarray:
        """Total dB added to each input level."""
        if self.absorption_dB.size:
            return self.geometric_dB + self.absorption_dB
        return np.array([self.geometric_dB])

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "levels_dB": [round(float(x), 2) for x in self.levels_dB],
            "geometric_dB": round(self.geometric_dB, 3),
            "absorption_dB": [round(float(x), 3) for x in self.absorption_dB],
            "measured_distance_m": self.measured_distance_m,
            "reference_distance_m": self.reference_distance_m,
            "absorption_applied": self.absorption_applied,
            "refusal": self.refusal,
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }

    def summary(self) -> str:
        if not self.valid:
            return f"  Distance normalisation REFUSED: {self.refusal}"
        lines = [
            f"  Normalised {self.measured_distance_m:g} m -> "
            f"{self.reference_distance_m:g} m",
            f"    Geometric spreading: {self.geometric_dB:+.2f} dB",
        ]
        if self.absorption_applied and self.absorption_dB.size:
            lo, hi = float(self.absorption_dB.min()), float(self.absorption_dB.max())
            lines.append(f"    Atmospheric absorption: {lo:+.2f} to {hi:+.2f} dB across bands")
        else:
            lines.append("    Atmospheric absorption: NOT applied (needs a spectrum)")
        for a in self.assumptions:
            lines.append(f"    Assumes: {a}")
        for w in self.warnings:
            lines.append(f"    WARNING: {w}")
        return "\n".join(lines)


# ---- Atmosphere ----

@dataclass
class Atmosphere:
    """
    The state of the air a measurement was made in.

    Defaults are the ISO 9613-1 reference conditions. `from_metadata` builds one
    from a TestMetadata record and reports which fields had to be defaulted, so a
    correction computed from assumed weather is never mistaken for one computed
    from recorded weather.
    """
    temperature_C: float = 20.0
    humidity_pct: float = 50.0
    pressure_kPa: float = P_REF_kPa
    # Fields that were not supplied and so took the reference value.
    defaulted: tuple = ()

    def __post_init__(self) -> None:
        if self.temperature_C + T_ZERO_C_K <= 0:
            raise AtmosphereError(
                f"temperature must be above absolute zero, got {self.temperature_C} C"
            )
        if not (0.0 <= self.humidity_pct <= 100.0):
            raise AtmosphereError(
                f"humidity_pct must be 0-100, got {self.humidity_pct}"
            )
        if self.pressure_kPa <= 0:
            raise AtmosphereError(f"pressure must be positive, got {self.pressure_kPa} kPa")

    # -- construction --

    @classmethod
    def from_metadata(cls, metadata) -> "Atmosphere":
        """
        Build from a TestMetadata (or any object/dict with the same field names).

        Missing fields fall back to the ISO reference conditions and are listed in
        `defaulted`. Nothing is invented silently.
        """
        def get(name):
            if isinstance(metadata, dict):
                return metadata.get(name)
            return getattr(metadata, name, None)

        defaulted: List[str] = []
        temperature = get("temperature_C")
        humidity = get("humidity_pct")
        pressure = get("pressure_kPa")

        if temperature is None:
            temperature = 20.0
            defaulted.append("temperature_C")
        if humidity is None:
            humidity = 50.0
            defaulted.append("humidity_pct")
        if pressure is None:
            pressure = P_REF_kPa
            defaulted.append("pressure_kPa")

        return cls(
            temperature_C=float(temperature),
            humidity_pct=float(humidity),
            pressure_kPa=float(pressure),
            defaulted=tuple(defaulted),
        )

    # -- derived quantities --

    @property
    def temperature_K(self) -> float:
        return self.temperature_C + T_ZERO_C_K

    @property
    def speed_of_sound_m_per_s(self) -> float:
        return speed_of_sound(self.temperature_C)

    @property
    def density_kg_per_m3(self) -> float:
        return air_density(self.temperature_C, self.pressure_kPa)

    @property
    def is_reference_conditions(self) -> bool:
        """Whether every field was defaulted, i.e. no weather was actually recorded."""
        return len(self.defaulted) == 3

    def absorption_coefficient_dB_per_m(self, frequency_Hz) -> np.ndarray | float:
        """Attenuation coefficient at this atmosphere, dB/m."""
        return absorption_coefficient_dB_per_m(
            frequency_Hz, self.temperature_C, self.humidity_pct, self.pressure_kPa
        )

    def absorption_dB(self, frequency_Hz, distance_m: float) -> np.ndarray | float:
        """Total absorption over a path at this atmosphere, dB."""
        return absorption_dB(
            frequency_Hz, distance_m,
            self.temperature_C, self.humidity_pct, self.pressure_kPa,
        )

    # -- validity --

    def out_of_standard_range(self) -> List[str]:
        """Conditions outside the range ISO 9613-1 states its accuracy for."""
        out: List[str] = []
        lo, hi = ISO_9613_TEMPERATURE_RANGE_C
        if not (lo <= self.temperature_C <= hi):
            out.append(
                f"temperature {self.temperature_C:g} C is outside the {lo:g} to {hi:g} C "
                f"range ISO 9613-1 states its accuracy over"
            )
        lo, hi = ISO_9613_HUMIDITY_RANGE_PCT
        if not (lo <= self.humidity_pct <= hi):
            out.append(
                f"humidity {self.humidity_pct:g}% is outside the {lo:g} to {hi:g}% "
                f"range ISO 9613-1 states its accuracy over"
            )
        return out

    def _shared_preamble(
        self,
        measured_distance_m: Optional[float],
        reference_distance_m: float,
    ) -> Optional[str]:
        """Refusal reason common to every normalisation, or None if it can proceed."""
        if measured_distance_m is None:
            return (
                "mic_distance_m was not recorded. A level cannot be referred to another "
                "distance without knowing the distance it was measured at."
            )
        if measured_distance_m <= 0:
            return f"mic_distance_m must be positive, got {measured_distance_m} m"
        if reference_distance_m <= 0:
            return f"reference distance must be positive, got {reference_distance_m} m"
        return None

    def _assumptions(self) -> List[str]:
        out = ["free-field spherical spreading (no ground reflection or barrier)"]
        if self.defaulted:
            out.append(
                "weather not recorded; ISO reference conditions assumed for "
                + ", ".join(self.defaulted)
            )
        return out

    # -- normalisation --

    def normalise_band_levels(
        self,
        band_levels_dB: np.ndarray,
        band_frequencies_Hz: np.ndarray,
        *,
        measured_distance_m: Optional[float],
        reference_distance_m: float = 1.0,
        peak_Pa: Optional[float] = None,
        allow_nonlinear: bool = False,
    ) -> NormalisationResult:
        """
        Refer a 1/3-octave spectrum measured at one distance to another distance.

        Each band is corrected with its own absorption coefficient evaluated at its
        own midband frequency, which is the only defensible way to apply a
        frequency-dependent loss to band data:

            L(d_ref, f) = L(d, f) + 20*log10(d/d_ref) + alpha(f) * (d - d_ref)

        Args:
            band_levels_dB: Band levels measured at `measured_distance_m`.
            band_frequencies_Hz: Midband frequencies, same length.
            measured_distance_m: Distance the levels were measured at.
            reference_distance_m: Distance to refer them to.
            peak_Pa: Peak overpressure of the event, if known, for the
                     nonlinearity check.
            allow_nonlinear: Proceed even when propagation was not linear. The
                             result is still labelled.

        Returns:
            NormalisationResult. Check `.valid` before using `.levels_dB`.
        """
        levels = np.asarray(band_levels_dB, dtype=np.float64)
        freqs = np.asarray(band_frequencies_Hz, dtype=np.float64)

        refusal = self._shared_preamble(measured_distance_m, reference_distance_m)
        if refusal:
            return NormalisationResult(valid=False, refusal=refusal)
        if levels.shape != freqs.shape:
            return NormalisationResult(
                valid=False,
                refusal=(
                    f"band levels have shape {levels.shape} but band frequencies have "
                    f"shape {freqs.shape}; they must describe the same filter bank"
                ),
            )
        if levels.size == 0:
            return NormalisationResult(valid=False, refusal="no band levels supplied")

        warnings = list(self.out_of_standard_range())

        nonlinear = _nonlinearity_check(peak_Pa)
        if nonlinear and not allow_nonlinear:
            return NormalisationResult(valid=False, refusal=nonlinear)
        if nonlinear:
            warnings.append(nonlinear + " Proceeding was explicitly requested.")

        out_of_band = freqs[(freqs > 0) & (
            (freqs < ISO_9613_FREQUENCY_RANGE_HZ[0]) | (freqs > ISO_9613_FREQUENCY_RANGE_HZ[1])
        )]
        if out_of_band.size:
            warnings.append(
                f"{out_of_band.size} band(s) lie outside the "
                f"{ISO_9613_FREQUENCY_RANGE_HZ[0]:g}-{ISO_9613_FREQUENCY_RANGE_HZ[1]:g} Hz "
                f"range ISO 9613-1 tabulates; their absorption is an extrapolation"
            )

        geometric = geometric_spreading_dB(measured_distance_m, reference_distance_m)
        path_difference_m = measured_distance_m - reference_distance_m
        absorption = self.absorption_coefficient_dB_per_m(freqs) * path_difference_m

        return NormalisationResult(
            valid=True,
            levels_dB=levels + geometric + absorption,
            geometric_dB=geometric,
            absorption_dB=np.asarray(absorption, dtype=np.float64),
            measured_distance_m=float(measured_distance_m),
            reference_distance_m=float(reference_distance_m),
            absorption_applied=True,
            warnings=warnings,
            assumptions=self._assumptions(),
        )

    def normalise_peak_level(
        self,
        peak_dB: float,
        *,
        measured_distance_m: Optional[float],
        reference_distance_m: float = 1.0,
        peak_Pa: Optional[float] = None,
        allow_nonlinear: bool = False,
    ) -> NormalisationResult:
        """
        Refer a broadband peak level to another distance, GEOMETRICALLY ONLY.

        A peak has no single frequency, so no single absorption coefficient
        applies to it. This applies spherical spreading and reports absorption as
        not applied. For a spectrum, use normalise_band_levels(), which corrects
        each band at its own frequency.

        Args:
            peak_dB: Peak level measured at `measured_distance_m`.
            measured_distance_m: Distance the level was measured at.
            reference_distance_m: Distance to refer it to.
            peak_Pa: Peak overpressure, for the nonlinearity check. When omitted
                     the check falls back to `peak_dB`, which is only meaningful
                     if that level is calibrated dB SPL.
            allow_nonlinear: Proceed even when propagation was not linear.

        Returns:
            NormalisationResult with a single-element `levels_dB`.
        """
        refusal = self._shared_preamble(measured_distance_m, reference_distance_m)
        if refusal:
            return NormalisationResult(valid=False, refusal=refusal)

        warnings = list(self.out_of_standard_range())

        nonlinear = _nonlinearity_check(peak_Pa)
        if nonlinear and not allow_nonlinear:
            return NormalisationResult(valid=False, refusal=nonlinear)
        if nonlinear:
            warnings.append(nonlinear + " Proceeding was explicitly requested.")

        geometric = geometric_spreading_dB(measured_distance_m, reference_distance_m)
        warnings.append(
            "Atmospheric absorption was NOT applied: a broadband peak has no single "
            "frequency. The corrected level is therefore an upper bound when "
            "normalising inward and a lower bound when normalising outward."
        )

        return NormalisationResult(
            valid=True,
            levels_dB=np.array([peak_dB + geometric], dtype=np.float64),
            geometric_dB=geometric,
            measured_distance_m=float(measured_distance_m),
            reference_distance_m=float(reference_distance_m),
            absorption_applied=False,
            warnings=warnings,
            assumptions=self._assumptions(),
        )

    def to_dict(self) -> dict:
        return {
            "temperature_C": self.temperature_C,
            "humidity_pct": self.humidity_pct,
            "pressure_kPa": self.pressure_kPa,
            "speed_of_sound_m_per_s": round(self.speed_of_sound_m_per_s, 2),
            "density_kg_per_m3": round(self.density_kg_per_m3, 4),
            "defaulted": list(self.defaulted),
            "out_of_standard_range": self.out_of_standard_range(),
        }

    def summary(self) -> str:
        recorded = "assumed" if self.is_reference_conditions else "recorded"
        lines = [
            f"  Atmosphere ({recorded}): {self.temperature_C:g} C, "
            f"{self.humidity_pct:g}% RH, {self.pressure_kPa:g} kPa",
            f"    Speed of sound: {self.speed_of_sound_m_per_s:.1f} m/s",
        ]
        if self.defaulted:
            lines.append("    Defaulted to ISO reference: " + ", ".join(self.defaulted))
        for w in self.out_of_standard_range():
            lines.append(f"    WARNING: {w}")
        return "\n".join(lines)


# ---- What the atmosphere did to this measurement ----

# How far each condition is perturbed when reporting sensitivity.
#
# These are not error bars on the physics; they are the size of the mistake an
# operator actually makes. A pocket thermometer left in the sun reads a few
# degrees high, a hygrometer that has not been calibrated is out by tens of
# percent, and a barometer set to sea level rather than field elevation is out by
# a couple of kilopascals. Reporting the result's sensitivity to each answers the
# question "does it matter that I guessed the humidity?".
SENSITIVITY_TEMPERATURE_C: float = 5.0
SENSITIVITY_HUMIDITY_PCT: float = 20.0
SENSITIVITY_PRESSURE_kPa: float = 2.0


@dataclass
class AtmosphericEffect:
    """
    What the air between the muzzle and the microphone did to the measurement.

    This is the answer to "how much of what I measured is the atmosphere?" - a
    question that has a different answer in every band, and one that a single
    broadband number cannot express.
    """
    frequencies_Hz: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    alpha_dB_per_m: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    absorption_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    geometric_dB: float = 0.0
    distance_m: float = float("nan")
    reference_distance_m: float = 1.0
    temperature_sensitivity_dB: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False)
    humidity_sensitivity_dB: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False)
    pressure_sensitivity_dB: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False)
    atmosphere: Optional["Atmosphere"] = None
    notes: List[str] = field(default_factory=list)

    @property
    def total_dB(self) -> np.ndarray:
        """Total propagation loss per band from the reference distance."""
        if self.absorption_dB.size:
            return self.geometric_dB + self.absorption_dB
        return np.array([])

    @property
    def worst_band_Hz(self) -> float:
        """Frequency at which the atmosphere took the most out of the signal."""
        if not self.absorption_dB.size:
            return float("nan")
        return float(self.frequencies_Hz[int(np.argmax(self.absorption_dB))])

    @property
    def worst_absorption_dB(self) -> float:
        if not self.absorption_dB.size:
            return float("nan")
        return float(np.max(self.absorption_dB))

    @property
    def largest_sensitivity_dB(self) -> float:
        """
        The biggest change any single mis-recorded condition would cause.

        This is the number that decides whether the weather had to be recorded
        accurately for THIS measurement, or whether it barely mattered.
        """
        candidates = [
            arr for arr in (self.temperature_sensitivity_dB,
                            self.humidity_sensitivity_dB,
                            self.pressure_sensitivity_dB)
            if arr.size
        ]
        if not candidates:
            return float("nan")
        return float(max(np.max(np.abs(arr)) for arr in candidates))

    @property
    def matters(self) -> bool:
        """
        Whether the atmosphere is worth arguing about for this measurement.

        Half a decibel is the drift limit a session is invalidated at, so an
        atmospheric term smaller than that is not what is wrong with the result.
        """
        worst = self.worst_absorption_dB
        return bool(math.isfinite(worst) and worst >= 0.5)

    def to_dict(self) -> dict:
        def arr(values, digits=3):
            return [round(float(v), digits) for v in values]
        return {
            "distance_m": None if math.isnan(self.distance_m) else self.distance_m,
            "reference_distance_m": self.reference_distance_m,
            "frequencies_Hz": arr(self.frequencies_Hz, 1),
            # 4 decimals, not 2. This coefficient spans five orders of
            # magnitude across the audio range -- 0.0136 dB/km at 20 Hz to
            # 545 dB/km at 20 kHz -- so a fixed 2 decimal places is 26 % error
            # at the bottom of the range and none at the top. On a log axis
            # that lands as a visible staircase through the first two decades.
            "alpha_dB_per_km": arr(self.alpha_dB_per_m * 1000.0, 4),
            "absorption_dB": arr(self.absorption_dB, 3),
            "geometric_dB": round(self.geometric_dB, 3),
            "total_dB": arr(self.total_dB, 3),
            "worst_band_Hz": (
                None if math.isnan(self.worst_band_Hz) else self.worst_band_Hz
            ),
            "worst_absorption_dB": (
                None if math.isnan(self.worst_absorption_dB)
                else round(self.worst_absorption_dB, 3)
            ),
            "sensitivity": {
                "temperature_delta_C": SENSITIVITY_TEMPERATURE_C,
                "humidity_delta_pct": SENSITIVITY_HUMIDITY_PCT,
                "pressure_delta_kPa": SENSITIVITY_PRESSURE_kPa,
                "temperature_dB": arr(self.temperature_sensitivity_dB, 3),
                "humidity_dB": arr(self.humidity_sensitivity_dB, 3),
                "pressure_dB": arr(self.pressure_sensitivity_dB, 3),
                "largest_dB": (
                    None if math.isnan(self.largest_sensitivity_dB)
                    else round(self.largest_sensitivity_dB, 3)
                ),
            },
            "matters": self.matters,
            "atmosphere": self.atmosphere.to_dict() if self.atmosphere else None,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        if not self.absorption_dB.size:
            return "  Atmospheric effect: not computed (no path length recorded)"
        lines = [
            f"  Atmospheric effect over {self.distance_m:g} m",
            f"    Absorption ranges {float(np.min(self.absorption_dB)):.2f} to "
            f"{self.worst_absorption_dB:.2f} dB across the bands, worst at "
            f"{self.worst_band_Hz:.0f} Hz",
        ]
        if math.isfinite(self.largest_sensitivity_dB):
            lines.append(
                f"    Getting a condition wrong by "
                f"{SENSITIVITY_TEMPERATURE_C:g} C, "
                f"{SENSITIVITY_HUMIDITY_PCT:g}% RH or "
                f"{SENSITIVITY_PRESSURE_kPa:g} kPa would move a band by up to "
                f"{self.largest_sensitivity_dB:.2f} dB"
            )
        if not self.matters:
            lines.append(
                "    At this distance the atmosphere took less than 0.5 dB from every "
                "band, so it is not what limits this measurement."
            )
        for note in self.notes:
            lines.append(f"    {note}")
        return "\n".join(lines)


def describe_atmospheric_effect(
    frequencies_Hz: np.ndarray,
    *,
    atmosphere: "Atmosphere",
    distance_m: Optional[float],
    reference_distance_m: float = 1.0,
) -> AtmosphericEffect:
    """
    Quantify what the air did to a measurement, band by band.

    Reports the absorption the path actually cost, and how much that answer would
    change if each recorded condition were wrong by a realistic amount. The
    second part is what tells an operator whether the weather had to be measured
    or merely noted.

    Args:
        frequencies_Hz: Band centre frequencies.
        atmosphere: The air the measurement was made in.
        distance_m: Path length from source to microphone.
        reference_distance_m: Distance the geometric term is referred to.

    Returns:
        AtmosphericEffect. Empty arrays when the distance was not recorded.
    """
    freqs = np.asarray(frequencies_Hz, dtype=np.float64).ravel()
    effect = AtmosphericEffect(
        frequencies_Hz=freqs,
        atmosphere=atmosphere,
        reference_distance_m=float(reference_distance_m),
    )

    if distance_m is None or not math.isfinite(distance_m) or distance_m <= 0:
        effect.notes.append(
            "mic_distance_m was not recorded, so the absorption this measurement "
            "actually accumulated cannot be computed."
        )
        return effect
    if freqs.size == 0:
        return effect

    effect.distance_m = float(distance_m)
    alpha = np.atleast_1d(atmosphere.absorption_coefficient_dB_per_m(freqs))
    effect.alpha_dB_per_m = alpha
    effect.absorption_dB = alpha * float(distance_m)
    effect.geometric_dB = geometric_spreading_dB(
        float(distance_m), float(reference_distance_m)
    )

    # Sensitivity: recompute the absorption with each condition perturbed, and
    # report the change. The perturbation is applied in the direction that
    # increases absorption, so the figure is the magnitude of the risk.
    def shifted(temperature=None, humidity=None, pressure=None):
        candidate = Atmosphere(
            temperature_C=atmosphere.temperature_C if temperature is None else temperature,
            humidity_pct=atmosphere.humidity_pct if humidity is None else humidity,
            pressure_kPa=atmosphere.pressure_kPa if pressure is None else pressure,
        )
        return np.atleast_1d(
            candidate.absorption_coefficient_dB_per_m(freqs)
        ) * float(distance_m)

    def worst_of(low, high):
        return np.where(
            np.abs(low - effect.absorption_dB) >= np.abs(high - effect.absorption_dB),
            low - effect.absorption_dB,
            high - effect.absorption_dB,
        )

    effect.temperature_sensitivity_dB = worst_of(
        shifted(temperature=atmosphere.temperature_C - SENSITIVITY_TEMPERATURE_C),
        shifted(temperature=atmosphere.temperature_C + SENSITIVITY_TEMPERATURE_C),
    )
    effect.humidity_sensitivity_dB = worst_of(
        shifted(humidity=max(0.0, atmosphere.humidity_pct - SENSITIVITY_HUMIDITY_PCT)),
        shifted(humidity=min(100.0, atmosphere.humidity_pct + SENSITIVITY_HUMIDITY_PCT)),
    )
    effect.pressure_sensitivity_dB = worst_of(
        shifted(pressure=max(1.0, atmosphere.pressure_kPa - SENSITIVITY_PRESSURE_kPa)),
        shifted(pressure=atmosphere.pressure_kPa + SENSITIVITY_PRESSURE_kPa),
    )

    if atmosphere.defaulted:
        effect.notes.append(
            "Conditions were assumed, not recorded: "
            + ", ".join(atmosphere.defaulted)
            + ". The sensitivity figures above are the size of the resulting risk."
        )

    for problem in atmosphere.out_of_standard_range():
        effect.notes.append(problem.capitalize() + ".")

    # Bands above the tabulated range are where the formula grows fastest, so an
    # extrapolated figure there can be large AND wrong. A 40 kHz band at a
    # 96 kHz sample rate is four times beyond where the standard states accuracy.
    lo, hi = ISO_9613_FREQUENCY_RANGE_HZ
    outside = freqs[(freqs > 0) & ((freqs < lo) | (freqs > hi))]
    if outside.size:
        worst_outside = float(np.max(outside))
        effect.notes.append(
            f"{outside.size} band(s) lie outside the {lo:g}-{hi:g} Hz range "
            f"ISO 9613-1 tabulates, the highest at {worst_outside:.0f} Hz. Absorption "
            f"there is an extrapolation of the formula, and it is exactly where the "
            f"formula grows fastest, so those figures carry unquantified error."
        )
        in_range = freqs[(freqs >= lo) & (freqs <= hi)]
        if in_range.size:
            mask = (freqs >= lo) & (freqs <= hi)
            effect.notes.append(
                f"Within the tabulated range the worst absorption is "
                f"{float(np.max(effect.absorption_dB[mask])):.2f} dB at "
                f"{float(freqs[mask][int(np.argmax(effect.absorption_dB[mask]))]):.0f} Hz."
            )
    return effect


@dataclass
class NormalisedComparison:
    """
    A reference/test band pair referred to a common distance and each one's own
    atmosphere, so the difference between them is the suppressor rather than the
    setup.

    Both the raw and the normalised insertion loss are carried. The raw figure is
    never discarded: a corrected number that cannot be checked against what was
    actually measured is not a measurement record.
    """
    valid: bool
    frequencies_Hz: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    reference_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    test_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    insertion_loss_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    raw_insertion_loss_dB: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    normalisation_distance_m: float = 1.0
    reference_distance_m: float = float("nan")
    test_distance_m: float = float("nan")
    reference_result: Optional[NormalisationResult] = None
    test_result: Optional[NormalisationResult] = None
    refusal: str = ""
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    @property
    def shift_dB(self) -> np.ndarray:
        """How much normalisation moved the insertion loss, band by band."""
        if self.insertion_loss_dB.size and self.raw_insertion_loss_dB.size:
            return self.insertion_loss_dB - self.raw_insertion_loss_dB
        return np.array([])

    @property
    def largest_shift_dB(self) -> float:
        shift = self.shift_dB
        if not shift.size:
            return float("nan")
        return float(shift[int(np.argmax(np.abs(shift)))])

    def to_dict(self) -> dict:
        def arr(values, digits=2):
            return [round(float(v), digits) for v in values]
        return {
            "valid": self.valid,
            "normalisation_distance_m": self.normalisation_distance_m,
            "reference_distance_m": (
                None if math.isnan(self.reference_distance_m) else self.reference_distance_m
            ),
            "test_distance_m": (
                None if math.isnan(self.test_distance_m) else self.test_distance_m
            ),
            "frequencies_Hz": arr(self.frequencies_Hz, 1),
            "reference_dB": arr(self.reference_dB, 1),
            "test_dB": arr(self.test_dB, 1),
            "insertion_loss_dB": arr(self.insertion_loss_dB, 1),
            "raw_insertion_loss_dB": arr(self.raw_insertion_loss_dB, 1),
            "shift_dB": arr(self.shift_dB, 2),
            "largest_shift_dB": (
                None if math.isnan(self.largest_shift_dB) else round(self.largest_shift_dB, 2)
            ),
            "refusal": self.refusal,
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }

    def summary(self) -> str:
        if not self.valid:
            return f"  Distance normalisation of the comparison REFUSED: {self.refusal}"
        lines = [
            f"  Insertion loss normalised to {self.normalisation_distance_m:g} m "
            f"(reference measured at {self.reference_distance_m:g} m, test at "
            f"{self.test_distance_m:g} m)",
        ]
        if self.shift_dB.size:
            lines.append(
                f"    Normalisation moved the per-band insertion loss by up to "
                f"{self.largest_shift_dB:+.2f} dB"
            )
        for assumption in self.assumptions:
            lines.append(f"    Assumes: {assumption}")
        for warning in self.warnings:
            lines.append(f"    WARNING: {warning}")
        return "\n".join(lines)


def normalise_insertion_loss_bands(
    reference_bands_dB: np.ndarray,
    test_bands_dB: np.ndarray,
    frequencies_Hz: np.ndarray,
    *,
    reference_atmosphere: Atmosphere,
    test_atmosphere: Atmosphere,
    reference_distance_m: Optional[float],
    test_distance_m: Optional[float],
    normalisation_distance_m: float = 1.0,
    reference_peak_Pa: Optional[float] = None,
    test_peak_Pa: Optional[float] = None,
    allow_nonlinear: bool = False,
) -> NormalisedComparison:
    """
    Refer both strings of a comparison to a common distance before differencing.

    Each spectrum is corrected with ITS OWN measured distance and ITS OWN
    recorded atmosphere, then the difference is taken. That is what removes the
    setup from the insertion loss: if the microphone sat half a metre further
    back for the suppressed string, spherical spreading alone credits the
    suppressor with 3.5 dB it did not earn.

    The normalisation distance must be stated and matters. When the two strings
    were shot in different air, the absorption they each accumulated over the
    shared path differs, so the insertion loss genuinely depends on the distance
    it is quoted at. One metre is the default because it is the standard
    reporting distance and because it is short enough that absorption over the
    remaining path is small.

    When both strings were measured at the same distance in the same air the
    corrections are identical and cancel exactly in the difference, so this is a
    no-op on the insertion loss - which is the correctness property to check.

    Args:
        reference_bands_dB: Band levels of the unsuppressed reference.
        test_bands_dB: Band levels of the suppressed test.
        frequencies_Hz: Midband frequencies shared by both.
        reference_atmosphere: Air the reference was recorded in.
        test_atmosphere: Air the test was recorded in.
        reference_distance_m: Distance the reference was measured at.
        test_distance_m: Distance the test was measured at.
        normalisation_distance_m: Distance to refer both to.
        reference_peak_Pa: Reference peak overpressure, for the linearity check.
        test_peak_Pa: Test peak overpressure, for the linearity check.
        allow_nonlinear: Proceed through the nonlinear guard, still labelled.

    Returns:
        NormalisedComparison. Check `.valid` before using `.insertion_loss_dB`.
    """
    reference = np.asarray(reference_bands_dB, dtype=np.float64).ravel()
    test = np.asarray(test_bands_dB, dtype=np.float64).ravel()
    freqs = np.asarray(frequencies_Hz, dtype=np.float64).ravel()

    if not (reference.shape == test.shape == freqs.shape):
        return NormalisedComparison(
            valid=False,
            refusal=(
                f"the two spectra and the frequency axis must describe the same "
                f"filter bank; got {reference.shape}, {test.shape} and {freqs.shape}"
            ),
        )
    if reference.size == 0:
        return NormalisedComparison(valid=False, refusal="no band levels supplied")

    raw_il = reference - test

    reference_result = reference_atmosphere.normalise_band_levels(
        reference, freqs,
        measured_distance_m=reference_distance_m,
        reference_distance_m=normalisation_distance_m,
        peak_Pa=reference_peak_Pa,
        allow_nonlinear=allow_nonlinear,
    )
    if not reference_result.valid:
        return NormalisedComparison(
            valid=False,
            raw_insertion_loss_dB=raw_il,
            frequencies_Hz=freqs,
            refusal=f"reference string could not be normalised: {reference_result.refusal}",
            reference_result=reference_result,
        )

    test_result = test_atmosphere.normalise_band_levels(
        test, freqs,
        measured_distance_m=test_distance_m,
        reference_distance_m=normalisation_distance_m,
        peak_Pa=test_peak_Pa,
        allow_nonlinear=allow_nonlinear,
    )
    if not test_result.valid:
        return NormalisedComparison(
            valid=False,
            raw_insertion_loss_dB=raw_il,
            frequencies_Hz=freqs,
            refusal=f"test string could not be normalised: {test_result.refusal}",
            reference_result=reference_result,
            test_result=test_result,
        )

    warnings = list(dict.fromkeys(reference_result.warnings + test_result.warnings))
    assumptions = list(dict.fromkeys(reference_result.assumptions + test_result.assumptions))

    if reference_atmosphere.defaulted or test_atmosphere.defaulted:
        warnings.append(
            "At least one string's weather was not recorded, so its absorption "
            "correction was computed from assumed conditions."
        )

    return NormalisedComparison(
        valid=True,
        frequencies_Hz=freqs,
        reference_dB=reference_result.levels_dB,
        test_dB=test_result.levels_dB,
        insertion_loss_dB=reference_result.levels_dB - test_result.levels_dB,
        raw_insertion_loss_dB=raw_il,
        normalisation_distance_m=float(normalisation_distance_m),
        reference_distance_m=float(reference_distance_m),
        test_distance_m=float(test_distance_m),
        reference_result=reference_result,
        test_result=test_result,
        warnings=warnings,
        assumptions=assumptions,
    )


def _nonlinearity_check(peak_Pa: Optional[float]) -> str:
    """
    Refusal text if the wave was too strong for linear propagation, else "".

    Returns "" when peak_Pa is None, because an unknown peak is not evidence of
    nonlinearity; the caller is told separately that the check did not run.
    """
    if peak_Pa is None or not math.isfinite(peak_Pa):
        return ""
    mach = abs(peak_Pa) / (GAMMA_AIR * P_ATM_PA)
    if mach <= NONLINEAR_MACH_LIMIT:
        return ""
    return (
        f"Peak overpressure {abs(peak_Pa):.0f} Pa is an acoustic Mach number of "
        f"{mach:.3f}, above the linear limit of {NONLINEAR_MACH_LIMIT:g} "
        f"(about {NONLINEAR_PEAK_dB:.0f} dB). Propagation between the muzzle and the "
        f"microphone was not linear, so the 1/r law understates the true decay and a "
        f"distance-normalised level would be wrong by an unmodelled amount."
    )


def crack_blast_delay_s(
    mic_distance_m: float,
    mic_angle_deg: float,
    projectile_velocity_m_per_s: float,
    temperature_C: float = 20.0,
) -> Optional[float]:
    """
    Time between the ballistic crack and the muzzle blast at a microphone.

    A supersonic projectile trails a Mach cone of half-angle mu = arcsin(c/v).
    For a microphone at downrange distance x and perpendicular offset y, the
    first cone arrival is found by minimising over the shedding time: the
    disturbance that reaches the microphone first was shed at

        x_shed = x - y * tan(mu)

    and travelled the slant path y / cos(mu) at the speed of sound, so

        t_crack = x_shed / v  +  ( y / cos(mu) ) / c

    while the muzzle blast, radiating spherically from the muzzle at t = 0,
    arrives at

        t_blast = sqrt(x^2 + y^2) / c

    The crack arrives first wherever the microphone is inside the cone. The
    returned delay is t_blast - t_crack, positive when the crack leads.

    Geometry convention matches TestMetadata.mic_angle_deg: 0 degrees is
    downrange along the line of fire, 90 degrees is abeam of the muzzle.

    Args:
        mic_distance_m: Straight-line distance from muzzle to microphone.
        mic_angle_deg: Angle from the line of fire, degrees.
        projectile_velocity_m_per_s: Muzzle velocity.
        temperature_C: Ambient temperature, for the speed of sound.

    Returns:
        Delay in seconds, positive when the crack precedes the blast, or None if
        the projectile is subsonic (no crack exists) or the microphone lies
        outside the Mach cone.
    """
    c = speed_of_sound(temperature_C)
    v = float(projectile_velocity_m_per_s)
    if v <= c:
        return None
    if mic_distance_m <= 0:
        raise AtmosphereError(f"mic_distance_m must be positive, got {mic_distance_m} m")

    theta = math.radians(mic_angle_deg)
    x = mic_distance_m * math.cos(theta)
    y = abs(mic_distance_m * math.sin(theta))

    mu = math.asin(c / v)
    if y <= 0.0:
        # Directly on the line of fire: the cone never sweeps across the mic.
        return None

    # Downrange position at which the cone that reaches this mic was shed.
    x_shed = x - y * math.tan(mu)
    if x_shed < 0.0:
        # The cone would have had to be shed behind the muzzle: the microphone
        # lies ahead of the Mach cone and never hears a crack from this shot.
        return None

    t_crack = x_shed / v + (y / math.cos(mu)) / c
    t_blast = math.sqrt(x * x + y * y) / c
    return float(t_blast - t_crack)


# ---- CLI for testing ----

def main() -> int:
    """Print absorption over the 1/3-octave bands for a given atmosphere."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ISO 9613-1 atmospheric absorption and distance normalisation"
    )
    parser.add_argument("--temperature", type=float, default=20.0, help="Air temperature (C)")
    parser.add_argument("--humidity", type=float, default=50.0, help="Relative humidity (%%)")
    parser.add_argument("--pressure", type=float, default=P_REF_kPa, help="Pressure (kPa)")
    parser.add_argument("--distance", type=float, default=100.0, help="Path length (m)")
    args = parser.parse_args()

    air = Atmosphere(
        temperature_C=args.temperature,
        humidity_pct=args.humidity,
        pressure_kPa=args.pressure,
    )
    print(air.summary())
    print(f"\n  Absorption over {args.distance:g} m:")
    print(f"    {'Freq (Hz)':>10}  {'alpha (dB/km)':>14}  {'A (dB)':>9}")

    freqs = np.array([
        50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    ], dtype=np.float64)
    alpha = air.absorption_coefficient_dB_per_m(freqs)
    for f, a in zip(freqs, np.atleast_1d(alpha)):
        print(f"    {f:>10.0f}  {a * 1000.0:>14.2f}  {a * args.distance:>9.2f}")

    print(f"\n  Speed of sound: {air.speed_of_sound_m_per_s:.2f} m/s")
    print(f"  Air density:    {air.density_kg_per_m3:.4f} kg/m^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
