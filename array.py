#!/usr/bin/env python3
"""
array.py - Multi-Microphone Array Geometry and Consistency

SASA has always let the operator pick a channel. A multi-microphone array is
more than several channels: it is several microphones at KNOWN, DIFFERENT places,
and that is what makes it worth having. Two microphones at the same distance on
opposite sides of the muzzle should read the same level. If they do not, one of
them is wrong, and knowing which is worth more than any average of the two.

WHAT THIS MODULE DOES

It does not combine channels into a single number. There is no defensible way to
average microphones that are measuring different things, and a mean of a good
microphone and an obstructed one is worse than either. Instead:

  * expected_arrival_delays()  From the recorded geometry and the speed of
                               sound, when each microphone SHOULD hear the
                               blast, relative to the first.

  * check_array_consistency()  Refers every channel to a common distance using
                               its own geometry and the recorded atmosphere,
                               then asks whether the channels then agree. They
                               should: the same blast referred to the same
                               distance is the same number. A channel that
                               disagrees is carrying something the others are
                               not - an obstruction, a reflection, a wrong
                               distance in the metadata, or a failing capsule.

  * The verdict is per channel, with the disagreement stated in decibels, so an
    operator can drop the bad microphone rather than the measurement.

WHY DISAGREEMENT IS THE USEFUL OUTPUT

A single microphone cannot tell you it was wrong. An array can: it is the only
configuration in this instrument where an error becomes visible rather than
merely possible. Reporting the spread between channels is therefore the point of
having them, and averaging it away is the one thing that would waste them.

Usage:
    from array import ArrayGeometry, MicPosition, check_array_consistency

    geometry = ArrayGeometry([
        MicPosition(channel=0, distance_m=1.0, angle_deg=90.0),
        MicPosition(channel=1, distance_m=3.0, angle_deg=90.0),
    ])
    report = check_array_consistency({0: ref_bands, 1: test_bands}, freqs,
                                     geometry=geometry, atmosphere=air)
    print(report.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from atmosphere import Atmosphere, speed_of_sound

# How far two channels may disagree, after both have been referred to a common
# distance, before the array is called inconsistent.
#
# This is the same 0.5 dB a session is invalidated at for calibration drift: it
# is the point at which the instrument can no longer be treated as stable. Below
# it, the difference is inside what the chain itself contributes.
CONSISTENCY_TOLERANCE_dB: float = 0.5

# Channels needed before consistency means anything. With one microphone there
# is nothing to disagree with.
MIN_CHANNELS: int = 2


class ArrayError(ValueError):
    """Raised when an array description is physically impossible."""


@dataclass
class MicPosition:
    """One microphone's place in the array."""
    channel: int
    distance_m: float
    angle_deg: float
    height_m: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.distance_m is None or not math.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ArrayError(
                f"channel {self.channel}: distance_m must be positive and finite, "
                f"got {self.distance_m}"
            )
        if self.angle_deg is None or not math.isfinite(self.angle_deg):
            raise ArrayError(
                f"channel {self.channel}: angle_deg must be finite, got {self.angle_deg}"
            )
        if not self.label:
            self.label = f"channel {self.channel}"

    @property
    def position_m(self) -> tuple:
        """Cartesian position, with 0 degrees downrange along the line of fire."""
        theta = math.radians(self.angle_deg)
        return (self.distance_m * math.cos(theta), self.distance_m * math.sin(theta))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "label": self.label,
            "distance_m": self.distance_m,
            "angle_deg": self.angle_deg,
            "height_m": self.height_m,
        }


@dataclass
class ArrayGeometry:
    """Where every microphone in the array was."""
    positions: List[MicPosition] = field(default_factory=list)

    def __post_init__(self) -> None:
        channels = [p.channel for p in self.positions]
        duplicates = {c for c in channels if channels.count(c) > 1}
        if duplicates:
            raise ArrayError(
                f"each channel may appear once; these appear more than once: "
                f"{sorted(duplicates)}"
            )

    @property
    def channels(self) -> List[int]:
        return [p.channel for p in self.positions]

    def get(self, channel: int) -> Optional[MicPosition]:
        for position in self.positions:
            if position.channel == channel:
                return position
        return None

    @property
    def is_array(self) -> bool:
        """
        Whether this describes microphones at DIFFERENT places.

        Channel count alone is not the test. A record that carries one
        microphone position and is applied to two channels produces two
        positions that are identical, and two microphones at the same point are
        not an array: comparing them shows only that a splitter works. An array
        requires at least two distinct positions.
        """
        if len(self.positions) < MIN_CHANNELS:
            return False
        distinct = {
            (round(p.distance_m, 6), round(p.angle_deg, 6),
             None if p.height_m is None else round(p.height_m, 6))
            for p in self.positions
        }
        return len(distinct) >= MIN_CHANNELS

    @classmethod
    def from_metadata(cls, metadata: Any, channels: Sequence[int]) -> "ArrayGeometry":
        """
        Build from a TestMetadata that records ONE microphone position.

        A single recorded position cannot describe an array. Every channel is
        given that same position, which is only correct for a single microphone;
        `describes_an_array` on the consistency report says so, so the result is
        never mistaken for a real array description.
        """
        def get(name):
            if isinstance(metadata, dict):
                return metadata.get(name)
            return getattr(metadata, name, None)

        distance, angle = get("mic_distance_m"), get("mic_angle_deg")
        if distance is None or angle is None:
            return cls([])
        try:
            distance, angle = float(distance), float(angle)
        except (TypeError, ValueError):
            return cls([])
        if not (math.isfinite(distance) and math.isfinite(angle)) or distance <= 0:
            return cls([])

        height = get("mic_height_m")
        try:
            height = float(height) if height is not None else None
        except (TypeError, ValueError):
            height = None

        return cls([
            MicPosition(channel=int(c), distance_m=distance, angle_deg=angle,
                        height_m=height)
            for c in channels
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_channels": len(self.positions),
            "is_array": self.is_array,
            "positions": [p.to_dict() for p in self.positions],
        }


def expected_arrival_delays(
    geometry: ArrayGeometry,
    *,
    temperature_C: float = 20.0,
) -> Dict[int, float]:
    """
    When each microphone should hear the muzzle blast, relative to the earliest.

    The blast radiates spherically from the muzzle, so a microphone at distance d
    hears it at d/c. The differences between channels are a direct, checkable
    prediction of the recorded geometry: if the measured delays do not match
    these, the geometry in the metadata is not the geometry on the range.

    Args:
        geometry: Microphone positions.
        temperature_C: Ambient temperature, which sets the speed of sound.

    Returns:
        {channel: delay in seconds relative to the nearest microphone}.
    """
    if not geometry.positions:
        return {}
    c = speed_of_sound(temperature_C)
    arrivals = {p.channel: p.distance_m / c for p in geometry.positions}
    earliest = min(arrivals.values())
    return {channel: t - earliest for channel, t in arrivals.items()}


@dataclass
class ChannelAgreement:
    """One channel's verdict against the rest of the array."""
    channel: int
    label: str
    level_dB: float = float("nan")
    normalised_dB: float = float("nan")
    deviation_dB: float = float("nan")
    agrees: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        def num(x):
            return round(float(x), 3) if math.isfinite(x) else None
        return {
            "channel": self.channel,
            "label": self.label,
            "level_dB": num(self.level_dB),
            "normalised_dB": num(self.normalised_dB),
            "deviation_dB": num(self.deviation_dB),
            "agrees": self.agrees,
            "reason": self.reason,
        }


@dataclass
class ArrayConsistencyReport:
    """Whether the microphones in an array are measuring the same event."""
    channels: List[ChannelAgreement] = field(default_factory=list)
    reference_distance_m: float = 1.0
    spread_dB: float = float("nan")
    consistent: bool = False
    describes_an_array: bool = False
    refusal: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def disagreeing(self) -> List[ChannelAgreement]:
        return [c for c in self.channels if not c.agrees]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consistent": self.consistent,
            "describes_an_array": self.describes_an_array,
            "reference_distance_m": self.reference_distance_m,
            "spread_dB": None if math.isnan(self.spread_dB) else round(self.spread_dB, 3),
            "tolerance_dB": CONSISTENCY_TOLERANCE_dB,
            "channels": [c.to_dict() for c in self.channels],
            "disagreeing_channels": [c.channel for c in self.disagreeing],
            "refusal": self.refusal,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        if self.refusal:
            return f"  Array consistency: NOT CHECKED - {self.refusal}"
        lines = [
            f"  Array consistency across {len(self.channels)} channel(s), "
            f"referred to {self.reference_distance_m:g} m"
        ]
        for channel in self.channels:
            mark = "  " if channel.agrees else "!!"
            lines.append(
                f"    {mark} {channel.label:<16s} {channel.level_dB:8.2f} dB measured "
                f"-> {channel.normalised_dB:8.2f} dB referred "
                f"({channel.deviation_dB:+.2f} from the array median)"
            )
        if self.consistent:
            lines.append(
                f"    Channels agree to {self.spread_dB:.2f} dB, within the "
                f"{CONSISTENCY_TOLERANCE_dB:g} dB tolerance."
            )
        else:
            lines.append(
                f"    Channels DISAGREE by {self.spread_dB:.2f} dB, beyond the "
                f"{CONSISTENCY_TOLERANCE_dB:g} dB tolerance. The same blast referred to "
                f"the same distance must give the same level, so at least one channel "
                f"is measuring something the others are not."
            )
        for note in self.notes:
            lines.append(f"    {note}")
        return "\n".join(lines)


def check_array_consistency(
    channel_levels_dB: Dict[int, float],
    *,
    geometry: ArrayGeometry,
    atmosphere: Optional[Atmosphere] = None,
    reference_distance_m: float = 1.0,
    tolerance_dB: float = CONSISTENCY_TOLERANCE_dB,
) -> ArrayConsistencyReport:
    """
    Ask whether every microphone in the array measured the same blast.

    Each channel's level is referred to a common distance using its own recorded
    geometry. Referred to the same place, they must agree: it is one blast. A
    channel that then sits outside the tolerance is carrying something the others
    are not, and it is named rather than averaged in.

    Absorption is NOT applied here, because these are broadband levels and a
    broadband level has no single absorption coefficient. Over the few metres an
    array spans that term is small; over a long baseline it is not, and the
    report says so.

    Args:
        channel_levels_dB: {channel: broadband level} for one event or string.
        geometry: Where each microphone was.
        atmosphere: Recorded air, used only to note when absorption would matter.
        reference_distance_m: Distance to refer every channel to.
        tolerance_dB: Agreement required.

    Returns:
        ArrayConsistencyReport. Check `.refusal` first.
    """
    report = ArrayConsistencyReport(reference_distance_m=float(reference_distance_m))

    if not geometry.positions:
        report.refusal = (
            "no microphone positions were recorded, so the channels cannot be "
            "referred to a common distance"
        )
        return report
    report.describes_an_array = geometry.is_array

    usable = {
        channel: float(level) for channel, level in channel_levels_dB.items()
        if isinstance(level, (int, float)) and math.isfinite(level)
        and geometry.get(channel) is not None
    }
    if len(usable) < MIN_CHANNELS:
        report.refusal = (
            f"only {len(usable)} channel(s) have both a level and a recorded "
            f"position, and consistency needs at least {MIN_CHANNELS}"
        )
        return report

    if reference_distance_m <= 0:
        report.refusal = (
            f"reference distance must be positive, got {reference_distance_m} m"
        )
        return report

    # Refer each channel to the common distance by spherical spreading only.
    normalised: Dict[int, float] = {}
    for channel, level in usable.items():
        position = geometry.get(channel)
        normalised[channel] = level + 20.0 * math.log10(
            position.distance_m / reference_distance_m
        )

    values = np.array(list(normalised.values()), dtype=np.float64)
    median = float(np.median(values))
    report.spread_dB = float(np.max(values) - np.min(values))

    for channel in sorted(usable):
        position = geometry.get(channel)
        deviation = normalised[channel] - median
        agrees = abs(deviation) <= tolerance_dB
        reason = ""
        if not agrees:
            reason = (
                f"sits {deviation:+.2f} dB from the array median once referred to "
                f"{reference_distance_m:g} m; check this microphone for an "
                f"obstruction, a reflecting surface, a wrong distance in the "
                f"metadata, or a failing capsule"
            )
        report.channels.append(ChannelAgreement(
            channel=channel,
            label=position.label,
            level_dB=usable[channel],
            normalised_dB=normalised[channel],
            deviation_dB=deviation,
            agrees=agrees,
            reason=reason,
        ))

    report.consistent = all(c.agrees for c in report.channels)

    # With two microphones the median sits exactly between them, so a
    # disagreement flags BOTH and neither can be blamed. Three is the first
    # count at which the odd one out is identifiable.
    if not report.consistent and len(report.channels) == 2:
        report.notes.append(
            "With two microphones the disagreement is detectable but not "
            "attributable: each sits the same distance from the midpoint, so "
            "nothing here says which one is wrong. A third microphone would "
            "identify the odd one out."
        )

    if not report.describes_an_array:
        report.notes.append(
            "Every channel was given the same recorded position, so this is not a "
            "described array: the comparison shows only that the channels of one "
            "microphone agree, not that separate microphones do."
        )

    # Does the omitted absorption term matter over this baseline?
    distances = [geometry.get(c).distance_m for c in usable]
    baseline = max(distances) - min(distances)
    if atmosphere is not None and baseline > 0:
        alpha_4k = float(np.atleast_1d(
            atmosphere.absorption_coefficient_dB_per_m(np.array([4000.0]))
        )[0])
        absorption_over_baseline = alpha_4k * baseline
        if absorption_over_baseline > 0.1:
            report.notes.append(
                f"The channels span {baseline:g} m, over which the air absorbs about "
                f"{absorption_over_baseline:.2f} dB at 4 kHz. These are broadband "
                f"levels so no absorption was applied; part of the disagreement above "
                f"is that term rather than the microphones."
            )

    return report


# ---- CLI for testing ----

def main() -> int:
    """Demonstrate a consistency check on a synthetic two-microphone array."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-microphone array consistency")
    parser.add_argument("--near-m", type=float, default=1.0, help="Near mic distance")
    parser.add_argument("--far-m", type=float, default=3.0, help="Far mic distance")
    parser.add_argument("--far-error-dB", type=float, default=0.0,
                        help="Error to plant on the far microphone")
    args = parser.parse_args()

    geometry = ArrayGeometry([
        MicPosition(channel=0, distance_m=args.near_m, angle_deg=90.0, label="near"),
        MicPosition(channel=1, distance_m=args.far_m, angle_deg=90.0, label="far"),
    ])
    # A single blast: the far mic reads lower by exactly the spreading loss.
    near_level = 150.0
    far_level = near_level - 20.0 * math.log10(args.far_m / args.near_m) + args.far_error_dB

    air = Atmosphere(temperature_C=18.0, humidity_pct=62.0)
    report = check_array_consistency(
        {0: near_level, 1: far_level}, geometry=geometry, atmosphere=air)
    print(report.summary())
    print()
    delays = expected_arrival_delays(geometry, temperature_C=18.0)
    print("  Expected arrival delays, relative to the nearest microphone:")
    for channel, delay in sorted(delays.items()):
        print(f"    channel {channel}: {delay * 1000.0:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
