#!/usr/bin/env python3
"""
provenance.py - Measurement Provenance and Test Conditions

A number without provenance is not a measurement. This module carries the two kinds
of context that make a SASA result defensible:

  SoftwareInfo   what produced the number - version, commit, platform, library
                 versions, and a hash of the input file, so a result can be
                 reproduced exactly and so two results can be compared knowing
                 whether the instrument itself changed between them.

  TestMetadata   the physical conditions the number was measured under - microphone
                 model and position, weapon and ammunition, and the atmosphere.
                 Sound pressure from a muzzle blast varies with distance, angle,
                 temperature, humidity and barometric pressure, so a peak level
                 quoted without them cannot be compared against anyone else's, or
                 even against the same rig on a different day.

Fields are recorded as supplied, and completeness is reported rather than enforced:
the operator is told what is missing from a defensible record, and nothing is
silently invented.

Usage:
    from provenance import SoftwareInfo, TestMetadata

    software = SoftwareInfo.capture()
    meta = TestMetadata(operator="G. Wilson", mic_distance_m=1.0, ...)
    print(meta.completeness_report())
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from textutil import count

# The version of SASA, for the whole application: main.py re-exports this one
# rather than keeping its own. See the note there -- this constant was stale by
# two releases and every record it stamped named the wrong producer.
__version__ = "2.4.2"

# Fields required for a measurement record that another lab could reproduce.
REQUIRED_FOR_DEFENSIBLE: tuple[str, ...] = (
    "operator",
    "date",
    "weapon",
    "ammunition",
    "configuration",
    "mic_model",
    "mic_distance_m",
    "mic_angle_deg",
    "temperature_C",
    "humidity_pct",
    "pressure_kPa",
)

# Fields that materially change the measured level and so are worth prompting for.
AFFECTS_LEVEL: tuple[str, ...] = (
    "mic_distance_m", "mic_angle_deg", "mic_height_m",
    "temperature_C", "humidity_pct", "pressure_kPa", "wind_mps",
)


def _run_git(args: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a git command, returning stripped stdout or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd or Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> Optional[str]:
    """
    SHA-256 of a file, read in chunks so a multi-gigabyte recording is affordable.

    Returns None if the file cannot be read.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


@dataclass
class SoftwareInfo:
    """Identifies the exact instrument that produced a result."""
    name: str = "SASA"
    version: str = __version__
    git_commit: Optional[str] = None
    git_dirty: bool = False
    python_version: str = ""
    platform: str = ""
    libraries: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> "SoftwareInfo":
        """Capture the running software's identity."""
        commit = _run_git(["rev-parse", "HEAD"])
        status = _run_git(["status", "--porcelain"])

        libs: Dict[str, str] = {}
        for mod in ("numpy", "scipy", "soundfile", "matplotlib", "plotly"):
            try:
                libs[mod] = __import__(mod).__version__
            except Exception:  # noqa: BLE001 - a missing optional library is not fatal
                pass

        return cls(
            git_commit=commit,
            git_dirty=bool(status),
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            libraries=libs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        commit = (self.git_commit or "unknown")[:12]
        dirty = " +local-changes" if self.git_dirty else ""
        return f"{self.name} {self.version} ({commit}{dirty})"


@dataclass
class TestMetadata:
    """
    Physical conditions of a measurement.

    Every field is optional so an exploratory analysis is never blocked, but
    completeness_report() states plainly what is missing from a record that could be
    defended or reproduced.
    """
    # Session
    operator: str = ""
    date: str = ""
    location: str = ""
    test_id: str = ""

    # Weapon / ammunition
    weapon: str = ""
    barrel_length_in: Optional[float] = None
    ammunition: str = ""
    suppressor: str = ""
    configuration: str = ""          # "suppressed" | "unsuppressed"

    # Microphone and geometry
    mic_model: str = ""
    mic_serial: str = ""
    mic_distance_m: Optional[float] = None
    mic_angle_deg: Optional[float] = None    # 0 = downrange, 90 = left of muzzle
    mic_height_m: Optional[float] = None
    ground_surface: str = ""
    windscreen: str = ""

    # Environment
    temperature_C: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_kPa: Optional[float] = None
    wind_mps: Optional[float] = None

    # Calibration record
    calibrator_model: str = ""
    calibrator_level_dB: Optional[float] = None
    calibration_pre_dB: Optional[float] = None
    calibration_post_dB: Optional[float] = None

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.date:
            self.date = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
        if self.configuration:
            self.configuration = self.configuration.strip().lower()

    @property
    def calibration_drift_dB(self) -> Optional[float]:
        """
        Drift between pre- and post-test calibration.

        Most measurement protocols invalidate a session when this exceeds 0.5 dB,
        because the chain cannot be assumed stable across the test.
        """
        if self.calibration_pre_dB is None or self.calibration_post_dB is None:
            return None
        return self.calibration_post_dB - self.calibration_pre_dB

    def missing_required(self) -> List[str]:
        """Required fields that were not supplied."""
        missing = []
        for name in REQUIRED_FOR_DEFENSIBLE:
            value = getattr(self, name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(name)
        return missing

    def warnings(self) -> List[str]:
        """Conditions that undermine the record."""
        out: List[str] = []

        drift = self.calibration_drift_dB
        if drift is not None and abs(drift) > 0.5:
            out.append(
                f"Calibration drifted {drift:+.2f} dB between pre- and post-test checks "
                f"(limit 0.5 dB). The measurement chain was not stable across this session."
            )

        if self.configuration and self.configuration not in ("suppressed", "unsuppressed"):
            out.append(
                f"configuration is '{self.configuration}'; expected 'suppressed' or "
                f"'unsuppressed' so that reference and test runs can be paired."
            )

        if self.mic_distance_m is not None and self.mic_distance_m <= 0:
            out.append(f"mic_distance_m must be positive, got {self.mic_distance_m}")

        if self.humidity_pct is not None and not (0 <= self.humidity_pct <= 100):
            out.append(f"humidity_pct must be 0-100, got {self.humidity_pct}")

        missing_env = [f for f in AFFECTS_LEVEL if getattr(self, f, None) is None]
        if missing_env:
            out.append(
                "Not recorded, and each of these changes the measured level: "
                + ", ".join(missing_env)
            )

        return out

    def is_defensible(self) -> bool:
        """Whether this record could stand as evidence."""
        return not self.missing_required()

    def completeness_report(self) -> str:
        """Human-readable statement of what the record does and does not establish."""
        missing = self.missing_required()
        lines = []
        if missing:
            lines.append(
                f"  Measurement record INCOMPLETE - {count(len(missing), 'required field')} missing:"
            )
            lines.append("    " + ", ".join(missing))
            lines.append(
                "    Results remain numerically valid, but cannot be reproduced or "
                "compared against another session."
            )
        else:
            lines.append("  Measurement record complete.")
        for w in self.warnings():
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestMetadata":
        """
        Build from a dictionary, ignoring unknown keys.

        Unknown keys are ignored rather than raising, so a metadata file written by a
        newer version does not break an older one.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["calibration_drift_dB"] = self.calibration_drift_dB
        data["is_defensible"] = self.is_defensible()
        data["missing_required"] = self.missing_required()
        return data


@dataclass
class SourceInfo:
    """The audio file a result was derived from."""
    path: str = ""
    sha256: Optional[str] = None
    sample_rate: int = 0
    channels: int = 0
    subtype: str = ""
    frames: int = 0
    duration_s: float = 0.0
    channel_used: str = ""      # "mono mix", "channel 0", ...

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_provenance_block(
    software: SoftwareInfo,
    source: SourceInfo,
    test_metadata: TestMetadata,
    *,
    timestamp: Optional[str] = None,
    elapsed_s: float = 0.0,
    output_dir: str = "",
) -> Dict[str, Any]:
    """Assemble the provenance portion of the analysis record."""
    return {
        "software": software.to_dict(),
        "analysis": {
            "timestamp": timestamp or datetime.now(timezone.utc).astimezone().isoformat(),
            "input_file": source.path,
            "input_sha256": source.sha256,
            "output_dir": output_dir,
            "elapsed_s": round(elapsed_s, 3),
        },
        "source": source.to_dict(),
        "test_metadata": test_metadata.to_dict(),
    }


def main() -> int:
    """Print the current provenance block."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Show measurement provenance")
    parser.add_argument("file", type=Path, nargs="?", help="Optional file to hash")
    args = parser.parse_args()

    software = SoftwareInfo.capture()
    print(software.summary())
    print(json.dumps(software.to_dict(), indent=2))

    if args.file and args.file.exists():
        print(f"\nsha256({args.file.name}) = {file_sha256(args.file)}")

    meta = TestMetadata()
    print("\nEmpty metadata record:")
    print(meta.completeness_report())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
