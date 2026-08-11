#!/usr/bin/env python3
"""
SASA - Shot Acoustic Spectral Analysis: analysis pipeline and command line.

This module turns a recording into a measurement record. Its job is to make every
choice that affects the numbers explicit, recorded and reproducible:

  Calibration is never assumed.  There is no default Pa/FS anywhere in this file.
      A run states how the microphone was calibrated - a calibrator tone, a
      recording chain, a direct factor, a named preset, or explicitly nothing -
      and an uncalibrated run is labelled "dB re FS" everywhere rather than being
      passed off as dB SPL.

  Admissibility is checked before analysis.  Clipping, DC offset, inadequate
      sample rate and poor signal-to-noise are assessed on the FULL-SCALE samples.
      A recording that fails a hard check still produces output, but that output
      says, in every artifact, that the measurement is inadmissible.

  Channels are not averaged.  Averaging microphones that are not co-located
      destroys the measurement, and mixing a mono source held in a stereo file
      reads 6 dB low. One channel is analysed at a time unless the operator asks
      in writing for a mix.

  Data is written before pictures.  Metrics reach disk before any plot is drawn,
      so a plotting failure can never discard a computed result.

Output contract (analysis_metadata.json, schema_version 2.0) and the progress
protocol ([SASA-PROGRESS] <percent> <message>, plus [SASA-OUTPUT] <dir>) are
consumed by the desktop UI; see write_record() and progress().

Exit codes:
    0  analysis completed and the measurement is admissible
    1  usage, configuration or runtime error - no result
    2  analysis completed but no shots were detected - no result
    3  analysis completed but the measurement is INADMISSIBLE (see quality.errors)

Author: Ridgeback Defense
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Plot rendering must never require a display; this is set before matplotlib is
# imported anywhere (plots.py is imported lazily, inside the plotting stage).
os.environ.setdefault("MPLBACKEND", "Agg")

from WavLoader import get_wav_info, load_wav, load_wav_chunk  # noqa: E402
from calibration import (  # noqa: E402
    Calibration,
    SignalQuality,
    amplitude_to_dB_SPL,
    assess_signal_quality,
)
from shot_detect import (  # noqa: E402
    DetectionReport,
    ShotEvent,
    bandpass_for_detection,
    compute_envelope,
    detect_shots,
)
from metrics import (  # noqa: E402
    AggregateMetrics,
    MetricStats,
    ShotMetrics,
    compute_aggregate_metrics,
    compute_insertion_loss,
    compute_shot_metrics,
)
from bands import ThirdOctaveAnalyzer, band_insertion_loss  # noqa: E402
from STFT import STFTResult, analyze_stft, recommended_nperseg  # noqa: E402
from weighting import weighting_settling_samples  # noqa: E402
from provenance import (  # noqa: E402
    SoftwareInfo,
    SourceInfo,
    TestMetadata,
    file_sha256,
    make_provenance_block,
)

__version__ = "2.0.0"

SCHEMA_VERSION = "2.0"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_SHOTS = 2
EXIT_INVALID = 3

logger = logging.getLogger("sasa")

# Extensions handled without extraction, and those that must go through ffmpeg.
AUDIO_EXTS = {
    ".wav", ".wave", ".bwf", ".rf64", ".w64", ".flac", ".aif", ".aiff", ".aifc",
    ".caf", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".wma", ".alac",
}
VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v", ".mpeg",
    ".mpg", ".mts", ".m2ts", ".3gp",
}

# Chunked processing (long recordings). Chunks OVERLAP by CHUNK_CONTEXT_S so that
# no shot window is cut at a chunk boundary and no filter or detector state has to
# restart cold in the middle of the signal.
MAX_DURATION_FULL_LOAD_S = 600.0
CHUNK_DURATION_S = 120.0
CHUNK_CONTEXT_S = 2.0

# Display-only thinning. Analysis always uses every sample at the native rate;
# these bounds exist because a 192 kHz spectrogram at full frame rate makes a
# 26 MB HTML file that no browser can pan smoothly.
MAX_WAVEFORM_POINTS = 150_000
MAX_SPECTROGRAM_FRAMES = 1500
SPECTROGRAM_DOWNSAMPLE = 40
# Full sample resolution is retained around each shot so the blast can be
# inspected by zooming in; this is the span kept, relative to the peak.
WAVEFORM_FULLRES_PRE_S = 0.010
WAVEFORM_FULLRES_POST_S = 0.020

# Formats matplotlib is known to write, used when matplotlib cannot be imported
# during argument validation.
_FALLBACK_STATIC_FORMATS = {
    "png", "pdf", "svg", "svgz", "eps", "ps", "jpg", "jpeg",
    "tif", "tiff", "webp", "raw", "rgba", "pgf",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Errors
# ═══════════════════════════════════════════════════════════════════════════

class SasaError(Exception):
    """Base class for errors that should be reported without a traceback."""


class ConfigurationError(SasaError):
    """The run was described incorrectly (bad flag, bad file, impossible value)."""


class CalibrationRequired(ConfigurationError):
    """No calibration was specified, and SASA will not invent one."""


# ═══════════════════════════════════════════════════════════════════════════
#  Progress protocol, human output and logging
# ═══════════════════════════════════════════════════════════════════════════

# Progress is reported on a 0-100 scale by each analysis pass. When several
# passes run in one invocation (--channels all), each pass is mapped into its
# own slice of the overall bar so the percentage never travels backwards -
# the UI treats a decrease as a new run and resets its progress indicator.
_PROGRESS_WINDOW: Tuple[float, float] = (0.0, 100.0)


@contextlib.contextmanager
def progress_window(low: float, high: float):
    """Scale progress emitted inside this block into the range [low, high]."""
    global _PROGRESS_WINDOW
    previous = _PROGRESS_WINDOW
    _PROGRESS_WINDOW = (float(low), float(high))
    try:
        yield
    finally:
        _PROGRESS_WINDOW = previous


def progress(percent: float, message: str) -> None:
    """
    Emit one machine-readable progress line.

    The Node bridge parses exactly this shape and treats the line as a control
    line rather than log output, so nothing else may share the prefix.
    """
    if not math.isfinite(percent):
        percent = 0.0
    fraction = min(100.0, max(0.0, percent)) / 100.0
    low, high = _PROGRESS_WINDOW
    pct = int(round(low + (high - low) * fraction))
    pct = min(100, max(0, pct))
    text = " ".join(str(message).split())
    print(f"[SASA-PROGRESS] {pct} {text}", flush=True)


def announce_output_dir(path: Path) -> None:
    """Publish the output directory on its own clearly-marked line."""
    print(f"[SASA-OUTPUT] {path}", flush=True)


def say(message: str = "") -> None:
    """Human-facing progress on stdout (kept separate from the log file)."""
    print(message, flush=True)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure console diagnostics.

    Diagnostics go to stderr (and, once an output directory exists, to a log file
    inside it). Human-facing progress stays on stdout so the two streams can be
    consumed independently.
    """
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if getattr(handler, "_sasa_console", False):
            handler.setLevel(level)
            return

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console._sasa_console = True  # type: ignore[attr-defined]
    logger.addHandler(console)


def add_log_file(log_path: Path) -> logging.Handler:
    """
    Attach a log file inside the output directory.

    Every run leaves a full-detail log next to its results, independently of how
    verbose the console was told to be.
    """
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
    return file_handler


# ═══════════════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _finite(value: Any, name: str) -> float:
    """Coerce to a finite float or raise. NaN and infinity are rejected explicitly."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number, got {value!r}") from exc
    if math.isnan(v):
        raise ConfigurationError(f"{name} must be a real number, got NaN")
    if math.isinf(v):
        raise ConfigurationError(f"{name} must be finite, got {v}")
    return v


def _positive(value: Any, name: str) -> float:
    v = _finite(value, name)
    if v <= 0:
        raise ConfigurationError(f"{name} must be greater than zero, got {v}")
    return v


def _non_negative(value: Any, name: str) -> float:
    v = _finite(value, name)
    if v < 0:
        raise ConfigurationError(f"{name} must be zero or greater, got {v}")
    return v


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a whole number, got {value!r}") from exc
    if float(value) != v:
        raise ConfigurationError(f"{name} must be a whole number, got {value!r}")
    if not (minimum <= v <= maximum):
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}, got {v}")
    return v


def supported_static_formats() -> set:
    """Formats matplotlib can actually write in this installation."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig = plt.figure()
        try:
            return set(fig.canvas.get_supported_filetypes().keys())
        finally:
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - fall back to the known-good list
        logger.debug("Could not query matplotlib formats (%s); using fallback list", exc)
        return set(_FALLBACK_STATIC_FORMATS)


def validate_formats(raw: Any) -> List[str]:
    """
    Parse and validate a plot-format specification.

    Accepts a comma-separated string or a list. Entries are trimmed and lower-cased
    so that "png, pdf" is valid rather than aborting the run halfway through
    plotting, and every entry is checked against what matplotlib can really write
    (plus "html", which SASA renders itself with Plotly).
    """
    if raw is None:
        return ["png"]
    items = raw.split(",") if isinstance(raw, str) else list(raw)

    allowed = supported_static_formats() | {"html"}
    cleaned: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ConfigurationError(f"--formats entries must be text, got {item!r}")
        fmt = item.strip().lower().lstrip(".")
        if not fmt:
            continue
        if fmt not in allowed:
            raise ConfigurationError(
                f"Unsupported plot format {fmt!r}. Supported: "
                + ", ".join(sorted(allowed))
            )
        if fmt not in cleaned:
            cleaned.append(fmt)

    if not cleaned:
        raise ConfigurationError("--formats listed no usable formats")
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
#  Calibration presets and the profile store
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationPreset:
    """A calibration that was measured once for one specific rig."""
    name: str
    Pa_per_FS: float
    provenance: str
    builtin: bool = False

    def to_calibration(self) -> Calibration:
        return Calibration.preset(self.Pa_per_FS, self.name, self.provenance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "Pa_per_FS": self.Pa_per_FS,
            "provenance": self.provenance,
            "builtin": self.builtin,
        }


# The 143.96 Pa/FS constant that used to be the hidden default for every run.
# It is a real measurement, but of one microphone, one preamp gain and one
# recorder in 2012 - so it ships as a named preset carrying its own provenance,
# and it is never selected unless the operator asks for it by name.
BUILTIN_PRESETS: Dict[str, CalibrationPreset] = {
    "ST2012-114dB-20120226": CalibrationPreset(
        name="ST2012-114dB-20120226",
        Pa_per_FS=143.96,
        provenance=(
            "Derived from the 114 dB SPL calibrator tone recorded in "
            "Audio/260212_0010-1.wav on 2012-02-26 with the ST2012 stereo "
            "microphone pair. Valid ONLY for that microphone, preamp gain and "
            "recorder setting; applying it to any other rig produces levels "
            "scaled by someone else's hardware."
        ),
        builtin=True,
    ),
}


def config_dir() -> Path:
    """Per-user configuration directory, overridable with SASA_CONFIG_DIR."""
    env = os.environ.get("SASA_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SASA"
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "SASA"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sasa"


def profiles_path(override: Optional[Path] = None) -> Path:
    """Location of the calibration profile store."""
    if override is not None:
        return Path(override).expanduser()
    return config_dir() / "calibration_profiles.json"


def load_profiles(path: Optional[Path] = None, warnings: Optional[List[str]] = None) -> Dict[str, CalibrationPreset]:
    """
    Load saved calibration profiles, merged over the built-in presets.

    A corrupt or unreadable store is reported as a warning rather than failing the
    run: the operator can still calibrate explicitly.
    """
    presets: Dict[str, CalibrationPreset] = dict(BUILTIN_PRESETS)
    store = profiles_path(path)
    if not store.is_file():
        return presets

    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"Calibration profile store {store} could not be read ({exc}); ignoring it."
        logger.warning(message)
        if warnings is not None:
            warnings.append(message)
        return presets

    entries = data.get("profiles", data) if isinstance(data, dict) else {}
    if not isinstance(entries, dict):
        message = f"Calibration profile store {store} is not a profile map; ignoring it."
        logger.warning(message)
        if warnings is not None:
            warnings.append(message)
        return presets

    for name, entry in entries.items():
        try:
            presets[str(name)] = CalibrationPreset(
                name=str(name),
                Pa_per_FS=_positive(entry["Pa_per_FS"], f"profile {name}: Pa_per_FS"),
                provenance=str(entry.get("provenance", "") or "saved profile"),
            )
        except (KeyError, TypeError, ConfigurationError) as exc:
            message = f"Calibration profile {name!r} is unusable and was skipped ({exc})."
            logger.warning(message)
            if warnings is not None:
                warnings.append(message)
    return presets


def save_profile(name: str, calibration: Calibration, note: str = "", path: Optional[Path] = None) -> Path:
    """Persist a resolved calibration under a name so a rig is configured once."""
    if not name or not name.strip():
        raise ConfigurationError("A profile name is required")
    name = name.strip()
    if name in BUILTIN_PRESETS:
        raise ConfigurationError(f"{name!r} is a built-in preset and cannot be overwritten")
    if not calibration.calibrated:
        raise ConfigurationError("An uncalibrated run cannot be saved as a calibration profile")

    store = profiles_path(path)
    store.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"version": 1, "profiles": {}}
    if store.is_file():
        try:
            existing = json.loads(store.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("profiles"), dict):
                data = existing
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Replacing unreadable profile store %s (%s)", store, exc)

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    provenance = note.strip() or calibration.description
    data["profiles"][name] = {
        "Pa_per_FS": calibration.Pa_per_FS,
        "provenance": f"{provenance} [saved {stamp} via {calibration.method}]",
        "method": calibration.method,
        "saved": stamp,
    }
    store.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return store


def delete_profile(name: str, path: Optional[Path] = None) -> bool:
    """Remove a saved profile. Returns False if it was not there."""
    store = profiles_path(path)
    if not store.is_file():
        return False
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Profile store {store} could not be read: {exc}") from exc
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict) or name not in profiles:
        return False
    del profiles[name]
    store.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


CALIBRATION_OPTIONS_TEXT = """\
Calibration is required and will not be guessed. Choose one:

  --calibrator-tone FILE [--calibrator-level-dB 94|114]
        Preferred. Measures the whole acquisition chain as it was configured for
        this test. Add --calibrator-post FILE for the post-test drift check.

  --sensitivity-mV V [--preamp-gain-dB G] [--adc-full-scale-V V]
        From the microphone datasheet and the recorder's front panel.

  --Pa-per-FS V
        A conversion factor you already trust.

  --preset NAME
        A named profile measured earlier for this rig (--list-presets).
        Save one with --save-preset NAME after calibrating.

  --uncalibrated
        No calibration. Every level is then reported as "dB re FS", which is a
        relative number and is not sound pressure level."""


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisConfig:
    """
    Every parameter that affects the numbers, validated on construction.

    There is deliberately no default for any calibration quantity: a run must say
    how it was calibrated or say that it was not.
    """

    # ---- Calibration inputs (mutually exclusive; see resolve_calibration) ----
    Pa_per_FS: Optional[float] = None
    sensitivity_mV_per_Pa: Optional[float] = None
    preamp_gain_dB: float = 0.0
    adc_full_scale_V: Optional[float] = None
    V_per_FS: Optional[float] = None            # legacy alias for adc_full_scale_V
    calibrator_tone_file: Optional[str] = None
    calibrator_post_file: Optional[str] = None
    calibrator_level_dB: float = 114.0
    calibrator_frequency_Hz: float = 1000.0
    preset: Optional[str] = None
    uncalibrated: bool = False
    calibration_description: str = ""

    # ---- Channel selection ----
    channel: int = 0
    mono_mix: bool = False

    # ---- Shot detection ----
    detection_threshold_dB: Optional[float] = None
    threshold_relative_dB: Optional[float] = None
    refractory_ms: float = 200.0
    pre_shot_ms: float = 50.0
    post_shot_ms: float = 200.0
    min_shots: int = 0
    max_shots: int = 1000

    # ---- STFT ----
    nperseg: Optional[int] = None       # None selects recommended_nperseg()
    noverlap: Optional[int] = None      # None derives from overlap_fraction
    overlap_fraction: float = 0.75
    stft_window: str = "hann"

    # ---- Analysis ----
    load_dtype: str = "float32"
    compute_bands: bool = True
    compute_time_series: bool = True
    band_hop_ms: float = 10.0
    band_time_weighting: str = "fast"
    protection_NRR_dB: float = 0.0

    # ---- Output ----
    make_plots: bool = True
    save_per_shot_plots: bool = True
    save_aggregate_plots: bool = True
    plot_formats: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.plot_formats is None:
            self.plot_formats = ["png"]
        else:
            self.plot_formats = validate_formats(self.plot_formats)

        # Calibration quantities
        if self.Pa_per_FS is not None:
            self.Pa_per_FS = _positive(self.Pa_per_FS, "Pa_per_FS")
        if self.sensitivity_mV_per_Pa is not None:
            self.sensitivity_mV_per_Pa = _positive(self.sensitivity_mV_per_Pa, "sensitivity_mV_per_Pa")
        if self.adc_full_scale_V is not None:
            self.adc_full_scale_V = _positive(self.adc_full_scale_V, "adc_full_scale_V")
        if self.V_per_FS is not None:
            self.V_per_FS = _positive(self.V_per_FS, "V_per_FS")
        self.preamp_gain_dB = _finite(self.preamp_gain_dB, "preamp_gain_dB")
        self.calibrator_level_dB = _finite(self.calibrator_level_dB, "calibrator_level_dB")
        if not (40.0 <= self.calibrator_level_dB <= 180.0):
            raise ConfigurationError(
                f"calibrator_level_dB of {self.calibrator_level_dB} dB is not a plausible "
                f"calibrator output (expected roughly 94 or 114 dB)"
            )
        self.calibrator_frequency_Hz = _positive(self.calibrator_frequency_Hz, "calibrator_frequency_Hz")

        # Channels
        self.channel = _integer(self.channel, "channel", minimum=0, maximum=1023)

        # Detection
        if self.detection_threshold_dB is not None:
            self.detection_threshold_dB = _finite(self.detection_threshold_dB, "detection_threshold_dB")
        if self.threshold_relative_dB is not None:
            self.threshold_relative_dB = _positive(self.threshold_relative_dB, "threshold_relative_dB")
        self.refractory_ms = _positive(self.refractory_ms, "refractory_ms")
        self.pre_shot_ms = _non_negative(self.pre_shot_ms, "pre_shot_ms")
        self.post_shot_ms = _positive(self.post_shot_ms, "post_shot_ms")
        self.min_shots = _integer(self.min_shots, "min_shots", minimum=0, maximum=100_000)
        self.max_shots = _integer(self.max_shots, "max_shots", minimum=1, maximum=100_000)
        if self.min_shots > self.max_shots:
            raise ConfigurationError(
                f"min_shots ({self.min_shots}) cannot exceed max_shots ({self.max_shots})"
            )

        # STFT
        if self.nperseg is not None:
            self.nperseg = _integer(self.nperseg, "nperseg", minimum=64, maximum=1 << 20)
        self.overlap_fraction = _finite(self.overlap_fraction, "overlap_fraction")
        if not (0.0 <= self.overlap_fraction < 1.0):
            raise ConfigurationError(
                f"overlap_fraction must be at least 0 and below 1, got {self.overlap_fraction}"
            )
        if self.noverlap is not None:
            self.noverlap = _integer(self.noverlap, "noverlap", minimum=0, maximum=(1 << 20) - 1)
            if self.nperseg is not None and self.noverlap >= self.nperseg:
                raise ConfigurationError(
                    f"noverlap ({self.noverlap}) must be smaller than nperseg ({self.nperseg})"
                )
        if not isinstance(self.stft_window, str) or not self.stft_window:
            raise ConfigurationError("stft_window must be a window name")

        # Analysis
        if self.load_dtype not in ("float32", "float64"):
            raise ConfigurationError(
                f"load_dtype must be float32 or float64, got {self.load_dtype!r}"
            )
        self.band_hop_ms = _positive(self.band_hop_ms, "band_hop_ms")
        if self.band_time_weighting not in ("fast", "slow", "impulse"):
            raise ConfigurationError(
                f"band_time_weighting must be fast, slow or impulse, got {self.band_time_weighting!r}"
            )
        self.protection_NRR_dB = _non_negative(self.protection_NRR_dB, "protection_NRR_dB")

    # ---- Derived values ----

    def resolved_nperseg(self, sample_rate: int) -> int:
        """FFT size actually used: explicit if given, otherwise scaled to the rate."""
        if self.nperseg is not None:
            return int(self.nperseg)
        return int(recommended_nperseg(sample_rate, target_ms=2.0))

    def resolved_noverlap(self, nperseg: int) -> int:
        """
        Overlap actually used, always derived from the window it belongs to.

        A fixed overlap is what made the UI's 512- and 1024-sample windows abort:
        1536 is not a legal overlap for a 512-point window. This derives it and
        then clamps it into the legal range rather than failing.
        """
        if self.noverlap is not None:
            noverlap = int(self.noverlap)
        else:
            noverlap = int(round(nperseg * self.overlap_fraction))
        return int(min(max(noverlap, 0), max(nperseg - 1, 0)))

    def static_formats(self) -> List[str]:
        """Formats handled by matplotlib (html is rendered separately by Plotly)."""
        return [f for f in (self.plot_formats or []) if f != "html"]

    # ---- Serialisation ----

    @classmethod
    def from_json(cls, path: Path, warnings: Optional[List[str]] = None) -> "AnalysisConfig":
        """
        Load configuration from JSON, ignoring unknown keys with a warning.

        An unknown key used to reach the constructor and raise TypeError, so a
        config written by a newer build broke an older one outright.
        """
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(f"Config file {path} could not be read: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Config file {path} is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigurationError(f"Config file {path} must contain a JSON object")

        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(data) - known)
        if unknown:
            message = (
                f"Config file {path.name} contains unrecognised key(s): "
                f"{', '.join(unknown)}. They were ignored."
            )
            logger.warning(message)
            if warnings is not None:
                warnings.append(message)
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]

    def to_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Calibration resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_calibration(
    config: AnalysisConfig,
    *,
    presets: Optional[Dict[str, CalibrationPreset]] = None,
    warnings: Optional[List[str]] = None,
) -> Calibration:
    """
    Turn the configured calibration inputs into a Calibration, or refuse.

    Exactly one source must be given. There is no fallback, because a silent
    fallback is how every uncalibrated run in the previous build came to be
    published as absolute dB SPL.
    """
    presets = presets if presets is not None else load_profiles(warnings=warnings)

    chosen: List[str] = []
    if config.calibrator_tone_file:
        chosen.append("--calibrator-tone")
    if config.sensitivity_mV_per_Pa is not None:
        chosen.append("--sensitivity-mV")
    if config.Pa_per_FS is not None:
        chosen.append("--Pa-per-FS")
    if config.preset:
        chosen.append("--preset")
    if config.uncalibrated:
        chosen.append("--uncalibrated")

    if not chosen:
        raise CalibrationRequired(CALIBRATION_OPTIONS_TEXT)
    if len(chosen) > 1:
        raise ConfigurationError(
            "Calibration was specified more than once (" + ", ".join(chosen) + "). "
            "Give exactly one calibration source."
        )

    method = chosen[0]

    if method == "--calibrator-tone":
        return _calibration_from_tone(config)

    if method == "--sensitivity-mV":
        adc_V = config.adc_full_scale_V if config.adc_full_scale_V is not None else config.V_per_FS
        if adc_V is None:
            adc_V = 1.0
            message = (
                "--adc-full-scale-V was not given; assuming 1.0 V at digital full scale. "
                "If the recorder's full-scale input is not 1 V, every level is wrong by "
                "20*log10(actual/1.0) dB."
            )
            logger.warning(message)
            if warnings is not None:
                warnings.append(message)
        return Calibration.from_recording_chain(
            sensitivity_mV_per_Pa=float(config.sensitivity_mV_per_Pa),
            adc_full_scale_V=float(adc_V),
            preamp_gain_dB=float(config.preamp_gain_dB),
            description=config.calibration_description,
        )

    if method == "--Pa-per-FS":
        return Calibration(
            Pa_per_FS=float(config.Pa_per_FS),
            calibrated=True,
            method="direct",
            description=config.calibration_description
            or f"Direct: {float(config.Pa_per_FS):.6g} Pa/FS (supplied by operator)",
        )

    if method == "--preset":
        name = str(config.preset)
        preset = presets.get(name)
        if preset is None:
            raise ConfigurationError(
                f"Unknown calibration preset {name!r}. Available: "
                + (", ".join(sorted(presets)) or "(none)")
                + ". Use --list-presets for details."
            )
        message = (
            f"Using calibration preset {preset.name!r}. It is only valid for the rig it "
            f"was measured on: {preset.provenance}"
        )
        logger.info(message)
        if warnings is not None:
            warnings.append(message)
        return preset.to_calibration()

    return Calibration.uncalibrated()


def _calibration_from_tone(config: AnalysisConfig) -> Calibration:
    """Derive calibration from a recorded calibrator tone (and optional post-test check)."""
    tone_path = Path(str(config.calibrator_tone_file)).expanduser()
    if not tone_path.is_file():
        raise ConfigurationError(f"Calibrator recording not found: {tone_path}")

    tone, sr = _read_calibrator(tone_path, config)

    post = None
    if config.calibrator_post_file:
        post_path = Path(str(config.calibrator_post_file)).expanduser()
        if not post_path.is_file():
            raise ConfigurationError(f"Post-test calibrator recording not found: {post_path}")
        post, post_sr = _read_calibrator(post_path, config)
        if post_sr != sr:
            raise ConfigurationError(
                f"Pre-test calibrator is {sr} Hz but post-test is {post_sr} Hz; "
                f"they must come from the same recorder configuration."
            )

    try:
        return Calibration.from_calibrator_tone(
            tone,
            sr,
            calibrator_level_dB=float(config.calibrator_level_dB),
            tone_frequency_Hz=float(config.calibrator_frequency_Hz),
            description=config.calibration_description,
            post_test_samples=post,
        )
    except ValueError as exc:
        raise ConfigurationError(f"Calibrator tone in {tone_path.name} is unusable: {exc}") from exc


def _read_calibrator(path: Path, config: AnalysisConfig) -> Tuple[np.ndarray, int]:
    """Read a calibrator recording, taking the same channel the test will use."""
    try:
        frames, sr, _duration, channels = get_wav_info(path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a configuration error
        raise ConfigurationError(f"Calibrator recording {path} could not be read: {exc}") from exc
    if frames <= 0:
        raise ConfigurationError(f"Calibrator recording {path} is empty")

    channel = config.channel if config.channel < channels else 0
    data = load_wav(path, dtype="float64", channel=channel)
    return np.asarray(data.samples, dtype=np.float64), int(sr)


# ═══════════════════════════════════════════════════════════════════════════
#  Input preparation (audio, video, channel selection)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PreparedInput:
    """The audio file that will actually be analysed, and where it came from."""
    audio_path: Path
    original_path: Path
    extracted: bool = False
    from_cache: bool = False


def prepare_input(path: Path, warnings: Optional[List[str]] = None) -> PreparedInput:
    """
    Resolve the user's file to a readable audio file.

    A video given on the command line is extracted exactly as one chosen in the
    picker: previously only the picker branch extracted, so `main.py clip.mp4`
    walked straight into the loader and failed.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"File not found: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    needs_extraction = suffix in VIDEO_EXTS

    if not needs_extraction:
        try:
            import soundfile as sf  # noqa: PLC0415

            sf.info(str(path))
            return PreparedInput(audio_path=path, original_path=path)
        except Exception as exc:  # noqa: BLE001 - may still be a container ffmpeg can read
            logger.info("%s is not directly readable as audio (%s); trying extraction", path.name, exc)
            needs_extraction = True
            if suffix not in AUDIO_EXTS and warnings is not None:
                warnings.append(f"{path.name} is not a recognised audio file; attempting extraction.")

    from ExtractAudio import ExtractionError, extract_audio_cached  # noqa: PLC0415

    say(f"  Extracting audio from {path.name} (original rate, depth and channels preserved)...")
    try:
        audio_path, cached = extract_audio_cached(path)
    except ExtractionError as exc:
        raise ConfigurationError(str(exc)) from exc

    if cached:
        say(f"  Using previously extracted audio: {audio_path.name}")
    else:
        say(f"  Extracted: {audio_path.name}")
    return PreparedInput(
        audio_path=audio_path, original_path=path, extracted=True, from_cache=cached
    )


def describe_channel(channel: Optional[int], mono_mix: bool, n_channels: int) -> str:
    """Human/machine description of which channel produced the numbers."""
    if mono_mix:
        return f"mono mix of {n_channels} channels"
    return f"channel {channel}"


def read_samples(
    path: Path,
    start: int,
    count: int,
    *,
    channel: Optional[int],
    mono_mix: bool,
    dtype: str,
) -> np.ndarray:
    """Read a span of full-scale samples for the selected channel."""
    samples, _sr = load_wav_chunk(
        path, start, count,
        dtype=dtype,
        mono=mono_mix,
        channel=None if mono_mix else channel,
    )
    return np.asarray(samples)


# ═══════════════════════════════════════════════════════════════════════════
#  Output directory and artifact writing
# ═══════════════════════════════════════════════════════════════════════════

def create_output_directory(base_dir: Path, input_file: Path, suffix: str = "") -> Path:
    """
    Create a unique output directory.

    Directory names used to be keyed to one-second resolution and created with
    exist_ok=True, so two runs in the same second silently merged their results
    into one directory. Names now carry milliseconds and a short random tag, and
    the directory is created exclusively, so a collision is impossible.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in input_file.stem)[:64]
    tail = f"_{suffix}" if suffix else ""

    for _attempt in range(64):
        token = uuid.uuid4().hex[:6]
        candidate = base_dir / f"{stem}_{stamp}{tail}_{token}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SasaError(f"Could not create a unique output directory under {base_dir}")


def _json_default(obj: Any) -> Any:
    """Serialise numpy scalars, arrays and paths that reach the JSON writer."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _sanitise(obj: Any) -> Any:
    """Replace non-finite floats with None so the JSON stays strictly valid."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating) and not math.isfinite(float(obj)):
        return None
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically, so a reader never sees a half-written record."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_sanitise(data), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


CSV_COLUMNS: List[Tuple[str, str, str]] = [
    # (column stem, unit placeholder, ShotMetrics attribute)
    ("shot_number", "", "shot_number"),
    ("time", "s", "_time_s"),
    ("valid", "", "valid"),
    ("clipped", "", "clipped"),
    ("window_truncated", "", "window_truncated"),
    ("rise_time_resolved", "", "rise_time_resolved"),
    ("window_duration", "ms", "_duration_ms"),
    ("integration_window", "ms", "_integration_ms"),
    ("Lpeak_Z", "{unit}", "Lpeak_Z"),
    ("Lpeak_A", "{unit}", "Lpeak_A"),
    ("Lpeak_C", "{unit}", "Lpeak_C"),
    ("LZE", "{unit}", "LZE"),
    ("LAE", "{unit}", "LAE"),
    ("LCE", "{unit}", "LCE"),
    ("LZFmax", "{unit}", "LZFmax"),
    ("LZSmax", "{unit}", "LZSmax"),
    ("LAFmax", "{unit}", "LAFmax"),
    ("LASmax", "{unit}", "LASmax"),
    ("LAImax", "{unit}", "LAImax"),
    ("LZImax", "{unit}", "LZImax"),
    ("rise_time", "us", "rise_time_us"),
    ("a_duration", "ms", "a_duration_ms"),
    ("b_duration", "ms", "b_duration_ms"),
    ("specific_impulse", "Pa*s", "specific_impulse_Pa_s"),
    ("peak_overpressure", "Pa", "peak_overpressure_Pa"),
    ("crest_factor", "dB", "crest_factor_dB"),
    ("spectral_centroid", "Hz", "spectral_centroid_Hz"),
    ("kurtosis", "", "kurtosis"),
    ("noise_floor", "{unit}", "noise_floor_dB"),
    ("snr", "dB", "snr_dB"),
    ("notes", "", "_notes"),
]


def save_csv_summary(
    output_path: Path,
    shot_metrics: Sequence[ShotMetrics],
    shots: Sequence[ShotEvent],
    *,
    record: Dict[str, Any],
) -> Optional[Path]:
    """
    Write the per-shot metrics table.

    Every level header carries its unit, every row carries the shot's time and its
    validity flags, and the file opens with a commented provenance preamble, so a
    CSV that has been emailed on its own still states what instrument produced it,
    how it was calibrated, and whether the measurement was admissible.

    Returns the path written, or None when there is nothing to write - the caller
    must not claim a file that does not exist.
    """
    if not shot_metrics:
        return None

    unit = record["calibration"]["level_unit"]
    software = record["software"]
    analysis = record["analysis"]
    settings = record["settings"]
    quality = record["quality"]
    detection = record["detection"]

    preamble = [
        "SASA per-shot acoustic metrics",
        f"schema_version: {record['schema_version']}",
        f"software: {software['name']} {software['version']} "
        f"(commit {software.get('git_commit') or 'unknown'}"
        f"{', local changes' if software.get('git_dirty') else ''})",
        f"analysis_timestamp: {analysis['timestamp']}",
        f"input_file: {analysis['input_file']}",
        f"input_sha256: {analysis['input_sha256']}",
        f"sample_rate_Hz: {record['source']['sample_rate']}",
        f"channel_used: {record['source']['channel_used']}",
        f"calibration_method: {record['calibration']['method']}",
        f"calibration_Pa_per_FS: {record['calibration']['Pa_per_FS']!r}",
        f"calibration_description: {record['calibration']['description']}",
        f"level_unit: {unit}",
        f"detection_threshold_dB: {detection['threshold_dB']} ({detection['threshold_mode']})",
        f"refractory_ms: {settings['refractory_ms']}",
        f"pre_shot_ms: {settings['pre_shot_ms']}",
        f"post_shot_ms: {settings['post_shot_ms']}",
        f"measurement_valid: {record['validity']['measurement_valid']}",
    ]
    if not record["calibration"]["calibrated"]:
        preamble.append(
            "NOTE: this analysis is UNCALIBRATED. Levels are dB re full scale, "
            "not sound pressure level, and are comparable only within this file."
        )
    for err in quality.get("errors", []):
        preamble.append(f"INVALID: {err}")
    for warn in quality.get("warnings", []):
        preamble.append(f"WARNING: {warn}")

    headers = []
    for stem, unit_spec, _attr in CSV_COLUMNS:
        label = unit_spec.format(unit=unit) if unit_spec else ""
        headers.append(f"{stem} ({label})" if label else stem)

    times = {s.shot_number: s.time_s for s in shots}

    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        for line in preamble:
            handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow(headers)

        for metric in shot_metrics:
            row = []
            for _stem, _unit_spec, attr in CSV_COLUMNS:
                if attr == "_time_s":
                    value = times.get(metric.shot_number)
                    row.append("" if value is None else f"{value:.5f}")
                elif attr == "_duration_ms":
                    row.append(f"{metric.duration_s * 1000.0:.2f}")
                elif attr == "_integration_ms":
                    row.append(f"{metric.integration_window_s * 1000.0:.2f}")
                elif attr == "_notes":
                    row.append("; ".join(metric.notes))
                else:
                    value = getattr(metric, attr)
                    if isinstance(value, bool):
                        row.append("true" if value else "false")
                    elif isinstance(value, (int, np.integer)):
                        row.append(str(int(value)))
                    elif isinstance(value, (float, np.floating)):
                        number = float(value)
                        if not math.isfinite(number):
                            row.append("")
                        elif number != 0.0 and abs(number) < 1.0:
                            # Small quantities (specific impulse in Pa*s) would round to
                            # 0.01 at two decimals and lose the measurement entirely.
                            row.append(f"{number:.6g}")
                        else:
                            row.append(f"{number:.2f}")
                    else:
                        row.append(str(value))
            writer.writerow(row)

    return output_path


def save_insertion_loss_csv(output_path: Path, comparison: Dict[str, Any]) -> Optional[Path]:
    """Write the insertion-loss table (the deliverable) as its own CSV."""
    metrics = comparison.get("metrics") or []
    bands = comparison.get("bands") or {}
    if not metrics and not bands.get("frequencies_Hz"):
        return None

    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        handle.write("# SASA insertion loss (reference minus test; positive = quieter)\n")
        handle.write(f"# reference_dir: {comparison.get('reference_dir', '')}\n")
        handle.write(f"# reference_input: {comparison.get('reference_input', '')}\n")
        for warn in comparison.get("warnings", []):
            handle.write(f"# WARNING: {warn}\n")

        writer = csv.writer(handle)
        if metrics:
            writer.writerow([
                "metric", "reference (dB)", "test (dB)", "insertion_loss (dB)",
                "ci95 (dB)", "reference_n", "test_n",
            ])
            for row in metrics:
                writer.writerow([
                    row["metric"], row["reference_dB"], row["test_dB"],
                    row["reduction_dB"], row["ci95_dB"], row["reference_n"], row["test_n"],
                ])
        if bands.get("frequencies_Hz"):
            writer.writerow([])
            writer.writerow(["band_centre (Hz)", "reference (dB)", "test (dB)", "insertion_loss (dB)"])
            for freq, ref, test, il in zip(
                bands["frequencies_Hz"], bands["reference_dB"],
                bands["test_dB"], bands["insertion_loss_dB"],
            ):
                writer.writerow([freq, ref, test, il])
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  Reference comparison (insertion loss)
# ═══════════════════════════════════════════════════════════════════════════

def load_reference_record(reference_dir: Path) -> Dict[str, Any]:
    """Read a previous analysis output directory's record."""
    reference_dir = Path(reference_dir).expanduser().resolve()
    if reference_dir.is_file() and reference_dir.name.endswith(".json"):
        record_path = reference_dir
    else:
        record_path = reference_dir / "analysis_metadata.json"
    if not record_path.is_file():
        raise ConfigurationError(
            f"--reference {reference_dir} does not contain analysis_metadata.json. "
            f"Point it at the output directory of a previous UNSUPPRESSED analysis."
        )
    try:
        data = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Reference record {record_path} could not be read: {exc}") from exc
    if not isinstance(data, dict) or "aggregate" not in data:
        raise ConfigurationError(f"Reference record {record_path} is not a SASA analysis record")
    return data


def aggregate_from_record(record: Dict[str, Any]) -> AggregateMetrics:
    """Rebuild an AggregateMetrics from a stored record so it can be compared."""
    aggregate = record.get("aggregate") or {}
    stats: Dict[str, MetricStats] = {}
    for name, entry in (aggregate.get("statistics") or {}).items():
        try:
            stats[name] = MetricStats(
                name=entry.get("name", name),
                unit=entry.get("unit", ""),
                n=int(entry.get("n", 0)),
                mean=float(entry.get("mean", float("nan"))),
                std=float(entry.get("std", 0.0)),
                minimum=float(entry.get("min", float("nan"))),
                maximum=float(entry.get("max", float("nan"))),
                median=float(entry.get("median", float("nan"))),
                ci95_half_width=float(entry.get("ci95_half_width", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Reference statistic %r was unusable and is skipped (%s)", name, exc)

    return AggregateMetrics(
        n_shots=int(aggregate.get("n_shots", 0)),
        n_valid=int(aggregate.get("n_valid", 0)),
        stats=stats,
        band_frequencies=np.asarray(aggregate.get("band_frequencies_Hz") or [], dtype=np.float64),
        band_exposure_mean_dB=np.asarray(aggregate.get("band_exposure_mean_dB") or [], dtype=np.float64),
    )


def compare_with_reference(
    reference_dir: Path,
    test_aggregate: AggregateMetrics,
    test_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute insertion loss against a previous unsuppressed analysis.

    Preconditions that would invalidate the comparison - different level units,
    different sample rates, a suppressed reference, an empty reference - are
    checked and reported rather than assumed.
    """
    reference = load_reference_record(reference_dir)
    ref_aggregate = aggregate_from_record(reference)
    warnings: List[str] = []

    ref_cal = reference.get("calibration") or {}
    test_cal = test_record["calibration"]
    ref_source = reference.get("source") or {}
    test_source = test_record["source"]

    if ref_cal.get("level_unit") != test_cal["level_unit"]:
        raise ConfigurationError(
            f"Reference levels are in {ref_cal.get('level_unit')!r} but this analysis is in "
            f"{test_cal['level_unit']!r}. Insertion loss between different level units is meaningless."
        )
    if ref_aggregate.n_valid == 0:
        raise ConfigurationError(
            f"The reference analysis in {reference_dir} has no valid shots to compare against."
        )

    if ref_cal.get("Pa_per_FS") != test_cal["Pa_per_FS"]:
        warnings.append(
            f"Reference and test use different calibration factors "
            f"({ref_cal.get('Pa_per_FS')} vs {test_cal['Pa_per_FS']} Pa/FS). "
            f"The comparison is only meaningful if both are genuinely calibrated."
        )
    if ref_source.get("sample_rate") != test_source.get("sample_rate"):
        warnings.append(
            f"Reference was recorded at {ref_source.get('sample_rate')} Hz and this test at "
            f"{test_source.get('sample_rate')} Hz; band comparisons may not align."
        )
    if not ref_cal.get("calibrated", False):
        warnings.append(
            "The reference analysis is uncalibrated, so the insertion loss is a difference "
            "of relative levels. It is valid only if both recordings used identical gain."
        )

    ref_meta = reference.get("test_metadata") or {}
    test_meta = test_record.get("test_metadata") or {}
    if ref_meta.get("configuration") == "suppressed":
        warnings.append(
            "The reference is recorded as 'suppressed'. Insertion loss is defined against an "
            "UNSUPPRESSED reference; the sign of the result is probably inverted."
        )
    for field_name in ("mic_distance_m", "mic_angle_deg", "mic_model"):
        ref_value, test_value = ref_meta.get(field_name), test_meta.get(field_name)
        if ref_value is not None and test_value is not None and ref_value != test_value:
            warnings.append(
                f"Reference and test differ in {field_name} ({ref_value} vs {test_value}); "
                f"the difference in level is not attributable to the suppressor alone."
            )

    losses = compute_insertion_loss(ref_aggregate, test_aggregate)

    # The band block always carries its four documented keys, empty when the
    # comparison was refused, so a consumer can index them without first
    # testing whether per-band insertion loss happened to be computable.
    bands: Dict[str, Any] = {
        "frequencies_Hz": [],
        "reference_dB": [],
        "test_dB": [],
        "insertion_loss_dB": [],
    }
    ref_bands = ref_aggregate.band_exposure_mean_dB
    test_bands = test_aggregate.band_exposure_mean_dB
    if ref_bands.size and test_bands.size:
        ref_freqs = ref_aggregate.band_frequencies
        test_freqs = test_aggregate.band_frequencies
        if ref_bands.shape == test_bands.shape and np.allclose(ref_freqs, test_freqs):
            try:
                il = band_insertion_loss(ref_bands, test_bands)
                bands = {
                    "frequencies_Hz": [float(f) for f in test_freqs],
                    "reference_dB": [round(float(v), 1) for v in ref_bands],
                    "test_dB": [round(float(v), 1) for v in test_bands],
                    "insertion_loss_dB": [round(float(v), 1) for v in il],
                }
            except ValueError as exc:
                warnings.append(f"Per-band insertion loss could not be computed: {exc}")
        else:
            warnings.append(
                "Reference and test band vectors describe different filter banks "
                "(different sample rates), so per-band insertion loss was not computed."
            )
    else:
        warnings.append("Per-band insertion loss needs band analysis in both runs; it was skipped.")

    return {
        "reference_dir": str(Path(reference_dir).resolve()),
        "reference_input": (reference.get("analysis") or {}).get("input_file", ""),
        "reference_sha256": (reference.get("analysis") or {}).get("input_sha256"),
        "reference_n_shots": ref_aggregate.n_valid,
        "test_n_shots": test_aggregate.n_valid,
        "level_unit": test_cal["level_unit"],
        "metrics": [loss.to_dict() for loss in losses],
        "bands": bands,
        "warnings": warnings,
    }


def print_insertion_loss(comparison: Dict[str, Any]) -> None:
    """Print the deliverable: how much quieter the test configuration is."""
    unit = comparison.get("level_unit", "dB")
    say("")
    say("  Insertion loss (reference - test; positive means quieter)")
    say(f"    Reference: {comparison.get('reference_input', '(unknown)')}")
    say(f"    Shots:     {comparison.get('reference_n_shots')} reference / "
        f"{comparison.get('test_n_shots')} test")
    if not comparison.get("metrics"):
        say("    No comparable metrics were found in the reference record.")
    for row in comparison.get("metrics", []):
        say(f"    {row['metric']:<10} {row['reference_dB']:>8.1f} -> {row['test_dB']:>8.1f} "
            f"{unit}   IL = {row['reduction_dB']:+6.1f} dB  (+/-{row['ci95_dB']:.1f})")
    bands = comparison.get("bands") or {}
    if bands.get("insertion_loss_dB"):
        il = bands["insertion_loss_dB"]
        freqs = bands["frequencies_Hz"]
        best = int(np.argmax(il))
        worst = int(np.argmin(il))
        say(f"    Per-band IL: best {il[best]:+.1f} dB at {freqs[best]:.0f} Hz, "
            f"worst {il[worst]:+.1f} dB at {freqs[worst]:.0f} Hz "
            f"({len(il)} bands written to the record)")
    for warn in comparison.get("warnings", []):
        say(f"    WARNING: {warn}")


# ═══════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    """Everything one analysis produced, including the record written to disk."""
    input_file: Path
    output_dir: Path
    calibration: Calibration
    sample_rate: int
    duration_s: float
    n_shots: int
    shots: List[ShotEvent]
    shot_metrics: List[ShotMetrics]
    aggregate: AggregateMetrics
    config: AnalysisConfig
    timestamp: str
    quality: Optional[SignalQuality] = None
    detection: Optional[DetectionReport] = None
    measurement_valid: bool = True
    reference_requested: bool = False
    insertion_loss_produced: bool = False
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.record


# ═══════════════════════════════════════════════════════════════════════════
#  The pipeline
# ═══════════════════════════════════════════════════════════════════════════

def analyze_file(
    wav_path: Path,
    config: AnalysisConfig,
    output_base: Optional[Path] = None,
    *,
    output_dir: Optional[Path] = None,
    dir_suffix: str = "",
    metadata: Optional[TestMetadata] = None,
    reference_dir: Optional[Path] = None,
    calibration: Optional[Calibration] = None,
    original_path: Optional[Path] = None,
    warnings: Optional[List[str]] = None,
    config_file: Optional[Path] = None,
) -> AnalysisResult:
    """
    Run the complete analysis pipeline on one audio file and one channel.

    Args:
        wav_path: Audio file (already extracted, if the input was a video).
        config: Validated analysis configuration.
        output_base: Directory to create the run directory under.
        output_dir: Use this exact directory instead of creating one.
        dir_suffix: Tag appended to the created directory name (e.g. "ch01").
        metadata: Test conditions record.
        reference_dir: Previous unsuppressed analysis to compute insertion loss against.
        calibration: Pre-resolved calibration; resolved from config when omitted.
        original_path: The file the operator actually supplied (video before extraction).
        warnings: Collector for non-fatal problems; they are written into the record.
        config_file: Config file the settings came from, for the record.

    Returns:
        AnalysisResult, including the record that was written to disk.
    """
    started = time.time()
    run_warnings: List[str] = list(warnings) if warnings else []
    wav_path = Path(wav_path)

    if calibration is None:
        calibration = resolve_calibration(config, warnings=run_warnings)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # ---- Output directory (created before anything can fail, so logs land somewhere) ----
    named_after = Path(original_path) if original_path else wav_path
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(output_base) if output_base is not None else wav_path.parent / "analysis"
        out_dir = create_output_directory(base, named_after, dir_suffix)

    file_handler = add_log_file(out_dir / "analysis.log")
    logger.info("SASA %s analysing %s -> %s", __version__, wav_path, out_dir)

    try:
        return _run_pipeline(
            wav_path=wav_path,
            config=config,
            calibration=calibration,
            out_dir=out_dir,
            timestamp=timestamp,
            started=started,
            metadata=metadata or TestMetadata(),
            reference_dir=reference_dir,
            original_path=Path(original_path) if original_path else wav_path,
            run_warnings=run_warnings,
            config_file=config_file,
        )
    finally:
        if file_handler is not None:
            file_handler.close()
            logger.removeHandler(file_handler)


def _run_pipeline(
    *,
    wav_path: Path,
    config: AnalysisConfig,
    calibration: Calibration,
    out_dir: Path,
    timestamp: str,
    started: float,
    metadata: TestMetadata,
    reference_dir: Optional[Path],
    original_path: Path,
    run_warnings: List[str],
    config_file: Optional[Path],
) -> AnalysisResult:
    say("")
    say("=" * 64)
    say("SASA - Shot Acoustic Spectral Analysis")
    say("=" * 64)
    say(f"Input:  {original_path}")
    say(f"Output: {out_dir}")
    announce_output_dir(out_dir)
    progress(2, "Reading source file")

    # ---- Source metadata ----
    try:
        frames, sample_rate, duration_s, n_channels = get_wav_info(wav_path)
    except Exception as exc:  # noqa: BLE001 - a bad file is a configuration error
        raise ConfigurationError(f"{wav_path} could not be read as audio: {exc}") from exc

    if frames <= 0:
        raise ConfigurationError(f"{wav_path.name} contains no audio samples (0 frames).")
    if sample_rate <= 0:
        raise ConfigurationError(f"{wav_path.name} declares an impossible sample rate ({sample_rate} Hz).")

    subtype = ""
    try:
        import soundfile as sf  # noqa: PLC0415

        subtype = sf.info(str(wav_path)).subtype or ""
    except Exception as exc:  # noqa: BLE001 - the subtype is documentation, not a blocker
        run_warnings.append(f"Sample format could not be read from {wav_path.name}: {exc}")
        logger.warning("Could not read subtype of %s: %s", wav_path, exc)

    # ---- Channel selection ----
    channel: Optional[int] = None if config.mono_mix else config.channel
    if not config.mono_mix:
        if config.channel >= n_channels:
            raise ConfigurationError(
                f"--channel {config.channel} was requested but {wav_path.name} has "
                f"{n_channels} channel(s) (0-{n_channels - 1})."
            )
        if n_channels > 1:
            message = (
                f"{wav_path.name} has {n_channels} channels; analysing channel "
                f"{config.channel} only. Use --channel N to pick another, or "
                f"--channels all to analyse each separately."
            )
            logger.warning(message)
            run_warnings.append(message)
            say(f"  NOTE: {message}")
    else:
        message = (
            f"Averaging {n_channels} channels to mono was explicitly requested. If the "
            f"microphones are not co-located this destroys the measurement."
        )
        logger.warning(message)
        run_warnings.append(message)

    channel_used = describe_channel(config.channel, config.mono_mix, n_channels)

    say(f"  Sample rate: {sample_rate} Hz   Duration: {duration_s:.2f} s   "
        f"Channels: {n_channels} ({channel_used})")

    progress(5, "Hashing input for provenance")
    # The hash identifies the file named in `path` - the one the operator supplied.
    # When that was a video, the extracted audio is recorded separately so the
    # analysed bytes are also identified, and neither hash is attributed to the
    # wrong file.
    source_hash = file_sha256(original_path)
    source = SourceInfo(
        path=str(original_path),
        sha256=source_hash,
        sample_rate=int(sample_rate),
        channels=int(n_channels),
        subtype=subtype,
        frames=int(frames),
        duration_s=round(float(duration_s), 4),
        channel_used=channel_used,
    )

    # ---- Calibration report ----
    say("")
    say(f"  Calibration: {calibration.description}")
    say(f"    method {calibration.method}, {calibration.Pa_per_FS:.6g} Pa/FS, "
        f"levels in {calibration.level_unit} (full scale = {calibration.full_scale_dB:.1f} dB)")
    if not calibration.calibrated:
        say("    UNCALIBRATED - every level below is relative (dB re FS), NOT sound pressure level.")
    if calibration.residual_dB is not None:
        say(f"    Post-test calibrator drift: {calibration.residual_dB:+.2f} dB")
        if abs(calibration.residual_dB) > 0.5:
            message = (
                f"Calibrator drifted {calibration.residual_dB:+.2f} dB between pre- and post-test "
                f"(limit 0.5 dB); the chain was not stable across this session."
            )
            run_warnings.append(message)
            logger.warning(message)

    chunked = duration_s > MAX_DURATION_FULL_LOAD_S
    nperseg = config.resolved_nperseg(sample_rate)
    noverlap = config.resolved_noverlap(nperseg)
    logger.info("STFT: nperseg=%d noverlap=%d (%.0f%% overlap, %.2f ms window)",
                nperseg, noverlap, 100.0 * noverlap / nperseg, 1000.0 * nperseg / sample_rate)

    progress(8, "Assessing recording quality")

    if chunked:
        say("")
        say(f"  Long recording ({duration_s / 60.0:.1f} min): using overlapping "
            f"{CHUNK_DURATION_S:.0f} s chunks with {CHUNK_CONTEXT_S:.0f} s of context.")
        analysis = _analyze_chunked(
            wav_path=wav_path, config=config, calibration=calibration,
            sample_rate=sample_rate, total_frames=frames, channel=channel,
            run_warnings=run_warnings,
        )
    else:
        analysis = _analyze_in_memory(
            wav_path=wav_path, config=config, calibration=calibration,
            sample_rate=sample_rate, total_frames=frames, channel=channel,
            run_warnings=run_warnings,
        )

    quality: SignalQuality = analysis["quality"]
    shots: List[ShotEvent] = analysis["shots"]
    detection: DetectionReport = analysis["detection"]
    shot_metrics: List[ShotMetrics] = analysis["shot_metrics"]

    say("")
    say("  Recording quality:")
    say(quality.summary())

    measurement_valid = quality.is_valid
    if not measurement_valid:
        say("")
        banner = [
            "MEASUREMENT INADMISSIBLE - the checks below failed.",
            "Numbers are still produced so the fault can be seen,",
            "but they must not be reported as a measurement.",
        ]
        width = max(len(line) for line in banner) + 8
        say("  " + "!" * width)
        for line in banner:
            say("  !!  " + line.ljust(width - 8) + "  !!")
        say("  " + "!" * width)
        for err in quality.errors:
            say(f"  INVALID: {err}")
        for err in quality.errors:
            logger.error("Measurement invalid: %s", err)

    say("")
    say("  Detection:")
    say(detection.summary())
    for shot in shots:
        flags = []
        if shot.clipped:
            flags.append("CLIPPED")
        if shot.truncated:
            flags.append("TRUNCATED")
        if shot.has_multiple_arrivals:
            flags.append(f"{len(shot.arrivals)} arrivals")
        suffix = ("  [" + ", ".join(flags) + "]") if flags else ""
        say(f"    Shot {shot.shot_number}: t={shot.time_s:.3f} s, "
            f"peak={shot.peak_dB:.1f} {calibration.level_unit}{suffix}")

    progress(62, "Computing aggregate statistics")
    aggregate = compute_aggregate_metrics(
        shot_metrics, protection_NRR_dB=config.protection_NRR_dB
    )

    if shot_metrics:
        say("")
        say(f"  Aggregate over {aggregate.n_valid} valid of {aggregate.n_shots} shots "
            f"(levels in {calibration.level_unit}):")
        for key in ("Lpeak_Z", "Lpeak_C", "LAE", "LAFmax"):
            stat = aggregate.stats.get(key)
            if stat and math.isfinite(stat.mean):
                say(f"    {key:<9} mean {stat.mean:7.1f}  +/-{stat.ci95_half_width:.1f} (95% CI)  "
                    f"min {stat.minimum:7.1f}  max {stat.maximum:7.1f}")
        if aggregate.n_valid < aggregate.n_shots:
            say(f"    {aggregate.n_shots - aggregate.n_valid} shot(s) were excluded as invalid.")
        if aggregate.hazard and math.isfinite(aggregate.hazard.LAeq8h_dB):
            hazard = aggregate.hazard
            say(f"    LAeq8h {hazard.LAeq8h_dB:.1f} dB, dose {hazard.dose_percent:.1f}% of the "
                f"{hazard.criterion_dB:.0f} dB criterion, "
                f"{hazard.allowable_rounds:.0f} rounds/day allowable")

    # ---- Test metadata completeness ----
    say("")
    say("  Measurement record:")
    say(metadata.completeness_report())
    for warn in metadata.warnings():
        logger.warning("Metadata: %s", warn)

    # ---- Assemble the record ----
    software = SoftwareInfo.capture()
    elapsed = time.time() - started
    record = make_provenance_block(
        software, source, metadata,
        timestamp=timestamp, elapsed_s=elapsed, output_dir=str(out_dir),
    )
    source_block = dict(record["source"])
    if Path(wav_path) != Path(original_path):
        source_block["extracted_audio"] = str(wav_path)
        source_block["extracted_sha256"] = file_sha256(wav_path)

    record = {
        "schema_version": SCHEMA_VERSION,
        "software": record["software"],
        "analysis": record["analysis"],
        "source": source_block,
        "calibration": calibration.to_dict(),
        "quality": quality.to_dict(),
        "detection": detection.to_dict(),
        "settings": _settings_block(
            config, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap,
            channel_used=channel_used, chunked=chunked, detection=detection,
            reference_dir=reference_dir, config_file=config_file,
        ),
        "test_metadata": record["test_metadata"],
        "shots": [s.to_dict() for s in shots],
        "per_shot_metrics": [m.to_dict() for m in shot_metrics],
        "aggregate": aggregate.to_dict(),
        "artifacts": {},
        "validity": {
            "measurement_valid": bool(measurement_valid and bool(shots)),
            "calibrated": bool(calibration.calibrated),
            "level_unit": calibration.level_unit,
            "reasons": list(quality.errors) + ([] if shots else ["No shots were detected"]),
            "n_shots_detected": len(shots),
            "n_shots_valid": aggregate.n_valid,
        },
        "warnings": run_warnings,
        "insertion_loss": None,
    }

    # ---- Insertion loss, the actual deliverable ----
    if reference_dir is not None:
        progress(66, "Computing insertion loss against the reference")
        try:
            comparison = compare_with_reference(reference_dir, aggregate, record)
            record["insertion_loss"] = comparison
            print_insertion_loss(comparison)
        except ConfigurationError as exc:
            message = f"Insertion loss was not computed: {exc}"
            logger.error(message)
            run_warnings.append(message)
            say(f"  ERROR: {message}")

    # ---- DATA FIRST: metrics reach disk before any plot is drawn ----
    progress(70, "Writing data artifacts")
    say("")
    say("  Writing data...")
    artifacts: Dict[str, str] = {}

    csv_path = save_csv_summary(
        out_dir / "metrics_summary.csv", shot_metrics, shots, record=record
    )
    if csv_path is not None:
        artifacts["metrics_csv"] = csv_path.name
        say(f"    metrics: {csv_path.name}")
    else:
        say("    metrics: not written (no shots detected)")

    if record["insertion_loss"]:
        il_path = save_insertion_loss_csv(out_dir / "insertion_loss.csv", record["insertion_loss"])
        if il_path is not None:
            artifacts["insertion_loss_csv"] = il_path.name
            say(f"    insertion loss: {il_path.name}")

    config_path = out_dir / "config.json"
    try:
        config.to_json(config_path)
        artifacts["config_json"] = config_path.name
    except OSError as exc:
        message = f"Config could not be written: {exc}"
        logger.warning(message)
        run_warnings.append(message)

    record["artifacts"] = dict(artifacts)
    record_path = out_dir / "analysis_metadata.json"
    write_json(record_path, record)
    artifacts["metadata_json"] = record_path.name
    say(f"    record:  {record_path.name}")

    # ---- Plots (never able to lose a computed metric) ----
    if config.make_plots:
        progress(74, "Generating plots")
        plot_artifacts = _generate_plots(
            wav_path=wav_path, config=config, calibration=calibration,
            sample_rate=sample_rate, total_frames=frames, channel=channel,
            shots=shots, shot_metrics=shot_metrics, aggregate=aggregate,
            quality=quality, record=record, out_dir=out_dir,
            nperseg=nperseg, noverlap=noverlap, chunked=chunked,
            run_warnings=run_warnings,
            preloaded=analysis.get("samples_FS"),
        )
        artifacts.update(plot_artifacts)
    else:
        say("  Plots skipped (--no-plots).")

    if (out_dir / "analysis.log").exists():
        artifacts["log"] = "analysis.log"

    # ---- Final record, now naming every artifact that really exists ----
    progress(97, "Finalising record")
    record["artifacts"] = artifacts
    record["warnings"] = run_warnings
    record["analysis"]["elapsed_s"] = round(time.time() - started, 3)
    write_json(record_path, record)

    say("")
    say("=" * 64)
    say(f"Analysis complete in {record['analysis']['elapsed_s']:.1f} s")
    say(f"Output directory: {out_dir}")
    say("=" * 64)
    announce_output_dir(out_dir)
    progress(100, "Complete")

    return AnalysisResult(
        input_file=original_path,
        output_dir=out_dir,
        calibration=calibration,
        sample_rate=int(sample_rate),
        duration_s=float(duration_s),
        n_shots=len(shots),
        shots=shots,
        shot_metrics=shot_metrics,
        aggregate=aggregate,
        config=config,
        timestamp=timestamp,
        quality=quality,
        detection=detection,
        measurement_valid=bool(measurement_valid),
        reference_requested=reference_dir is not None,
        insertion_loss_produced=bool(record.get("insertion_loss")),
        warnings=run_warnings,
        artifacts=artifacts,
        record=record,
    )


def _settings_block(
    config: AnalysisConfig,
    *,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    channel_used: str,
    chunked: bool,
    detection: DetectionReport,
    reference_dir: Optional[Path],
    config_file: Optional[Path],
) -> Dict[str, Any]:
    """Every analysis parameter actually used, as used - not as requested."""
    return {
        "channel": None if config.mono_mix else config.channel,
        "channel_used": channel_used,
        "mono_mix": config.mono_mix,
        "load_dtype": config.load_dtype,
        "detection_threshold_dB_requested": config.detection_threshold_dB,
        "threshold_relative_dB": config.threshold_relative_dB,
        "detection_threshold_dB_used": round(detection.threshold_dB, 2),
        "detection_threshold_mode": detection.threshold_mode,
        "refractory_ms": config.refractory_ms,
        "pre_shot_ms": config.pre_shot_ms,
        "post_shot_ms": config.post_shot_ms,
        "min_shots": config.min_shots,
        "max_shots": config.max_shots,
        "nperseg": nperseg,
        "nperseg_mode": "explicit" if config.nperseg is not None else "auto",
        "nperseg_ms": round(1000.0 * nperseg / sample_rate, 3),
        "noverlap": noverlap,
        "noverlap_mode": "explicit" if config.noverlap is not None else "derived",
        "overlap_fraction": round(noverlap / nperseg, 4) if nperseg else 0.0,
        "stft_window": config.stft_window,
        "stft_scaling": "rms",
        "compute_bands": config.compute_bands,
        "compute_time_series": config.compute_time_series,
        "band_hop_ms": config.band_hop_ms,
        "band_time_weighting": config.band_time_weighting,
        "protection_NRR_dB": config.protection_NRR_dB,
        "high_pass_10Hz": True,
        "make_plots": config.make_plots,
        "plot_formats": list(config.plot_formats or []),
        "save_per_shot_plots": config.save_per_shot_plots,
        "save_aggregate_plots": config.save_aggregate_plots,
        "chunked": chunked,
        "chunk_duration_s": CHUNK_DURATION_S if chunked else None,
        "chunk_context_s": CHUNK_CONTEXT_S if chunked else None,
        "reference_dir": str(reference_dir) if reference_dir else None,
        "config_file": str(config_file) if config_file else None,
        "sasa_version": __version__,
    }


# ── In-memory analysis ─────────────────────────────────────────────────────

def _analyze_in_memory(
    *,
    wav_path: Path,
    config: AnalysisConfig,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    channel: Optional[int],
    run_warnings: List[str],
) -> Dict[str, Any]:
    """Full-load path: the whole channel is held in memory."""
    samples_FS = read_samples(
        wav_path, 0, total_frames,
        channel=channel, mono_mix=config.mono_mix, dtype=config.load_dtype,
    )
    if samples_FS.size == 0:
        raise ConfigurationError(f"{wav_path.name} produced no samples for {describe_channel(channel, config.mono_mix, 1)}")

    # Quality is assessed on the FULL-SCALE samples, which is the only place
    # clipping is visible: after multiplication by Pa/FS nothing is at 1.0 any more.
    quality = assess_signal_quality(samples_FS, sample_rate, calibration)
    for warn in quality.warnings:
        logger.warning("Quality: %s", warn)

    pressure = calibration.to_pascals(samples_FS)

    progress(20, "Detecting shots")
    reports: List[DetectionReport] = []
    shots = detect_shots(
        pressure, sample_rate,
        threshold_dB=config.detection_threshold_dB,
        threshold_relative_dB=config.threshold_relative_dB,
        pre_ms=config.pre_shot_ms,
        post_ms=config.post_shot_ms,
        refractory_ms=config.refractory_ms,
        min_shots=config.min_shots,
        max_shots=config.max_shots,
        full_scale_dB=calibration.full_scale_dB if calibration.calibrated else None,
        samples_FS=samples_FS,          # so individual shots carry a clipped flag
        report=reports,
    )
    detection = reports[0] if reports else DetectionReport(
        len(shots), len(shots), 0, float("nan"), "unknown", float("nan"), float("nan")
    )
    for warn in detection.warnings:
        logger.warning("Detection: %s", warn)

    shot_metrics: List[ShotMetrics] = []
    for index, shot in enumerate(shots):
        progress(25 + 35.0 * (index + 1) / max(len(shots), 1),
                 f"Computing metrics for shot {shot.shot_number} of {len(shots)}")
        window = pressure[shot.window_start:shot.window_end]
        metric = compute_shot_metrics(
            window, sample_rate,
            compute_bands=config.compute_bands,
            compute_time_series=config.compute_time_series,
            shot_number=shot.shot_number,
            full_signal=pressure,
            window_start=shot.window_start,
            window_truncated=shot.truncated,
            clipped=shot.clipped,
        )
        shot_metrics.append(metric)
        logger.info(
            "Shot %d: Lpeak_Z=%.1f LAE=%.1f LAFmax=%.1f%s",
            shot.shot_number, metric.Lpeak_Z, metric.LAE, metric.LAFmax,
            " [INVALID]" if not metric.valid else "",
        )

    return {
        "quality": quality,
        "shots": shots,
        "detection": detection,
        "shot_metrics": shot_metrics,
        "samples_FS": samples_FS,
        "pressure": pressure,
    }


# ── Chunked analysis for long recordings ───────────────────────────────────

def _chunk_plan(
    total_frames: int, chunk_frames: int, context_frames: int
) -> Iterable[Tuple[int, int, int, int]]:
    """
    Yield (read_start, read_stop, core_start, core_stop) for overlapping chunks.

    Each chunk is read with `context_frames` of extra signal on both sides but only
    "owns" its core span. That is what stops a shot near a boundary from getting a
    truncated window, and gives every filter and detector real signal to settle on
    instead of restarting cold at each boundary.
    """
    start = 0
    while start < total_frames:
        core_stop = min(start + chunk_frames, total_frames)
        read_start = max(0, start - context_frames)
        read_stop = min(total_frames, core_stop + context_frames)
        yield read_start, read_stop, start, core_stop
        start = core_stop


def _analyze_chunked(
    *,
    wav_path: Path,
    config: AnalysisConfig,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    channel: Optional[int],
    run_warnings: List[str],
) -> Dict[str, Any]:
    """
    Analyse a long recording without loading it whole.

    Two properties the previous chunked path did not have:
      1. Detection uses ONE global threshold, computed in a first pass, so a quiet
         chunk cannot get a different threshold from a loud one.
      2. Shot windows and per-shot metrics are read directly from the file around
         each shot, with weighting-filter warm-up context, so a shot next to a
         120 s boundary is measured exactly like one in the middle.
    """
    chunk_frames = max(1, int(CHUNK_DURATION_S * sample_rate))
    context_frames = max(
        int(CHUNK_CONTEXT_S * sample_rate),
        int(3.0 * (config.pre_shot_ms + config.post_shot_ms) * sample_rate / 1000.0),
    )
    dtype = config.load_dtype

    def read(start: int, count: int) -> np.ndarray:
        return read_samples(
            wav_path, start, count,
            channel=channel, mono_mix=config.mono_mix, dtype=dtype,
        )

    # ---- Pass 1: global quality and a single global detection threshold ----
    env_window = max(1, int(1.0 * sample_rate / 1000.0))
    env_hop = max(1, int(0.25 * sample_rate / 1000.0))

    peak_FS = 0.0
    clipped_samples = 0
    clipped_runs = 0
    dc_sum = 0.0
    sq_sum = 0.0
    n_total = 0
    peak_envelope = 0.0
    noise_estimates: List[float] = []
    quality_chunks: List[SignalQuality] = []

    from calibration import detect_clipping  # noqa: PLC0415

    n_chunks = max(1, math.ceil(total_frames / chunk_frames))
    for index, (_rs, _re, core_start, core_stop) in enumerate(
        _chunk_plan(total_frames, chunk_frames, 0)
    ):
        block = read(core_start, core_stop - core_start)
        if block.size == 0:
            continue
        progress(8 + 8.0 * (index + 1) / n_chunks, "Scanning recording (pass 1 of 2)")

        peak_FS = max(peak_FS, float(np.max(np.abs(block))))
        counts = detect_clipping(block)
        clipped_samples += counts[0]
        clipped_runs += counts[1]
        dc_sum += float(np.sum(block, dtype=np.float64))
        sq_sum += float(np.sum(np.asarray(block, dtype=np.float64) ** 2))
        n_total += block.size

        pressure_block = calibration.to_pascals(block)
        envelope, _ = compute_envelope(
            bandpass_for_detection(pressure_block, sample_rate), env_window, env_hop
        )
        if envelope.size:
            peak_envelope = max(peak_envelope, float(envelope.max()))
            noise_estimates.append(float(np.percentile(envelope, 10.0)))

        # Assess a representative slice for the qualitative checks that need spectra
        if len(quality_chunks) < 8:
            quality_chunks.append(assess_signal_quality(block, sample_rate, calibration))
        del block, pressure_block, envelope
        gc.collect()

    quality = _merge_quality(
        quality_chunks, calibration,
        n_samples=n_total, sample_rate=sample_rate, peak_FS=peak_FS,
        clipped_samples=clipped_samples, clipped_runs=clipped_runs,
        dc=dc_sum / max(n_total, 1), rms=math.sqrt(sq_sum / max(n_total, 1)),
    )

    peak_dB = float(amplitude_to_dB_SPL(max(peak_envelope, 1e-12)))
    if config.detection_threshold_dB is not None:
        absolute_threshold = float(config.detection_threshold_dB)
        mode = "absolute"
    else:
        relative = config.threshold_relative_dB if config.threshold_relative_dB is not None else 30.0
        absolute_threshold = peak_dB - float(relative)
        mode = "relative (resolved globally over all chunks)"
    logger.info("Chunked detection threshold: %.1f dB (%s), global envelope peak %.1f dB",
                absolute_threshold, mode, peak_dB)

    # ---- Pass 2: detection on overlapping chunks, keeping each chunk's core ----
    pre_samples = int(config.pre_shot_ms * sample_rate / 1000.0)
    post_samples = int(config.post_shot_ms * sample_rate / 1000.0)
    shots: List[ShotEvent] = []
    n_candidates = 0
    n_suppressed = 0
    detection_warnings: List[str] = []

    for index, (read_start, read_stop, core_start, core_stop) in enumerate(
        _chunk_plan(total_frames, chunk_frames, context_frames)
    ):
        progress(16 + 9.0 * (index + 1) / n_chunks, "Detecting shots (pass 2 of 2)")
        block = read(read_start, read_stop - read_start)
        if block.size == 0:
            continue
        pressure_block = calibration.to_pascals(block)

        reports: List[DetectionReport] = []
        found = detect_shots(
            pressure_block, sample_rate,
            threshold_dB=absolute_threshold,
            pre_ms=config.pre_shot_ms,
            post_ms=config.post_shot_ms,
            refractory_ms=config.refractory_ms,
            max_shots=config.max_shots,
            full_scale_dB=calibration.full_scale_dB if calibration.calibrated else None,
            samples_FS=block,
            report=reports,
        )
        if reports:
            n_candidates += reports[0].n_candidates
            n_suppressed += reports[0].n_suppressed_by_refractory
            for warn in reports[0].warnings:
                if warn not in detection_warnings:
                    detection_warnings.append(warn)

        for shot in found:
            global_index = read_start + shot.index
            if not (core_start <= global_index < core_stop):
                continue        # another chunk owns this event
            window_start = global_index - pre_samples
            window_end = global_index + post_samples
            truncated = window_start < 0 or window_end > total_frames
            shots.append(ShotEvent(
                index=int(global_index),
                time_s=global_index / sample_rate,
                peak_Pa=shot.peak_Pa,
                peak_dB=shot.peak_dB,
                window_start=max(0, window_start),
                window_end=min(total_frames, window_end),
                shot_number=0,
                truncated=truncated,
                clipped=shot.clipped,
                snr_dB=shot.snr_dB,
                arrivals=shot.arrivals,
            ))
        del block, pressure_block
        gc.collect()

    # Deduplicate across seams (an event within one refractory period of the last)
    shots.sort(key=lambda s: s.time_s)
    refractory_samples = max(1, int(config.refractory_ms * sample_rate / 1000.0))
    deduped: List[ShotEvent] = []
    for shot in shots:
        if deduped and (shot.index - deduped[-1].index) < refractory_samples:
            n_suppressed += 1
            continue
        shot.shot_number = len(deduped) + 1
        deduped.append(shot)
    shots = deduped[: config.max_shots]

    if len(shots) < config.min_shots:
        detection_warnings.append(f"Expected at least {config.min_shots} shots, found {len(shots)}")

    detection_warnings.append(
        f"Chunked analysis: {n_chunks} chunk(s) of {CHUNK_DURATION_S:.0f} s with "
        f"{context_frames / sample_rate:.1f} s of overlap; the detection threshold was "
        f"resolved once over the whole recording."
    )

    detection = DetectionReport(
        n_detected=len(shots),
        n_candidates=n_candidates,
        n_suppressed_by_refractory=n_suppressed,
        threshold_dB=absolute_threshold,
        threshold_mode=mode,
        peak_level_dB=peak_dB,
        noise_floor_dB=float(amplitude_to_dB_SPL(max(min(noise_estimates, default=1e-12), 1e-12))),
        full_scale_dB=calibration.full_scale_dB if calibration.calibrated else None,
        warnings=detection_warnings,
    )

    # ---- Per-shot metrics, each read with warm-up context ----
    warmup = weighting_settling_samples(sample_rate, "A")
    shot_metrics: List[ShotMetrics] = []
    for position, shot in enumerate(shots):
        progress(25 + 35.0 * (position + 1) / max(len(shots), 1),
                 f"Computing metrics for shot {shot.shot_number} of {len(shots)}")
        context_start = max(0, shot.window_start - warmup)
        block = read(context_start, shot.window_end - context_start)
        if block.size == 0:
            run_warnings.append(f"Shot {shot.shot_number} could not be re-read from the file.")
            continue
        pressure_block = calibration.to_pascals(block)
        offset = shot.window_start - context_start
        metric = compute_shot_metrics(
            pressure_block[offset:], sample_rate,
            compute_bands=config.compute_bands,
            compute_time_series=config.compute_time_series,
            shot_number=shot.shot_number,
            full_signal=pressure_block,
            window_start=offset,
            window_truncated=shot.truncated,
            clipped=shot.clipped,
        )
        shot_metrics.append(metric)
        del block, pressure_block
        gc.collect()

    return {
        "quality": quality,
        "shots": shots,
        "detection": detection,
        "shot_metrics": shot_metrics,
        "samples_FS": None,
        "pressure": None,
    }


def _merge_quality(
    chunks: Sequence[SignalQuality],
    calibration: Calibration,
    *,
    n_samples: int,
    sample_rate: int,
    peak_FS: float,
    clipped_samples: int,
    clipped_runs: int,
    dc: float,
    rms: float,
) -> SignalQuality:
    """
    Combine per-chunk quality assessments into one whole-recording verdict.

    Clipping, peak and DC are computed exactly across the whole file; the
    spectral checks are taken from the sampled chunks. Errors and warnings are
    unioned, so a single clipped chunk still invalidates the measurement.
    """
    eps = 1e-30
    peak_level_dB = float(amplitude_to_dB_SPL(max(peak_FS * calibration.Pa_per_FS, eps)))
    headroom_dB = float(-20.0 * np.log10(max(peak_FS, eps)))
    noise_floor_dB = min((c.noise_floor_dB for c in chunks), default=peak_level_dB)
    lf_fraction = float(np.mean([c.lf_energy_fraction for c in chunks])) if chunks else 0.0

    warnings: List[str] = []
    errors: List[str] = []
    for chunk in chunks:
        for warn in chunk.warnings:
            if warn not in warnings:
                warnings.append(warn)

    if clipped_runs > 0:
        errors.append(
            f"Recording is CLIPPED ({clipped_samples} samples in {clipped_runs} runs). "
            f"Peak levels are understated and rise time, crest factor and kurtosis are invalid. "
            f"Re-record with lower input gain."
        )

    quality = SignalQuality(
        n_samples=n_samples,
        sample_rate=sample_rate,
        duration_s=n_samples / sample_rate if sample_rate else 0.0,
        peak_FS=peak_FS,
        headroom_dB=headroom_dB,
        clipped_samples=clipped_samples,
        clipped_runs=clipped_runs,
        clipping_ratio=clipped_samples / max(n_samples, 1),
        dc_offset_FS=dc,
        dc_offset_dB=float(20.0 * np.log10(max(abs(dc), eps) / max(rms, eps))),
        noise_floor_dB=noise_floor_dB,
        peak_level_dB=peak_level_dB,
        snr_dB=peak_level_dB - noise_floor_dB,
        lf_energy_fraction=lf_fraction,
        nyquist_Hz=sample_rate / 2.0,
        sample_rate_adequate=sample_rate >= 48000,
        warnings=warnings,
        errors=errors,
    )
    return quality


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting (isolated: a failure here can never lose a metric)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_step(name: str, run_warnings: List[str]):
    """
    Context manager factory that turns any plot failure into a recorded warning.

    Nothing in the plotting stage is allowed to abort the run or vanish silently:
    the previous build wrapped whole plot blocks in `except Exception: pass`.
    """
    class _Step:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, _tb):
            if exc is None:
                return False
            message = f"{name} could not be produced: {exc}"
            logger.warning(message)
            logger.debug("%s failed", name, exc_info=(exc_type, exc, _tb))
            run_warnings.append(message)
            say(f"    WARNING: {message}")
            return True    # suppressed, recorded

    return _Step()


def _make_reader(
    wav_path: Path,
    config: AnalysisConfig,
    channel: Optional[int],
    preloaded: Optional[np.ndarray],
):
    """
    Return a (start, count) -> full-scale samples reader.

    When the channel is already in memory (the short-file path) it is sliced
    rather than re-read, so plotting costs no extra I/O.
    """
    if preloaded is not None:
        def reader(start: int, count: int) -> np.ndarray:
            start = max(0, int(start))
            return preloaded[start:start + max(0, int(count))]
        return reader

    def reader(start: int, count: int) -> np.ndarray:
        return read_samples(
            wav_path, start, count,
            channel=channel, mono_mix=config.mono_mix, dtype=config.load_dtype,
        )
    return reader


def _generate_plots(
    *,
    wav_path: Path,
    config: AnalysisConfig,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    channel: Optional[int],
    shots: List[ShotEvent],
    shot_metrics: List[ShotMetrics],
    aggregate: AggregateMetrics,
    quality: SignalQuality,
    record: Dict[str, Any],
    out_dir: Path,
    nperseg: int,
    noverlap: int,
    chunked: bool,
    run_warnings: List[str],
    preloaded: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Draw every figure, recording (never raising) whatever fails."""
    artifacts: Dict[str, str] = {}
    say("  Generating plots...")
    reader = _make_reader(wav_path, config, channel, preloaded)

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import plots as plot_module  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - plotting is optional; data is already safe
        message = f"Plotting is unavailable ({exc}); the data artifacts are complete."
        logger.warning(message)
        run_warnings.append(message)
        say(f"    WARNING: {message}")
        return artifacts

    with _plot_step("Plot style", run_warnings):
        plot_module.setup_plot_style()

    static_formats = config.static_formats()
    level_unit = calibration.level_unit
    want_html = "html" in (config.plot_formats or [])
    plotly_available = _plotly_available()
    if want_html and not plotly_available:
        message = "HTML plots were requested but Plotly is not installed; PNG was used instead."
        logger.warning(message)
        run_warnings.append(message)

    def call(func_name: str, *args, **kwargs):
        """
        Call an optional plots.py function only if it exists, passing only the
        keyword arguments its current signature accepts.

        plots.py is being rewritten in parallel, so this stays tolerant of a
        function that has not landed yet or whose keywords have moved.
        """
        func = getattr(plot_module, func_name, None)
        if func is None:
            logger.info("plots.%s is not available in this build; skipping", func_name)
            return None
        try:
            import inspect  # noqa: PLC0415

            accepted = set(inspect.signature(func).parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            pass
        result = func(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    # ---- Full-recording waveform ----
    with _plot_step("Full waveform", run_warnings):
        time_axis, pressure_plot = _waveform_for_display(
            reader, calibration, sample_rate, total_frames, shots
        )
        if time_axis.size:
            saved = False
            if plotly_available and want_html:
                path = out_dir / "waveform_full.html"
                if plot_module.save_interactive_waveform_html(
                    path, time_axis, pressure_plot, shots=shots,
                    title=f"Pressure Waveform: {wav_path.name}",
                ):
                    artifacts["waveform_html"] = path.name
                    saved = True
            if static_formats:
                figure, _ = plot_module.plot_waveform_pa(
                    time_axis, pressure_plot, shots=shots,
                    title=f"Pressure Waveform: {wav_path.name}",
                )
                for path in plot_module.save_figure(
                    figure, out_dir / "waveform_full", formats=static_formats
                ):
                    artifacts.setdefault(f"waveform_{path.suffix.lstrip('.')}", path.name)
                plt.close(figure)
                saved = True
            if saved:
                say("    waveform")
        del time_axis, pressure_plot
        gc.collect()

    # ---- Full-recording spectrograms ----
    for weighting, key in (("Z", "spectrogram_z"), ("C", "spectrogram_c")):
        with _plot_step(f"{weighting}-weighted spectrogram", run_warnings):
            stft = _spectrogram_for_display(
                reader, config, calibration, sample_rate, total_frames,
                nperseg=nperseg, noverlap=noverlap, weighting=weighting, chunked=chunked,
            )
            if stft is None:
                raise RuntimeError("the recording is shorter than one FFT window")
            if plotly_available and want_html:
                path = out_dir / f"{key}_full.html"
                if plot_module.save_interactive_spectrogram_html(
                    path, stft, shots=shots,
                    title=f"{weighting}-Weighted Spectrogram: {wav_path.name}",
                ):
                    artifacts[f"{key}_html"] = path.name
            if static_formats:
                figure, _ = plot_module.plot_spectrogram_dB(
                    stft, shots=shots,
                    title=f"{weighting}-Weighted Spectrogram: {wav_path.name}",
                )
                for path in plot_module.save_figure(
                    figure, out_dir / f"{key}_full", formats=static_formats
                ):
                    artifacts.setdefault(f"{key}_{path.suffix.lstrip('.')}", path.name)
                plt.close(figure)
            say(f"    {weighting}-weighted spectrogram")
            del stft
            gc.collect()

    # ---- 1/3-octave heatmap ----
    if config.compute_bands and static_formats:
        with _plot_step("1/3-octave band heatmap", run_warnings):
            times, freqs, levels = _bands_for_display(
                reader, config, calibration, sample_rate, total_frames, chunked
            )
            if times.size:
                figure, _ = plot_module.plot_third_octave_heatmap(
                    times, freqs, levels, shots=shots,
                    title=f"1/3-Octave Band Levels: {wav_path.name}",
                )
                for path in plot_module.save_figure(
                    figure, out_dir / "bands_full", formats=static_formats
                ):
                    artifacts.setdefault(f"bands_{path.suffix.lstrip('.')}", path.name)
                plt.close(figure)
                say("    1/3-octave bands")
            del times, freqs, levels
            gc.collect()

    # ---- Per-shot summaries ----
    if config.save_per_shot_plots and shots and static_formats:
        shot_dir = out_dir / "shots"
        shot_dir.mkdir(exist_ok=True)
        produced = 0
        for position, (shot, metric) in enumerate(zip(shots, shot_metrics)):
            progress(78 + 16.0 * (position + 1) / max(len(shots), 1),
                     f"Plotting shot {shot.shot_number} of {len(shots)}")
            with _plot_step(f"Shot {shot.shot_number} summary", run_warnings):
                block = reader(shot.window_start, shot.window_end - shot.window_start)
                window = calibration.to_pascals(block)
                window_time = np.arange(window.size) / sample_rate
                shot_nperseg = min(nperseg, max(64, 1 << int(math.log2(max(window.size, 64)))))
                shot_noverlap = config.resolved_noverlap(shot_nperseg)
                stft_z = analyze_stft(window, sample_rate, nperseg=shot_nperseg,
                                      noverlap=shot_noverlap, weighting="Z",
                                      calibrated=calibration.calibrated)
                stft_c = analyze_stft(window, sample_rate, nperseg=shot_nperseg,
                                      noverlap=shot_noverlap, weighting="C",
                                      calibrated=calibration.calibrated)
                figure = plot_module.create_shot_summary_figure(
                    window_time, window, stft_z, stft_c, metric,
                    title=f"Shot {shot.shot_number} Analysis ({level_unit})",
                )
                paths = plot_module.save_figure(
                    figure, shot_dir / f"shot_{shot.shot_number:02d}_summary",
                    formats=static_formats,
                )
                plt.close(figure)
                if paths:
                    artifacts.setdefault(
                        f"shot_{shot.shot_number:02d}_summary",
                        str(paths[0].relative_to(out_dir)),
                    )
                produced += 1
                del block, window, stft_z, stft_c
                gc.collect()
        if produced:
            say(f"    per-shot summaries ({produced})")

    # ---- Optional figures that plots.py is gaining in parallel ----
    def save_optional(figure, stem: str, key_prefix: str) -> None:
        if figure is None:
            return
        for path in plot_module.save_figure(figure, out_dir / stem, formats=static_formats):
            artifacts.setdefault(f"{key_prefix}_{path.suffix.lstrip('.')}", path.name)
        plt.close(figure)
        say(f"    {stem.replace('_', ' ')}")

    if static_formats:
        with _plot_step("Measurement quality figure", run_warnings):
            save_optional(
                call("plot_measurement_quality", quality.to_dict(),
                     title="Measurement Quality", level_unit=level_unit),
                "measurement_quality", "quality",
            )

        if len(shots) > 1:
            with _plot_step("Shot overlay figure", run_warnings):
                # plots.plot_shot_overlay overlays WAVEFORMS, not shot events.
                traces = []
                for shot in shots[:16]:
                    block = reader(shot.window_start, shot.window_end - shot.window_start)
                    if block.size:
                        traces.append(calibration.to_pascals(block))
                save_optional(
                    call("plot_shot_overlay", traces, sample_rate=sample_rate,
                         labels=[f"Shot {s.shot_number}" for s in shots[:16]],
                         level_unit=level_unit),
                    "shot_overlay", "overlay",
                )
                del traces
                gc.collect()

        comparison = record.get("insertion_loss")
        if comparison and (comparison.get("bands") or {}).get("insertion_loss_dB"):
            with _plot_step("Insertion loss figure", run_warnings):
                bands = comparison["bands"]
                save_optional(
                    call(
                        "plot_insertion_loss",
                        np.asarray(bands["reference_dB"], dtype=float),
                        np.asarray(bands["test_dB"], dtype=float),
                        np.asarray(bands["frequencies_Hz"], dtype=float),
                        level_unit=level_unit,
                    ),
                    "insertion_loss", "insertion_loss",
                )

    return artifacts


def _plotly_available() -> bool:
    try:
        import plotly.graph_objects  # noqa: F401,PLC0415

        return True
    except ImportError as exc:
        logger.info("Plotly is not installed (%s); interactive HTML plots are unavailable", exc)
        return False


def _waveform_for_display(
    reader,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    shots: Sequence[ShotEvent],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a display waveform: full resolution around every shot, thinned elsewhere.

    This affects the picture only. Every metric is computed from every sample.
    """
    step = max(1, total_frames // MAX_WAVEFORM_POINTS)
    pre = int(WAVEFORM_FULLRES_PRE_S * sample_rate)
    post = int(WAVEFORM_FULLRES_POST_S * sample_rate)

    regions: List[Tuple[int, int]] = []
    for shot in shots:
        start = max(0, shot.index - pre)
        stop = min(total_frames, shot.index + post)
        if stop > start:
            regions.append((start, stop))
    regions.sort()
    merged: List[Tuple[int, int]] = []
    for start, stop in regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))

    times: List[np.ndarray] = []
    values: List[np.ndarray] = []

    def thinned(start: int, stop: int) -> None:
        if stop <= start:
            return
        block = reader(start, stop - start)
        if block.size == 0:
            return
        times.append((start + np.arange(0, block.size, step)) / sample_rate)
        values.append(calibration.to_pascals(block[::step]))

    def full(start: int, stop: int) -> None:
        block = reader(start, stop - start)
        if block.size == 0:
            return
        times.append((start + np.arange(block.size)) / sample_rate)
        values.append(calibration.to_pascals(block))

    cursor = 0
    for start, stop in merged:
        thinned(cursor, start)
        full(start, stop)
        cursor = stop
    thinned(cursor, total_frames)

    if not times:
        return np.array([]), np.array([])
    # float32 for the plot: an interactive chart serialises the array verbatim,
    # and 64-bit precision on a picture doubles the file for nothing.
    return (
        np.concatenate(times).astype(np.float32),
        _round_for_display(np.concatenate(values)).astype(np.float32),
    )


def _spectrogram_for_display(
    reader,
    config: AnalysisConfig,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    *,
    nperseg: int,
    noverlap: int,
    weighting: str,
    chunked: bool,
) -> Optional[STFTResult]:
    """
    Compute a full-recording spectrogram, chunk by chunk when the file is long.

    The result is thinned to MAX_SPECTROGRAM_FRAMES for display; every metric is
    computed elsewhere, from the full-rate signal.
    """
    if not chunked:
        block = reader(0, total_frames)
        if block.size < nperseg:
            return None
        stft = analyze_stft(
            calibration.to_pascals(block), sample_rate,
            nperseg=nperseg, noverlap=noverlap, window=config.stft_window,
            weighting=weighting, calibrated=calibration.calibrated,
        )
        return _thin_spectrogram(stft)

    chunk_frames = max(nperseg * 4, int(CHUNK_DURATION_S * sample_rate))
    times: List[np.ndarray] = []
    magnitudes: List[np.ndarray] = []
    freqs = np.array([])
    template: Optional[STFTResult] = None

    for _rs, _re, core_start, core_stop in _chunk_plan(total_frames, chunk_frames, 0):
        block = reader(core_start, core_stop - core_start)
        if block.size < nperseg:
            continue
        stft = analyze_stft(
            calibration.to_pascals(block), sample_rate,
            nperseg=nperseg, noverlap=noverlap, window=config.stft_window,
            weighting=weighting, calibrated=calibration.calibrated,
        )
        keep = max(1, SPECTROGRAM_DOWNSAMPLE)
        times.append(stft.time_s[::keep] + core_start / sample_rate)
        magnitudes.append(stft.magnitude_dB[:, ::keep])
        freqs = stft.frequencies_Hz
        template = stft
        del block, stft
        gc.collect()

    if template is None or not times:
        return None
    return _thin_spectrogram(STFTResult(
        time_s=np.concatenate(times),
        frequencies_Hz=freqs,
        magnitude_dB=np.concatenate(magnitudes, axis=1),
        weighting=weighting,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=config.stft_window,
        scaling=template.scaling,
        enbw_Hz=template.enbw_Hz,
        calibrated=calibration.calibrated,
    ))


def _thin_spectrogram(stft: STFTResult) -> STFTResult:
    """
    Prepare a spectrogram for display: at most MAX_SPECTROGRAM_FRAMES frames, and
    levels rounded to 0.1 dB.

    Both are display concerns only. Rounding matters because an interactive plot
    serialises every value as text: at full precision a five-second 192 kHz
    spectrogram becomes a 26 MB HTML file that no browser can pan.
    """
    frames = stft.time_s.size
    if frames > MAX_SPECTROGRAM_FRAMES:
        stft = stft.decimate_frames(math.ceil(frames / MAX_SPECTROGRAM_FRAMES))
    stft.magnitude_dB = np.round(stft.magnitude_dB, 1).astype(np.float32)
    stft.time_s = stft.time_s.astype(np.float32)
    return stft


def _round_for_display(values: np.ndarray, significant: int = 5) -> np.ndarray:
    """
    Round a display waveform to a fixed relative precision.

    Absolute rounding would flatten the noise floor of an uncalibrated waveform
    (peak ~0.3 FS) while barely touching a calibrated one (peak ~30 Pa), so the
    number of decimals is derived from the signal's own magnitude.
    """
    if values.size == 0:
        return values
    peak = float(np.max(np.abs(values)))
    if not math.isfinite(peak) or peak <= 0:
        return values
    decimals = int(np.clip(significant - math.floor(math.log10(peak)) - 1, 0, 12))
    return np.round(values, decimals)


def _bands_for_display(
    reader,
    config: AnalysisConfig,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    chunked: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Band-level time history for the heatmap.

    In chunked mode each chunk is read with a second of leading context and the
    settling frames are discarded, so the filter bank and its detectors are never
    restarted mid-signal - which is what produced a visible step at every 120 s
    boundary in the previous build.
    """
    analyzer = ThirdOctaveAnalyzer(sample_rate=sample_rate)

    def analyze(block: np.ndarray) -> Dict[str, Any]:
        return analyzer.analyze(
            calibration.to_pascals(block),
            time_weighting=config.band_time_weighting,
            hop_ms=config.band_hop_ms,
        )

    if not chunked:
        block = reader(0, total_frames)
        if block.size == 0:
            return np.array([]), analyzer.center_frequencies, np.array([[]])
        result = analyze(block)
        return result["time_s"], result["center_frequencies"], result["band_levels_dB"]

    context_frames = int(1.0 * sample_rate)
    chunk_frames = max(1, int(CHUNK_DURATION_S * sample_rate))
    times: List[np.ndarray] = []
    levels: List[np.ndarray] = []

    for read_start, read_stop, core_start, core_stop in _chunk_plan(
        total_frames, chunk_frames, context_frames
    ):
        read_start = max(0, core_start - context_frames)   # leading context only
        block = reader(read_start, core_stop - read_start)
        if block.size == 0:
            continue
        result = analyze(block)
        chunk_times = result["time_s"] + read_start / sample_rate
        keep = chunk_times >= core_start / sample_rate       # drop the settling region
        if np.any(keep):
            times.append(chunk_times[keep])
            levels.append(result["band_levels_dB"][:, keep])
        del block, result
        gc.collect()

    if not times:
        return np.array([]), analyzer.center_frequencies, np.array([[]])

    time_axis = np.concatenate(times)
    level_matrix = np.concatenate(levels, axis=1)
    if time_axis.size > MAX_WAVEFORM_POINTS:
        step = max(1, time_axis.size // MAX_WAVEFORM_POINTS)
        time_axis = time_axis[::step]
        level_matrix = level_matrix[:, ::step]
    return time_axis, analyzer.center_frequencies, level_matrix


# ═══════════════════════════════════════════════════════════════════════════
#  Command line
# ═══════════════════════════════════════════════════════════════════════════

# Metadata flags mirror TestMetadata's fields exactly; dest is prefixed so a
# metadata field can never collide with an analysis parameter.
METADATA_FLAGS: List[Tuple[str, type, str]] = [
    ("operator", str, "Who ran the test"),
    ("date", str, "Test date (YYYY-MM-DD)"),
    ("location", str, "Test location"),
    ("test-id", str, "Test or session identifier"),
    ("weapon", str, "Weapon make/model"),
    ("barrel-length-in", float, "Barrel length in inches"),
    ("ammunition", str, "Ammunition make, type and lot"),
    ("suppressor", str, "Suppressor under test"),
    ("configuration", str, "'suppressed' or 'unsuppressed'"),
    ("mic-model", str, "Microphone model"),
    ("mic-serial", str, "Microphone serial number"),
    ("mic-distance-m", float, "Microphone distance from the muzzle, metres"),
    ("mic-angle-deg", float, "Microphone angle, degrees (0 = downrange)"),
    ("mic-height-m", float, "Microphone height, metres"),
    ("ground-surface", str, "Ground surface under the measurement"),
    ("windscreen", str, "Windscreen fitted"),
    ("temperature-C", float, "Air temperature, degrees C"),
    ("humidity-pct", float, "Relative humidity, percent"),
    ("pressure-kPa", float, "Barometric pressure, kPa"),
    ("wind-mps", float, "Wind speed, m/s"),
    ("calibrator-model", str, "Acoustic calibrator model"),
    ("calibration-pre-dB", float, "Pre-test calibration check reading, dB"),
    ("calibration-post-dB", float, "Post-test calibration check reading, dB"),
    ("notes", str, "Free-text notes"),
]


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command-line surface."""
    parser = argparse.ArgumentParser(
        prog="sasa",
        description="SASA - Shot Acoustic Spectral Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Calibrated with a pistonphone recording (preferred)
  sasa string.wav --calibrator-tone cal_114dB.wav --calibrator-level-dB 114

  # Calibrated from the recording chain
  sasa string.wav --sensitivity-mV 12.5 --preamp-gain-dB 20 --adc-full-scale-V 2.0

  # Relative analysis, clearly labelled dB re FS
  sasa string.wav --uncalibrated

  # The deliverable: suppressed run measured against an unsuppressed reference
  sasa suppressed.wav --preset MyRig --reference analysis/unsuppressed_2026.../

  # Every channel of a multi-microphone recording, separately
  sasa array.wav --preset MyRig --channels all

Exit codes: 0 ok, 1 error, 2 no shots detected, 3 measurement inadmissible.
""",
    )

    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="Audio or video file. Omit to choose one interactively.")
    parser.add_argument("--version", action="version", version=f"SASA {__version__}")

    cal = parser.add_argument_group(
        "Calibration (exactly one is required; there is no default)"
    )
    cal.add_argument("--calibrator-tone", type=Path, default=None, metavar="FILE",
                     help="Recording of an acoustic calibrator; measures the whole chain.")
    cal.add_argument("--calibrator-level-dB", type=float, default=114.0, metavar="dB",
                     help="Calibrator's stated output level (typically 94 or 114). Default: 114.")
    cal.add_argument("--calibrator-post", type=Path, default=None, metavar="FILE",
                     help="Post-test calibrator recording, for the drift check.")
    cal.add_argument("--calibrator-freq-Hz", type=float, default=1000.0, metavar="Hz",
                     help="Calibrator tone frequency. Default: 1000.")
    cal.add_argument("--sensitivity-mV", type=float, default=None, metavar="mV/Pa",
                     help="Microphone sensitivity in mV/Pa.")
    cal.add_argument("--preamp-gain-dB", type=float, default=0.0, metavar="dB",
                     help="Preamp/recorder input gain. Default: 0.")
    cal.add_argument("--adc-full-scale-V", type=float, default=None, metavar="V",
                     help="Input voltage corresponding to digital full scale.")
    cal.add_argument("--V-per-FS", type=float, default=None, metavar="V",
                     help="Deprecated alias for --adc-full-scale-V.")
    cal.add_argument("--Pa-per-FS", type=float, default=None, metavar="Pa",
                     help="Direct conversion factor, Pascals per full scale.")
    cal.add_argument("--preset", type=str, default=None, metavar="NAME",
                     help="Named calibration profile for a known rig.")
    cal.add_argument("--uncalibrated", action="store_true",
                     help="No calibration: every level is reported as dB re FS.")
    cal.add_argument("--cal-desc", type=str, default="", metavar="TEXT",
                     help="Description recorded alongside the calibration.")
    cal.add_argument("--list-presets", action="store_true",
                     help="List the calibration presets and profiles, then exit.")
    cal.add_argument("--save-preset", type=str, default=None, metavar="NAME",
                     help="Save this run's calibration as a reusable profile.")
    cal.add_argument("--delete-preset", type=str, default=None, metavar="NAME",
                     help="Delete a saved profile, then exit.")
    cal.add_argument("--profiles-file", type=Path, default=None, metavar="FILE",
                     help="Override the calibration profile store location.")

    chan = parser.add_argument_group("Channels")
    chan.add_argument("--channel", type=int, default=0, metavar="N",
                      help="Zero-based channel to analyse. Default: 0.")
    chan.add_argument("--channels", type=str, default=None, metavar="all",
                      help="'all' analyses every channel separately into its own directory.")
    chan.add_argument("--mono-mix", action="store_true",
                      help="Average all channels. Destroys a multi-microphone measurement; "
                           "never the default.")

    det = parser.add_argument_group("Shot detection")
    det.add_argument("--threshold-dB", type=float, default=None, metavar="dB",
                     help="Absolute detection threshold; requires a real calibration.")
    det.add_argument("--threshold-relative-dB", type=float, default=None, metavar="dB",
                     help="Detection threshold this many dB below the loudest event. Default: 30.")
    det.add_argument("--refractory-ms", type=float, default=200.0, metavar="ms",
                     help="Minimum spacing between shots. Default: 200.")
    det.add_argument("--pre-ms", type=float, default=50.0, metavar="ms",
                     help="Window before each peak. Default: 50.")
    det.add_argument("--post-ms", type=float, default=200.0, metavar="ms",
                     help="Window after each peak. Default: 200.")
    det.add_argument("--min-shots", type=int, default=0, metavar="N",
                     help="Warn if fewer than N shots are found. Default: 0.")
    det.add_argument("--max-shots", type=int, default=1000, metavar="N",
                     help="Safety limit on detections. Default: 1000.")

    ana = parser.add_argument_group("Analysis")
    ana.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                     help="Sample precision when reading. Default: float32.")
    ana.add_argument("--nperseg", type=str, default="auto", metavar="N|auto",
                     help="STFT window size, or 'auto' to scale it to the sample rate. Default: auto.")
    ana.add_argument("--noverlap", type=int, default=None, metavar="N",
                     help="STFT overlap in samples. Default: derived from --overlap-fraction.")
    ana.add_argument("--overlap-fraction", type=float, default=0.75, metavar="F",
                     help="STFT overlap as a fraction of the window, 0 to <1. Default: 0.75.")
    ana.add_argument("--band-hop-ms", type=float, default=10.0, metavar="ms",
                     help="1/3-octave time resolution. Default: 10.")
    ana.add_argument("--band-weighting", type=str, default="fast",
                     choices=["fast", "slow", "impulse"],
                     help="1/3-octave time weighting. Default: fast.")
    ana.add_argument("--nrr-dB", type=float, default=0.0, metavar="dB",
                     help="Hearing protector NRR used in the hazard assessment. Default: 0.")
    ana.add_argument("--no-bands", action="store_true", help="Skip 1/3-octave band analysis.")
    ana.add_argument("--no-time-series", action="store_true",
                     help="Do not retain level time series (smaller memory footprint).")
    ana.add_argument("--no-per-shot", action="store_true", help="Skip per-shot summary figures.")

    out = parser.add_argument_group("Output")
    out.add_argument("--output", "-o", type=Path, default=None, metavar="DIR",
                     help="Directory to create the run directory under.")
    out.add_argument("--config", type=Path, default=None, metavar="FILE",
                     help="Load settings from JSON. Explicit flags override the file.")
    out.add_argument("--formats", type=str, default="png", metavar="LIST",
                     help="Plot formats, comma-separated (png, pdf, svg, html...). Default: png.")
    out.add_argument("--no-plots", action="store_true",
                     help="Write data artifacts only; draw nothing.")
    out.add_argument("--reference", type=Path, default=None, metavar="DIR",
                     help="Previous UNSUPPRESSED analysis directory; computes insertion loss.")
    out.add_argument("--verbose", "-v", action="store_true", help="Debug-level diagnostics.")
    out.add_argument("--quiet", "-q", action="store_true", help="Warnings and errors only.")

    meta = parser.add_argument_group("Test metadata (recorded with the result)")
    meta.add_argument("--metadata", type=Path, default=None, metavar="FILE",
                      help="JSON file of test conditions; individual flags override it.")
    for flag, kind, help_text in METADATA_FLAGS:
        dest = "meta_" + flag.replace("-", "_")
        kwargs: Dict[str, Any] = {
            "type": kind, "default": None, "dest": dest, "help": help_text,
            "metavar": "TEXT" if kind is str else "VALUE",
        }
        if flag == "configuration":
            kwargs["choices"] = ["suppressed", "unsuppressed"]
            kwargs.pop("metavar")
        meta.add_argument(f"--{flag}", **kwargs)

    return parser


def _explicit_flags(argv: Sequence[str]) -> set:
    """
    Which option strings the operator actually typed.

    --config used to discard every other flag silently. Knowing what was typed
    lets explicit flags win over the file, and lets the run say so.
    """
    typed = set()
    for token in argv:
        if token.startswith("--"):
            typed.add(token.split("=", 1)[0])
        elif token.startswith("-") and len(token) > 1:
            typed.add(token.split("=", 1)[0])
    return typed


def _config_from_args(args: argparse.Namespace, typed: set, warnings: List[str]) -> AnalysisConfig:
    """Build the AnalysisConfig, letting explicit flags override any config file."""
    if args.config is not None:
        config = AnalysisConfig.from_json(args.config, warnings)
        overrides: List[str] = []
    else:
        config = AnalysisConfig()
        overrides = []

    def apply(flag: str, attribute: str, value: Any, *, always: bool = False) -> None:
        if always or flag in typed:
            if args.config is not None and flag in typed:
                overrides.append(flag)
            setattr(config, attribute, value)

    always = args.config is None

    apply("--Pa-per-FS", "Pa_per_FS", args.Pa_per_FS, always=always and args.Pa_per_FS is not None)
    apply("--sensitivity-mV", "sensitivity_mV_per_Pa", args.sensitivity_mV,
          always=always and args.sensitivity_mV is not None)
    apply("--preamp-gain-dB", "preamp_gain_dB", args.preamp_gain_dB)
    apply("--adc-full-scale-V", "adc_full_scale_V", args.adc_full_scale_V,
          always=always and args.adc_full_scale_V is not None)
    apply("--V-per-FS", "V_per_FS", args.V_per_FS, always=always and args.V_per_FS is not None)
    apply("--calibrator-tone", "calibrator_tone_file",
          str(args.calibrator_tone) if args.calibrator_tone else None,
          always=always and args.calibrator_tone is not None)
    apply("--calibrator-post", "calibrator_post_file",
          str(args.calibrator_post) if args.calibrator_post else None,
          always=always and args.calibrator_post is not None)
    apply("--calibrator-level-dB", "calibrator_level_dB", args.calibrator_level_dB)
    apply("--calibrator-freq-Hz", "calibrator_frequency_Hz", args.calibrator_freq_Hz)
    apply("--preset", "preset", args.preset, always=always and args.preset is not None)
    apply("--uncalibrated", "uncalibrated", bool(args.uncalibrated),
          always=always and args.uncalibrated)
    apply("--cal-desc", "calibration_description", args.cal_desc)

    apply("--channel", "channel", args.channel, always=always)
    apply("--mono-mix", "mono_mix", bool(args.mono_mix), always=always and args.mono_mix)

    apply("--threshold-dB", "detection_threshold_dB", args.threshold_dB,
          always=always and args.threshold_dB is not None)
    apply("--threshold-relative-dB", "threshold_relative_dB", args.threshold_relative_dB,
          always=always and args.threshold_relative_dB is not None)
    apply("--refractory-ms", "refractory_ms", args.refractory_ms, always=always)
    apply("--pre-ms", "pre_shot_ms", args.pre_ms, always=always)
    apply("--post-ms", "post_shot_ms", args.post_ms, always=always)
    apply("--min-shots", "min_shots", args.min_shots, always=always)
    apply("--max-shots", "max_shots", args.max_shots, always=always)

    nperseg = _parse_nperseg(args.nperseg)
    apply("--nperseg", "nperseg", nperseg, always=always)
    apply("--noverlap", "noverlap", args.noverlap, always=always and args.noverlap is not None)
    apply("--overlap-fraction", "overlap_fraction", args.overlap_fraction, always=always)
    apply("--dtype", "load_dtype", args.dtype, always=always)
    apply("--band-hop-ms", "band_hop_ms", args.band_hop_ms, always=always)
    apply("--band-weighting", "band_time_weighting", args.band_weighting, always=always)
    apply("--nrr-dB", "protection_NRR_dB", args.nrr_dB, always=always)
    apply("--no-bands", "compute_bands", not args.no_bands, always=always and args.no_bands)
    apply("--no-time-series", "compute_time_series", not args.no_time_series,
          always=always and args.no_time_series)
    apply("--no-per-shot", "save_per_shot_plots", not args.no_per_shot,
          always=always and args.no_per_shot)
    apply("--no-plots", "make_plots", not args.no_plots, always=always and args.no_plots)
    apply("--formats", "plot_formats", validate_formats(args.formats), always=always)

    if overrides:
        message = (
            f"Config file {args.config} was loaded; these command-line flags override it: "
            + ", ".join(sorted(set(overrides)))
        )
        logger.info(message)
        warnings.append(message)
        say(f"  {message}")

    # Re-validate after mutation: the dataclass validated its initial values only.
    return AnalysisConfig(**config.to_dict())


def _parse_nperseg(value: Any) -> Optional[int]:
    """'auto' means derive from the sample rate; anything else must be an integer."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("auto", ""):
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigurationError(
                f"--nperseg must be a whole number or 'auto', got {value!r}"
            ) from exc
    return int(value)


def _metadata_from_args(args: argparse.Namespace, warnings: List[str]) -> TestMetadata:
    """Build TestMetadata from a file plus individual flags (flags win)."""
    data: Dict[str, Any] = {}
    if args.metadata is not None:
        path = Path(args.metadata)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(f"Metadata file {path} could not be read: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Metadata file {path} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigurationError(f"Metadata file {path} must contain a JSON object")
        known = {f.name for f in TestMetadata.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(loaded) - known)
        if unknown:
            message = (
                f"Metadata file {path.name} contains unrecognised key(s): "
                f"{', '.join(unknown)}. They were ignored."
            )
            logger.warning(message)
            warnings.append(message)
        data.update({k: v for k, v in loaded.items() if k in known})

    for flag, _kind, _help in METADATA_FLAGS:
        dest = "meta_" + flag.replace("-", "_")
        value = getattr(args, dest, None)
        if value is not None:
            data[flag.replace("-", "_")] = value

    # The calibrator level given for calibration is also part of the record.
    if "calibrator_level_dB" not in data and args.calibrator_tone is not None:
        data["calibrator_level_dB"] = args.calibrator_level_dB

    return TestMetadata.from_dict(data)


def _print_presets(presets: Dict[str, CalibrationPreset], store: Path) -> None:
    say(f"Calibration presets (store: {store})")
    if not presets:
        say("  (none)")
        return
    for name in sorted(presets):
        preset = presets[name]
        kind = "built-in" if preset.builtin else "saved"
        say("")
        say(f"  {name}   [{kind}]")
        say(f"    Pa/FS: {preset.Pa_per_FS:.6g}   "
            f"(full scale = {20.0 * math.log10(preset.Pa_per_FS / 2e-5):.1f} dB SPL)")
        say(f"    {preset.provenance}")


def _pick_file_interactively() -> Optional[Path]:
    """Open the native picker, if this installation has one."""
    try:
        from FileSelector import choose_media_file  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - headless installs have no tkinter
        raise ConfigurationError(
            f"No input file was given and the file picker is unavailable ({exc}). "
            f"Pass the file as an argument."
        ) from exc
    say("Select a recording to analyse...")
    return choose_media_file()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code; never raises for user error."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    typed = _explicit_flags(argv)

    configure_logging(args.verbose, args.quiet)
    warnings: List[str] = []

    try:
        store = profiles_path(args.profiles_file)

        if args.delete_preset:
            removed = delete_profile(args.delete_preset, args.profiles_file)
            say(f"{'Deleted' if removed else 'No such profile:'} {args.delete_preset}")
            return EXIT_OK if removed else EXIT_ERROR

        presets = load_profiles(args.profiles_file, warnings)
        if args.list_presets:
            _print_presets(presets, store)
            return EXIT_OK

        config = _config_from_args(args, typed, warnings)

        # ---- Resolve the input ----
        if args.input is not None:
            chosen = args.input
        else:
            chosen = _pick_file_interactively()
            if chosen is None:
                say("No file selected.")
                return EXIT_ERROR

        prepared = prepare_input(Path(chosen), warnings)

        # ---- Resolve the calibration up front, before any analysis time is spent ----
        calibration = resolve_calibration(config, presets=presets, warnings=warnings)

        if args.save_preset:
            saved_to = save_profile(
                args.save_preset, calibration,
                note=config.calibration_description or f"from {prepared.original_path.name}",
                path=args.profiles_file,
            )
            say(f"Saved calibration profile {args.save_preset!r} to {saved_to}")

        metadata = _metadata_from_args(args, warnings)

        if args.reference is not None:
            if not Path(args.reference).exists():
                raise ConfigurationError(f"--reference directory not found: {args.reference}")
            # Fail before spending analysis time on a reference that cannot be used.
            load_reference_record(args.reference)

        # ---- Which channels ----
        _frames, _sr, _duration, n_channels = get_wav_info(prepared.audio_path)
        if args.channels is not None:
            if str(args.channels).strip().lower() != "all":
                raise ConfigurationError(
                    f"--channels accepts only 'all' (got {args.channels!r}); "
                    f"use --channel N for a single channel."
                )
            if config.mono_mix:
                raise ConfigurationError("--channels all and --mono-mix are contradictory")
            channels = list(range(n_channels))
        else:
            channels = [None]   # a single run using config.channel / config.mono_mix

        results: List[AnalysisResult] = []
        for position, channel_index in enumerate(channels):
            run_config = config
            suffix = ""
            if channel_index is not None:
                run_config = AnalysisConfig(**{**config.to_dict(), "channel": channel_index})
                suffix = f"ch{channel_index:02d}"
                say("")
                say(f"===== Channel {channel_index} of {n_channels} "
                    f"({position + 1}/{len(channels)}) =====")

            # Give each channel its own slice of the overall progress bar, so
            # the reported percentage rises once across the whole invocation
            # instead of restarting at 0 for every channel.
            span = 100.0 / len(channels)
            with progress_window(position * span, (position + 1) * span):
                results.append(analyze_file(
                    prepared.audio_path,
                    run_config,
                    args.output,
                    dir_suffix=suffix,
                    metadata=metadata,
                    reference_dir=args.reference,
                    calibration=calibration,
                    original_path=prepared.original_path,
                    warnings=warnings,
                    config_file=args.config,
                ))

        return _final_status(results)

    except CalibrationRequired as exc:
        say("")
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (ConfigurationError, SasaError) as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - last resort; the traceback goes to the log
        logger.exception("Unhandled error during analysis")
        print(f"ERROR: analysis failed: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _final_status(results: Sequence[AnalysisResult]) -> int:
    """
    Decide the process exit code and say plainly what happened.

    Zero detections is a failure, not a success with a 0.0 dB result: nothing is
    published, no CSV is claimed, and the exit code says so.
    """
    if not results:
        return EXIT_ERROR

    say("")
    say("Summary")
    any_shots = False
    any_invalid = False
    deliverable_missing = False
    for result in results:
        label = result.record["source"]["channel_used"]
        if result.n_shots == 0:
            say(f"  {label}: NO SHOTS DETECTED - no measurement was produced.")
            continue
        any_shots = True
        unit = result.calibration.level_unit
        stat = result.aggregate.stats.get("Lpeak_Z")
        lae = result.aggregate.stats.get("LAE")
        state = "VALID" if result.measurement_valid else "INADMISSIBLE"
        if not result.measurement_valid:
            any_invalid = True
        if result.reference_requested and not result.insertion_loss_produced:
            deliverable_missing = True
        say(f"  {label}: {result.n_shots} shot(s), {state}")
        if stat and math.isfinite(stat.maximum):
            say(f"    Peak (Z) max {stat.maximum:.1f} {unit}")
        if lae and math.isfinite(lae.mean):
            say(f"    LAE mean     {lae.mean:.1f} {unit}")
        say(f"    Output       {result.output_dir}")

    if not any_shots:
        say("")
        say("No shots were detected in any channel. Nothing was measured.")
        say("  Try --threshold-relative-dB 20 (more sensitive), check --channel,")
        say("  or confirm the recording really contains the string.")
        return EXIT_NO_SHOTS

    if any_invalid:
        say("")
        say("At least one channel produced an INADMISSIBLE measurement; see quality.errors "
            "in analysis_metadata.json. The numbers must not be reported.")
        return EXIT_INVALID

    if deliverable_missing:
        say("")
        say("Insertion loss was requested but could not be computed; see the warnings above. "
            "The per-shot measurement was still written.")
        return EXIT_ERROR

    return EXIT_OK


def cli_main() -> None:
    """Console-script entry point (pyproject declares main:cli_main)."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
