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
import base64
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
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Plot rendering must never require a display; this is set before matplotlib is
# imported anywhere (plots.py is imported lazily, inside the plotting stage).
os.environ.setdefault("MPLBACKEND", "Agg")

from WavLoader import get_wav_info, load_wav, load_wav_chunk  # noqa: E402
from calibration import (  # noqa: E402
    dB_SPL_to_amplitude,
    Calibration,
    CeilingClipping,
    SignalQuality,
    amplitude_to_dB_SPL,
    assess_signal_quality,
    ceiling_clipping_error,
)
from shot_detect import (  # noqa: E402
    DetectionReport,
    DetectionTuning,
    ShotEvent,
    autotune_detection,
    autotune_from_envelope,
    bandpass_for_detection,
    compute_envelope,
    detect_shots,
    TrimSpan,
    find_shot_string_span,
)
from metrics import (  # noqa: E402
    AggregateMetrics,
    MetricStats,
    ShotMetrics,
    compute_aggregate_metrics,
    compute_insertion_loss,
    compute_shot_metrics,
)
from ahaah import VALIDATION_STATUS as AHAAH_STATUS, compute_ahaah_both  # noqa: E402
from anomaly import review_shot_string  # noqa: E402
from atmosphere import (  # noqa: E402
    Atmosphere,
    describe_atmospheric_effect,
    normalise_insertion_loss_bands,
)
from bands import ThirdOctaveAnalyzer, band_insertion_loss  # noqa: E402
from pairing import assess_comparability  # noqa: E402
from stringstats import string_summary  # noqa: E402
from STFT import STFTResult, analyze_stft, recommended_nperseg  # noqa: E402
from weighting import weighting_settling_samples  # noqa: E402
from provenance import (  # noqa: E402
    SoftwareInfo,
    SourceInfo,
    TestMetadata,
    file_sha256,
    make_provenance_block,
)
from provenance import __version__ as _PROVENANCE_VERSION  # noqa: E402
from textutil import count, plural

# ONE version for the whole application, and provenance.py owns it.
#
# There used to be two constants. This one tracked the releases; the one in
# provenance.py was left at 2.0.0, and it is the one that goes into every
# record's software block -- so every analysis produced after 2.0.0 stated,
# in its own provenance, that it had been produced by 2.0.0. A record whose
# account of what made it is wrong is the failure this file exists to prevent.
#
# main imports provenance, so the constant lives there and is re-exported
# here. tests/test_packaging.py holds it to pyproject.toml.
__version__ = _PROVENANCE_VERSION

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
# Context kept either side of the shot string in the full-recording figures.
# Wide enough to show the ambient the shots stand against, narrow enough that a
# ten-minute range recording still collapses to the part with shooting in it.
PLOT_FOCUS_MARGIN_S = 0.25

WAVEFORM_FULLRES_PRE_S = 0.010
WAVEFORM_FULLRES_POST_S = 0.020

# The waveform data artifact the interface plots from.
#
# It is a MIN/MAX ENVELOPE, not a subsampling. Taking every Nth sample of a
# blast recording throws the peak away whenever the peak lands between two
# kept samples, and at 280 samples per column that is almost every column --
# so the picture would under-report exactly the thing being measured. Storing
# the extreme of each column instead means the drawn envelope always contains
# the true peak.
#
# A RESOLUTION PYRAMID, the way a map has zoom levels.
#
# One envelope is not enough. 2048 columns across a twelve-second string is
# 5.9 ms per column, so zooming into a single blast just magnifies the same
# columns and the picture stops gaining detail exactly where the operator
# started looking closely. Each level here has four times the columns of the
# one before, and the interface picks the coarsest level that still puts at
# least one column in every pixel of the plot -- so detail appears as the zoom
# goes in, and nothing larger than the screen is ever decoded.
#
# Three levels and no more. A fourth at 131072 columns took the file to 7 MB
# and bought nothing: by then the visible span is inside a shot window, and the
# per-shot envelope below is finer still. The pyramid only has to carry the eye
# from the whole string down to one blast; from there the shot takes over.
WAVEFORM_ENVELOPE_LEVELS = (2048, 8192, 32768)

# Per shot, over its own window only. A 250 ms window at 48 kHz is 12000
# samples, so 4096 columns is about three samples per column -- finer than any
# display can resolve, and the level the deep zoom actually lands on.
SHOT_ENVELOPE_COLUMNS = 4096

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
    #
    # The four tunable knobs default to None, meaning "measure it from the
    # recording". A number here is the operator overriding that measurement, and
    # is always obeyed; None with auto_detect off falls back to the constants
    # below, which are what SASA shipped before it could tune itself.
    detection_threshold_dB: Optional[float] = None
    threshold_relative_dB: Optional[float] = None
    refractory_ms: Optional[float] = None
    pre_shot_ms: Optional[float] = None
    post_shot_ms: Optional[float] = None
    auto_detect: bool = True
    expected_shots: Optional[int] = None
    min_snr_dB: Optional[float] = None
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
    # "shots" draws the full-recording figures over the shot string only;
    # "full" draws the whole file. Metrics are unaffected either way.
    plot_span: str = "shots"
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
        if self.refractory_ms is not None:
            self.refractory_ms = _positive(self.refractory_ms, "refractory_ms")
        if self.pre_shot_ms is not None:
            self.pre_shot_ms = _non_negative(self.pre_shot_ms, "pre_shot_ms")
        if self.post_shot_ms is not None:
            self.post_shot_ms = _positive(self.post_shot_ms, "post_shot_ms")
        if self.expected_shots is not None:
            self.expected_shots = _integer(
                self.expected_shots, "expected_shots", minimum=1, maximum=100_000
            )
        if self.min_snr_dB is not None:
            self.min_snr_dB = _non_negative(self.min_snr_dB, "min_snr_dB")
            if self.min_snr_dB > 80.0:
                raise ConfigurationError(
                    f"min_snr_dB of {self.min_snr_dB} dB would reject every real shot; "
                    f"muzzle blast clears its noise floor by 30-60 dB"
                )
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
        if self.plot_span not in ("shots", "full"):
            raise ConfigurationError(
                f"plot_span must be shots or full, got {self.plot_span!r}"
            )

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
                f"Config file {path.name} contains unrecognised {plural(len(unknown), 'key')}: "
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


# The rate below which a muzzle blast's rise time cannot be resolved. Kept
# here as well as in calibration.assess_signal_quality so the probe can say so
# BEFORE a run rather than in the verdict afterwards.
MIN_USEFUL_SAMPLE_RATE_HZ: int = 48000


def probe_input(path: Path) -> Dict[str, Any]:
    """
    Describe a recording without analysing it.

    The interface needs the sample rate, channel count and duration the moment
    a file is chosen, not after a run: the bin spacing it quotes, the band
    range it can offer and the "this rate cannot resolve a rise time" warning
    are all functions of properties that are in the file's header. Reading
    them costs milliseconds; discovering them by running an analysis and
    reading the verdict costs minutes and a wasted output directory.

    Nothing is decoded. A container is probed through ffmpeg's stream
    metadata, an audio file through its header, so this stays cheap on a file
    of any length.

    Returns a JSON-safe dict. `readable` is false when the file cannot be
    described at all, and `problem` then says why -- this never raises for a
    file the operator picked by mistake.
    """
    path = Path(path).expanduser()
    out: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "readable": False,
        "problem": None,
        "needs_extraction": None,
        "sample_rate": None,
        "channels": None,
        "subtype": None,
        "frames": None,
        "duration_s": None,
        "size_bytes": None,
        "nyquist_Hz": None,
        "sample_rate_adequate": None,
        "notes": [],
    }

    if not path.exists():
        out["problem"] = f"File not found: {path}"
        return out
    if not path.is_file():
        out["problem"] = f"Not a file: {path}"
        return out

    try:
        out["size_bytes"] = path.stat().st_size
    except OSError:
        pass

    suffix = path.suffix.lower()

    # Audio first: a header read, no decoding, no ffmpeg.
    if suffix not in VIDEO_EXTS:
        try:
            import soundfile as sf  # noqa: PLC0415

            info = sf.info(str(path))
            out.update(
                readable=True,
                needs_extraction=False,
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                subtype=str(info.subtype or ""),
                frames=int(info.frames),
                duration_s=round(float(info.frames) / float(info.samplerate), 4)
                if info.samplerate else None,
            )
        except Exception as exc:  # noqa: BLE001 - may still be a container ffmpeg can read
            logger.info("%s is not directly readable as audio (%s)", path.name, exc)

    # Container: ask ffmpeg about the first audio stream. Still no decoding.
    if not out["readable"]:
        try:
            from ExtractAudio import ExtractionError, find_ffmpeg, probe_audio_stream  # noqa: PLC0415

            exe = find_ffmpeg()
            if not exe:
                out["problem"] = (
                    "No ffmpeg is available to read this container, so its audio "
                    "track cannot be described or extracted."
                )
                return out
            info = probe_audio_stream(path, exe)
            frames = (
                int(round(info.duration_s * info.sample_rate))
                if info.duration_s and info.sample_rate else None
            )
            out.update(
                readable=True,
                needs_extraction=True,
                sample_rate=int(info.sample_rate) if info.sample_rate else None,
                channels=int(info.channels) if info.channels else None,
                subtype=str(info.codec or ""),
                frames=frames,
                duration_s=round(float(info.duration_s), 4) if info.duration_s else None,
            )
            out["notes"].append(
                f"The audio track will be extracted from this {suffix.lstrip('.') or 'container'} "
                f"before analysis, at its own rate, depth and channel count."
            )
        except ExtractionError as exc:
            out["problem"] = str(exc)
            return out
        except Exception as exc:  # noqa: BLE001 - a probe must not raise for a bad file
            out["problem"] = f"{path.name} could not be read as audio or as a container: {exc}"
            return out

    rate = out["sample_rate"]
    if rate:
        out["nyquist_Hz"] = rate / 2.0
        out["sample_rate_adequate"] = rate >= MIN_USEFUL_SAMPLE_RATE_HZ
        if not out["sample_rate_adequate"]:
            out["notes"].append(
                f"{rate} Hz cannot resolve a muzzle blast's rise time: one sample is "
                f"{1e6 / rate:.1f} us against a typical 1-50 us rise. Rise time, crest "
                f"factor and A-duration from this recording will be upper bounds. "
                f"{MIN_USEFUL_SAMPLE_RATE_HZ} Hz or more is wanted."
            )

    if out["channels"] and out["channels"] > 1:
        out["notes"].append(
            f"{count(out['channels'], 'channel')}; channel 0 is analysed unless another "
            f"is chosen. The channels are not mixed, because mixing two microphones "
            f"changes the level."
        )

    return out


def describe_channel(channel: Optional[int], mono_mix: bool, n_channels: int) -> str:
    """Human/machine description of which channel produced the numbers."""
    if mono_mix:
        return f"mono mix of {n_channels} channels"
    return f"channel {channel}"


def read_samples(
    path: Path,
    start: int,
    n_frames: int,
    *,
    channel: Optional[int],
    mono_mix: bool,
    dtype: str,
) -> np.ndarray:
    """Read a span of full-scale samples for the selected channel."""
    samples, _sr = load_wav_chunk(
        path, start, n_frames,
        dtype=dtype,
        mono=mono_mix,
        channel=None if mono_mix else channel,
    )
    return np.asarray(samples)


# ═══════════════════════════════════════════════════════════════════════════
#  Detection preview
# ═══════════════════════════════════════════════════════════════════════════
#
# Why this exists.
#
# Detection settings used to be knobs you turned before a run and judged after
# it, by reading a shot count in a verdict. That is a slow loop over an
# expensive operation, and it puts the one decision that determines every
# subsequent number - which events are shots - furthest from the evidence.
#
# The preview does detection and nothing else: no metrics, no weighting
# filters, no figures, no output directory. It answers "how many shots, where,
# and how sure" in a fraction of a second, so the setting can be moved against
# the answer instead of against a guess.
#
# It is deliberately UNCALIBRATED. Detection is relative to the recording's own
# peak, so a calibration would change none of it, and requiring one would mean
# an operator could not look at their recording until they had resolved a
# question that has nothing to do with looking at it. Every level it reports is
# therefore dB re FS and is labelled as such.

# The envelope the preview draws. Enough columns to see individual blasts in a
# minute-long recording; few enough that the response stays small.
PREVIEW_COLUMNS: int = 1600

# Preview levels are relative to full scale, never sound pressure. Stated once
# here so no caller has to decide.
PREVIEW_LEVEL_UNIT: str = "dB re FS"


def detect_preview(
    path: Path,
    *,
    channel: int = 0,
    auto_detect: bool = True,
    expected_shots: Optional[int] = None,
    threshold_relative_dB: Optional[float] = None,
    refractory_ms: Optional[float] = None,
    pre_ms: Optional[float] = None,
    post_ms: Optional[float] = None,
    min_snr_dB: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Detect shots and report what was found, without analysing anything.

    Returns a dictionary in every case, including failure: the interface calls
    this while the operator is still choosing a file, and an exception there is
    a dead page rather than a message.
    """
    result: Dict[str, Any] = {
        "readable": False,
        "problem": None,
        "path": str(path),
        "name": Path(path).name,
        "level_unit": PREVIEW_LEVEL_UNIT,
        "sample_rate": None,
        "duration_s": None,
        "n_shots": 0,
        "shots": [],
        "tuning": None,
        "detection": None,
        "envelope": None,
        "quality": None,
    }

    try:
        import soundfile as sf  # noqa: PLC0415

        prepared = prepare_input(Path(path))
        info = sf.info(str(prepared.audio_path))
        sample_rate = int(info.samplerate)
        samples = read_samples(
            prepared.audio_path, 0, int(info.frames),
            channel=channel, mono_mix=False, dtype="float32",
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a message, not a crash
        result["problem"] = str(exc) or exc.__class__.__name__
        return result

    if samples.size == 0 or sample_rate <= 0:
        result["problem"] = "The recording contains no samples on this channel."
        return result

    x = np.asarray(samples, dtype=np.float64).ravel()
    result["readable"] = True
    result["sample_rate"] = sample_rate
    result["duration_s"] = round(x.size / sample_rate, 4)

    reference = Calibration.uncalibrated()
    # Detection works in the same dB-above-P_REF space a calibrated run does,
    # because that is what detect_shots() reports. Subtracting the level of a
    # full-scale sample turns each one back into dB re FS, which is the only
    # thing an uncalibrated recording can honestly be measured in.
    full_scale = reference.full_scale_dB
    quality = assess_signal_quality(x, sample_rate, reference)
    result["quality"] = {
        "is_clipped": bool(quality.is_clipped),
        "ceiling_clipped": bool(quality.ceiling.detected),
        "ceiling_dBFS": (round(quality.ceiling.ceiling_dBFS, 2)
                         if quality.ceiling.detected else None),
        "clipped_runs": int(quality.clipped_runs),
        "errors": list(quality.errors),
    }

    gate = FALLBACK_MIN_SNR_DB if min_snr_dB is None else float(min_snr_dB)
    tuning: Optional[DetectionTuning] = None
    if auto_detect:
        tuning = autotune_detection(
            x, sample_rate, expected_shots=expected_shots, min_snr_dB=gate,
        )
        # DetectionTuning reports levels the way detect_shots() does, which on a
        # calibrated run is dB SPL. Here there is no calibration, so those two
        # fields are neither SPL nor dB re FS until the full-scale offset is
        # taken off. They are replaced rather than supplemented: leaving a
        # number in the payload whose unit depends on how it got there is how a
        # dB re FS value ends up printed as a sound pressure level.
        payload = tuning.to_dict()
        for absolute, relative in (("peak_dB", "peak_dBFS"),
                                   ("noise_floor_dB", "noise_floor_dBFS")):
            value = payload.pop(absolute, None)
            payload[relative] = round(value - full_scale, 2) if value is not None else None
        result["tuning"] = payload

    config = AnalysisConfig(
        uncalibrated=True,
        auto_detect=auto_detect,
        expected_shots=expected_shots,
        threshold_relative_dB=threshold_relative_dB,
        refractory_ms=refractory_ms,
        pre_shot_ms=pre_ms,
        post_shot_ms=post_ms,
        min_snr_dB=gate,
    )
    resolved = resolve_detection(config, tuning)

    reports: List[DetectionReport] = []
    shots = detect_shots(
        x, sample_rate,
        threshold_relative_dB=resolved.threshold_relative_dB,
        threshold_dB=resolved.threshold_dB,
        pre_ms=resolved.pre_ms,
        post_ms=resolved.post_ms,
        refractory_ms=resolved.refractory_ms,
        min_snr_dB=resolved.min_snr_dB,
        samples_FS=x,
        ceiling_FS=quality.ceiling.ceiling_FS if quality.ceiling.detected else None,
        report=reports,
    )
    detection = reports[0] if reports else None

    result["settings"] = resolved.to_dict()
    result["n_shots"] = len(shots)
    result["shots"] = [
        {
            "shot_number": shot.shot_number,
            "time_s": round(shot.time_s, 5),
            "peak_dBFS": round(shot.peak_dB - full_scale, 2),
            "snr_dB": (round(shot.snr_dB, 1) if math.isfinite(shot.snr_dB) else None),
            "clipped": bool(shot.clipped),
            "window_start_s": round(shot.window_start / sample_rate, 5),
            "window_end_s": round(shot.window_end / sample_rate, 5),
            "n_arrivals": len(shot.arrivals),
        }
        for shot in shots
    ]
    if detection is not None:
        result["detection"] = {
            "n_candidates": detection.n_candidates,
            "n_suppressed_by_refractory": detection.n_suppressed_by_refractory,
            "threshold_dBFS": round(detection.threshold_dB - full_scale, 2),
            "threshold_mode": detection.threshold_mode,
            "peak_dBFS": round(detection.peak_level_dB - full_scale, 2),
            "noise_floor_dBFS": round(detection.noise_floor_dB - full_scale, 2),
            "warnings": list(detection.warnings),
        }

    lo, hi = _column_envelope(x, PREVIEW_COLUMNS)
    result["envelope"] = {"columns": len(lo), "lo": lo, "hi": hi}
    return result


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


# Distance the per-band insertion loss is referred to when both strings recorded
# their microphone distance. One metre is the standard reporting distance, and it
# is short enough that absorption over the remaining path is small. The value is
# always stated in the record, because when the two strings were shot in
# different air the insertion loss genuinely depends on where it is quoted.
NORMALISATION_DISTANCE_M: float = 1.0


def _num_or_none(value: Any) -> Optional[float]:
    """Coerce a metadata value to a float, or None when it was not recorded."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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

    # Is this pair the same experiment? Each objection that physics can price is
    # priced, so the operator sees how many of the decibels below belong to the
    # suppressor and how many belong to the setup having moved.
    ref_meta = reference.get("test_metadata") or {}
    test_meta = test_record.get("test_metadata") or {}
    comparability = assess_comparability(
        ref_meta, test_meta, ref_aggregate, test_aggregate,
        reference_label=str(Path(reference_dir).name),
        test_label=str((test_record.get("source") or {}).get("path") or "test"),
    )
    for objection in comparability.objections:
        prefix = {
            "blocking": "NOT COMPARABLE: ",
            "material": "",
            "advisory": "",
        }.get(objection.severity, "")
        amount = (
            f" This accounts for {objection.quantified_dB:+.2f} dB of the difference."
            if objection.quantified_dB is not None else ""
        )
        warnings.append(f"{prefix}{objection.message}{amount}")

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

    # ---- Refer both strings to a common distance before differencing ----
    #
    # The raw per-band figure above is what was measured. This is what the
    # suppressor did, with the geometry and the weather taken back out. Both are
    # reported: a corrected number that cannot be checked against the
    # measurement it came from is not a measurement record.
    normalised: Dict[str, Any] = {"valid": False, "refusal": "not attempted"}
    if bands["insertion_loss_dB"]:
        ref_peak_Pa = test_peak_Pa = None
        if test_cal.get("calibrated"):
            # Only meaningful in Pascals, so the linearity guard only runs when
            # the levels are real sound pressure.
            with contextlib.suppress(ValueError, OverflowError):
                ref_peak_Pa = float(dB_SPL_to_amplitude(ref_aggregate.Lpeak_Z_max))
                test_peak_Pa = float(dB_SPL_to_amplitude(test_aggregate.Lpeak_Z_max))

        comparison_result = normalise_insertion_loss_bands(
            np.asarray(bands["reference_dB"], dtype=float),
            np.asarray(bands["test_dB"], dtype=float),
            np.asarray(bands["frequencies_Hz"], dtype=float),
            reference_atmosphere=Atmosphere.from_metadata(ref_meta),
            test_atmosphere=Atmosphere.from_metadata(test_meta),
            reference_distance_m=_num_or_none(ref_meta.get("mic_distance_m")),
            test_distance_m=_num_or_none(test_meta.get("mic_distance_m")),
            normalisation_distance_m=NORMALISATION_DISTANCE_M,
            reference_peak_Pa=ref_peak_Pa,
            test_peak_Pa=test_peak_Pa,
        )
        normalised = comparison_result.to_dict()
        if comparison_result.valid:
            say("")
            for line in comparison_result.summary().splitlines():
                say(line)
        else:
            warnings.append(
                f"Insertion loss was NOT normalised to a common distance: "
                f"{comparison_result.refusal} The per-band figures are as measured, "
                f"so any difference in microphone position or weather between the "
                f"two strings is still in them."
            )

    return {
        "reference_dir": str(Path(reference_dir).resolve()),
        "reference_input": (reference.get("analysis") or {}).get("input_file", ""),
        "reference_sha256": (reference.get("analysis") or {}).get("input_sha256"),
        "reference_n_shots": ref_aggregate.n_valid,
        "test_n_shots": test_aggregate.n_valid,
        "level_unit": test_cal["level_unit"],
        "metrics": [loss.to_dict() for loss in losses],
        "bands": bands,
        "bands_normalised": normalised,
        "comparability": comparability.to_dict(),
        "warnings": warnings,
    }


def print_insertion_loss(comparison: Dict[str, Any]) -> None:
    """Print the deliverable: how much quieter the test configuration is."""
    unit = comparison.get("level_unit", "dB")
    comparability = comparison.get("comparability") or {}
    say("")
    if comparability and not comparability.get("comparable", True):
        say("  INSERTION LOSS IS NOT VALID FOR THIS PAIR")
        say("  The two strings are not the same experiment. The numbers below are")
        say("  the arithmetic difference between them, which is not insertion loss:")
        for objection in comparability.get("objections", []):
            if objection.get("severity") == "blocking":
                say(f"    - {objection['message']}")
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
    unexplained = float(comparability.get("unexplained_dB") or 0.0)
    if unexplained > 0.0:
        say(f"    Of the difference above, {unexplained:.2f} dB is attributable to the "
            f"two setups not matching, not to the suppressor.")
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
                f"{count(n_channels, 'channel')} (0-{n_channels - 1})."
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
    resolved: ResolvedDetection = analysis["resolved_detection"]
    shot_metrics: List[ShotMetrics] = analysis["shot_metrics"]

    say("")
    say("  Recording quality:")
    say(quality.summary())

    if resolved.tuning is not None:
        say("")
        say(resolved.tuning.summary())
        for note in resolved.tuning.notes:
            say(f"    {note}")
            run_warnings.append(note)

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
            say(f"    {count(aggregate.n_shots - aggregate.n_valid, 'shot')} "
                f"{plural(aggregate.n_shots - aggregate.n_valid, 'was')} excluded as invalid.")
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

    # ---- How the string behaved: first-round pop, drift, distribution ----
    #
    # Computed per level metric, because a suppressor can pop on peak and not on
    # exposure, and because the with/without-first-round pair is the honest way
    # to quote a suppressor that pops at all.
    string_breakdown: Dict[str, Any] = {}
    for level_name in ("Lpeak_Z", "Lpeak_A", "LAE"):
        levels = [
            float(getattr(m, level_name)) for m in shot_metrics
            if m.valid and math.isfinite(getattr(m, level_name, float("nan")))
        ]
        if levels:
            string_breakdown[level_name] = string_summary(levels, metric=level_name).to_dict()

    headline = string_breakdown.get("Lpeak_Z")
    if headline:
        say("")
        say(string_summary(
            [float(m.Lpeak_Z) for m in shot_metrics
             if m.valid and math.isfinite(m.Lpeak_Z)],
            metric="Lpeak_Z",
        ).summary())

    # ---- Which shots do not belong to this string, and why ----
    #
    # Run AFTER the string breakdown so the review knows whether the first round
    # popped: an established pop makes shot one an outlier by definition, and
    # reporting it as a possible squib would send the technician after the one
    # thing that was expected.
    pop_established = bool(
        ((string_breakdown.get("Lpeak_Z") or {}).get("first_round_pop") or {})
        .get("established", False)
    )
    shot_review = review_shot_string(
        shot_metrics, first_round_pop_established=pop_established
    )
    for line in shot_review.summary().splitlines():
        say(line)
    for flag in shot_review.flags_for_review():
        logger.info("Shot %d: %s", flag.shot_number, flag.message)

    # ---- The air the measurement was made in ----
    #
    # Recorded here whether or not it was used, so a later distance or
    # atmosphere normalisation can state what it assumed, and so two sessions
    # can be compared knowing whether the weather differed between them.
    air = Atmosphere.from_metadata(metadata)

    # What that air actually did to this measurement, band by band, and how much
    # the answer would move if a condition were mis-recorded. At bench distance
    # this is a fraction of a decibel; downrange it is the dominant term in the
    # top bands, and the operator needs to be able to see which case they are in.
    atmospheric_effect = describe_atmospheric_effect(
        aggregate.band_frequencies,
        atmosphere=air,
        distance_m=_num_or_none(getattr(metadata, "mic_distance_m", None)),
    )
    if atmospheric_effect.absorption_dB.size:
        say("")
        for line in atmospheric_effect.summary().splitlines():
            say(line)

    ahaah_block = _ahaah_block(
        analysis.get("pressure"), shots, shot_metrics, sample_rate,
        calibrated=bool(calibration.calibrated),
    )
    if ahaah_block.get("attempted"):
        say("")
        say("  Auditory Risk Unit (AHAAH):")
        say(f"    {ahaah_block.get('headline', 'unavailable')}")
        for note in (ahaah_block.get("notes") or [])[:2]:
            say(f"    {note}")

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
            resolved=resolved, reference_dir=reference_dir, config_file=config_file,
        ),
        "test_metadata": record["test_metadata"],
        "atmosphere": air.to_dict(),
        "atmospheric_effect": atmospheric_effect.to_dict(),
        "ahaah": ahaah_block,
        "shots": [s.to_dict() for s in shots],
        "per_shot_metrics": [m.to_dict() for m in shot_metrics],
        "shot_review": shot_review.to_dict(),
        "string_statistics": string_breakdown,
        "aggregate": aggregate.to_dict(),
        "artifacts": {},
        "validity": {
            "measurement_valid": bool(measurement_valid and bool(shots)),
            "calibrated": bool(calibration.calibrated),
            "level_unit": calibration.level_unit,
            "reasons": list(quality.errors) + ([] if shots else ["No shots were detected"]),
            "n_shots_detected": len(shots),
            "n_shots_valid": aggregate.n_valid,
            "shots_to_exclude": shot_review.shots_to_exclude(),
            "shots_to_review": shot_review.shots_to_review(),
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

    levels_block = _shot_levels_block(shot_metrics, level_unit=calibration.level_unit)
    if levels_block is not None:
        levels_path = out_dir / "shot_levels.json"
        try:
            write_json(levels_path, levels_block)
            artifacts["shot_levels"] = levels_path.name
            say(f"    levels:   {levels_path.name} ({count(len(levels_block['shots']), 'shot')})")
        except OSError as exc:
            message = f"The shot level curves could not be written: {exc}"
            logger.warning(message)
            run_warnings.append(message)

    waveform_block = _waveform_envelope_block(
        analysis.get("pressure"), shots, sample_rate,
        level_unit=calibration.level_unit,
        calibrated=bool(calibration.calibrated),
        focus=_plot_focus(config, shots, frames, sample_rate),
    )
    if waveform_block is not None:
        waveform_path = out_dir / "waveform_envelope.json"
        try:
            write_json(waveform_path, waveform_block)
            artifacts["waveform_envelope"] = waveform_path.name
            say(f"    waveform: {waveform_path.name} "
                f"({waveform_block['columns']} columns, "
                f"{count(len(waveform_block['shots']), 'shot window')})")
        except OSError as exc:
            message = f"The waveform envelope could not be written: {exc}"
            logger.warning(message)
            run_warnings.append(message)

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


def _column_envelope(values: np.ndarray, columns: int) -> Tuple[List[float], List[float]]:
    """
    Reduce a signal to the minimum and maximum within each of `columns` spans.

    This is the only honest way to draw a waveform smaller than its sample
    count. Subsampling picks one sample per column and discards the rest, so a
    transient shorter than the column -- which is what a muzzle blast IS at
    this scale -- is kept only if it happens to land on the chosen sample. The
    envelope keeps the extremes, so the drawn band always contains the peak
    and the reader can trust the height of it.

    Rounded to a RELATIVE precision, not to a fixed number of decimals. A
    fixed 3 dp is millipascals on a calibrated recording that peaks near 60 Pa
    -- finer than any display resolves -- and a catastrophe on an uncalibrated
    one that peaks near 0.75 full scale, where it rounds the entire noise floor
    to zero and draws a flat line where the signal is.
    """
    n = int(values.size)
    if n == 0:
        return [], []
    columns = max(1, min(int(columns), n))
    edges = np.linspace(0, n, columns + 1).astype(np.int64)
    # reduceat needs strictly increasing starts; the clamp above guarantees it.
    starts = edges[:-1]
    lo = np.minimum.reduceat(values, starts)
    hi = np.maximum.reduceat(values, starts)

    # Five decimal digits below the peak: about 0.001 % of full deflection,
    # which no display can resolve, whatever the units happen to be.
    peak = float(max(abs(float(lo.min())), abs(float(hi.max())), 1e-12))
    digits = max(0, 5 - int(math.floor(math.log10(peak))) - 1)
    digits = min(digits, 12)
    return ([round(float(v), digits) for v in lo], [round(float(v), digits) for v in hi])


def _waveform_envelope_block(
    pressure,
    shots,
    sample_rate: float,
    *,
    level_unit: str,
    calibrated: bool,
    focus: Optional[TrimSpan] = None,
) -> Optional[Dict[str, Any]]:
    """
    The waveform, as data the interface can plot and interrogate.

    Written alongside the PNG rather than instead of it: the PNG is what goes
    into the printed report, and this is what the operator points at. Both are
    drawn from the same samples, so they cannot disagree.

    Emitted from the DATA stage, not the plotting stage, so `--no-plots` still
    produces an interactive chart -- the flag turns off figure rendering, not
    the measurement.

    Returns None when there is no contiguous signal to reduce, which is the
    chunked path for a long recording.
    """
    if pressure is None or getattr(pressure, "size", 0) == 0:
        return None

    levels: List[Dict[str, Any]] = []
    for columns in WAVEFORM_ENVELOPE_LEVELS:
        if levels and levels[-1]["columns"] >= int(pressure.size):
            break            # the previous level already resolves every sample
        lo, hi = _column_envelope(pressure, columns)
        levels.append({"columns": len(lo), "lo": lo, "hi": hi})

    block: Dict[str, Any] = {
        "sample_rate_Hz": float(sample_rate),
        "n_samples": int(pressure.size),
        "duration_s": round(float(pressure.size) / float(sample_rate), 6),
        # Calibration.to_pascals is a PASS-THROUGH when the chain was never
        # calibrated -- its own docstring says so -- and the result is in
        # full-scale units. Labelling that axis "Pa" would put an absolute
        # pressure on a relative measurement, which is the one thing this
        # application exists not to do.
        "unit": "Pa" if calibrated else "FS",
        "calibrated": bool(calibrated),
        "level_unit": level_unit,
        "columns": levels[0]["columns"],
        "levels": levels,
        # The span the chart should OPEN at. Every sample is still in the file
        # above; this says where to look first, so a five-second string inside a
        # ten-minute recording is not presented as five hairlines on a rule.
        # Zooming out is one gesture; noticing that you needed to is not.
        "focus": focus.to_dict() if focus is not None else None,
        "shots": [],
    }

    # One higher-resolution envelope per shot window, so zooming into a blast
    # resolves it instead of showing the same overview columns magnified.
    for shot in shots or []:
        start = int(getattr(shot, "window_start", 0))
        stop = int(getattr(shot, "window_end", 0))
        if stop <= start or start < 0 or stop > pressure.size:
            continue
        shot_lo, shot_hi = _column_envelope(pressure[start:stop], SHOT_ENVELOPE_COLUMNS)
        block["shots"].append({
            "shot_number": getattr(shot, "shot_number", None),
            "t0_s": round(start / float(sample_rate), 6),
            "t1_s": round(stop / float(sample_rate), 6),
            "peak_time_s": round(float(getattr(shot, "index", start)) / float(sample_rate), 6),
            "columns": len(shot_lo),
            "lo": shot_lo,
            "hi": shot_hi,
        })
    return block


def _shot_levels_block(shot_metrics, *, level_unit: str) -> Optional[Dict[str, Any]]:
    """
    The time-weighted level curves for each shot, as data.

    These are the LAF/LAS/LZF/LZS histories that the per-shot summary figure
    draws in its top-right panel. They exist on ShotMetrics already and were
    simply never serialised, so the one panel of that figure that shows how a
    shot DEVELOPS -- the rise, the plateau, the decay the suppressor actually
    changes -- was available only as a picture.

    They are cheap: the hop is one millisecond, so a 250 ms window is 250
    points per curve, and eight shots of four curves is a few thousand numbers.

    Written to its own file rather than into analysis_metadata.json because the
    history endpoint re-serialises every record it lists, and this would be
    paid on every visit to the History view for no benefit there.
    """
    shots: List[Dict[str, Any]] = []
    for metric in shot_metrics or []:
        times = np.asarray(getattr(metric, "time_s", []), dtype=float).ravel()
        if times.size == 0:
            continue
        curves: Dict[str, Any] = {}
        for name in ("LAF", "LAS", "LZF", "LZS"):
            values = np.asarray(getattr(metric, name, []), dtype=float).ravel()
            if values.size != times.size:
                continue
            # -inf is what a silent sample gives; JSON has no word for it, and
            # null is the honest translation.
            curves[name] = [None if not math.isfinite(v) else round(float(v), 2)
                            for v in values]
        if not curves:
            continue
        shots.append({
            "shot_number": getattr(metric, "shot_number", None),
            "time_s": [round(float(t), 5) for t in times],
            "curves": curves,
        })

    if not shots:
        return None
    return {"level_unit": level_unit, "shots": shots}


def _ahaah_block(
    pressure,
    shots,
    shot_metrics,
    sample_rate: float,
    *,
    calibrated: bool,
) -> Dict[str, Any]:
    """
    Run the AHAAH model over the loudest valid shot and record what came back.

    WHAT THIS IS FOR, GIVEN THAT IT NEVER RETURNS A NUMBER
    ------------------------------------------------------
    MIL-STD-1474E approves two impulse-noise metrics: the A-weighted energy
    method, which SASA computes exactly, and the Auditory Risk Unit from ARL's
    AHAAH model. Customers ask for the ARU. ahaah.py implements the model to
    the limit of what ARL has published and then REFUSES to emit the number,
    because four specifications the answer depends on are absent from the
    public release and there is one reference case to test against -- see the
    header of ahaah.py and docs/AHAAH-SPEC.md section 11.

    Until this ran, none of that reached the operator: the module existed and
    nothing called it, so the application was silent on a metric its customers
    ask for by name. It now runs on every analysis and the record carries the
    refusal, the reason, and the model's own diagnostics. A refusal that is
    visible and argued is a deliverable. A number would not be.

    The LOUDEST shot is the one submitted, because ARU is assessed per impulse
    and the worst impulse is the one that governs.

    Args:
        pressure: the full calibrated pressure history, or None on the chunked
            path where no contiguous array is retained.
        shots: detected shot events, carrying their window bounds.
        shot_metrics: per-shot metrics, for choosing the loudest.
        sample_rate: sample rate of `pressure`, Hz.
        calibrated: False produces the model's uncalibrated refusal rather than
            a silent skip, so the reason is still on the record.

    Returns:
        A JSON-serialisable block. `attempted` is False when there was nothing
        to submit, and then `reason` says which.
    """
    block: Dict[str, Any] = {
        "attempted": False,
        "validation_status": AHAAH_STATUS,
        "reason": None,
        "shot_number": None,
        "headline": None,
        "notes": [],
        "unwarned": None,
        "warned": None,
    }

    if pressure is None:
        block["reason"] = ("The recording was analysed in chunks, so no contiguous "
                           "pressure history was retained to submit.")
        return block
    if not shots or not shot_metrics:
        block["reason"] = "No shots were detected."
        return block

    valid = [(m, s) for m, s in zip(shot_metrics, shots) if getattr(m, "valid", True)]
    if not valid:
        block["reason"] = "No valid shot to submit."
        return block

    metric, shot = max(
        valid,
        key=lambda pair: (pair[0].Lpeak_Z if math.isfinite(pair[0].Lpeak_Z) else -math.inf),
    )
    window = pressure[shot.window_start:shot.window_end]
    if window.size == 0:
        block["reason"] = "The loudest shot's window was empty."
        return block

    try:
        unwarned, warned = compute_ahaah_both(
            window, sample_rate,
            calibrated=calibrated,
            # 96 kHz is the model's own preference; refusing outright on a
            # 48 kHz recording would replace the model's reasoned refusal with
            # a gate, and the reasoned one is the more useful of the two.
            allow_low_rate=True,
        )
    except Exception as err:                                   # noqa: BLE001
        logger.warning("AHAAH did not run: %s", err)
        block["reason"] = f"The model raised: {err}"
        return block

    block["attempted"] = True
    block["shot_number"] = getattr(shot, "shot_number", None)
    block["headline"] = unwarned.headline_label
    block["notes"] = list(unwarned.notes)
    block["unwarned"] = unwarned.to_dict()
    block["warned"] = warned.to_dict()
    return block


# What the detection knobs fall back to when the operator has not set them and
# tuning is off or could not reach an answer. These are the values SASA shipped
# before it could measure them, kept so that --no-auto-detect reproduces the
# older behaviour exactly rather than something new.
FALLBACK_REFRACTORY_MS: float = 200.0
FALLBACK_PRE_SHOT_MS: float = 50.0
FALLBACK_POST_SHOT_MS: float = 200.0
FALLBACK_THRESHOLD_RELATIVE_DB: float = 30.0

# The impulsiveness gate. A candidate that does not clear the noise floor by
# this much is not a shot: a Gaussian noise peak sits about 13 dB above the RMS,
# so without a gate a recording with no gunfire in it yields a confident "shot"
# and a full metrics record. Real muzzle blast clears its floor by 30-60 dB, so
# 15 dB rejects noise while remaining far below anything genuine.
FALLBACK_MIN_SNR_DB: float = 15.0


@dataclass
class ResolvedDetection:
    """
    The detection settings a run will actually use, and where each came from.

    `source` carries one word per field - "operator", "measured" or "default" -
    so a result can never present a tuned number as though the operator chose
    it, or a fallback as though it were measured.
    """
    threshold_dB: Optional[float]
    threshold_relative_dB: Optional[float]
    refractory_ms: float
    pre_ms: float
    post_ms: float
    min_snr_dB: float = FALLBACK_MIN_SNR_DB
    source: Dict[str, str] = field(default_factory=dict)
    tuning: Optional[DetectionTuning] = None

    @property
    def tuned(self) -> bool:
        return bool(self.tuning and self.tuning.applied)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold_dB": self.threshold_dB,
            "threshold_relative_dB": self.threshold_relative_dB,
            "refractory_ms": self.refractory_ms,
            "pre_ms": self.pre_ms,
            "post_ms": self.post_ms,
            "min_snr_dB": self.min_snr_dB,
            "source": dict(self.source),
            "tuning": self.tuning.to_dict() if self.tuning else None,
        }


def resolve_min_snr(config: AnalysisConfig) -> float:
    """
    The impulsiveness gate, resolved on its own.

    The tuner needs this before it runs - its sweep stops where the gate would
    reject everything below - and the gate is never something the tuner
    chooses, so it can be settled without one.
    """
    return FALLBACK_MIN_SNR_DB if config.min_snr_dB is None else float(config.min_snr_dB)


def resolve_detection(
    config: AnalysisConfig,
    tuning: Optional[DetectionTuning],
) -> ResolvedDetection:
    """
    Decide each detection setting, in the order operator, measured, default.

    An explicit value is never overridden - tuning informs the knobs the
    operator left alone and no others. This is also why the tuner runs even when
    an absolute threshold was given: the refractory period and the post-trigger
    window are measurements of the recording, not of the threshold, and stay
    useful.
    """
    source: Dict[str, str] = {}

    def pick(name: str, given: Optional[float], measured: Optional[float],
             fallback: float) -> float:
        if given is not None:
            source[name] = "operator"
            return float(given)
        if measured is not None and tuning is not None and tuning.applied:
            source[name] = "measured"
            return float(measured)
        source[name] = "default"
        return float(fallback)

    usable = tuning if (tuning and tuning.applied) else None

    if config.detection_threshold_dB is not None:
        source["threshold"] = "operator"
        threshold_dB: Optional[float] = float(config.detection_threshold_dB)
        threshold_relative_dB: Optional[float] = None
    else:
        threshold_dB = None
        threshold_relative_dB = pick(
            "threshold", config.threshold_relative_dB,
            usable.threshold_relative_dB if usable else None,
            FALLBACK_THRESHOLD_RELATIVE_DB,
        )

    return ResolvedDetection(
        threshold_dB=threshold_dB,
        threshold_relative_dB=threshold_relative_dB,
        refractory_ms=pick("refractory", config.refractory_ms,
                           usable.refractory_ms if usable else None,
                           FALLBACK_REFRACTORY_MS),
        pre_ms=pick("pre", config.pre_shot_ms,
                    usable.pre_ms if usable else None, FALLBACK_PRE_SHOT_MS),
        post_ms=pick("post", config.post_shot_ms,
                     usable.post_ms if usable else None, FALLBACK_POST_SHOT_MS),
        # Not tuned: this one is a policy about what counts as impulsive, and
        # nothing in the recording can decide it. It is resolved here anyway so
        # the value the run used is reported beside the ones that were measured.
        min_snr_dB=pick("min_snr", config.min_snr_dB, None, FALLBACK_MIN_SNR_DB),
        source=source,
        tuning=tuning,
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
    resolved: ResolvedDetection,
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
        "threshold_relative_dB": resolved.threshold_relative_dB,
        "detection_threshold_dB_used": round(detection.threshold_dB, 2),
        "detection_threshold_mode": detection.threshold_mode,
        "refractory_ms": resolved.refractory_ms,
        "pre_shot_ms": resolved.pre_ms,
        "post_shot_ms": resolved.post_ms,
        "auto_detect": config.auto_detect,
        "expected_shots": config.expected_shots,
        "min_snr_dB": resolved.min_snr_dB,
        "detection_source": dict(resolved.source),
        "detection_tuning": resolved.tuning.to_dict() if resolved.tuning else None,
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
        "plot_span": config.plot_span,
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

    tuning: Optional[DetectionTuning] = None
    if config.auto_detect:
        progress(18, "Measuring detection settings")
        tuning = autotune_detection(
            pressure, sample_rate,
            expected_shots=config.expected_shots,
            min_snr_dB=resolve_min_snr(config),
        )
        logger.info("%s", tuning.summary().strip())
    resolved = resolve_detection(config, tuning)

    progress(20, "Detecting shots")
    reports: List[DetectionReport] = []
    shots = detect_shots(
        pressure, sample_rate,
        threshold_dB=resolved.threshold_dB,
        threshold_relative_dB=resolved.threshold_relative_dB,
        pre_ms=resolved.pre_ms,
        post_ms=resolved.post_ms,
        refractory_ms=resolved.refractory_ms,
        min_snr_dB=resolved.min_snr_dB,
        min_shots=config.min_shots,
        max_shots=config.max_shots,
        full_scale_dB=calibration.full_scale_dB if calibration.calibrated else None,
        samples_FS=samples_FS,          # so individual shots carry a clipped flag
        ceiling_FS=quality.ceiling.ceiling_FS if quality.ceiling.detected else None,
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
        "resolved_detection": resolved,
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
    dtype = config.load_dtype

    def read(start: int, n_frames: int) -> np.ndarray:
        return read_samples(
            wav_path, start, n_frames,
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
    # The envelope of the WHOLE recording, stitched from the chunks as they are
    # read. Tuning has to see the whole string: a threshold chosen from one
    # chunk is exactly the per-chunk threshold this analyser exists to avoid.
    tune_envelope: List[np.ndarray] = []
    tune_indices: List[np.ndarray] = []

    from calibration import CeilingClippingScan, detect_clipping  # noqa: PLC0415

    ceiling_scan = CeilingClippingScan()
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
        ceiling_scan.feed(block)
        dc_sum += float(np.sum(block, dtype=np.float64))
        sq_sum += float(np.sum(np.asarray(block, dtype=np.float64) ** 2))
        n_total += block.size

        pressure_block = calibration.to_pascals(block)
        envelope, frames = compute_envelope(
            bandpass_for_detection(pressure_block, sample_rate), env_window, env_hop
        )
        if envelope.size:
            peak_envelope = max(peak_envelope, float(envelope.max()))
            noise_estimates.append(float(np.percentile(envelope, 10.0)))
            if config.auto_detect:
                tune_envelope.append(envelope.astype(np.float32))
                tune_indices.append((frames + core_start).astype(np.int64))

        # Assess a representative slice for the qualitative checks that need spectra
        if len(quality_chunks) < 8:
            quality_chunks.append(assess_signal_quality(block, sample_rate, calibration))
        del block, pressure_block, envelope, frames
        gc.collect()

    tuning: Optional[DetectionTuning] = None
    if config.auto_detect and tune_envelope:
        progress(16, "Measuring detection settings")
        tuning = autotune_from_envelope(
            np.concatenate(tune_envelope).astype(np.float64),
            np.concatenate(tune_indices),
            sample_rate, env_hop,
            expected_shots=config.expected_shots,
            min_snr_dB=resolve_min_snr(config),
        )
        logger.info("%s", tuning.summary().strip())
    tune_envelope.clear()
    tune_indices.clear()
    gc.collect()
    resolved = resolve_detection(config, tuning)

    context_frames = max(
        int(CHUNK_CONTEXT_S * sample_rate),
        int(3.0 * (resolved.pre_ms + resolved.post_ms) * sample_rate / 1000.0),
    )

    quality = _merge_quality(
        quality_chunks, calibration,
        n_samples=n_total, sample_rate=sample_rate, peak_FS=peak_FS,
        clipped_samples=clipped_samples, clipped_runs=clipped_runs,
        dc=dc_sum / max(n_total, 1), rms=math.sqrt(sq_sum / max(n_total, 1)),
        ceiling=ceiling_scan.result(),
    )

    peak_dB = float(amplitude_to_dB_SPL(max(peak_envelope, 1e-12)))
    if resolved.threshold_dB is not None:
        absolute_threshold = float(resolved.threshold_dB)
        mode = "absolute"
    else:
        absolute_threshold = peak_dB - float(resolved.threshold_relative_dB or 0.0)
        mode = "relative (resolved globally over all chunks)"
    logger.info("Chunked detection threshold: %.1f dB (%s), global envelope peak %.1f dB",
                absolute_threshold, mode, peak_dB)

    # ---- Pass 2: detection on overlapping chunks, keeping each chunk's core ----
    pre_samples = int(resolved.pre_ms * sample_rate / 1000.0)
    post_samples = int(resolved.post_ms * sample_rate / 1000.0)
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
            pre_ms=resolved.pre_ms,
            post_ms=resolved.post_ms,
            refractory_ms=resolved.refractory_ms,
            min_snr_dB=resolved.min_snr_dB,
            max_shots=config.max_shots,
            full_scale_dB=calibration.full_scale_dB if calibration.calibrated else None,
            samples_FS=block,
            ceiling_FS=quality.ceiling.ceiling_FS if quality.ceiling.detected else None,
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
    refractory_samples = max(1, int(resolved.refractory_ms * sample_rate / 1000.0))
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
        f"Chunked analysis: {count(n_chunks, 'chunk')} of {CHUNK_DURATION_S:.0f} s with "
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
        "resolved_detection": resolved,
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
    ceiling: Optional[CeilingClipping] = None,
) -> SignalQuality:
    """
    Combine per-chunk quality assessments into one whole-recording verdict.

    Clipping, peak and DC are computed exactly across the whole file; the
    spectral checks are taken from the sampled chunks. Errors and warnings are
    unioned, so a single clipped chunk still invalidates the measurement.

    ``ceiling`` is the whole-file limiter scan. It cannot be merged from the
    chunks: a chunk's own maximum is not a ceiling, so the scan has to be run
    across the file with one global extreme.
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

    ceiling = ceiling or CeilingClipping()
    if clipped_runs > 0:
        errors.append(
            f"Recording is CLIPPED ({clipped_samples} samples in {clipped_runs} runs). "
            f"Peak levels are understated and rise time, crest factor and kurtosis are invalid. "
            f"Re-record with lower input gain."
        )
    elif ceiling.detected:
        errors.append(ceiling_clipping_error(ceiling))

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
        ceiling=ceiling,
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
        def reader(start: int, n_frames: int) -> np.ndarray:
            start = max(0, int(start))
            return preloaded[start:start + max(0, int(n_frames))]
        return reader

    def reader(start: int, n_frames: int) -> np.ndarray:
        return read_samples(
            wav_path, start, n_frames,
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

    # ---- The span the full-recording figures cover ----
    #
    # A shot is a few hundred milliseconds. Drawn against a recording that is
    # mostly the walk to the line, it is a hairline. So these figures default to
    # the shot string, and their titles say which span they are of.
    focus = _plot_focus(config, shots, total_frames, sample_rate)
    view_start, view_stop = (focus.start, focus.end) if focus.applied else (0, total_frames)
    # The record is written after this function returns, so the interface can
    # read which span its charts cover instead of assuming the whole file.
    if isinstance(record.get("settings"), dict):
        record["settings"]["plot_focus"] = focus.to_dict()
    if focus.applied:
        say(f"    figures cover {focus.start_s:.2f}-{focus.end_s:.2f} s, the shot string "
            f"({focus.removed_s:.1f} s outside it is not drawn)")
        artifacts_span = f" ({focus.start_s:.2f}\u2013{focus.end_s:.2f} s)"
    else:
        artifacts_span = ""

    # ---- Full-recording waveform ----
    with _plot_step("Full waveform", run_warnings):
        time_axis, pressure_plot = _waveform_for_display(
            reader, calibration, sample_rate, total_frames, shots,
            frame_start=view_start, frame_stop=view_stop,
        )
        if time_axis.size:
            saved = False
            if plotly_available and want_html:
                path = out_dir / "waveform_full.html"
                if plot_module.save_interactive_waveform_html(
                    path, time_axis, pressure_plot, shots=shots,
                    title=f"Pressure Waveform: {wav_path.name}{artifacts_span}",
                ):
                    artifacts["waveform_html"] = path.name
                    saved = True
            if static_formats:
                figure, _ = plot_module.plot_waveform_pa(
                    time_axis, pressure_plot, shots=shots,
                    title=f"Pressure Waveform: {wav_path.name}{artifacts_span}",
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
                frame_start=view_start, frame_stop=view_stop,
            )
            if stft is None:
                raise RuntimeError("the recording is shorter than one FFT window")

            # Emitted BEFORE the figure branches, deliberately. _plot_step
            # swallows an exception from anything below it, so a matplotlib
            # failure must not be able to take the data with it.
            matrix = _spectrogram_matrix_block(stft)
            if matrix:
                matrix_path = out_dir / f"{key}_matrix.json"
                try:
                    write_json(matrix_path, matrix)
                    artifacts[f"{key}_matrix"] = matrix_path.name
                except OSError as exc:
                    logger.warning("The %s matrix could not be written: %s", key, exc)

            if plotly_available and want_html:
                path = out_dir / f"{key}_full.html"
                if plot_module.save_interactive_spectrogram_html(
                    path, stft, shots=shots,
                    title=f"{weighting}-Weighted Spectrogram: {wav_path.name}{artifacts_span}",
                ):
                    artifacts[f"{key}_html"] = path.name
            if static_formats:
                figure, _ = plot_module.plot_spectrogram_dB(
                    stft, shots=shots,
                    title=f"{weighting}-Weighted Spectrogram: {wav_path.name}{artifacts_span}",
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
    # No longer gated on static_formats: the heatmap is a live chart in the
    # interface now, so the data has to be written even when nobody asked for a
    # picture. The figure below it remains optional.
    if config.compute_bands:
        with _plot_step("1/3-octave band heatmap", run_warnings):
            times, freqs, levels = _bands_for_display(
                reader, config, calibration, sample_rate, total_frames, chunked,
                frame_start=view_start, frame_stop=view_stop,
            )
            if times.size:
                # Emitted BEFORE the figure branch: a matplotlib failure below
                # must not be able to take the data with it.
                matrix = _band_matrix_block(
                    times, freqs, levels,
                    level_unit=level_unit,
                    time_weighting=config.band_time_weighting,
                    hop_ms=config.band_hop_ms,
                )
                if matrix:
                    matrix_path = out_dir / "bands_matrix.json"
                    try:
                        write_json(matrix_path, matrix)
                        artifacts["bands_matrix"] = matrix_path.name
                    except OSError as exc:
                        logger.warning("The band matrix could not be written: %s", exc)

                if static_formats:
                    figure, _ = plot_module.plot_third_octave_heatmap(
                        times, freqs, levels, shots=shots,
                        title=f"1/3-Octave Band Levels: {wav_path.name}{artifacts_span}",
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
    # As above, the data is no longer conditional on wanting a picture. Each
    # shot's two spectrograms are written as their own file so the interface
    # fetches only the shot being looked at, however long the string is.
    if config.save_per_shot_plots and shots:
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

                for stft, suffix in ((stft_z, "z"), (stft_c, "c")):
                    matrix = _spectrogram_matrix_block(stft)
                    if not matrix:
                        continue
                    matrix["time_offset_s"] = round(shot.window_start / sample_rate, 6)
                    name = f"shot_{shot.shot_number:02d}_spectrogram_{suffix}.json"
                    try:
                        write_json(shot_dir / name, matrix)
                        artifacts[f"shot_{shot.shot_number:02d}_spectrogram_{suffix}"] = (
                            f"shots/{name}"
                        )
                    except OSError as exc:
                        logger.warning("The shot %s spectrogram could not be written: %s",
                                       shot.shot_number, exc)

                if static_formats:
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

        # The string as a population rather than an average: how the levels are
        # distributed, and whether the first round left the group.
        valid_peaks = [
            float(m.Lpeak_Z) for m in shot_metrics
            if m.valid and math.isfinite(m.Lpeak_Z)
        ]
        if len(valid_peaks) > 1:
            with _plot_step("Level distribution figure", run_warnings):
                save_optional(
                    call("plot_level_distribution", valid_peaks,
                         metric_label="Peak level, Z-weighted",
                         label="This string", level_unit=level_unit),
                    "level_distribution", "distribution",
                )

        pop_block = ((record.get("string_statistics") or {}).get("Lpeak_Z") or {}).get(
            "first_round_pop"
        )
        if pop_block and not pop_block.get("refusal"):
            with _plot_step("First-round pop figure", run_warnings):
                ordered_peaks = [
                    float(m.Lpeak_Z) for m in shot_metrics
                    if m.valid and math.isfinite(m.Lpeak_Z)
                ]
                save_optional(
                    call("plot_first_round_pop", ordered_peaks,
                         pop=SimpleNamespace(**pop_block),
                         metric_label="Peak level, Z-weighted",
                         level_unit=level_unit),
                    "first_round_pop", "first_round_pop",
                )

        # What the air took out of the signal on its way to the microphone.
        effect_block = record.get("atmospheric_effect") or {}
        if effect_block.get("absorption_dB"):
            with _plot_step("Atmospheric absorption figure", run_warnings):
                save_optional(
                    call("plot_atmospheric_absorption", effect_block,
                         level_unit=level_unit),
                    "atmospheric_absorption", "atmosphere",
                )

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


def _view_bounds(
    frame_start: int, frame_stop: Optional[int], total_frames: int
) -> Tuple[int, int]:
    """Clamp a requested display span to the recording, never returning an empty one."""
    start = max(0, min(int(frame_start), max(0, total_frames - 1)))
    stop = total_frames if frame_stop is None else min(int(frame_stop), total_frames)
    if stop <= start:
        return 0, total_frames
    return start, stop


def _plot_focus(
    config: AnalysisConfig,
    shots: Sequence[ShotEvent],
    total_frames: int,
    sample_rate: int,
) -> TrimSpan:
    """
    The part of the recording the full-recording figures should show.

    A range recording is mostly not shooting. Drawn end to end, five shots in a
    ten-minute file are five hairlines against a flat rule, and the spectrogram
    that should show a muzzle blast's decay shows nine minutes of wind. So the
    figures default to the span that holds the string - which is what the
    figures are of - and say so in their titles. Every metric is still computed
    from the whole recording; this is the picture only.

    Returns an unapplied span when there is nothing to focus on, which
    TrimSpan.reason explains.
    """
    if config.plot_span == "full":
        return TrimSpan(
            start=0, end=int(total_frames), sample_rate=int(sample_rate),
            n_shots=len(shots), original_samples=int(total_frames),
            reason="the whole recording was requested",
        )
    return find_shot_string_span(
        shots, total_frames, sample_rate, margin_s=PLOT_FOCUS_MARGIN_S
    )


def _waveform_for_display(
    reader,
    calibration: Calibration,
    sample_rate: int,
    total_frames: int,
    shots: Sequence[ShotEvent],
    *,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a display waveform: full resolution around every shot, thinned elsewhere.

    frame_start/frame_stop restrict the picture to part of the recording; the
    time axis stays absolute, so a focused figure still reads in the
    recording's own clock. This affects the picture only. Every metric is
    computed from every sample.
    """
    view_start, view_stop = _view_bounds(frame_start, frame_stop, total_frames)
    step = max(1, (view_stop - view_start) // MAX_WAVEFORM_POINTS)
    pre = int(WAVEFORM_FULLRES_PRE_S * sample_rate)
    post = int(WAVEFORM_FULLRES_POST_S * sample_rate)

    regions: List[Tuple[int, int]] = []
    for shot in shots:
        start = max(view_start, shot.index - pre)
        stop = min(view_stop, shot.index + post)
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

    cursor = view_start
    for start, stop in merged:
        thinned(cursor, start)
        full(start, stop)
        cursor = stop
    thinned(cursor, view_stop)

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
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> Optional[STFTResult]:
    """
    Compute a full-recording spectrogram, chunk by chunk when the file is long.

    The result is thinned to MAX_SPECTROGRAM_FRAMES for display; every metric is
    computed elsewhere, from the full-rate signal.
    """
    view_start, view_stop = _view_bounds(frame_start, frame_stop, total_frames)
    view_frames = view_stop - view_start

    if not chunked:
        block = reader(view_start, view_frames)
        if block.size < nperseg:
            return None
        stft = analyze_stft(
            calibration.to_pascals(block), sample_rate,
            nperseg=nperseg, noverlap=noverlap, window=config.stft_window,
            weighting=weighting, calibrated=calibration.calibrated,
        )
        stft.time_s = stft.time_s + view_start / sample_rate
        return _thin_spectrogram(stft)

    chunk_frames = max(nperseg * 4, int(CHUNK_DURATION_S * sample_rate))
    times: List[np.ndarray] = []
    magnitudes: List[np.ndarray] = []
    freqs = np.array([])
    template: Optional[STFTResult] = None

    for _rs, _re, core_start, core_stop in _chunk_plan(view_frames, chunk_frames, 0):
        core_start += view_start
        core_stop += view_start
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


MATRIX_QUANTISATION_STEP_dB = 0.1
MATRIX_MISSING_COUNT = 65535


def _quantised_matrix(matrix: np.ndarray) -> Dict[str, Any]:
    """
    A 2-D decibel array as data the interface can read a level off, not a picture.

    Stored as unsigned 16-bit counts of 0.1 dB. For a spectrogram that is
    LOSSLESS: _thin_spectrogram has already rounded the matrix to 0.1 dB for
    display, so this reproduces the array Python plotted exactly. For an array
    that has not been rounded -- the band-level history -- the rounding happens
    here and is therefore visible in the output: what is stored is what the
    readout shows, to a tenth of a decibel, and the step is declared in the
    block so nothing downstream has to assume it.

    Base64 of big-endian uint16 rather than a PNG. A PNG is smaller, but it
    carries the measurement through colour management: a stray gAMA or iCCP
    chunk, or a wide-gamut display, shifts the bytes a canvas reads back while
    leaving the picture looking correct. A readout that is quietly wrong is the
    one failure this application exists to avoid. JSON numbers would be honest
    too and about five times larger.

    65535 is reserved for a non-finite cell so silence and NaN stay
    distinguishable from a real level.

    Bin-major (rows are frequency, columns are time) so one column of the
    display is a stride, and the browser can slice a frequency row without
    walking the whole array.
    """
    values = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(values)
    if values.ndim != 2 or not finite.any():
        return {}

    step = MATRIX_QUANTISATION_STEP_dB
    offset = float(np.floor(values[finite].min() * 10.0) / 10.0)
    counts = np.full(values.shape, MATRIX_MISSING_COUNT, dtype=np.uint16)
    scaled = np.round((values[finite] - offset) / step)
    counts[finite] = np.clip(scaled, 0, MATRIX_MISSING_COUNT - 1).astype(np.uint16)

    return {
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "quantisation": {
            "offset_dB": offset,
            "step_dB": step,
            "dtype": "uint16",
            "byte_order": "big",
            "missing": MATRIX_MISSING_COUNT,
        },
        "magnitude_dB_b64": base64.b64encode(counts.astype(">u2").tobytes()).decode("ascii"),
        "min_dB": round(float(values[finite].min()), 3),
        "max_dB": round(float(values[finite].max()), 3),
    }


def _spectrogram_matrix_block(stft: STFTResult) -> Dict[str, Any]:
    """The spectrogram as readable data. See _quantised_matrix for the format."""
    magnitude = np.asarray(stft.magnitude_dB, dtype=np.float64)
    payload = _quantised_matrix(magnitude)
    if not payload:
        return {}

    return {
        "schema": 1,
        "kind": "spectrogram",
        "weighting": stft.weighting,
        "level_label": stft.level_label,
        "calibrated": bool(stft.calibrated),
        "sample_rate_Hz": float(stft.sample_rate),
        "nperseg": int(stft.nperseg),
        "noverlap": int(stft.noverlap),
        "window": stft.window,
        "enbw_Hz": round(float(stft.enbw_Hz), 4),
        "frames": int(magnitude.shape[1]),
        "bins": int(magnitude.shape[0]),
        "time_s": [round(float(t), 6) for t in stft.time_s],
        "frequencies_Hz": [round(float(f), 3) for f in stft.frequencies_Hz],
        **payload,
    }


def _band_matrix_block(
    times: np.ndarray,
    frequencies: np.ndarray,
    levels: np.ndarray,
    *,
    level_unit: str,
    time_weighting: str,
    hop_ms: float,
) -> Dict[str, Any]:
    """
    The one-third-octave band history as readable data.

    This is the same picture as bands_full.png -- band level against time --
    and it is the last figure in the results that existed only as an image. A
    heatmap is exactly the plot an operator wants to interrogate: the question
    is always WHICH band was loud and WHEN, and neither is answerable by
    looking.
    """
    times = np.asarray(times, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if times.size == 0 or frequencies.size == 0 or levels.ndim != 2:
        return {}
    if levels.shape != (frequencies.size, times.size):
        logger.warning(
            "The band history is %s, expected (%d, %d); the heatmap block was not written",
            levels.shape, frequencies.size, times.size,
        )
        return {}

    payload = _quantised_matrix(levels)
    if not payload:
        return {}

    return {
        "schema": 1,
        "kind": "bands",
        "level_unit": level_unit,
        "time_weighting": time_weighting,
        "hop_ms": round(float(hop_ms), 4),
        "frames": int(times.size),
        "bins": int(frequencies.size),
        "time_s": [round(float(t), 6) for t in times],
        "frequencies_Hz": [round(float(f), 3) for f in frequencies],
        **payload,
    }


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
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
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

    view_start, view_stop = _view_bounds(frame_start, frame_stop, total_frames)
    view_frames = view_stop - view_start

    if not chunked:
        block = reader(view_start, view_frames)
        if block.size == 0:
            return np.array([]), analyzer.center_frequencies, np.array([[]])
        result = analyze(block)
        return (result["time_s"] + view_start / sample_rate,
                result["center_frequencies"], result["band_levels_dB"])

    context_frames = int(1.0 * sample_rate)
    chunk_frames = max(1, int(CHUNK_DURATION_S * sample_rate))
    times: List[np.ndarray] = []
    levels: List[np.ndarray] = []

    for read_start, read_stop, core_start, core_stop in _chunk_plan(
        view_frames, chunk_frames, context_frames
    ):
        core_start += view_start
        core_stop += view_start
        read_start = max(view_start, core_start - context_frames)   # leading context only
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

    det = parser.add_argument_group(
        "Shot detection (measured from the recording unless you say otherwise)"
    )
    det.add_argument("--no-auto-detect", dest="auto_detect", action="store_false",
                     help="Do not measure the detection settings; use the values below "
                          "or their defaults.")
    det.set_defaults(auto_detect=True)
    det.add_argument("--expected-shots", type=int, default=None, metavar="N",
                     help="Rounds you know were fired. Preferred when a threshold "
                          "produces it; never forced when none does.")
    det.add_argument("--threshold-dB", type=float, default=None, metavar="dB",
                     help="Absolute detection threshold; requires a real calibration.")
    det.add_argument("--threshold-relative-dB", type=float, default=None, metavar="dB",
                     help="Detection threshold this many dB below the loudest event. "
                          "Overrides the measured value.")
    det.add_argument("--refractory-ms", type=float, default=None, metavar="ms",
                     help="Minimum spacing between shots. Overrides the measured value.")
    det.add_argument("--pre-ms", type=float, default=None, metavar="ms",
                     help="Window before each peak. Overrides the measured value.")
    det.add_argument("--post-ms", type=float, default=None, metavar="ms",
                     help="Window after each peak. Overrides the measured value.")
    det.add_argument("--min-snr-dB", type=float, default=None, metavar="dB",
                     help="How far above the noise floor a candidate must be to count as "
                          "a shot. Default: 15. Raising it also narrows the range the "
                          "automatic tuner searches.")
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
    out.add_argument("--session", type=Path, default=None, metavar="DIR",
                     help="Analyse every recording in DIR as one range session, "
                          "then pair each suppressed string with its unsuppressed "
                          "reference and report the session as a whole.")
    out.add_argument("--config", type=Path, default=None, metavar="FILE",
                     help="Load settings from JSON. Explicit flags override the file.")
    out.add_argument("--formats", type=str, default="png", metavar="LIST",
                     help="Plot formats, comma-separated (png, pdf, svg, html...). Default: png.")
    out.add_argument("--no-plots", action="store_true",
                     help="Write data artifacts only; draw nothing.")
    out.add_argument("--plot-span", type=str, default="shots", choices=["shots", "full"],
                     help="Span the full-recording figures cover: the shot string, or "
                          "the whole file. Metrics are unaffected. Default: shots.")
    out.add_argument("--reference", type=Path, default=None, metavar="DIR",
                     help="Previous UNSUPPRESSED analysis directory; computes insertion loss.")
    out.add_argument("--verbose", "-v", action="store_true", help="Debug-level diagnostics.")
    out.add_argument("--quiet", "-q", action="store_true", help="Warnings and errors only.")
    out.add_argument("--probe", action="store_true",
                     help="Describe the input file as JSON and exit. Reads headers only: "
                          "no decoding, no extraction, no analysis.")
    out.add_argument("--detect-only", action="store_true",
                     help="Detect shots and print what was found as JSON, then exit. No "
                          "metrics, no figures, no output directory, and no calibration "
                          "needed - every level is dB re FS. Use it to see what a "
                          "detection setting does before committing to a run.")

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
    # Detection knobs are "measure it" until named, so they are applied only
    # when actually typed - `always` would write None over a config file's value.
    apply("--refractory-ms", "refractory_ms", args.refractory_ms,
          always=always and args.refractory_ms is not None)
    apply("--pre-ms", "pre_shot_ms", args.pre_ms, always=always and args.pre_ms is not None)
    apply("--post-ms", "post_shot_ms", args.post_ms, always=always and args.post_ms is not None)
    apply("--no-auto-detect", "auto_detect", bool(args.auto_detect),
          always=always and not args.auto_detect)
    apply("--expected-shots", "expected_shots", args.expected_shots,
          always=always and args.expected_shots is not None)
    apply("--min-snr-dB", "min_snr_dB", args.min_snr_dB,
          always=always and args.min_snr_dB is not None)
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
    apply("--plot-span", "plot_span", args.plot_span, always=always)
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
                f"Metadata file {path.name} contains unrecognised {plural(len(unknown), 'key')}: "
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

    # Answered before anything else is resolved: a probe needs no calibration,
    # no profiles and no output directory, and must not be able to fail
    # because one of them is missing.
    if args.probe:
        if args.input is None:
            print(json.dumps({"readable": False, "problem": "No file was given to probe."}))
            return EXIT_ERROR
        result = probe_input(Path(args.input))
        print(json.dumps(result))
        return EXIT_OK if result.get("readable") else EXIT_ERROR

    # Detection only, before any calibration or output directory is resolved:
    # the operator is looking at their recording, not measuring it yet.
    if args.detect_only:
        if args.input is None:
            print(json.dumps({"readable": False, "problem": "No file was given to detect in."}))
            return EXIT_ERROR
        preview = detect_preview(
            Path(args.input),
            channel=args.channel,
            auto_detect=args.auto_detect,
            expected_shots=args.expected_shots,
            threshold_relative_dB=args.threshold_relative_dB,
            refractory_ms=args.refractory_ms,
            pre_ms=args.pre_ms,
            post_ms=args.post_ms,
            min_snr_dB=args.min_snr_dB,
        )
        print(json.dumps(preview))
        return EXIT_OK if preview.get("readable") else EXIT_ERROR

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
        # A session names a directory rather than a file, so the single-file
        # resolution (and its interactive picker) is skipped entirely.
        if args.session is None:
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

        if args.session is not None:
            return _run_session_cli(
                args.session, config, calibration, warnings, output_base=args.output
            )

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
        say(f"  {label}: {count(result.n_shots, 'shot')}, {state}")
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


def _run_session_cli(
    directory: Path,
    config: AnalysisConfig,
    calibration: Calibration,
    warnings: List[str],
    output_base: Optional[Path] = None,
) -> int:
    """
    Analyse a whole range session and report it as one.

    Each recording carries its own metadata sidecar if it has one; the session
    is then paired and summarised by session.py, which refuses to guess when two
    references match a test equally well.
    """
    from session import discover_recordings, run_session, SessionError

    try:
        paths = discover_recordings(directory)
    except SessionError as exc:
        raise ConfigurationError(str(exc)) from exc

    if not paths:
        raise ConfigurationError(
            f"No recordings found in {directory}. A session directory must hold at "
            f"least one audio or video file."
        )

    say("")
    say(f"  Session: {count(len(paths), 'recording')} in {directory}")
    for path in paths:
        say(f"    {path.name}")
    say("")

    # One directory for the whole session, with each recording's run beneath it,
    # so a session is a single artefact that can be archived or handed over.
    base = Path(output_base) if output_base else Path(directory)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = base / f"session_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    def analyse_one(path: Path) -> Dict[str, Any]:
        """Run one recording, reusing the session's calibration and settings."""
        prepared = prepare_input(path, warnings)
        metadata = _metadata_from_sidecar(path) or TestMetadata()
        result = analyze_file(
            prepared.audio_path,
            config,
            output_base=session_dir,
            metadata=metadata,
            calibration=calibration,
            original_path=prepared.original_path,
            warnings=warnings,
        )
        return result.record

    result = run_session(
        paths,
        analyse=analyse_one,
        progress=lambda pct, message: progress(pct, message),
    )

    say("")
    say("=" * 64)
    for line in result.summary().splitlines():
        say(line)
    say("=" * 64)

    record_path = session_dir / "session.json"
    write_json(record_path, result.to_dict())
    say(f"  Session record: {record_path}")
    announce_output_dir(session_dir)

    if result.failed:
        return EXIT_ERROR
    return EXIT_OK


def _metadata_from_sidecar(path: Path) -> Optional[TestMetadata]:
    """
    Read a per-recording metadata sidecar, if one sits beside the audio.

    A session is many recordings with DIFFERENT metadata - that is the whole
    point, since one of them is the unsuppressed reference. So each file may
    carry its own <name>.metadata.json, and a missing one is not an error.
    """
    for candidate in (
        path.with_suffix(".metadata.json"),
        path.with_name(path.stem + ".metadata.json"),
        path.with_name(path.stem + ".json"),
    ):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return TestMetadata.from_dict(data)
    return None


def cli_main() -> None:
    """Console-script entry point (pyproject declares main:cli_main)."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
