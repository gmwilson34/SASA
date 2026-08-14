#!/usr/bin/env python3
"""
report.py - SASA Measurement Report

Turns an analysis record (analysis_metadata.json, schema 2.0) into ONE
self-contained HTML file that can be handed to a customer, archived, or printed to
PDF from any browser. Everything is inline: no stylesheets, no scripts, no fonts,
no remote images. The file works with the network cable pulled out, which is the
only way a measurement record can be trusted years after it was written.

The document is ordered the way a reader needs it, not the way the software
computes it:

  1. Title block - what this is, who measured it, when, and with which build
  2. MEASUREMENT VALIDITY - first, in plain words, before any number is quoted
  3. Test conditions - every required field, missing ones shown as "not recorded"
  4. Headline results - mean with 95% CI, n, std, min, max, median
  5. Insertion loss vs the unsuppressed reference - the suppressor deliverable
  6. Hearing hazard - LAeq8h, dose, allowable rounds, method named
  7. Per-shot table with validity flags
  8. Figures
  9. Methods and limitations

Two rules are absolute:

  * An uncalibrated measurement is NEVER presented as dB SPL. Relative levels are
    labelled "dB re FS" everywhere they appear, including axis captions, and the
    document says on its first page that they are not sound pressure levels.
  * A clipped measurement is NEVER presented as valid. Saturation censors exactly
    the peak the report exists to state.

Usage:
    python report.py <analysis_dir> [-o report.html]

    from report import generate_report
    generate_report(metadata_dict, "report.html", figures=["waveform_full.png"])
"""

from __future__ import annotations

import base64
import html
import json
import math
import mimetypes
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from textutil import count, join_list, plural

__all__ = [
    "generate_report",
    "build_report_from_directory",
    "assess_validity",
    "ValidityCheck",
    "ValidityReport",
]

REPORT_SCHEMA_SUPPORTED = ("2.0",)

# ═══════════════════════════════════════════════════════════════════════════
#  Palette
#
#  Transcribed from ui/renderer/tokens.css - that file is the source of truth
#  for the "Ridgeback Instrument" design system, and these hex values are its
#  LIGHT theme (:root / :root[data-theme="light"]). The report is deliberately
#  light-only: it is printed and shared as a document, and a dark-theme PDF is
#  unreadable on paper. If tokens.css changes, change these to match.
# ═══════════════════════════════════════════════════════════════════════════

C = {
    # surfaces
    "bg_canvas":     "#F4F6F9",
    "bg_surface":    "#FFFFFF",
    "bg_sunken":     "#EAEEF3",
    "bg_inset":      "#F7F9FB",
    # lines
    "border":        "#C8D1DE",
    "border_subtle": "#E1E7EF",
    "border_input":  "#7C8798",
    # text
    "text":          "#0F141A",
    "text_2":        "#48566A",
    "text_3":        "#556274",
    "text_on_accent": "#FFFFFF",
    # accent
    "accent":        "#14508C",
    "accent_wash":   "#E8F0F9",
    "accent_border": "#A9C4E2",
    # semantic - reserved for measurement validity, never decoration
    "ok":            "#1B6E3C", "ok_wash":     "#E4F3EA", "ok_border":     "#A5D6BA",
    "warn":          "#8A5A00", "warn_wash":   "#FBF0DC", "warn_border":   "#E4C489",
    "danger":        "#A32020", "danger_wash": "#FBE9E9", "danger_border": "#E8AFAF",
    "info":          "#0B5F73", "info_wash":   "#E2F1F5", "info_border":   "#9CCBD7",
    # data series
    "series_1":      "#14508C",   # Z-weighted
    "series_2":      "#B45309",   # A-weighted
    "series_3":      "#6D28D9",   # C-weighted
    "series_5":      "#9D174D",   # reference (unsuppressed)
    "series_6":      "#4D7C0F",   # test (suppressed)
}

FONT_SANS = ('ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", '
             '"Helvetica Neue", Arial, sans-serif')
FONT_MONO = ('ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Mono", '
             '"Roboto Mono", Menlo, Consolas, monospace')


# ═══════════════════════════════════════════════════════════════════════════
#  Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dig(data: Any, *path: str, default: Any = None) -> Any:
    """Fetch a nested key, tolerating absent or wrongly-typed intermediates."""
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def _map(data: Any, *path: str) -> Dict[str, Any]:
    """
    A nested block that is expected to be a mapping.

    A record written by another version - or hand-edited - may hold a string or a
    list where a block belongs. The report must still render, stating what is
    missing, rather than failing on a customer's machine with an AttributeError.
    """
    node = _dig(data, *path, default=None)
    return node if isinstance(node, dict) else {}


def _seq(data: Any, *path: str) -> List[Any]:
    """A nested block that is expected to be a sequence."""
    node = _dig(data, *path, default=None)
    if isinstance(node, (list, tuple)):
        return list(node)
    return []


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _num(value: Any, digits: int = 1, *, dash: str = "—") -> str:
    """Format a number, or a dash when it is absent or not finite."""
    if not _is_num(value):
        return dash
    return f"{float(value):.{digits}f}"


def _txt(value: Any, *, missing: str = "not recorded") -> str:
    """Text for a metadata field, stating explicitly when nothing was recorded."""
    if value is None:
        return missing
    if isinstance(value, str):
        return value.strip() or missing
    if isinstance(value, bool):
        return "yes" if value else "no"
    if _is_num(value):
        trimmed = f"{float(value):g}"
        return trimmed
    return str(value)


def _e(value: Any) -> str:
    """HTML-escape."""
    return html.escape(str(value), quote=True)


def _missing_span(text: str) -> str:
    """Render a not-recorded value visibly, so an absence cannot read as a value."""
    if text in ("not recorded", "—"):
        return f'<span class="missing">{_e(text)}</span>'
    return _e(text)


# ═══════════════════════════════════════════════════════════════════════════
#  Measurement validity
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidityCheck:
    """One validity question, its answer, and what the answer means."""
    label: str
    status: str            # "ok" | "warn" | "fail"
    finding: str           # the measured fact
    consequence: str = ""  # what it means for the numbers in this report


@dataclass
class ValidityReport:
    """The admissibility of the whole measurement."""
    verdict: str                    # "valid" | "qualified" | "invalid"
    headline: str
    statement: str
    calibrated: bool
    level_unit: str
    checks: List[ValidityCheck] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)

    @property
    def is_admissible(self) -> bool:
        return self.verdict != "invalid"


def assess_validity(metadata: Dict[str, Any]) -> ValidityReport:
    """
    Decide whether this measurement may be reported, and say why in plain words.

    Nothing here is advisory decoration. Each check either clears the measurement,
    qualifies it, or blocks it, and a blocked measurement is announced at the top of
    the document rather than being footnoted under the numbers.
    """
    calibration = _map(metadata, "calibration")
    quality = _map(metadata, "quality")
    detection = _map(metadata, "detection")
    test_meta = _map(metadata, "test_metadata")
    shots = _seq(metadata, "per_shot_metrics")
    source = _map(metadata, "source")

    calibrated = bool(calibration.get("calibrated"))
    level_unit = calibration.get("level_unit") or ("dB SPL" if calibrated else "dB re FS")

    checks: List[ValidityCheck] = []
    blocking: List[str] = []

    # ── 1. Calibration ────────────────────────────────────────────────
    method = str(calibration.get("method") or "unknown")
    if calibrated:
        pa = calibration.get("Pa_per_FS")
        detail = f"{method}, {_num(pa, 4)} Pa per full scale"
        if _is_num(calibration.get("reference_level_dB")):
            detail += f", referenced to a {_num(calibration['reference_level_dB'])} dB calibrator"
        checks.append(ValidityCheck(
            "Calibration", "ok",
            f"Calibrated ({detail}).",
            f"Levels are absolute sound pressure levels in {level_unit}.",
        ))
    else:
        checks.append(ValidityCheck(
            "Calibration", "fail",
            "UNCALIBRATED - no conversion from digital full scale to Pascals was established.",
            "Every level in this report is RELATIVE (dB re FS). These are NOT sound "
            "pressure levels. They cannot be compared with any other instrument, with a "
            "regulatory limit, or with a measurement made on a different day or gain "
            "setting. Only differences measured within this same recording are meaningful.",
        ))
        blocking.append("The measurement is uncalibrated; levels are relative, not dB SPL.")

    residual = calibration.get("residual_dB")
    drift = test_meta.get("calibration_drift_dB")
    if _is_num(drift):
        if abs(float(drift)) > 0.5:
            checks.append(ValidityCheck(
                "Calibration stability", "fail",
                f"Pre/post-test calibration drifted {float(drift):+.2f} dB (limit ±0.50 dB).",
                "The measurement chain was not stable across the session, so levels "
                "recorded at the end are not directly comparable with those at the start.",
            ))
            blocking.append(f"Calibration drifted {float(drift):+.2f} dB across the session.")
        else:
            checks.append(ValidityCheck(
                "Calibration stability", "ok",
                f"Pre/post-test calibration agreed within {abs(float(drift)):.2f} dB.",
                "The chain was stable across the session.",
            ))
    elif calibrated:
        checks.append(ValidityCheck(
            "Calibration stability", "warn",
            "No post-test calibration check was recorded.",
            "Drift during the session cannot be ruled out.",
        ))
    if _is_num(residual) and abs(float(residual)) > 0.5:
        checks.append(ValidityCheck(
            "Calibrator residual", "warn",
            f"Calibrator residual {float(residual):+.2f} dB.",
            "The calibration tone did not settle to its stated level.",
        ))

    # ── 2. Clipping ───────────────────────────────────────────────────
    clipped_shots = [s for s in shots if isinstance(s, dict) and s.get("clipped")]
    q_clipped = bool(quality.get("is_clipped")) or int(quality.get("clipped_runs") or 0) > 0
    if q_clipped or clipped_shots:
        where = count(len(clipped_shots), "shot") if clipped_shots else "the recording"
        # Two causes, and they call for two different corrections: a converter
        # that ran out of range wants less gain, a limiter wants switching off.
        if quality.get("ceiling_clipped"):
            ceiling = quality.get("ceiling_dBFS")
            detail = (
                f"The waveform is flat-topped at {_txt(ceiling, missing='?')} dBFS in {where} "
                f"({_txt(quality.get('ceiling_samples'), missing='?')} samples in "
                f"{_txt(quality.get('ceiling_runs'), missing='?')} plateaus). The ceiling is "
                f"below digital full scale, so a limiter or automatic gain control was active."
            )
            remedy = (
                "A clipped peak is censored: the true peak is unknown and is at least the "
                "value shown. Peak levels, crest factor, rise time and sound exposure from "
                "the affected shots are lower bounds, not measurements. The apparent "
                "headroom is not real headroom. Re-record with limiting and AGC switched off."
            )
        else:
            detail = (
                f"The signal reached digital full scale in {where} "
                f"({_txt(quality.get('clipped_samples'), missing='?')} samples in "
                f"{_txt(quality.get('clipped_runs'), missing='?')} runs)."
            )
            remedy = (
                "A clipped peak is censored: the true peak is unknown and is at least the "
                "value shown. Peak levels, crest factor, rise time and sound exposure from "
                "the affected shots are lower bounds, not measurements. Re-record with more "
                "headroom."
            )
        checks.append(ValidityCheck("Clipping", "fail", detail, remedy))
        blocking.append("The recording is clipped; peak levels are lower bounds, not measurements.")
    else:
        checks.append(ValidityCheck(
            "Clipping", "ok", "The waveform is not flat-topped, at the rail or below it.",
            "Peak levels are limited by neither the converter nor a limiter.",
        ))

    # ── 3. Headroom ───────────────────────────────────────────────────
    headroom = quality.get("headroom_dB")
    if _is_num(headroom):
        if float(headroom) < 1.0:
            status, consequence = "fail", (
                "The peak sat within 1 dB of full scale. Treat the peak levels as "
                "unreliable even where no sample was pinned.")
            blocking.append("Less than 1 dB of headroom remained at the peak.")
        elif float(headroom) < 6.0:
            status, consequence = "warn", (
                "Little margin remained. A slightly louder round in the same string "
                "would have clipped.")
        else:
            status, consequence = "ok", "Adequate margin to full scale."
        checks.append(ValidityCheck(
            "Headroom", status, f"{_num(headroom)} dB below digital full scale.", consequence))
    else:
        checks.append(ValidityCheck("Headroom", "warn", "Headroom was not assessed.", ""))

    # ── 4. Signal-to-noise ────────────────────────────────────────────
    snr = quality.get("snr_dB")
    if _is_num(snr):
        if float(snr) < 10.0:
            status, consequence = "fail", (
                "The event is barely above the noise floor. Exposure levels are "
                "dominated by background noise rather than by the shot.")
            blocking.append("Signal-to-noise ratio below 10 dB.")
        elif float(snr) < 20.0:
            status, consequence = "warn", (
                "Sound exposure levels (LAE, LZE) may be inflated by background noise "
                "inside the integration window.")
        else:
            status, consequence = "ok", "The event stands clear of the noise floor."
        checks.append(ValidityCheck(
            "Signal-to-noise", status,
            f"{_num(snr)} dB between the peak and the noise floor "
            f"({_num(quality.get('noise_floor_dB'))} {level_unit}).",
            consequence))
    else:
        checks.append(ValidityCheck("Signal-to-noise", "warn", "SNR was not assessed.", ""))

    # ── 5. Sample rate ────────────────────────────────────────────────
    sample_rate = quality.get("sample_rate") or source.get("sample_rate")
    adequate = quality.get("sample_rate_adequate")
    nyquist = (float(sample_rate) / 2.0) if _is_num(sample_rate) else None
    unresolved_rise = [s for s in shots
                       if isinstance(s, dict) and s.get("rise_time_resolved") is False]
    if adequate is False:
        checks.append(ValidityCheck(
            "Sample rate", "fail",
            f"{_txt(sample_rate)} Hz (Nyquist {_num(nyquist, 0)} Hz) is below what a "
            f"muzzle blast requires.",
            "The leading edge of the blast is not resolved. Peak level and rise time are "
            "understated, and the A-duration cannot be determined.",
        ))
        blocking.append(f"Sample rate {_txt(sample_rate)} Hz cannot resolve the impulse.")
    elif unresolved_rise:
        checks.append(ValidityCheck(
            "Sample rate", "warn",
            f"{_txt(sample_rate)} Hz (Nyquist {_num(nyquist, 0)} Hz); the rise time was "
            f"not resolvable for {count(len(unresolved_rise), 'shot')}.",
            "Rise time for those shots is an upper bound set by the sample interval.",
        ))
    else:
        checks.append(ValidityCheck(
            "Sample rate", "ok",
            f"{_txt(sample_rate)} Hz (Nyquist {_num(nyquist, 0)} Hz).",
            "Adequate to resolve the blast waveform.",
        ))

    # ── 6. Hard errors raised by the signal-quality assessment ────────
    for err in _seq(quality, "errors"):
        checks.append(ValidityCheck("Signal quality", "fail", str(err),
                                    "This is a hard validity failure."))
        blocking.append(str(err))
    for warn in _seq(quality, "warnings"):
        checks.append(ValidityCheck("Signal quality", "warn", str(warn), ""))

    # ── 6b. The analysis backend's own verdict and warnings ───────────
    # Nothing the instrument flagged during the run may be dropped on the way to
    # the customer, even if this module would not have flagged it itself.
    backend = _map(metadata, "validity")
    if backend.get("measurement_valid") is False:
        for reason in _seq(backend, "reasons"):
            if not any(str(reason) == c.finding for c in checks):
                checks.append(ValidityCheck("Analysis", "fail", str(reason),
                                            "Raised by the analysis backend."))
            if str(reason) not in blocking:
                blocking.append(str(reason))
        if not _seq(backend, "reasons"):
            checks.append(ValidityCheck(
                "Analysis", "fail",
                "The analysis marked this measurement invalid.",
                "Raised by the analysis backend without a stated reason."))
            blocking.append("The analysis marked this measurement invalid.")

    for warn in _seq(metadata, "warnings"):
        checks.append(ValidityCheck("Run warning", "warn", str(warn), ""))

    # ── 7. Detection ──────────────────────────────────────────────────
    n_detected = detection.get("n_detected")
    if _is_num(n_detected):
        det_warnings = _seq(detection, "warnings")
        checks.append(ValidityCheck(
            "Shot detection",
            "warn" if det_warnings else "ok",
            f"{count(int(n_detected), 'shot')} detected at a "
            f"{_num(detection.get('threshold_dB'))} {level_unit} threshold "
            f"({_txt(detection.get('threshold_mode'), missing='mode not recorded')}).",
            " ".join(str(w) for w in det_warnings),
        ))
        if int(n_detected) == 0:
            blocking.append("No shots were detected; there is nothing to report.")

    # Where the detection settings came from. A reader who is asked to accept a
    # shot count needs to know whether the setting that produced it was chosen
    # by a person, or measured - and, if measured, over how wide a span the
    # answer held. A count that only survives two decibels of threshold is a
    # different claim from one that survives twenty.
    settings = _map(metadata, "settings")
    tuning = _map(settings, "detection_tuning")
    if tuning.get("applied") is True:
        span = tuning.get("stable_width_dB")
        checks.append(ValidityCheck(
            "Detection settings", "ok",
            f"Measured from the recording: {_num(tuning.get('threshold_relative_dB'), 0)} dB "
            f"below peak, {_num(tuning.get('refractory_ms'), 0)} ms refractory, "
            f"{_num(tuning.get('post_ms'), 0)} ms post-trigger window.",
            f"The shot count is unchanged from {_num(tuning.get('stable_from_dB'), 0)} to "
            f"{_num(tuning.get('stable_to_dB'), 0)} dB below peak"
            + (f", a span of {_num(span, 0)} dB." if _is_num(span) else ".")
            + " The centre of that span was taken, so the count does not depend on a "
              "threshold chosen by hand.",
        ))
    elif settings.get("auto_detect") is False:
        checks.append(ValidityCheck(
            "Detection settings", "warn",
            "Chosen by the operator; nothing was measured from the recording.",
            "The shot count below is a consequence of those settings. Nothing here "
            "establishes that a different threshold would have found the same shots.",
        ))
    elif tuning.get("reason"):
        checks.append(ValidityCheck(
            "Detection settings", "warn",
            f"Could not be measured: {_txt(tuning.get('reason'))}.",
            "The defaults were used instead. Check the shot count against the waveform.",
        ))

    reflection = _map(tuning, "reflection")
    if reflection.get("detected") is True:
        followers = [str(v) for v in _seq(reflection, "followers")]
        checks.append(ValidityCheck(
            "Repeated delay", "warn",
            f"{count(len(followers), 'event')} arrive a near-constant "
            f"{_num(reflection.get('delay_ms'), 0)} ms after a louder one, "
            f"{_num(reflection.get('drop_dB'), 0)} dB quieter"
            + (f": {join_list(followers)}." if followers else "."),
            "A delay that repeats is the signature of a reflection off fixed geometry, "
            "not of separate discharges. If these are reflections they must be rejected "
            "before the aggregate is read, because a reflection averaged into a "
            "suppressor's rated level understates it.",
        ))

    if tuning.get("expectation_met") is False and _is_num(tuning.get("expected_shots")):
        checks.append(ValidityCheck(
            "Rounds expected", "warn",
            f"{count(int(tuning['expected_shots']), 'round')} were expected; "
            f"{count(int(tuning.get('n_shots') or 0), 'event')} were found.",
            "No threshold produced the expected count, and none was forced to. Either "
            "the expectation is wrong, or shots are missing from the recording.",
        ))

    # A record can carry a detection count and still have no measured shots - a
    # truncated run, or every shot rejected. Reporting statistics over nothing
    # would print an empty table under a "valid" heading.
    aggregate = _map(metadata, "aggregate")
    n_valid = aggregate.get("n_valid")
    n_agg = aggregate.get("n_shots")
    if _is_num(n_valid) and int(n_valid) == 0:
        checks.append(ValidityCheck(
            "Measured shots", "fail",
            f"No valid shots contributed to the results "
            f"({_txt(n_agg, missing='0')} detected, 0 usable).",
            "There is nothing to report. Every statistic below is empty.",
        ))
        blocking.append("No valid shots contributed to the reported statistics.")
    elif _is_num(n_valid) and _is_num(n_agg) and int(n_valid) < int(n_agg):
        checks.append(ValidityCheck(
            "Measured shots", "warn",
            f"{int(n_agg) - int(n_valid)} of {int(n_agg)} shots were excluded from the "
            f"aggregate statistics.",
            "The reported means describe only the shots that passed their per-shot "
            "validity checks; the excluded shots are listed in the per-shot table.",
        ))

    # ── 8. Is the record defensible? ──────────────────────────────────
    missing_required = _seq(test_meta, "missing_required")
    if test_meta.get("is_defensible") is True and not missing_required:
        checks.append(ValidityCheck(
            "Measurement record", "ok", "All required test conditions were recorded.",
            "The measurement can be reproduced and compared against another session.",
        ))
    else:
        checks.append(ValidityCheck(
            "Measurement record", "warn",
            f"Incomplete - {count(len(missing_required), 'required field')} missing: "
            f"{', '.join(str(m) for m in missing_required) or 'unspecified'}.",
            "The numbers remain arithmetically valid, but the measurement cannot be "
            "reproduced by another party or compared against another session, because "
            "the conditions that set the level were not recorded.",
        ))

    # ── Verdict ───────────────────────────────────────────────────────
    has_fail = any(c.status == "fail" for c in checks)
    has_warn = any(c.status == "warn" for c in checks)

    if has_fail:
        verdict = "invalid"
        if not calibrated and len(blocking) == 1:
            headline = "RELATIVE MEASUREMENT - NOT SOUND PRESSURE LEVELS"
            statement = (
                "This analysis was run without a calibration, so every level below is "
                "relative to digital full scale (dB re FS). The numbers are internally "
                "consistent and differences within this recording are real, but they are "
                "NOT sound pressure levels and must not be quoted as dB SPL, compared "
                "with another instrument, or checked against a hearing-hazard limit."
            )
        else:
            headline = "MEASUREMENT NOT VALID FOR REPORTING"
            statement = (
                "This measurement fails one or more validity checks and must not be "
                "presented as a sound pressure measurement. The results below are "
                "retained for diagnostic purposes only."
            )
    elif has_warn:
        verdict = "qualified"
        headline = "VALID - WITH QUALIFICATIONS"
        statement = (
            "The measurement is admissible. Some conditions limit how far the results "
            "can be generalised; each is stated below and applies to every number in "
            "this report."
        )
    else:
        verdict = "valid"
        headline = "VALID MEASUREMENT"
        statement = (
            "The recording passed every validity check applied by this instrument: it is "
            "calibrated, unclipped, clear of the noise floor, sampled fast enough to "
            "resolve the blast, and accompanied by a complete record of test conditions."
        )

    return ValidityReport(
        verdict=verdict, headline=headline, statement=statement,
        calibrated=calibrated, level_unit=level_unit,
        checks=checks, blocking=blocking,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Insertion loss
# ═══════════════════════════════════════════════════════════════════════════

_IL_METRICS = ("Lpeak_C", "Lpeak_Z", "Lpeak_A", "LAE", "LZE", "LCE", "LAImax", "LAFmax")


def _load_reference(metadata: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find the unsuppressed reference analysis, if one was supplied.

    Accepted, in order: an embedded 'reference' block, a reference directory named
    in the settings, or nothing at all. Returns (reference_metadata, source_note).
    """
    ref = metadata.get("reference")
    if isinstance(ref, dict) and ref:
        if "aggregate" in ref or "calibration" in ref:
            return ref, "embedded reference record"
        inner = ref.get("metadata")
        if isinstance(inner, dict):
            return inner, "embedded reference record"
        path = ref.get("path") or ref.get("dir")
        if isinstance(path, str):
            loaded = _load_metadata_file(Path(path))
            if loaded:
                return loaded, str(path)

    # main.py records the reference it actually used inside the comparison block.
    block = _map(metadata, "insertion_loss")
    recorded = block.get("reference_dir")
    if isinstance(recorded, str) and recorded:
        loaded = _load_metadata_file(Path(recorded))
        if loaded:
            return loaded, recorded

    for key in ("reference_dir", "reference_analysis_dir", "reference_analysis"):
        value = _dig(metadata, "settings", key)
        if isinstance(value, str) and value:
            loaded = _load_metadata_file(Path(value))
            if loaded:
                return loaded, value
    return None, None


def _load_metadata_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load analysis_metadata.json from a file or a directory."""
    try:
        candidate = path / "analysis_metadata.json" if path.is_dir() else path
        if not candidate.is_file():
            return None
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def compute_insertion_loss_rows(
    metadata: Dict[str, Any],
    reference: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Per-metric insertion loss (reference minus test), with a combined 95% interval.

    Preconditions are checked rather than assumed: two recordings can only be
    differenced if they share a calibration state and a microphone position.
    """
    notes: List[str] = []

    precomputed = metadata.get("insertion_loss")
    if isinstance(precomputed, list) and precomputed:
        return [r for r in precomputed if isinstance(r, dict)], notes
    if isinstance(precomputed, dict) and isinstance(precomputed.get("metrics"), list):
        # The analysis already computed this against a verified reference; its own
        # caveats travel with it rather than being silently dropped.
        notes.extend(str(w) for w in _seq(precomputed, "warnings"))
        return [r for r in precomputed["metrics"] if isinstance(r, dict)], notes

    if not reference:
        return [], notes

    test_cal = _map(metadata, "calibration")
    ref_cal = _map(reference, "calibration")
    if bool(test_cal.get("calibrated")) != bool(ref_cal.get("calibrated")):
        notes.append(
            "Insertion loss was NOT computed: one recording is calibrated and the other "
            "is not, so their levels are in different units and cannot be differenced.")
        return [], notes

    test_meta = _map(metadata, "test_metadata")
    ref_meta = _map(reference, "test_metadata")
    for field_name, label in (("mic_distance_m", "microphone distance"),
                              ("mic_angle_deg", "microphone angle"),
                              ("mic_height_m", "microphone height")):
        a, b = test_meta.get(field_name), ref_meta.get(field_name)
        if _is_num(a) and _is_num(b) and abs(float(a) - float(b)) > 1e-9:
            notes.append(
                f"The {label} differs between the reference ({_txt(b)}) and the test "
                f"({_txt(a)}). The difference below includes that geometry change, not "
                f"suppressor performance alone.")
    sr_a = _dig(metadata, "source", "sample_rate")
    sr_b = _dig(reference, "source", "sample_rate")
    if _is_num(sr_a) and _is_num(sr_b) and int(sr_a) != int(sr_b):
        notes.append(
            f"The reference was recorded at {int(sr_b)} Hz and the test at {int(sr_a)} Hz. "
            f"Peak-based metrics are sensitive to sample rate and are not strictly comparable.")

    test_stats = _map(metadata, "aggregate", "statistics")
    ref_stats = _map(reference, "aggregate", "statistics")

    rows: List[Dict[str, Any]] = []
    for metric in _IL_METRICS:
        t, r = test_stats.get(metric), ref_stats.get(metric)
        if not isinstance(t, dict) or not isinstance(r, dict):
            continue
        if not (_is_num(t.get("mean")) and _is_num(r.get("mean"))):
            continue
        t_ci = float(t.get("ci95_half_width") or 0.0)
        r_ci = float(r.get("ci95_half_width") or 0.0)
        rows.append({
            "metric": metric,
            "reference_dB": float(r["mean"]),
            "test_dB": float(t["mean"]),
            "reduction_dB": float(r["mean"]) - float(t["mean"]),
            "ci95_dB": math.sqrt(t_ci ** 2 + r_ci ** 2),
            "reference_n": r.get("n"),
            "test_n": t.get("n"),
        })
    return rows, notes


def compute_band_insertion_loss(
    metadata: Dict[str, Any],
    reference: Optional[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return (frequencies_Hz, reference_dB, test_dB, insertion_loss_dB)."""
    for precomputed in (_map(metadata, "insertion_loss", "bands"),
                        metadata.get("band_insertion_loss")):
        if not isinstance(precomputed, dict) or not precomputed:
            continue
        freqs = precomputed.get("frequencies_Hz") or precomputed.get("band_frequencies_Hz") or []
        il = precomputed.get("insertion_loss_dB") or []
        ref = precomputed.get("reference_dB") or []
        test = precomputed.get("test_dB") or []
        if freqs and il and len(freqs) == len(il):
            return list(freqs), list(ref), list(test), list(il)

    precomputed = None
    if isinstance(precomputed, dict):
        freqs = precomputed.get("frequencies_Hz") or precomputed.get("band_frequencies_Hz") or []
        il = precomputed.get("insertion_loss_dB") or []
        ref = precomputed.get("reference_dB") or []
        test = precomputed.get("test_dB") or []
        if freqs and il and len(freqs) == len(il):
            return list(freqs), list(ref), list(test), list(il)

    if not reference:
        return [], [], [], []

    freqs = _seq(metadata, "aggregate", "band_frequencies_Hz")
    test = _seq(metadata, "aggregate", "band_exposure_mean_dB")
    ref_freqs = _seq(reference, "aggregate", "band_frequencies_Hz")
    ref = _seq(reference, "aggregate", "band_exposure_mean_dB")

    if not (freqs and test and ref):
        return [], [], [], []
    if len(freqs) != len(ref_freqs) or len(test) != len(ref):
        # Different filter banks mean different sample rates: not comparable.
        return [], [], [], []

    il = [float(a) - float(b) for a, b in zip(ref, test)]
    return list(freqs), list(ref), list(test), il


# ═══════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════

_EMBEDDABLE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}

_FIGURE_TITLES = {
    "waveform_full": "Pressure waveform, full recording",
    "spectrogram_z_full": "Z-weighted spectrogram, full recording",
    "spectrogram_c_full": "C-weighted spectrogram, full recording",
    "bands_full": "1/3-octave band levels, full recording",
    "insertion_loss": "Insertion loss by 1/3-octave band",
    "shot_overlay": "Shot overlay, all detected shots",
    "measurement_quality": "Measurement quality",
}

# Figures read in this order: the recording, then its spectrum, then the comparison,
# then the diagnostics. Artifact keys are a machine index, not a running order.
_FIGURE_ORDER = (
    "waveform_full", "spectrogram_z_full", "spectrogram_c_full", "bands_full",
    "insertion_loss", "shot_overlay", "measurement_quality",
)


def _figure_title(key: Any, path: Path) -> str:
    """
    A caption a reader can use.

    The file's own stem identifies the figure ("waveform_full"); the artifact key
    that points at it is an index entry ("waveform_png") and makes a poor caption.
    """
    stem_title = _FIGURE_TITLES.get(path.stem)
    if stem_title:
        return stem_title
    name = str(key or path.stem)
    if name in _FIGURE_TITLES:
        return _FIGURE_TITLES[name]
    for suffix in ("_png", "_svg", "_jpg", "_figure", "_plot"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    cleaned = name.replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else path.stem


def _figure_sort_key(title_and_path: Tuple[str, Path]) -> Tuple[int, str]:
    stem = title_and_path[1].stem
    try:
        return (_FIGURE_ORDER.index(stem), stem)
    except ValueError:
        return (len(_FIGURE_ORDER), stem)

# A megabyte-scale figure is fine; a 200 MB one would make the report unopenable.
MAX_FIGURE_BYTES = 24 * 1024 * 1024


def _normalise_figures(figures: Any, base_dir: Optional[Path]) -> List[Tuple[str, Path]]:
    """Accept a list of paths, a dict {title: path}, or a list of (title, path)."""
    items: List[Tuple[str, Path]] = []

    def add(title: Any, path: Any) -> None:
        if path is None:
            return
        p = Path(path)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        label = str(title) if title else _figure_title(None, p)
        items.append((label, p))

    if figures is None:
        return items
    if isinstance(figures, dict):
        for title, path in figures.items():
            add(title, path)
        return items
    if isinstance(figures, (str, Path)):
        add(None, figures)
        return items
    if isinstance(figures, Iterable):
        for entry in figures:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                add(entry[0], entry[1])
            else:
                add(None, entry)
    return items


def _discover_figures(metadata: Dict[str, Any], base_dir: Optional[Path]) -> List[Tuple[str, Path]]:
    """Fall back to the artifacts block, then to whatever images sit in the directory."""
    found: List[Tuple[str, Path]] = []
    seen: set = set()

    artifacts = _map(metadata, "artifacts")
    if isinstance(artifacts, dict) and base_dir is not None:
        for name, rel in artifacts.items():
            if not isinstance(rel, str):
                continue
            path = (base_dir / rel)
            if path.suffix.lower() in _EMBEDDABLE and path.is_file():
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append((_figure_title(name, path), path))

    if not found and base_dir is not None and base_dir.is_dir():
        for path in sorted(base_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in _EMBEDDABLE:
                found.append((_figure_title(None, path), path))

    found.sort(key=_figure_sort_key)
    return found


def _embed_figure(path: Path) -> Optional[str]:
    """Read an image and return a data: URI, or None when it cannot be embedded."""
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size == 0 or size > MAX_FIGURE_BYTES:
            return None
        mime = _EMBEDDABLE.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith("image/"):
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    except OSError:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════

def _css() -> str:
    return f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{
  margin: 0; padding: 0;
  background: {C['bg_canvas']}; color: {C['text']};
  font-family: {FONT_SANS}; font-size: 15px; line-height: 1.5;
}}
.page {{ max-width: 1000px; margin: 0 auto; padding: 32px 28px 64px; }}
h1, h2, h3 {{ line-height: 1.25; margin: 0; font-weight: 600; }}
h1 {{ font-size: 28px; letter-spacing: -0.01em; }}
h2 {{ font-size: 22px; margin: 40px 0 12px; padding-bottom: 8px;
     border-bottom: 2px solid {C['border']}; }}
h3 {{ font-size: 17px; margin: 24px 0 8px; }}
p {{ margin: 0 0 12px; }}
.sub {{ color: {C['text_2']}; }}
.mono {{ font-family: {FONT_MONO}; font-variant-numeric: tabular-nums; }}
.missing {{ color: {C['text_3']}; font-style: italic; }}
.caps {{ font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
        color: {C['text_3']}; font-weight: 600; }}

/* ---- Title block ---- */
.masthead {{
  background: {C['bg_surface']}; border: 1px solid {C['border']};
  border-top: 4px solid {C['accent']}; border-radius: 7px;
  padding: 24px 26px; margin-bottom: 24px;
}}
.brand {{ display: flex; justify-content: space-between; align-items: baseline;
         gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }}
.brand-name {{ font-size: 13px; font-weight: 700; letter-spacing: 0.12em;
              text-transform: uppercase; color: {C['accent']}; }}
.title-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px 24px; margin-top: 18px; padding-top: 18px;
  border-top: 1px solid {C['border_subtle']};
}}
.title-grid div span {{ display: block; }}
.title-grid .v {{ font-size: 15px; font-weight: 500; margin-top: 2px; }}

/* ---- Validity ---- */
.verdict {{ border-radius: 7px; padding: 20px 24px; margin-bottom: 20px;
           border: 2px solid; }}
.verdict h2 {{ border: 0; margin: 0 0 8px; padding: 0; font-size: 20px;
              letter-spacing: 0.02em; }}
.verdict.valid {{ background: {C['ok_wash']}; border-color: {C['ok_border']}; }}
.verdict.valid h2 {{ color: {C['ok']}; }}
.verdict.qualified {{ background: {C['warn_wash']}; border-color: {C['warn_border']}; }}
.verdict.qualified h2 {{ color: {C['warn']}; }}
.verdict.invalid {{ background: {C['danger_wash']}; border-color: {C['danger']}; }}
.verdict.invalid h2 {{ color: {C['danger']}; }}
.verdict p {{ margin: 0 0 8px; }}
.verdict ul {{ margin: 10px 0 0; padding-left: 20px; }}
.verdict li {{ margin-bottom: 5px; font-weight: 500; }}

.checks {{ display: grid; gap: 10px; }}
.check {{ display: grid; grid-template-columns: 96px 1fr; gap: 14px;
         background: {C['bg_surface']}; border: 1px solid {C['border_subtle']};
         border-left: 4px solid {C['border']}; border-radius: 5px; padding: 12px 14px; }}
.check.ok {{ border-left-color: {C['ok']}; }}
.check.warn {{ border-left-color: {C['warn']}; }}
.check.fail {{ border-left-color: {C['danger']}; }}
.check .label {{ font-size: 12px; font-weight: 600; color: {C['text_3']};
                text-transform: uppercase; letter-spacing: 0.04em; padding-top: 2px; }}
.check .finding {{ font-weight: 500; }}
.check.fail .finding {{ color: {C['danger']}; }}
.check .consequence {{ color: {C['text_2']}; font-size: 14px; margin-top: 4px; }}

/* ---- Tables ---- */
.tablewrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; background: {C['bg_surface']};
        border: 1px solid {C['border']}; border-radius: 5px; font-size: 14px; }}
caption {{ caption-side: top; text-align: left; padding: 0 0 8px;
          color: {C['text_2']}; font-size: 14px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid {C['border_subtle']}; }}
thead th {{ background: {C['bg_sunken']}; color: {C['text_3']}; font-size: 11px;
           letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
           border-bottom: 1px solid {C['border']}; white-space: nowrap; }}
tbody tr:last-child td {{ border-bottom: 0; }}
tbody tr:nth-child(even) {{ background: {C['bg_inset']}; }}
td.num, th.num {{ text-align: right; font-family: {FONT_MONO};
                 font-variant-numeric: tabular-nums; white-space: nowrap; }}
th.num {{ font-family: {FONT_SANS}; }}
td.field {{ width: 32%; color: {C['text_2']}; }}

/* ---- Headline metrics ---- */
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
           gap: 14px; margin-bottom: 18px; }}
.metric {{ background: {C['bg_surface']}; border: 1px solid {C['border']};
          border-radius: 7px; padding: 16px 18px; }}
.metric .name {{ font-size: 12px; font-weight: 600; letter-spacing: 0.06em;
                text-transform: uppercase; color: {C['text_3']}; }}
.metric .value {{ font-family: {FONT_MONO}; font-size: 32px; font-weight: 600;
                 letter-spacing: -0.02em; margin: 6px 0 2px;
                 font-variant-numeric: tabular-nums; }}
.metric .unit {{ font-size: 14px; color: {C['text_3']}; margin-left: 6px; font-weight: 500; }}
.metric .ci {{ font-family: {FONT_MONO}; font-size: 13px; color: {C['text_2']}; }}
.metric .desc {{ font-size: 12px; color: {C['text_3']}; margin-top: 8px; }}
.metric.suspect {{ border-color: {C['danger_border']}; background: {C['danger_wash']}; }}

/* ---- Insertion loss ---- */
.bar {{ position: relative; background: {C['bg_sunken']}; border-radius: 3px;
       height: 14px; min-width: 90px; }}
.bar > i {{ position: absolute; top: 0; bottom: 0; left: 0; border-radius: 3px;
           background: {C['series_6']}; display: block; }}
.bar.negative > i {{ background: {C['danger']}; }}

/* ---- Callouts ---- */
.note {{ border-radius: 5px; padding: 12px 16px; margin: 12px 0; font-size: 14px;
        border: 1px solid {C['info_border']}; background: {C['info_wash']}; color: {C['text']}; }}
.note.warn {{ border-color: {C['warn_border']}; background: {C['warn_wash']}; }}
.note.danger {{ border-color: {C['danger_border']}; background: {C['danger_wash']}; }}
.note.ok {{ border-color: {C['ok_border']}; background: {C['ok_wash']}; }}
.note strong {{ color: {C['danger']}; }}
.note.warn strong {{ color: {C['warn']}; }}
.note.ok strong {{ color: {C['ok']}; }}

/* A list of per-shot or per-comparison findings. Each item carries its own
   severity flag, so the list is never colour alone. */
.findings {{ margin: 10px 0 0; padding-left: 20px; }}
.findings li {{ margin-bottom: 6px; line-height: 1.5; }}

.flag {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 7px;
        border-radius: 999px; border: 1px solid; margin-right: 4px; white-space: nowrap; }}
.flag.ok {{ color: {C['ok']}; border-color: {C['ok_border']}; background: {C['ok_wash']}; }}
.flag.warn {{ color: {C['warn']}; border-color: {C['warn_border']}; background: {C['warn_wash']}; }}
.flag.fail {{ color: {C['danger']}; border-color: {C['danger_border']}; background: {C['danger_wash']}; }}

/* ---- Figures ---- */
figure {{ margin: 0 0 22px; background: {C['bg_surface']}; border: 1px solid {C['border']};
         border-radius: 7px; padding: 14px; }}
figure img {{ width: 100%; height: auto; display: block; border-radius: 3px; }}
figcaption {{ margin-top: 10px; font-size: 13px; color: {C['text_2']}; }}

/* ---- Appendix ---- */
.appendix {{ font-size: 14px; }}
.appendix dt {{ font-weight: 600; margin-top: 14px; }}
.appendix dd {{ margin: 4px 0 0; color: {C['text_2']}; }}
.footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid {C['border']};
          font-size: 12px; color: {C['text_3']}; }}

/* ---- Print ---- */
@page {{ size: A4; margin: 14mm 12mm; }}
@media print {{
  body {{ background: {C['bg_surface']}; font-size: 10.5pt; }}
  .page {{ max-width: none; padding: 0; }}
  h2 {{ margin-top: 22px; }}
  h2, h3 {{ break-after: avoid; page-break-after: avoid; }}
  .verdict, .check, figure, .metric {{ break-inside: avoid; page-break-inside: avoid; }}
  table {{ font-size: 9pt; }}
  tr, td, th {{ break-inside: avoid; page-break-inside: avoid; }}
  thead {{ display: table-header-group; }}
  .pagebreak {{ break-before: page; page-break-before: always; }}
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Section builders
# ═══════════════════════════════════════════════════════════════════════════

def _section_title(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    meta = _map(metadata, "test_metadata")
    software = _map(metadata, "software")
    analysis = _map(metadata, "analysis")
    source = _map(metadata, "source")

    commit = str(software.get("git_commit") or "")
    commit_text = commit[:12] if commit else "not recorded"
    if software.get("git_dirty"):
        commit_text += " + uncommitted changes"

    fields = [
        ("Test ID", _txt(meta.get("test_id"))),
        ("Date of test", _txt(meta.get("date"))),
        ("Operator", _txt(meta.get("operator"))),
        ("Location", _txt(meta.get("location"))),
        ("Configuration", _txt(meta.get("configuration"))),
        ("Analysed", _txt(analysis.get("timestamp"))),
        ("Software", f"{_txt(software.get('name'), missing='SASA')} "
                     f"{_txt(software.get('version'), missing='version not recorded')}"),
        ("Build (commit)", commit_text),
    ]
    grid = "\n".join(
        f'<div><span class="caps">{_e(label)}</span>'
        f'<span class="v">{_missing_span(value)}</span></div>'
        for label, value in fields
    )

    source_name = Path(str(source.get("path") or analysis.get("input_file") or "")).name
    sha = str(source.get("sha256") or analysis.get("input_sha256") or "")
    sha_line = (f'<div class="sub mono" style="font-size:12px;margin-top:10px">'
                f'Source: {_e(source_name or "not recorded")}'
                + (f' &middot; SHA-256 {_e(sha[:32])}…' if sha else '')
                + '</div>')

    return f"""
<header class="masthead">
  <div class="brand">
    <span class="brand-name">Ridgeback Defense &middot; SASA</span>
    <span class="caps">Shot Acoustic Spectral Analysis</span>
  </div>
  <h1>Suppressor &amp; Gunshot Acoustic Measurement Report</h1>
  <p class="sub" style="margin-top:6px">
    Levels in this report are stated in <strong>{_e(validity.level_unit)}</strong>.
  </p>
  {sha_line}
  <div class="title-grid">{grid}</div>
</header>
"""


def _section_validity(validity: ValidityReport) -> str:
    blocking = ""
    if validity.blocking:
        items = "\n".join(f"<li>{_e(b)}</li>" for b in validity.blocking)
        blocking = f"<ul>{items}</ul>"

    checks = "\n".join(
        f'<div class="check {c.status}">'
        f'<div class="label">{_e(c.label)}</div>'
        f'<div><div class="finding">{_e(c.finding)}</div>'
        + (f'<div class="consequence">{_e(c.consequence)}</div>' if c.consequence else '')
        + '</div></div>'
        for c in validity.checks
    )

    return f"""
<section>
  <div class="verdict {validity.verdict}">
    <h2>{_e(validity.headline)}</h2>
    <p>{_e(validity.statement)}</p>
    {blocking}
  </div>
  <h2>Measurement validity</h2>
  <div class="checks">{checks}</div>
</section>
"""


_CONDITION_ROWS: Sequence[Tuple[str, str, str]] = (
    # (label, metadata key, unit)
    ("Weapon", "weapon", ""),
    ("Barrel length", "barrel_length_in", "in"),
    ("Ammunition", "ammunition", ""),
    ("Suppressor", "suppressor", ""),
    ("Configuration", "configuration", ""),
    ("Microphone model", "mic_model", ""),
    ("Microphone serial", "mic_serial", ""),
    ("Microphone distance", "mic_distance_m", "m"),
    ("Microphone angle", "mic_angle_deg", "° from bore"),
    ("Microphone height", "mic_height_m", "m"),
    ("Windscreen", "windscreen", ""),
    ("Ground surface", "ground_surface", ""),
    ("Temperature", "temperature_C", "°C"),
    ("Relative humidity", "humidity_pct", "%"),
    ("Barometric pressure", "pressure_kPa", "kPa"),
    ("Wind speed", "wind_mps", "m/s"),
    ("Calibrator", "calibrator_model", ""),
    ("Calibrator level", "calibrator_level_dB", "dB"),
    ("Calibration, pre-test", "calibration_pre_dB", "dB"),
    ("Calibration, post-test", "calibration_post_dB", "dB"),
)


def _section_conditions(metadata: Dict[str, Any]) -> str:
    meta = _map(metadata, "test_metadata")
    required = set(_seq(meta, "missing_required"))

    rows = []
    for label, key, unit in _CONDITION_ROWS:
        value = meta.get(key)
        text = _txt(value)
        if text != "not recorded" and unit:
            text = f"{text} {unit}"
        marker = ''
        if key in required:
            marker = ' <span class="flag fail">required</span>'
        # Every row is present even when empty: an omitted row reads as though the
        # condition did not matter to the measurement.
        rows.append(f'<tr><td class="field">{_e(label)}{marker}</td>'
                    f'<td>{_missing_span(text)}</td></tr>')

    notes = meta.get("notes")
    if isinstance(notes, str) and notes.strip():
        rows.append(f'<tr><td class="field">Notes</td><td>{_e(notes.strip())}</td></tr>')

    source = _map(metadata, "source")
    src_rows = []
    for label, value in (
        ("Sample rate", f"{_txt(source.get('sample_rate'))} Hz"),
        ("Channels", _txt(source.get("channels"))),
        ("Channel analysed", _txt(source.get("channel_used"))),
        ("Sample format", _txt(source.get("subtype"))),
        ("Duration", f"{_num(source.get('duration_s'), 2)} s"),
    ):
        src_rows.append(f'<tr><td class="field">{_e(label)}</td>'
                        f'<td>{_missing_span(value)}</td></tr>')

    return f"""
<section>
  <h2>Test conditions</h2>
  <div class="tablewrap">
    <table>
      <caption>Conditions as recorded by the operator. Sound pressure from a muzzle
      blast varies with distance, angle, temperature, humidity and barometric
      pressure, so a level quoted without them cannot be compared with anyone
      else's measurement.</caption>
      <thead><tr><th>Condition</th><th>Value</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <h3>Recording</h3>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>{''.join(src_rows)}</tbody>
    </table>
  </div>
</section>
"""


# Typeset names for every level this instrument reports, so a customer document
# never falls back to the internal identifier for some rows and not others.
_METRIC_LABELS: Dict[str, str] = {
    "Lpeak_Z": "L<sub>peak,Z</sub>",
    "Lpeak_A": "L<sub>peak,A</sub>",
    "Lpeak_C": "L<sub>Cpeak</sub>",
    "LAE": "L<sub>AE</sub>",
    "LZE": "L<sub>ZE</sub>",
    "LCE": "L<sub>CE</sub>",
    "LAFmax": "L<sub>AFmax</sub>",
    "LASmax": "L<sub>ASmax</sub>",
    "LAImax": "L<sub>AImax</sub>",
    "LZImax": "L<sub>ZImax</sub>",
    "LZFmax": "L<sub>ZFmax</sub>",
    "LZSmax": "L<sub>ZSmax</sub>",
}


def _metric_label(key: str) -> str:
    """Typeset name for a metric, falling back to the identifier itself."""
    return _METRIC_LABELS.get(str(key), _e(key))


_HEADLINE_METRICS: Sequence[Tuple[str, str, str]] = (
    ("Lpeak_C", "L<sub>Cpeak</sub>",
     "C-weighted peak sound pressure level - the regulatory impulse quantity."),
    ("Lpeak_Z", "L<sub>peak,Z</sub>",
     "Unweighted (Z) peak level - the physical peak of the blast wave."),
    ("LAE", "L<sub>AE</sub>",
     "A-weighted sound exposure level - the energy of one round in 1 s."),
    ("LAImax", "L<sub>AImax</sub>",
     "Maximum A-weighted level, Impulse time weighting (35 ms rise / 1500 ms decay)."),
)


def _section_results(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    aggregate = _map(metadata, "aggregate")
    stats = _map(aggregate, "statistics")
    unit = validity.level_unit
    suspect = validity.verdict == "invalid"

    cards = []
    for key, label, description in _HEADLINE_METRICS:
        s = stats.get(key) if isinstance(stats.get(key), dict) else None
        mean = s.get("mean") if s else None
        ci = s.get("ci95_half_width") if s else None
        n = s.get("n") if s else None
        ci_text = (f"± {_num(ci, 2)} (95% CI, n = {_txt(n, missing='?')})"
                   if _is_num(ci) and _is_num(n) else
                   (f"n = {_txt(n, missing='?')}" if n is not None else "no data"))
        cards.append(f"""
<div class="metric{' suspect' if suspect else ''}">
  <div class="name">{label}</div>
  <div class="value">{_num(mean)}<span class="unit">{_e(unit)}</span></div>
  <div class="ci">{_e(ci_text)}</div>
  <div class="desc">{description}</div>
</div>""")

    rows = []
    for key, label, _desc in _HEADLINE_METRICS:
        s = stats.get(key) if isinstance(stats.get(key), dict) else None
        if not s:
            continue
        rows.append(
            f'<tr><td>{label}</td>'
            f'<td class="num">{_txt(s.get("n"), missing="—")}</td>'
            f'<td class="num">{_num(s.get("mean"), 2)}</td>'
            f'<td class="num">± {_num(s.get("ci95_half_width"), 2)}</td>'
            f'<td class="num">{_num(s.get("std"), 2)}</td>'
            f'<td class="num">{_num(s.get("min"), 2)}</td>'
            f'<td class="num">{_num(s.get("median"), 2)}</td>'
            f'<td class="num">{_num(s.get("max"), 2)}</td></tr>'
        )
    # Any other metrics the analysis produced, for completeness.
    extra_levels = ("Lpeak_A", "LZE", "LCE", "LAFmax", "LASmax", "LZImax")
    for key in extra_levels:
        s = stats.get(key) if isinstance(stats.get(key), dict) else None
        if not s:
            continue
        rows.append(
            f'<tr><td>{_metric_label(key)}</td>'
            f'<td class="num">{_txt(s.get("n"), missing="—")}</td>'
            f'<td class="num">{_num(s.get("mean"), 2)}</td>'
            f'<td class="num">± {_num(s.get("ci95_half_width"), 2)}</td>'
            f'<td class="num">{_num(s.get("std"), 2)}</td>'
            f'<td class="num">{_num(s.get("min"), 2)}</td>'
            f'<td class="num">{_num(s.get("median"), 2)}</td>'
            f'<td class="num">{_num(s.get("max"), 2)}</td></tr>'
        )

    n_shots = aggregate.get("n_shots")
    n_valid = aggregate.get("n_valid")
    excluded = ""
    if _is_num(n_shots) and _is_num(n_valid) and int(n_valid) < int(n_shots):
        excluded = (f'<div class="note warn"><strong>{int(n_shots) - int(n_valid)} of '
                    f'{int(n_shots)} shots were excluded</strong> from these statistics '
                    f'because they failed a per-shot validity check (see the per-shot '
                    f'table). Statistics are computed over the {int(n_valid)} valid '
                    f'shots.</div>')

    unit_warning = ""
    if not validity.calibrated:
        unit_warning = (
            '<div class="note danger"><strong>These are not sound pressure levels.</strong> '
            'The analysis was uncalibrated, so every value is relative to digital full '
            'scale (dB re FS). They may not be quoted as dB SPL or compared with any '
            'external measurement or limit.</div>')

    return f"""
<section>
  <h2>Results</h2>
  {unit_warning}
  {excluded}
  <div class="metrics">{''.join(cards)}</div>
  <div class="tablewrap">
    <table>
      <caption>Distribution across the shot string. Levels are energy-averaged
      (ISO convention); dispersion is reported on the decibel values. The 95%
      confidence interval is on the mean, using the sample standard deviation.
      All values in {_e(unit)}.</caption>
      <thead><tr>
        <th>Metric</th><th class="num">n</th><th class="num">Mean</th>
        <th class="num">95% CI</th><th class="num">Std</th><th class="num">Min</th>
        <th class="num">Median</th><th class="num">Max</th>
      </tr></thead>
      <tbody>{''.join(rows) or '<tr><td colspan="8">No shot statistics available.</td></tr>'}</tbody>
    </table>
  </div>
</section>
"""


def _section_insertion_loss(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    reference, ref_source = _load_reference(metadata)
    rows, notes = compute_insertion_loss_rows(metadata, reference)
    freqs, ref_band, test_band, band_il = compute_band_insertion_loss(metadata, reference)

    if not rows and not band_il:
        message = (
            "No unsuppressed reference measurement was supplied with this analysis, so "
            "insertion loss cannot be reported. Insertion loss is the difference between "
            "a suppressed and an unsuppressed string recorded at the same microphone "
            "position, with the same calibration, on the same day.")
        extra = "".join(f'<div class="note warn">{_e(n)}</div>' for n in notes)
        return f"""
<section>
  <h2>Insertion loss</h2>
  <div class="note">{_e(message)}</div>
  {extra}
</section>
"""

    peak = max((abs(float(r.get("reduction_dB") or 0.0)) for r in rows), default=1.0) or 1.0
    metric_rows = []
    for r in rows:
        reduction = float(r.get("reduction_dB") or 0.0)
        width = min(100.0, abs(reduction) / peak * 100.0)
        negative = " negative" if reduction < 0 else ""
        metric_rows.append(
            f'<tr><td>{_metric_label(r.get("metric"))}</td>'
            f'<td class="num">{_num(r.get("reference_dB"), 1)}</td>'
            f'<td class="num">{_num(r.get("test_dB"), 1)}</td>'
            f'<td class="num"><strong>{_num(reduction, 1)}</strong></td>'
            f'<td class="num">± {_num(r.get("ci95_dB"), 2)}</td>'
            f'<td><div class="bar{negative}"><i style="width:{width:.1f}%"></i></div></td>'
            f'<td class="num">{_txt(r.get("reference_n"), missing="—")} / '
            f'{_txt(r.get("test_n"), missing="—")}</td></tr>')

    band_rows = []
    for i, f in enumerate(freqs):
        il = band_il[i] if i < len(band_il) else None
        band_rows.append(
            f'<tr><td class="num">{_num(f, 0)}</td>'
            f'<td class="num">{_num(ref_band[i], 1) if i < len(ref_band) else "—"}</td>'
            f'<td class="num">{_num(test_band[i], 1) if i < len(test_band) else "—"}</td>'
            f'<td class="num"><strong>{_num(il, 1)}</strong></td></tr>')

    band_table = ""
    if band_rows:
        band_table = f"""
  <h3>Insertion loss by 1/3-octave band</h3>
  <div class="tablewrap">
    <table>
      <caption>Band-by-band difference in mean band exposure level, reference minus
      test. Positive values are reduction. Nominal centre frequencies per ISO 266;
      filters per IEC 61260-1. All values in {_e(validity.level_unit)}.</caption>
      <thead><tr>
        <th class="num">Centre frequency (Hz)</th>
        <th class="num">Reference</th><th class="num">Test</th>
        <th class="num">Insertion loss</th>
      </tr></thead>
      <tbody>{''.join(band_rows)}</tbody>
    </table>
  </div>"""

    ref_note = ""
    if ref_source:
        ref_note = (f'<p class="sub" style="font-size:13px">Reference measurement: '
                    f'<span class="mono">{_e(ref_source)}</span>.</p>')
    caveats = "".join(f'<div class="note warn">{_e(n)}</div>' for n in notes)
    invalid_note = ""
    if not validity.is_admissible:
        invalid_note = (
            '<div class="note danger"><strong>This insertion loss is not a valid '
            'result.</strong> It is computed from a measurement that failed validity '
            'checks (see the top of this report) and is shown for diagnostic purposes '
            'only.</div>')

    return f"""
<section class="pagebreak">
  <h2>Insertion loss vs. unsuppressed reference</h2>
  {ref_note}
  {invalid_note}
  {caveats}
  {_comparability_block(metadata)}
  <div class="tablewrap">
    <table>
      <caption>Insertion loss = reference level − test level. Positive values mean
      the test configuration is quieter. The interval combines the 95% confidence
      intervals of both strings. All levels in {_e(validity.level_unit)}.</caption>
      <thead><tr>
        <th>Metric</th><th class="num">Reference</th><th class="num">Test</th>
        <th class="num">Reduction</th><th class="num">95% CI</th>
        <th>Relative</th><th class="num">n ref / test</th>
      </tr></thead>
      <tbody>{''.join(metric_rows) or '<tr><td colspan="7">No comparable metrics.</td></tr>'}</tbody>
    </table>
  </div>
  {band_table}
  {_normalised_bands_block(metadata, validity)}
</section>
"""


_STRING_METRIC_LABELS = {
    "Lpeak_Z": "Peak, Z-weighted",
    "Lpeak_A": "Peak, A-weighted",
    "LAE": "Sound exposure level",
}


def _section_string_behaviour(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    """
    How the string behaved over its length, rather than what it averaged to.

    A suppressor is bought on two numbers a single average hides: what the first
    round out of a cold can costs, and whether the string held together at all.
    Both are stated here, with the evidence for each, and both say plainly when
    the measurement could not establish them.
    """
    breakdown = _map(metadata, "string_statistics")
    review = _map(metadata, "shot_review")
    unit = validity.level_unit

    if not breakdown and not review:
        return ""

    # -- first-round pop --
    headline = None
    for key in ("Lpeak_Z", "Lpeak_A", "LAE"):
        if isinstance(breakdown.get(key), dict):
            headline = breakdown[key]
            break
    pop = _map(headline or {}, "first_round_pop")

    if not pop:
        pop_block = (
            '<div class="note">First-round pop was not evaluated for this '
            'measurement.</div>'
        )
    elif pop.get("refusal"):
        pop_block = (
            f'<div class="note warn"><strong>First-round pop: not measured.</strong> '
            f'{_e(str(pop["refusal"]))}</div>'
        )
    elif pop.get("established"):
        pop_block = f"""
  <div class="note danger">
    <strong>First-round pop established: {_e(_num(pop.get("observed_dB"), 2))} dB.</strong>
    The first round measured {_e(_num(pop.get("first_shot_dB"), 1))} {_e(unit)} against
    {_e(_num(pop.get("subsequent_mean_dB"), 1))} {_e(unit)} for the
    {_e(_txt(pop.get("n_subsequent"), missing="—"))} rounds that followed. A further
    shot from this string would have been expected between
    {_e(_num(pop.get("prediction_lower_dB"), 1))} and
    {_e(_num(pop.get("prediction_upper_dB"), 1))} {_e(unit)}; the first round fell
    outside that interval (one-sided p = {_e(_num(pop.get("p_value"), 4))}).
  </div>"""
    elif pop.get("first_shot_quieter"):
        pop_block = (
            '<div class="note warn"><strong>The first round was QUIETER than the rest '
            'of the string explains.</strong> That is not first-round pop; it points at '
            'a squib, a misfire or a detection error, and the string should be '
            'reviewed before this result is relied on.</div>'
        )
    else:
        pop_block = f"""
  <div class="note ok">
    <strong>No first-round pop this measurement can resolve.</strong>
    The first round measured {_e(_num(pop.get("first_shot_dB"), 1))} {_e(unit)}, inside
    the {_e(_num(pop.get("prediction_lower_dB"), 1))} to
    {_e(_num(pop.get("prediction_upper_dB"), 1))} {_e(unit)} interval a further shot
    from this string would have been expected to fall in
    (one-sided p = {_e(_num(pop.get("p_value"), 4))}). This is not proof that the
    suppressor does not pop: it means any pop is smaller than the shot-to-shot
    spread of this string.
  </div>"""

    single_string_caveat = ""
    if pop and pop.get("basis") == "single-string" and not pop.get("refusal"):
        single_string_caveat = (
            '<div class="note">This is one first round, from one string. A first-round '
            'pop figure that supports a published claim needs the first shot of several '
            'strings, each fired into a can that has been allowed to purge.</div>'
        )

    # -- with and without the first round --
    mean_rows = []
    for key, label in _STRING_METRIC_LABELS.items():
        stats = breakdown.get(key)
        if not isinstance(stats, dict):
            continue
        mean_rows.append(
            f'<tr><td>{_e(label)}</td>'
            f'<td class="num">{_num(stats.get("energy_mean_dB"), 1)}</td>'
            f'<td class="num">{_num(stats.get("energy_mean_excluding_first_dB"), 1)}</td>'
            f'<td class="num"><strong>{_num(stats.get("first_round_cost_dB"), 2)}</strong></td>'
            f'</tr>')

    means_table = ""
    if mean_rows:
        means_table = f"""
  <div class="tablewrap">
    <table>
      <caption>A suppressor that pops has two honest averages: the one a shooter
      hears including the first round of the day, and the one they hear thereafter.
      Quoting only the second is the commonest way a suppressor test flatters its
      subject, so both are given. All levels in {_e(unit)}.</caption>
      <thead><tr>
        <th>Metric</th><th class="num">All shots</th>
        <th class="num">Excluding first</th><th class="num">First-round cost</th>
      </tr></thead>
      <tbody>{''.join(mean_rows)}</tbody>
    </table>
  </div>"""

    # -- distribution and drift --
    spread_rows = []
    if isinstance(headline, dict):
        percentiles = headline.get("percentiles_dB") or {}
        percentile_text = "  ".join(
            f"p{k} {_num(v, 1)}" for k, v in
            sorted(percentiles.items(), key=lambda kv: float(kv[0]))
        ) if percentiles else "—"
        trend = headline.get("trend_dB_per_shot")
        if trend is None:
            trend_text = "not tested (too few shots)"
        elif headline.get("trend_established"):
            trend_text = (
                f'{_num(trend, 3)} dB per shot — <span class="flag warn">drift '
                f'established</span> (p = {_num(headline.get("trend_p_value"), 3)})'
            )
        else:
            trend_text = (
                f'{_num(trend, 3)} dB per shot — no drift established '
                f'(p = {_num(headline.get("trend_p_value"), 3)})'
            )
        for label, value in (
            ("Spread", f'{_num(headline.get("min_dB"), 1)} to '
                       f'{_num(headline.get("max_dB"), 1)} {unit} '
                       f'(range {_num(headline.get("range_dB"), 2)} dB)'),
            ("Standard deviation", f'{_num(headline.get("sd_dB"), 2)} dB'),
            ("Percentiles", percentile_text),
            ("Drift across the string", trend_text),
        ):
            spread_rows.append(
                f'<tr><td class="field">{_e(label)}</td><td>{value}</td></tr>')

    spread_table = ""
    if spread_rows:
        spread_table = f"""
  <div class="tablewrap">
    <table>
      <caption>Distribution of the {_metric_label("Lpeak_Z")} across the string.
      Drift is measured from the second shot onward, so first-round pop cannot be
      read as a heating trend.</caption>
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>{''.join(spread_rows)}</tbody>
    </table>
  </div>"""

    # -- shots flagged for review --
    flags = [f for f in _seq(review, "flags")
             if isinstance(f, dict) and f.get("severity") in ("exclude", "review")]
    if flags:
        items = "".join(
            f'<li><strong>Shot {_e(_txt(f.get("shot_number"), missing="?"))}</strong> '
            f'<span class="flag {"fail" if f.get("severity") == "exclude" else "warn"}">'
            f'{_e(str(f.get("severity")))}</span> {_e(str(f.get("message", "")))}</li>'
            for f in flags
        )
        review_block = f"""
  <h3>Shots flagged for review</h3>
  <ul class="findings">{items}</ul>"""
    else:
        review_block = (
            '<h3>Shots flagged for review</h3>'
            '<div class="note ok">No shot departs from the string.</div>'
        )

    sensitivity = review.get("sensitivity")
    sensitivity_block = (
        f'<div class="note">{_e(str(sensitivity))}</div>' if sensitivity else ""
    )

    return f"""
<section class="pagebreak">
  <h2>String behaviour</h2>
  <h3>First-round pop</h3>
  {pop_block}
  {single_string_caveat}
  {means_table}
  <h3>Distribution and drift</h3>
  {spread_table}
  {review_block}
  {sensitivity_block}
</section>
"""


def _comparability_block(metadata: Dict[str, Any]) -> str:
    """
    Whether the reference and test strings were the same experiment.

    Insertion loss inherits every way in which they were not, so the objections
    are stated with the number of decibels each one is worth wherever the physics
    can price it.
    """
    comparability = _map(metadata, "insertion_loss", "comparability")
    if not comparability:
        return ""

    objections = [o for o in _seq(comparability, "objections") if isinstance(o, dict)]
    if not objections:
        return ('<div class="note ok"><strong>The two strings describe the same '
                'experiment.</strong> No objection was raised against this '
                'comparison.</div>')

    blocking = [o for o in objections if o.get("severity") == "blocking"]
    header = ""
    if blocking:
        header = (
            '<div class="note danger"><strong>This is not a valid insertion loss.</strong> '
            'The two strings did not measure the same thing, so the difference between '
            'them is not attributable to the suppressor. The figures below are the '
            'arithmetic difference only.</div>'
        )

    rows = []
    for objection in sorted(
        objections,
        key=lambda o: {"blocking": 0, "material": 1, "advisory": 2}.get(o.get("severity"), 3),
    ):
        severity = str(objection.get("severity", "advisory"))
        flag_class = {"blocking": "fail", "material": "warn"}.get(severity, "")
        amount = objection.get("quantified_dB")
        amount_text = _num(amount, 2) if _is_num(amount) else "—"
        correctable = "yes" if objection.get("correctable") else "no"
        rows.append(
            f'<tr><td><span class="flag {flag_class}">{_e(severity)}</span></td>'
            f'<td>{_e(str(objection.get("message", "")))}</td>'
            f'<td class="num">{amount_text}</td>'
            f'<td>{correctable}</td></tr>')

    unexplained = comparability.get("unexplained_dB")
    total = ""
    if _is_num(unexplained) and float(unexplained) > 0:
        total = (
            f'<div class="note warn"><strong>{_num(unexplained, 2)} dB of the reported '
            f'reduction is attributable to the two setups not matching, not to the '
            f'suppressor.</strong></div>')

    return f"""
  <h3>Is this a valid comparison?</h3>
  {header}
  {total}
  <div class="tablewrap">
    <table>
      <caption>Every way in which the reference and test strings differed, and how
      many decibels each difference is worth where the physics can price it.
      "Correctable" means a stated correction exists; an uncorrectable difference
      cannot be removed from the result at all.</caption>
      <thead><tr>
        <th>Severity</th><th>Objection</th><th class="num">Worth (dB)</th>
        <th>Correctable</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>"""


def _normalised_bands_block(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    """Per-band insertion loss referred to a common distance and atmosphere."""
    normalised = _map(metadata, "insertion_loss", "bands_normalised")
    if not normalised:
        return ""

    if not normalised.get("valid"):
        refusal = normalised.get("refusal") or "not attempted"
        return (
            f'<div class="note warn"><strong>Insertion loss was not normalised to a '
            f'common distance.</strong> {_e(str(refusal))} The per-band figures above '
            f'are as measured, so any difference in microphone position or weather '
            f'between the two strings is still in them.</div>')

    freqs = _seq(normalised, "frequencies_Hz")
    raw = _seq(normalised, "raw_insertion_loss_dB")
    norm = _seq(normalised, "insertion_loss_dB")
    shift = _seq(normalised, "shift_dB")
    if not (freqs and norm):
        return ""

    rows = []
    for i, frequency in enumerate(freqs):
        rows.append(
            f'<tr><td class="num">{_num(frequency, 0)}</td>'
            f'<td class="num">{_num(raw[i] if i < len(raw) else None, 1)}</td>'
            f'<td class="num"><strong>{_num(norm[i], 1)}</strong></td>'
            f'<td class="num">{_num(shift[i] if i < len(shift) else None, 2)}</td></tr>')

    warnings = "".join(
        f'<div class="note warn">{_e(str(w))}</div>'
        for w in _seq(normalised, "warnings"))
    assumptions = "".join(
        f'<li>{_e(str(a))}</li>' for a in _seq(normalised, "assumptions"))
    assumption_block = (
        f'<p class="sub" style="font-size:13px">Assumptions: </p><ul class="findings">'
        f'{assumptions}</ul>' if assumptions else "")

    distance = normalised.get("normalisation_distance_m")
    ref_d = normalised.get("reference_distance_m")
    test_d = normalised.get("test_distance_m")

    return f"""
  <h3>Insertion loss referred to a common distance</h3>
  <div class="note">
    The reference was measured at {_e(_num(ref_d, 2))} m and the test at
    {_e(_num(test_d, 2))} m. Both spectra have been referred to
    {_e(_num(distance, 2))} m using free-field spherical spreading and each string's
    own ISO 9613-1 atmospheric absorption, so the difference below is the suppressor
    rather than the setup. The measured figures are retained alongside, because a
    corrected number that cannot be checked against the measurement it came from is
    not a measurement record.
  </div>
  {warnings}
  <div class="tablewrap">
    <table>
      <caption>Per-band insertion loss as measured, and referred to
      {_e(_num(distance, 2))} m. All levels in {_e(validity.level_unit)}.</caption>
      <thead><tr>
        <th class="num">Band (Hz)</th><th class="num">As measured</th>
        <th class="num">At {_e(_num(distance, 2))} m</th><th class="num">Shift</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  {assumption_block}"""


def _section_hazard(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    hazard = _dig(metadata, "aggregate", "hazard", default=None)
    if not isinstance(hazard, dict) or not hazard:
        return """
<section>
  <h2>Hearing hazard</h2>
  <div class="note">No hearing-hazard assessment was computed for this measurement.</div>
</section>
"""

    if not validity.calibrated:
        return """
<section>
  <h2>Hearing hazard</h2>
  <div class="note danger"><strong>No hazard assessment can be made.</strong>
  A hearing-hazard criterion is an absolute sound pressure limit, and this analysis
  is uncalibrated. Any dose computed from relative levels would be meaningless, so
  none is reported.</div>
</section>
"""

    laeq8h = hazard.get("LAeq8h_dB")
    criterion = hazard.get("criterion_dB")
    dose = hazard.get("dose_percent")
    allowable = hazard.get("allowable_rounds")
    nrr = hazard.get("protection_NRR_dB")
    exceeds = bool(hazard.get("exceeds_limit"))
    method = hazard.get("method") or ("Energy-based LAeq8h, 3 dB exchange rate "
                                      "(MIL-STD-1474E / NIOSH)")

    verdict_class = "danger" if exceeds else "warn"
    if exceeds:
        verdict = (f"The daily noise dose for {_txt(hazard.get('n_rounds'))} "
                   f"{plural(hazard.get('n_rounds') or 0, 'round')} "
                   f"EXCEEDS the {_num(criterion, 0)} dB criterion.")
    else:
        verdict = (f"The daily noise dose for {_txt(hazard.get('n_rounds'))} "
                   f"{plural(hazard.get('n_rounds') or 0, 'round')} is "
                   f"within the {_num(criterion, 0)} dB criterion.")

    protection = ("no hearing protection assumed" if not _is_num(nrr) or float(nrr) == 0.0
                  else f"assuming hearing protection rated NRR {_num(nrr, 0)} dB")

    return f"""
<section>
  <h2>Hearing hazard</h2>
  <div class="note {verdict_class}"><strong>{_e(verdict)}</strong> Computed with
  {_e(protection)}.</div>
  <div class="metrics">
    <div class="metric">
      <div class="name">L<sub>Aeq,8h</sub></div>
      <div class="value">{_num(laeq8h)}<span class="unit">dB</span></div>
      <div class="ci">criterion {_num(criterion, 0)} dB</div>
      <div class="desc">Energy of the whole string spread over an 8-hour day.</div>
    </div>
    <div class="metric">
      <div class="name">Daily dose</div>
      <div class="value">{_num(dose)}<span class="unit">%</span></div>
      <div class="ci">100% = the criterion</div>
      <div class="desc">Fraction of the permissible daily noise dose used.</div>
    </div>
    <div class="metric">
      <div class="name">Allowable rounds</div>
      <div class="value">{_num(allowable, 0)}</div>
      <div class="ci">per person per day</div>
      <div class="desc">Rounds at this exposure before the criterion is reached.</div>
    </div>
  </div>
  <p class="sub" style="font-size:13px">Method: {_e(method)}. This is the
  better-supported of the two impulse-noise metrics MIL-STD-1474E approves - see
  the methods appendix for what it does and does not establish.</p>
</section>
"""


def _section_shots(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    shots = _seq(metadata, "per_shot_metrics")
    events = {}
    for event in (_seq(metadata, "shots")):
        if isinstance(event, dict) and event.get("shot_number") is not None:
            events[event["shot_number"]] = event

    if not shots:
        return """
<section>
  <h2>Per-shot results</h2>
  <div class="note">No per-shot metrics are present in this analysis record.</div>
</section>
"""

    unit = validity.level_unit
    rows = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        number = shot.get("shot_number")
        event = events.get(number, {})

        flags = []
        if shot.get("clipped"):
            flags.append('<span class="flag fail">clipped</span>')
        if shot.get("window_truncated"):
            flags.append('<span class="flag warn">truncated</span>')
        if shot.get("rise_time_resolved") is False:
            flags.append('<span class="flag warn">rise unresolved</span>')
        if len(_seq(event, "arrivals")) > 1:
            flags.append('<span class="flag warn">multiple arrivals</span>')
        if shot.get("valid") is False:
            flags.insert(0, '<span class="flag fail">excluded</span>')
        if not flags:
            flags.append('<span class="flag ok">ok</span>')

        notes = _seq(shot, "notes")
        note_text = "; ".join(str(n) for n in notes) if notes else ""

        rows.append(
            f'<tr>'
            f'<td class="num">{_txt(number, missing="—")}</td>'
            f'<td class="num">{_num(event.get("time_s"), 3)}</td>'
            f'<td class="num">{_num(shot.get("Lpeak_C_dB"))}</td>'
            f'<td class="num">{_num(shot.get("Lpeak_Z_dB"))}</td>'
            f'<td class="num">{_num(shot.get("LAE_dB"))}</td>'
            f'<td class="num">{_num(shot.get("LAImax_dB"))}</td>'
            f'<td class="num">{_num(shot.get("rise_time_us"), 1)}</td>'
            f'<td class="num">{_num(shot.get("b_duration_ms"), 2)}</td>'
            f'<td class="num">{_num(shot.get("snr_dB"), 1)}</td>'
            f'<td>{"".join(flags)}</td>'
            f'<td>{_e(note_text)}</td>'
            f'</tr>')

    return f"""
<section class="pagebreak">
  <h2>Per-shot results</h2>
  <div class="tablewrap">
    <table>
      <caption>Every detected shot, including any excluded from the aggregate
      statistics. Levels in {_e(unit)}. A shot flagged "clipped" has a censored
      peak: its true level is at least the value shown.</caption>
      <thead><tr>
        <th class="num">#</th><th class="num">t (s)</th>
        <th class="num">L<sub>Cpeak</sub></th><th class="num">L<sub>peak,Z</sub></th>
        <th class="num">L<sub>AE</sub></th><th class="num">L<sub>AImax</sub></th>
        <th class="num">Rise (µs)</th><th class="num">B-dur (ms)</th>
        <th class="num">SNR (dB)</th><th>Validity</th><th>Notes</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>
"""


def _section_figures(figures: List[Tuple[str, Path]], validity: ValidityReport) -> str:
    blocks = []
    skipped: List[str] = []
    for title, path in figures:
        uri = _embed_figure(path)
        if uri is None:
            skipped.append(path.name)
            continue
        blocks.append(f"""
<figure>
  <img src="{uri}" alt="{_e(title)}" />
  <figcaption><strong>{_e(title)}</strong> &middot;
  <span class="mono">{_e(path.name)}</span> &middot; levels in
  {_e(validity.level_unit)}</figcaption>
</figure>""")

    if not blocks:
        return """
<section>
  <h2>Figures</h2>
  <div class="note">No figures were embedded in this report.</div>
</section>
"""

    note = ""
    if skipped:
        note = (f'<div class="note warn">Not embedded (unsupported type or too large): '
                f'{_e(", ".join(skipped))}. Interactive HTML plots are not embedded, '
                f'because they are separate documents; they remain in the analysis '
                f'directory alongside this report.</div>')

    return f"""
<section class="pagebreak">
  <h2>Figures</h2>
  {note}
  {''.join(blocks)}
</section>
"""


def _section_appendix(metadata: Dict[str, Any], validity: ValidityReport) -> str:
    settings = _map(metadata, "settings")
    software = _map(metadata, "software")
    libraries = _map(software, "libraries")

    setting_rows = []
    for key in sorted(settings):
        value = settings[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)[:400]
        setting_rows.append(f'<tr><td class="field mono">{_e(key)}</td>'
                            f'<td class="mono">{_e(value)}</td></tr>')

    lib_text = ", ".join(f"{k} {v}" for k, v in sorted(libraries.items())) or "not recorded"

    calibration = _map(metadata, "calibration")
    cal_desc = _txt(calibration.get("description"))

    return f"""
<section class="pagebreak">
  <h2>Methods and limitations</h2>

  <h3>Standards applied</h3>
  <dl class="appendix">
    <dt>IEC 61672-1 — Sound level meters</dt>
    <dd>A-, C- and Z-frequency weightings, and the F (Fast, 125 ms), S (Slow, 1 s) and
    I (Impulse, 35 ms rise / 1500 ms decay) time weightings, are implemented to the
    definitions in this standard. Weighting filters are applied as a single causal
    pass, so no acausal pre-ringing is introduced ahead of the blast front.</dd>

    <dt>IEC 61260-1 — Octave-band and fractional-octave-band filters</dt>
    <dd>The 1/3-octave filter bank used for band levels and band insertion loss is
    designed to this standard's class tolerances, with per-band decimation so that
    every band is filtered at an appropriate rate.</dd>

    <dt>ISO 266 — Preferred frequencies</dt>
    <dd>Band centre frequencies are the ISO 266 nominal series; the exact midband
    frequencies used by the filters are the base-10 values of IEC 61260-1.</dd>

    <dt>MIL-STD-1474E / NIOSH — Hearing-hazard criterion</dt>
    <dd>The daily dose is the energy-based 8-hour equivalent level (L<sub>Aeq,8h</sub>)
    with a 3 dB exchange rate against an 85 dB criterion.</dd>

    <dt>IEC 60942 — Sound calibrators</dt>
    <dd>Where a calibrator tone was used, calibration is derived from the recorded
    level of a calibrator of this class, capturing the entire acquisition chain as
    it was configured for the test.</dd>
  </dl>

  <h3>Limitations — stated plainly</h3>
  <dl class="appendix">
    <dt>Why the energy metric is the one reported.</dt>
    <dd>MIL-STD-1474E approves two impulse-noise metrics: the A-weighted energy
    method reported here, and the Auditory Risk Unit from ARL's AHAAH model. This
    instrument computes the energy method, and that is a deliberate choice. The
    2010 AIBS independent peer review, convened to compare four impulse-noise
    models, recommended the energy method as the standard until AHAAH could be
    verified, judged the correlation between AHAAH's predictions and observed
    hearing damage to be weak, and found that AHAAH handles repeated exposures
    less well than the energy method - which is the governing case for a string of
    shots. Later comparisons against human threshold-shift data ranked AHAAH the
    poorest of the three criteria tested and the energy method the best.</dd>

    <dt>The hazard figure is a screening criterion, not a compliance
    determination.</dt>
    <dd>MIL-STD-1474E specifies its energy metric, L<sub>IAeq,8h</sub>, on a 100 ms
    interval around each impulse, with a correction for long A-duration impulses.
    This instrument integrates each shot over its detection window instead
    (50 ms before to 200 ms after the arrival, by default), and applies no
    A-duration correction. A longer window captures decay and reverberant tail that
    a 100 ms window truncates, so the figure reported here is at least as high as
    the standard's and never lower - conservative, but not the standard's number.
    It must not be presented as a MIL-STD-1474E compliance determination.</dd>

    <dt>Free-field assumptions.</dt>
    <dd>Levels are reported as measured at the microphone. No correction is applied
    for ground reflection, atmospheric absorption, microphone incidence response, or
    distance scaling. A level measured at one position does not transfer to another
    without those corrections.</dd>

    <dt>Peak levels depend on the sample rate.</dt>
    <dd>A muzzle blast rises in tens of microseconds. Sampling always understates the
    true peak, and by more at lower rates; peak levels from recordings at different
    sample rates are not directly comparable.</dd>

    <dt>Supersonic rounds produce two arrivals.</dt>
    <dd>Where a projectile is supersonic, the ballistic crack and the muzzle blast
    arrive separately. A suppressor acts only on the muzzle blast. Shots with more
    than one detected arrival are flagged in the per-shot table; a peak that is
    actually the crack credits the suppressor with nothing.</dd>

    <dt>Statistics describe this string only.</dt>
    <dd>Confidence intervals characterise shot-to-shot variability within the string
    that was recorded. They do not account for round-to-round ammunition lot
    variation, barrel temperature, or day-to-day atmospheric change.</dd>

    {'<dt>These levels are relative.</dt><dd>This analysis was uncalibrated. Levels are '
     'in dB re FS and are not sound pressure levels. No comparison with any external '
     'measurement, standard or limit is valid.</dd>' if not validity.calibrated else ''}
  </dl>

  <h3>Calibration record</h3>
  <div class="tablewrap">
    <table>
      <tbody>
        <tr><td class="field">Method</td><td>{_missing_span(_txt(calibration.get('method')))}</td></tr>
        <tr><td class="field">Description</td><td>{_missing_span(cal_desc)}</td></tr>
        <tr><td class="field">Pa per full scale</td><td class="mono">{_num(calibration.get('Pa_per_FS'), 4)}</td></tr>
        <tr><td class="field">Full-scale level</td><td class="mono">{_num(calibration.get('full_scale_dB'))} {_e(validity.level_unit)}</td></tr>
        <tr><td class="field">Level unit</td><td>{_e(validity.level_unit)}</td></tr>
      </tbody>
    </table>
  </div>

  <h3>Analysis settings</h3>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Setting</th><th>Value</th></tr></thead>
      <tbody>{''.join(setting_rows) or '<tr><td colspan="2">Not recorded.</td></tr>'}</tbody>
    </table>
  </div>
  <p class="sub" style="font-size:13px; margin-top:10px">
    Libraries: <span class="mono">{_e(lib_text)}</span> &middot;
    Python {_e(_txt(software.get('python_version')))} &middot;
    {_e(_txt(software.get('platform')))}
  </p>
</section>
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    metadata_dict: Dict[str, Any],
    output_path: str | Path,
    figures: Any = None,
) -> Path:
    """
    Write a self-contained HTML measurement report.

    Args:
        metadata_dict: An analysis record (schema 2.0). Older or partial records are
            accepted; anything absent is reported as "not recorded" rather than
            being omitted or invented.
        output_path: Where to write the .html file.
        figures: Optional figures to embed - a sequence of paths, a mapping of
            {caption: path}, or a sequence of (caption, path). When None, figures
            are discovered from the record's artifacts block and then from the
            analysis directory.

    Returns:
        The path written.
    """
    if not isinstance(metadata_dict, dict):
        raise TypeError("metadata_dict must be a dict")

    output_path = Path(output_path)
    base_dir = _figure_base_dir(metadata_dict, output_path)

    validity = assess_validity(metadata_dict)

    figure_items = _normalise_figures(figures, base_dir)
    if not figure_items:
        figure_items = _discover_figures(metadata_dict, base_dir)

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    schema = str(metadata_dict.get("schema_version") or "unknown")
    test_id = _txt(_dig(metadata_dict, "test_metadata", "test_id"), missing="")
    title = f"SASA Measurement Report{f' — {test_id}' if test_id else ''}"

    body = "\n".join([
        _section_title(metadata_dict, validity),
        _section_validity(validity),
        _section_conditions(metadata_dict),
        # Hazard leads the substantive sections. It is the conclusion the rest of
        # the report supports, and the metric the reader is here for; the levels
        # that produce it follow immediately after.
        _section_hazard(metadata_dict, validity),
        _section_results(metadata_dict, validity),
        _section_insertion_loss(metadata_dict, validity),
        _section_string_behaviour(metadata_dict, validity),
        _section_shots(metadata_dict, validity),
        _section_figures(figure_items, validity),
        _section_appendix(metadata_dict, validity),
    ])

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="generator" content="SASA report.py" />
<title>{_e(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="page">
{body}
<footer class="footer">
  Generated by SASA (Ridgeback Defense) on {_e(generated)} from an analysis record of
  schema version {_e(schema)}. This document is self-contained: all styles and images
  are embedded and it makes no network requests. Levels are stated in
  {_e(validity.level_unit)}.
</footer>
</div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _figure_base_dir(metadata: Dict[str, Any], output_path: Path) -> Optional[Path]:
    """Where relative artifact paths are resolved from."""
    recorded = _dig(metadata, "analysis", "output_dir")
    if isinstance(recorded, str) and recorded:
        candidate = Path(recorded)
        if candidate.is_dir():
            return candidate
    parent = output_path.parent
    return parent if parent.exists() else None


def build_report_from_directory(
    analysis_dir: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Read analysis_metadata.json from a directory and write report.html into it."""
    analysis_dir = Path(analysis_dir)
    meta_path = analysis_dir / "analysis_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No analysis_metadata.json in {analysis_dir}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{meta_path} does not contain an analysis record")

    # The recorded output_dir may point at another machine's filesystem; the
    # directory we were actually handed is the authoritative one for figures.
    metadata.setdefault("analysis", {})
    if isinstance(metadata["analysis"], dict):
        recorded = metadata["analysis"].get("output_dir")
        if not recorded or not Path(str(recorded)).is_dir():
            metadata["analysis"]["output_dir"] = str(analysis_dir)

    target = Path(output_path) if output_path else analysis_dir / "report.html"
    return generate_report(metadata, target)


def check_self_contained(path: str | Path) -> List[str]:
    """
    Return every external reference found in a generated report.

    An empty list means the document renders identically with no network at all.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    findings: List[str] = []
    import re as _re
    for match in _re.finditer(r'(?:src|href|url\(|@import\s+)["\'(]?\s*(https?:)?//[^\s"\'()<>]+',
                              text, _re.IGNORECASE):
        findings.append(match.group(0)[:160])
    return findings


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML measurement report from a "
                    "SASA analysis directory.",
    )
    parser.add_argument("analysis_dir", type=Path,
                        help="Directory containing analysis_metadata.json")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output HTML path (default: <analysis_dir>/report.html)")
    parser.add_argument("--check", action="store_true",
                        help="Verify the output makes no external requests")
    args = parser.parse_args(argv)

    try:
        path = build_report_from_directory(args.analysis_dir, args.output)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    size_kb = path.stat().st_size / 1024.0
    print(f"Report written: {path}  ({size_kb:.1f} kB)")

    metadata = json.loads((Path(args.analysis_dir) / "analysis_metadata.json")
                          .read_text(encoding="utf-8"))
    validity = assess_validity(metadata)
    print(f"Validity: {validity.verdict.upper()} — {validity.headline}")
    print(f"Levels reported in: {validity.level_unit}")
    for reason in validity.blocking:
        print(f"  BLOCKING: {reason}")

    if args.check:
        external = check_self_contained(path)
        if external:
            print(f"NOT self-contained — {count(len(external), 'external reference')}:",
                  file=sys.stderr)
            for item in external[:20]:
                print(f"  {item}", file=sys.stderr)
            return 2
        print("Self-contained: no external references.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
