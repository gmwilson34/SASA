#!/usr/bin/env python3
"""
session.py - Whole-Range-Session Batching

A suppressor test is not one recording. It is a morning: a baseline string, then
a can, then another can, then the baseline again to prove the rig did not move.
Analysing those one at a time and pairing them by hand is where the operator's
day goes, and it is where mis-pairing happens - the wrong reference against the
wrong test produces a plausible insertion loss for an experiment nobody ran.

This module runs the whole session, pairs each suppressed string with its
unsuppressed reference by metadata, and reports what it did and what it refused
to do. It decides nothing on its own that it cannot defend:

  * Pairing is refused outright when two references match equally well, rather
    than picking one. See pairing.auto_pair.
  * A session-level trend is reported only when the baseline was shot more than
    once, because drift needs at least two observations of the same thing.
  * A recording that fails to analyse does not stop the session; it is recorded
    as failed and the rest continue.

The analysis itself is injected rather than imported, so this module stays
independent of the pipeline and can be exercised without touching audio.

Usage:
    from session import run_session
    from main import analyze_file

    result = run_session(paths, analyse=lambda p: analyze_file(p, config))
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from pairing import auto_pair
from stringstats import energy_average_dB, first_round_pop_across_strings

# Audio and video containers a range session is likely to hold.
SESSION_EXTENSIONS: tuple = (
    ".wav", ".flac", ".aiff", ".aif", ".mp4", ".mov", ".mkv", ".avi", ".mts", ".mxf",
)

# Strings needed before a session-level trend in the baseline is worth reporting.
MIN_BASELINES_FOR_TREND: int = 3


class SessionError(ValueError):
    """Raised when a session cannot be assembled at all."""


@dataclass
class SessionEntry:
    """One recording in a session, and what became of it."""
    path: Path
    index: int
    label: str = ""
    ok: bool = False
    error: str = ""
    record: Dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> Dict[str, Any]:
        block = self.record.get("test_metadata")
        return block if isinstance(block, dict) else {}

    @property
    def configuration(self) -> str:
        return str(self.metadata.get("configuration") or "").strip().lower()

    @property
    def n_valid(self) -> int:
        aggregate = self.record.get("aggregate") or {}
        try:
            return int(aggregate.get("n_valid") or 0)
        except (TypeError, ValueError):
            return 0

    def peak_levels(self) -> List[float]:
        """Per-shot Z-weighted peak levels, in shot order."""
        out: List[float] = []
        for shot in self.record.get("per_shot_metrics") or []:
            if not isinstance(shot, dict) or not shot.get("valid", True):
                continue
            value = shot.get("Lpeak_Z_dB")
            if isinstance(value, (int, float)) and math.isfinite(value):
                out.append(float(value))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "path": str(self.path),
            "label": self.label,
            "ok": self.ok,
            "error": self.error,
            "configuration": self.configuration,
            "n_valid_shots": self.n_valid,
            "output_dir": (self.record.get("analysis") or {}).get("output_dir", ""),
        }


@dataclass
class SessionResult:
    """Everything a range session produced."""
    entries: List[SessionEntry] = field(default_factory=list)
    pairings: List[Any] = field(default_factory=list)
    baseline_trend_dB_per_string: float = float("nan")
    baseline_levels_dB: List[float] = field(default_factory=list)
    baseline_labels: List[str] = field(default_factory=list)
    first_round_pop: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def analysed(self) -> List[SessionEntry]:
        return [e for e in self.entries if e.ok]

    @property
    def failed(self) -> List[SessionEntry]:
        return [e for e in self.entries if not e.ok]

    @property
    def paired(self) -> List[Any]:
        return [p for p in self.pairings if getattr(p, "paired", False)]

    @property
    def unpaired(self) -> List[Any]:
        return [p for p in self.pairings if not getattr(p, "paired", False)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_recordings": len(self.entries),
            "n_analysed": len(self.analysed),
            "n_failed": len(self.failed),
            "n_paired": len(self.paired),
            "n_unpaired": len(self.unpaired),
            "entries": [e.to_dict() for e in self.entries],
            "pairings": [p.to_dict() for p in self.pairings],
            "baseline": {
                "levels_dB": [round(v, 2) for v in self.baseline_levels_dB],
                "labels": list(self.baseline_labels),
                "trend_dB_per_string": (
                    None if math.isnan(self.baseline_trend_dB_per_string)
                    else round(self.baseline_trend_dB_per_string, 3)
                ),
            },
            "first_round_pop": (
                self.first_round_pop.to_dict() if self.first_round_pop else None
            ),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            f"  Session: {len(self.entries)} recording(s), "
            f"{len(self.analysed)} analysed, {len(self.failed)} failed"
        ]
        for entry in self.failed:
            lines.append(f"    FAILED  {entry.label}: {entry.error}")

        if self.pairings:
            lines.append(f"  Pairing: {len(self.paired)} of {len(self.pairings)} "
                         f"suppressed string(s) matched to a reference")
            for pairing in self.pairings:
                for line in pairing.summary().splitlines():
                    lines.append(f"  {line}")
        else:
            lines.append("  Pairing: no suppressed string was found to pair.")

        if math.isfinite(self.baseline_trend_dB_per_string):
            lines.append(
                f"  Baseline drift: {self.baseline_trend_dB_per_string:+.3f} dB per "
                f"string across {len(self.baseline_levels_dB)} unsuppressed string(s)"
            )
        elif self.baseline_levels_dB:
            lines.append(
                f"  Baseline drift: not tested - only "
                f"{len(self.baseline_levels_dB)} unsuppressed string(s), and drift "
                f"needs at least {MIN_BASELINES_FOR_TREND}"
            )

        if self.first_round_pop is not None:
            for line in self.first_round_pop.summary().splitlines():
                lines.append(f"  {line}")

        for note in self.notes:
            lines.append(f"    {note}")
        for warning in self.warnings:
            lines.append(f"    WARNING: {warning}")
        return "\n".join(lines)


def discover_recordings(directory: Path | str) -> List[Path]:
    """
    Find the recordings in a session directory, in a stable order.

    Sorted by name so a session runs the same way twice, which is what makes
    "the third string" mean the same thing on a re-run.
    """
    root = Path(directory)
    if not root.is_dir():
        raise SessionError(f"{root} is not a directory")
    found = [
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SESSION_EXTENSIONS
    ]
    return sorted(found, key=lambda p: p.name.lower())


def _label_for(entry: SessionEntry) -> str:
    test_id = str(entry.metadata.get("test_id") or "").strip()
    return test_id or entry.path.name


def run_session(
    paths: Sequence[Path | str],
    *,
    analyse: Callable[[Path], Dict[str, Any]],
    metadata_for: Optional[Callable[[Path], Dict[str, Any]]] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> SessionResult:
    """
    Analyse every recording in a session, then pair and summarise them.

    Args:
        paths: Recordings, in the order they were shot.
        analyse: Called with each path; must return an analysis record (the same
                 dict written to analysis_metadata.json). Raising is caught and
                 recorded as a failure for that recording only.
        metadata_for: Optional override supplying TestMetadata per path, used
                      when the metadata lives beside the audio rather than in the
                      record.
        progress: Optional callback(percent, message).

    Returns:
        SessionResult.
    """
    entries: List[SessionEntry] = []
    total = max(1, len(paths))

    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        entry = SessionEntry(path=path, index=index, label=path.name)
        if progress:
            progress(int(100 * index / total), f"Analysing {path.name}")
        try:
            record = analyse(path)
            if not isinstance(record, dict):
                raise SessionError(
                    f"analysis of {path.name} returned {type(record).__name__}, "
                    f"not an analysis record"
                )
            entry.record = record
            entry.ok = True
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the session
            entry.error = f"{type(exc).__name__}: {exc}"
            entry.ok = False
        entry.label = _label_for(entry)
        entries.append(entry)

    result = SessionResult(entries=entries)
    analysed = result.analysed

    if not analysed:
        result.warnings.append(
            "No recording in this session analysed successfully, so nothing could "
            "be paired or compared."
        )
        return result

    # -- pair suppressed strings to their references --
    metadatas = [
        (metadata_for(e.path) if metadata_for else e.metadata) for e in analysed
    ]
    aggregates = [_AggregateView(e.record) for e in analysed]
    labels = [e.label for e in analysed]

    unconfigured = [
        labels[i] for i, meta in enumerate(metadatas)
        if str(meta.get("configuration") or "").strip().lower()
        not in ("suppressed", "unsuppressed")
    ]
    if unconfigured:
        result.warnings.append(
            "These recordings have no usable 'configuration', so they can be "
            "neither a reference nor a test: " + ", ".join(unconfigured)
        )

    result.pairings = auto_pair(metadatas, labels=labels, aggregates=aggregates)

    # -- did the rig hold still across the session? --
    baselines = [
        (labels[i], analysed[i]) for i, meta in enumerate(metadatas)
        if str(meta.get("configuration") or "").strip().lower() == "unsuppressed"
    ]
    baseline_levels: List[float] = []
    baseline_labels: List[str] = []
    for label, entry in baselines:
        levels = entry.peak_levels()
        if levels:
            baseline_levels.append(energy_average_dB(levels))
            baseline_labels.append(label)

    result.baseline_levels_dB = baseline_levels
    result.baseline_labels = baseline_labels

    if len(baseline_levels) >= MIN_BASELINES_FOR_TREND:
        index = np.arange(len(baseline_levels), dtype=np.float64)
        slope, _ = np.polyfit(index, np.asarray(baseline_levels, dtype=np.float64), 1)
        result.baseline_trend_dB_per_string = float(slope)
        result.notes.append(
            "Baseline drift is measured on the unsuppressed strings only. They are "
            "the same experiment repeated, so any change across them is the rig, "
            "not the suppressor."
        )
    elif len(baseline_levels) == 2:
        result.notes.append(
            f"The baseline was shot twice and moved "
            f"{baseline_levels[1] - baseline_levels[0]:+.2f} dB between them. Two "
            f"points cannot establish a trend, but a large difference here means "
            f"the rig did not hold still."
        )

    # -- first-round pop across every suppressed string in the session --
    suppressed_strings = [
        analysed[i].peak_levels() for i, meta in enumerate(metadatas)
        if str(meta.get("configuration") or "").strip().lower() == "suppressed"
    ]
    suppressed_strings = [s for s in suppressed_strings if s]
    if suppressed_strings:
        result.first_round_pop = first_round_pop_across_strings(
            suppressed_strings, metric="Lpeak_Z"
        )

    return result


class _AggregateView:
    """
    Adapts an analysis record to the few attributes pairing.py reads.

    pairing.assess_comparability duck-types on `n_valid` and `band_frequencies`,
    so a whole AggregateMetrics does not need rebuilding from the record.
    """

    def __init__(self, record: Dict[str, Any]) -> None:
        aggregate = record.get("aggregate") or {}
        try:
            self.n_valid = int(aggregate.get("n_valid") or 0)
        except (TypeError, ValueError):
            self.n_valid = 0
        freqs = aggregate.get("band_frequencies_Hz") or []
        self.band_frequencies = np.asarray(freqs, dtype=float)


def compare_session_pairs(result: SessionResult) -> List[Dict[str, Any]]:
    """
    Re-state each successful pairing as a plain comparison row.

    The comparability report already travels with the pairing; this flattens it
    for a table or a session-level report.
    """
    rows: List[Dict[str, Any]] = []
    for pairing in result.paired:
        report = pairing.matched.report
        rows.append({
            "test": pairing.test_label,
            "reference": pairing.matched.reference_label,
            "match_score": round(pairing.matched.score, 2),
            "comparable": report.comparable,
            "unexplained_dB": round(report.unexplained_dB, 2),
            "n_objections": len(report.objections),
            "blocking": [o.message for o in report.blocking],
        })
    return rows


# ---- CLI for testing ----

def main() -> int:
    """List what a session directory holds and how it would pair, without analysing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Range-session batching")
    parser.add_argument("directory", type=Path, help="Session directory")
    parser.add_argument("--json", action="store_true", help="Emit the plan as JSON")
    args = parser.parse_args()

    try:
        paths = discover_recordings(args.directory)
    except SessionError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not paths:
        print(f"No recordings found in {args.directory}")
        return 1

    if args.json:
        print(json.dumps([str(p) for p in paths], indent=2))
    else:
        print(f"{len(paths)} recording(s) in {args.directory}:")
        for path in paths:
            print(f"  {path.name}")
        print("\nRun the session through main.py to analyse and pair them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
