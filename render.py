"""
Per-shot figure rendering, moved off the main process.

WHY THIS MODULE EXISTS
----------------------
Drawing is what an analysis spends its time on. Measured on a ten-shot,
22-second string, the whole run is 5.0 s; with ``--no-plots`` it is 1.0 s, and
with ``--no-per-shot`` it is 2.0 s. So the per-shot summaries alone are 3.0 s
of a 5.0 s run -- 60% of it -- and they are the one cost that grows with the
length of the string. A fifty-shot string spends about fifteen seconds drawing
fifty near-identical pages.

The work is also almost entirely serial Python. The profile is dominated by
matplotlib's text layout and tick construction (``Text._get_layout``,
``Axis._update_ticks``), not by the Agg rasteriser or the PNG encoder, so the
GIL is held for most of it and threads would not help. Separate processes
would.

THE UNIT OF WORK
----------------
One shot, complete: read its window from the file, transform it twice, write
the two spectrogram matrices, draw the summary figure, save it. The alternative
-- computing in the parent and shipping the arrays out to be drawn -- would put
a megabyte or two per shot through a pickle for no reason. Here the worker
opens the WAV itself and only the artifact NAMES come back, so the cost of
using a second process is the cost of starting it, not the cost of feeding it.

Starting one is the whole difficulty. A worker is a fresh interpreter that must
import numpy, scipy and matplotlib before it can draw a line, which is roughly
a second of wall clock. That is why ``render_shots`` renders a short string
serially: below a handful of shots the string finishes before a pool would have
finished importing. It is also why the pool is created once for the whole
string rather than per figure.

FROZEN BUILDS
-------------
Under PyInstaller a worker is the frozen executable re-invoked, and it only
becomes a worker because ``multiprocessing.freeze_support()`` runs early enough
in the entry point to intercept it -- ``main.cli_main`` and ``app.__main__``
both call it. The child's ``sys.path`` is PyInstaller's, not the parent's, and
on macOS the analysis modules may be shipped to ``Contents/Resources`` rather
than to ``_MEIPASS``; so every job carries the directory it came from and
``_ensure_importable`` restores it before anything project-local is imported.

None of that is trusted. If a worker cannot start, cannot import, or dies, the
jobs it did not finish are rendered in this process instead and the run
produces exactly the same files, more slowly. Parallelism here is an
optimisation, and it is never allowed to become a way to lose a figure.

``SASA_RENDER_WORKERS`` overrides the worker count for a run. Setting it to 1
draws every figure in this process, which is both the kill switch for a machine
where the pool misbehaves and the way the two paths are compared: with the
footer clock masked (it is stamped from ``datetime.now()`` at save time, so no
two runs agree byte for byte), the pooled figures are pixel-identical to the
serial ones.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Below this many shots, a pool costs more than it saves: every worker pays the
# same one-second import before it draws anything, and a three-shot string is
# finished in about that time.
MIN_SHOTS_FOR_POOL = 4

# Beyond this, added workers stop paying for themselves -- each one still pays
# the full import, and the figures are small enough that the machine runs out
# of useful work before it runs out of cores.
MAX_WORKERS = 8


@dataclass
class ShotFigureJob:
    """
    One shot's figures, described in terms a worker process can be handed.

    Every field is a plain value, a path as text, or a dataclass of plain
    values. Nothing here closes over the parent: no open file, no reader
    function, no matplotlib object, no numpy array. That is what makes the job
    cheap to send and the worker independent of when the parent gets round to
    sending it.
    """

    source_dir: str
    wav_path: str
    out_dir: str
    shot_dir: str
    shot_number: int
    window_start: int
    window_end: int
    sample_rate: int
    nperseg: int
    channel: Optional[int]
    level_unit: str
    static_formats: List[str]
    config: Any            # main.AnalysisConfig
    calibration: Any       # calibration.Calibration
    metric: Any            # metrics.ShotMetrics

    # The footer every figure must carry. It lives in a module global in the
    # parent (plots._DEFAULT_PROVENANCE), and a worker is a fresh interpreter
    # where that global is unset -- so it travels with the job rather than
    # being assumed. Today the pipeline leaves it None and each figure builds
    # its own; carrying it means the day the pipeline does set one, the pooled
    # figures do not quietly become the only ones without it.
    provenance: Any = None      # plots.FigureProvenance
    level_unit_default: Optional[str] = None


@dataclass
class ShotFigureResult:
    """What came back from one job. Warnings travel with it; they are not printed."""

    shot_number: int
    artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    drawn: bool = False
    # Both result types carry a label so one caller can report either without
    # asking which it has.
    label: str = ""


@dataclass
class FigureJob:
    """
    A figure whose data the parent has already computed: only the drawing is left.

    The full-recording figures are described this way rather than like a shot,
    because their data is not reconstructible from a file offset -- the
    waveform is thinned around the shots, the spectrogram is taken over the
    focus span, the heatmap is a band analysis -- and recomputing any of it in
    a worker would mean two implementations of the same picture.

    Sending the data instead is cheap: a full-recording spectrogram pickles in
    under a millisecond, and drawing and saving it takes three hundred. What
    the parent keeps is the ordering guarantee that matters -- the matrix JSON
    is written before the job is ever submitted, so a figure that fails cannot
    take the data with it.
    """

    source_dir: str
    out_dir: str
    stem: str                     # written as out_dir/stem.<fmt>
    func_name: str                # a plots.py drawing function
    args: tuple
    kwargs: Dict[str, Any]
    formats: List[str]
    artifact_prefix: str
    label: str                    # what to call it if it goes wrong
    provenance: Any = None
    level_unit_default: Optional[str] = None


@dataclass
class FigureResult:
    """What came back from one FigureJob."""

    label: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    drawn: bool = False


def _ensure_importable(source_dir: str) -> None:
    """
    Put the analysis modules within reach of this interpreter.

    A no-op in the parent. In a spawned worker -- particularly a frozen one,
    where sys.path is PyInstaller's and the modules may have been shipped
    alongside the bundle rather than into it -- this is what makes the imports
    below resolve at all.
    """
    if source_dir and os.path.isdir(source_dir) and source_dir not in sys.path:
        sys.path.insert(0, source_dir)


def render_shot_figures(job: ShotFigureJob) -> ShotFigureResult:
    """
    Produce one shot's two spectrogram matrices and its summary figure.

    Runs in a worker process and must assume nothing about the parent beyond
    what ``job`` carries. Raises nothing: a failure is recorded in the result's
    warnings, because the parent has no stack to attach it to and the run must
    not end because one page of a fifty-page appendix would not draw.

    The matrices are written BEFORE the figure, deliberately, and the same way
    round as in the serial path: a matplotlib failure must not be able to take
    the data with it.
    """
    result = ShotFigureResult(shot_number=job.shot_number,
                              label=f"shot {job.shot_number} summary")
    try:
        _ensure_importable(job.source_dir)

        import numpy as np  # noqa: PLC0415

        import main as main_module  # noqa: PLC0415
        import plots as plot_module  # noqa: PLC0415
        from STFT import analyze_stft  # noqa: PLC0415

        # Reached through main rather than reimplemented. _spectrogram_matrix_block
        # is private, and using it here is deliberate: the quantised matrix is a
        # format the interface parses, and a second copy of that encoding would
        # be free to drift from the one the serial path writes.
        read_samples = main_module.read_samples

        _install_style(plot_module, job)

        calibration = job.calibration
        config = job.config
        n_frames = job.window_end - job.window_start
        # The parent may have had this span in memory already (the short-file
        # path preloads the channel). Reading it again costs a few milliseconds
        # and returns the same samples: preloading IS this call over the whole
        # file, with these arguments.
        block = read_samples(
            Path(job.wav_path), job.window_start, n_frames,
            channel=job.channel, mono_mix=config.mono_mix, dtype=config.load_dtype,
        )
        window = calibration.to_pascals(block)
        window_time = np.arange(window.size) / job.sample_rate

        shot_nperseg = min(job.nperseg, max(64, 1 << int(math.log2(max(window.size, 64)))))
        shot_noverlap = config.resolved_noverlap(shot_nperseg)
        stft_z = analyze_stft(window, job.sample_rate, nperseg=shot_nperseg,
                              noverlap=shot_noverlap, weighting="Z",
                              calibrated=calibration.calibrated)
        stft_c = analyze_stft(window, job.sample_rate, nperseg=shot_nperseg,
                              noverlap=shot_noverlap, weighting="C",
                              calibrated=calibration.calibrated)

        shot_dir = Path(job.shot_dir)
        for stft, suffix in ((stft_z, "z"), (stft_c, "c")):
            matrix = main_module._spectrogram_matrix_block(stft)
            if not matrix:
                continue
            matrix["time_offset_s"] = round(job.window_start / job.sample_rate, 6)
            name = f"shot_{job.shot_number:02d}_spectrogram_{suffix}.json"
            try:
                main_module.write_json(shot_dir / name, matrix)
                result.artifacts[f"shot_{job.shot_number:02d}_spectrogram_{suffix}"] = (
                    f"shots/{name}"
                )
            except OSError as exc:
                result.warnings.append(
                    f"The shot {job.shot_number} spectrogram could not be written: {exc}"
                )

        if job.static_formats:
            figure = plot_module.create_shot_summary_figure(
                window_time, window, stft_z, stft_c, job.metric,
                title=f"Shot {job.shot_number} Analysis ({job.level_unit})",
            )
            try:
                paths = plot_module.save_figure(
                    figure, shot_dir / f"shot_{job.shot_number:02d}_summary",
                    formats=job.static_formats,
                )
            finally:
                import matplotlib.pyplot as plt  # noqa: PLC0415

                plt.close(figure)
            if paths:
                result.artifacts[f"shot_{job.shot_number:02d}_summary"] = str(
                    paths[0].relative_to(Path(job.out_dir))
                )

        result.drawn = True
    except Exception as exc:  # noqa: BLE001 - the parent has no stack to attach this to
        result.warnings.append(f"Shot {job.shot_number} summary could not be produced: {exc}")
    return result


def _requested_workers() -> Optional[int]:
    """
    SASA_RENDER_WORKERS, if the operator set it.

    An escape hatch rather than a feature: 1 forces every figure to be drawn in
    this process, which is how a machine where the pool misbehaves keeps
    working, and how the serial and parallel paths are compared to each other.
    A value that is not a number is ignored rather than fatal -- a typo in an
    environment variable must not stop an analysis.
    """
    raw = os.environ.get("SASA_RENDER_WORKERS", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("SASA_RENDER_WORKERS=%r is not a number; ignoring it", raw)
        return None


def _install_style(plot_module, job) -> None:
    """
    Give this interpreter the parent's matplotlib settings.

    A worker has not been styled at all. Without this a pooled figure would
    differ from a serial one in every colour and font it uses, and would carry
    a different attribution footer.
    """
    plot_module.setup_plot_style()
    if job.provenance is not None:
        plot_module.set_default_provenance(job.provenance)
    if job.level_unit_default:
        plot_module.set_default_level_unit(job.level_unit_default)


def render_figure(job: FigureJob) -> FigureResult:
    """
    Draw and save one already-computed figure.

    Runs in a worker process. Raises nothing, for the same reason
    ``render_shot_figures`` raises nothing: the parent has no stack to attach a
    failure to, and one figure that will not draw must not end the run.
    """
    result = FigureResult(label=job.label)
    try:
        _ensure_importable(job.source_dir)

        import matplotlib.pyplot as plt  # noqa: PLC0415

        import plots as plot_module  # noqa: PLC0415

        _install_style(plot_module, job)

        func = getattr(plot_module, job.func_name, None)
        if func is None:
            # plots.py has been rewritten alongside this before; a function
            # that has not landed yet is a skipped figure, not a failure.
            logger.info("plots.%s is not available in this build; skipping", job.func_name)
            return result

        figure = func(*job.args, **job.kwargs)
        if isinstance(figure, tuple):
            figure = figure[0]
        if figure is None:
            return result

        try:
            paths = plot_module.save_figure(
                figure, Path(job.out_dir) / job.stem, formats=job.formats,
            )
        finally:
            plt.close(figure)

        for path in paths:
            result.artifacts.setdefault(
                f"{job.artifact_prefix}_{path.suffix.lstrip('.')}", path.name
            )
        result.drawn = bool(paths)
    except Exception as exc:  # noqa: BLE001 - recorded, never raised across the pool
        result.warnings.append(f"{job.label} could not be produced: {exc}")
    return result


def _worker_count(n_jobs: int, requested: Optional[int]) -> int:
    """How many workers are worth starting for this many jobs."""
    if requested is None:
        requested = _requested_workers()
    if requested is not None:
        return max(1, min(int(requested), n_jobs))
    available = os.cpu_count() or 1
    # One core is left for the parent, which is still writing JSON and driving
    # progress while the workers draw.
    return max(1, min(MAX_WORKERS, available - 1, n_jobs))


def render_any(job: Any) -> Any:
    """
    Draw whatever kind of job this is.

    One entry point so that one pool can hold both kinds: a shot, which fetches
    its own audio and produces data as well as a picture, and a figure whose
    data the parent already computed.
    """
    if isinstance(job, ShotFigureJob):
        return render_shot_figures(job)
    return render_figure(job)


class FigurePool:
    """
    One pool for the whole plotting stage, or a stand-in that draws here.

    Every figure in a run goes through this. The full-recording figures are
    computed one after another and each takes about thirty milliseconds to
    compute and three hundred to draw; drawn in line, the parent spends nine
    tenths of that stage waiting for matplotlib while the machine's other cores
    do nothing. Submitted here, each is drawn while the parent computes the
    next -- and the pool is still warm when the per-shot summaries arrive, so
    those workers have already paid for their imports.

    That sharing is the reason this is a long-lived object rather than a call:
    a second pool for the per-shot stage would spend its entire saving starting
    up again.

    ``submit`` never raises and never blocks on the drawing. ``gather`` returns
    every result in submission order, having drawn here anything the pool did
    not deliver. A pool that cannot be created at all is not an error -- this
    object then draws everything itself, and the caller cannot tell the
    difference except by how long it took.
    """

    def __init__(
        self,
        *,
        workers: Optional[int] = None,
        expected: int = 0,
        on_result: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._explicit = workers if workers is not None else _requested_workers()
        self._expected = max(0, int(expected))
        self._on_result = on_result
        self._pool = None
        self._futures: List[Any] = []
        self._jobs: List[Any] = []
        self._done: Dict[int, Any] = {}

    def __enter__(self) -> "FigurePool":
        return self.start()

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False

    def start(self) -> "FigurePool":
        """
        Start the workers, if starting them is worth it.

        Separate from the context manager so a caller with a long stage to run
        can hold the pool across it without indenting the stage inside a
        ``with``. ``gather`` closes it either way.
        """
        n_workers = _worker_count(max(self._expected, 1), self._explicit)
        # The short-string rule is about economics, so an operator who names a
        # worker count overrides it: they have said what they want, and the
        # reason for the default no longer applies.
        if n_workers < 2 or (self._explicit is None and self._expected < MIN_SHOTS_FOR_POOL):
            return self
        try:
            import multiprocessing  # noqa: PLC0415
            from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

            # Spawn explicitly rather than by platform default. Fork is the
            # default on Linux and is not safe here: the parent has already
            # imported matplotlib and may be running threads, and a forked
            # child inherits both in a state it never initialised.
            context = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=context)
            logger.info("Figure pool started with %d workers for ~%d figures",
                        n_workers, self._expected)
        except Exception as exc:  # noqa: BLE001 - never fails the run, only slows it
            logger.warning("A figure pool was not available (%s); drawing in this process", exc)
            self._pool = None
        return self

    def close(self) -> None:
        """Release the workers. Idempotent, and never raises."""
        if self._pool is not None:
            pool, self._pool = self._pool, None
            try:
                pool.shutdown(wait=True)
            except Exception as exc:  # noqa: BLE001
                # Shutting down a pool that has already broken can raise, and
                # by this point every figure has been drawn one way or the
                # other. Failing here would lose a completed run to its own
                # cleanup.
                logger.warning("The figure pool did not shut down cleanly (%s)", exc)

    @property
    def parallel(self) -> bool:
        """True while there is a live pool to submit to."""
        return self._pool is not None

    def submit(self, job: Any) -> None:
        """
        Queue a job, or draw it here and now if there is no pool to queue it on.

        Drawing immediately in the serial case rather than deferring to gather()
        is deliberate: it releases the job's data as soon as the figure is
        written instead of holding every payload until the end of the stage,
        and it lets progress be reported as the work actually happens.

        A submission that fails -- a payload that will not pickle, a pool that
        has died since it was created -- costs the pool, not the figure.
        """
        index = len(self._jobs)
        self._jobs.append(job)
        if self._pool is None:
            self._finish(index, render_any(job))
            self._futures.append(None)
            return
        try:
            self._futures.append(self._pool.submit(render_any, job))
        except Exception as exc:  # noqa: BLE001
            logger.warning("A figure could not be queued (%s); drawing it in this process", exc)
            self._finish(index, render_any(job))
            self._futures.append(None)

    def gather(self) -> List[Any]:
        """
        Every result, in submission order, with nothing left owed.

        Results are collected as they finish rather than in order, so a caller
        driving a progress bar sees the string advance instead of waiting on
        whichever figure happens to be slowest. The list itself is still in
        submission order, so it can be zipped against the jobs.
        """
        pending = {future: index
                   for index, future in enumerate(self._futures) if future is not None}
        if pending:
            try:
                from concurrent.futures import as_completed  # noqa: PLC0415

                for future in as_completed(list(pending)):
                    index = pending[future]
                    try:
                        self._finish(index, future.result())
                    except Exception as exc:  # noqa: BLE001
                        # One worker died. Its job is redrawn below with the
                        # rest of whatever this pool did not deliver.
                        logger.warning("A figure worker did not return (%s); "
                                       "it will be redrawn here", exc)
            except Exception as exc:  # noqa: BLE001 - a broken pool, not a broken run
                logger.warning("The figure pool stopped responding (%s); "
                               "the rest is being drawn in this process", exc)

        missing = [index for index in range(len(self._jobs)) if index not in self._done]
        if missing:
            logger.info("%d of %d figures are being drawn in this process",
                        len(missing), len(self._jobs))
            for index in missing:
                self._finish(index, render_any(self._jobs[index]))

        # Nothing is owed now, so the workers are not needed. Closing here as
        # well as in __exit__ means a caller that holds the pool across a stage
        # rather than inside a `with` still releases it.
        self.close()
        return [self._done[index] for index in range(len(self._jobs))]

    def _finish(self, index: int, result: Any) -> None:
        self._done[index] = result
        if self._on_result is not None:
            self._on_result(result)


def render_shots(
    jobs: Sequence[ShotFigureJob],
    *,
    workers: Optional[int] = None,
    on_result: Optional[Callable[[ShotFigureResult], None]] = None,
) -> List[ShotFigureResult]:
    """
    Render a set of shots with a pool of their own.

    A convenience for callers with nothing else to draw. A caller that has
    other figures too should build one FigurePool and submit everything to it,
    so the workers are started once.
    """
    if not jobs:
        return []
    with FigurePool(workers=workers, expected=len(jobs), on_result=on_result) as pool:
        for job in jobs:
            pool.submit(job)
        return pool.gather()
