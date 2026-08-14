"""
test_render.py - that moving the drawing off this process cannot lose a figure.

render.py exists for speed, and speed is the only thing it is allowed to
change. Everything here is therefore about the contract rather than the
picture: that the jobs come back in the order they were given, that a worker
which dies costs a warning and not the run, that a pool which cannot be created
at all is indistinguishable from never having asked for one, and that a short
string does not pay for machinery it would not finish using.

The figures themselves are checked in test_plots.py, and the equivalence of the
two paths is a pixel comparison run against real output rather than a unit test
-- the footer carries a clock, so no two runs of anything agree byte for byte.
"""

from __future__ import annotations

import os
from typing import List

import pytest

import render
from render import ShotFigureJob, ShotFigureResult


def make_job(shot_number: int) -> ShotFigureJob:
    """A job that carries a number and nothing that would survive being run."""
    return ShotFigureJob(
        source_dir="",
        wav_path="/nonexistent.wav",
        out_dir="/out",
        shot_dir="/out/shots",
        shot_number=shot_number,
        window_start=0,
        window_end=1000,
        sample_rate=48000,
        nperseg=1024,
        channel=0,
        level_unit="dB SPL",
        static_formats=["png"],
        config=None,
        calibration=None,
        metric=None,
    )


@pytest.fixture(autouse=True)
def _clear_worker_env(monkeypatch):
    monkeypatch.delenv("SASA_RENDER_WORKERS", raising=False)


# ── The worker's failure contract ────────────────────────────────────────────

def test_worker_records_failure_instead_of_raising():
    """
    A job that cannot possibly succeed must come back, not blow up.

    The parent collects results from a pool; an exception crossing that
    boundary would take the remaining shots with it.
    """
    result = render.render_shot_figures(make_job(7))

    assert isinstance(result, ShotFigureResult)
    assert result.shot_number == 7
    assert result.drawn is False
    assert result.warnings, "a failure must say what failed"
    assert "Shot 7" in result.warnings[0]
    assert result.artifacts == {}


def test_every_job_appears_exactly_once_and_in_order():
    """The returned list is zippable against the jobs it was given."""
    jobs = [make_job(n) for n in (3, 1, 2)]

    results = render.render_shots(jobs)

    assert [r.shot_number for r in results] == [3, 1, 2]


def test_on_result_fires_once_per_job():
    seen: List[int] = []
    jobs = [make_job(n) for n in range(1, 4)]

    render.render_shots(jobs, on_result=lambda r: seen.append(r.shot_number))

    assert sorted(seen) == [1, 2, 3]


def test_no_jobs_is_not_an_error():
    assert render.render_shots([]) == []


# ── When to spend a process, and when not to ─────────────────────────────────

def test_short_string_never_starts_a_pool(monkeypatch):
    """
    Below the threshold the string is drawn here.

    A worker costs about a second of imports before it draws anything, which is
    longer than a three-shot string takes in total.
    """
    def explode(*_args, **_kwargs):
        raise AssertionError("a pool was created for a string too short to need one")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", explode)
    jobs = [make_job(n) for n in range(render.MIN_SHOTS_FOR_POOL - 1)]

    results = render.render_shots(jobs)

    assert len(results) == len(jobs)


def test_workers_1_forces_the_serial_path(monkeypatch):
    """SASA_RENDER_WORKERS=1 is the kill switch; it must actually kill it."""
    def explode(*_args, **_kwargs):
        raise AssertionError("a pool was created despite SASA_RENDER_WORKERS=1")

    monkeypatch.setenv("SASA_RENDER_WORKERS", "1")
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", explode)
    jobs = [make_job(n) for n in range(render.MIN_SHOTS_FOR_POOL + 4)]

    results = render.render_shots(jobs)

    assert len(results) == len(jobs)


def test_an_unparseable_worker_count_is_ignored_not_fatal(monkeypatch):
    """A typo in an environment variable must not stop an analysis."""
    monkeypatch.setenv("SASA_RENDER_WORKERS", "lots")

    assert render._requested_workers() is None


def test_worker_count_never_exceeds_the_work(monkeypatch):
    monkeypatch.setenv("SASA_RENDER_WORKERS", "64")

    assert render._worker_count(3, None) == 3


def test_worker_count_leaves_the_parent_a_core():
    """The parent is still writing JSON and driving progress while they draw."""
    count = render._worker_count(1000, None)

    assert 1 <= count <= render.MAX_WORKERS
    assert count <= max(1, (os.cpu_count() or 1) - 1)


# ── The fallback ─────────────────────────────────────────────────────────────

def test_a_pool_that_cannot_be_created_falls_back_silently(monkeypatch):
    """
    Parallelism is an optimisation. Losing it costs time, never a figure.

    This is the frozen-build case: a worker that cannot import the analysis
    modules breaks the pool, and every job must still be drawn here.
    """
    def refuse(*_args, **_kwargs):
        raise OSError("cannot allocate a process")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", refuse)
    jobs = [make_job(n) for n in range(render.MIN_SHOTS_FOR_POOL + 2)]
    seen: List[int] = []

    results = render.render_shots(jobs, on_result=lambda r: seen.append(r.shot_number))

    assert [r.shot_number for r in results] == [j.shot_number for j in jobs]
    assert sorted(seen) == sorted(j.shot_number for j in jobs)


def test_jobs_a_broken_pool_did_not_reach_are_drawn_here(monkeypatch):
    """
    A pool that dies half way leaves the rest owed, and they are still owed.

    The pool is made to hand back one result and then break, which is what a
    worker segfaulting mid-string looks like from up here.
    """
    delivered = {"n": 0}
    finished = ShotFigureResult(shot_number=1, artifacts={"shot_01_summary": "x.png"},
                                drawn=True)

    class HalfBrokenPool:
        def __init__(self, *_args, **_kwargs):
            pass

        def shutdown(self, wait=True):
            # A pool that has lost a worker often raises here too; the run is
            # finished by this point and must not die in its own cleanup.
            raise RuntimeError("shutdown of a broken pool")

        def submit(self, _fn, job):
            class _Future:
                def __init__(self, shot_number):
                    self.shot_number = shot_number

                def result(self_inner):
                    if delivered["n"] == 0:
                        delivered["n"] += 1
                        return finished
                    raise RuntimeError("a process in the process pool was terminated")

            return _Future(job.shot_number)

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", HalfBrokenPool)
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: list(fs))
    jobs = [make_job(n) for n in range(1, render.MIN_SHOTS_FOR_POOL + 2)]

    results = render.render_shots(jobs)

    assert [r.shot_number for r in results] == [j.shot_number for j in jobs]
    # The one the pool did deliver is kept; the rest were re-drawn here, where
    # they fail for their own reasons and say so rather than going missing.
    assert all(isinstance(r, ShotFigureResult) for r in results)


# ── The job itself ───────────────────────────────────────────────────────────

def test_a_job_is_picklable():
    """
    Whatever a job carries has to survive being sent to another interpreter.

    A field that cannot be pickled would not fail here, it would fail on a
    customer's fifty-shot string and silently fall back to serial.
    """
    import pickle

    from calibration import Calibration
    from main import AnalysisConfig

    job = make_job(1)
    job.config = AnalysisConfig()
    job.calibration = Calibration(Pa_per_FS=1.0)

    restored = pickle.loads(pickle.dumps(job))

    assert restored.shot_number == 1
    assert restored.calibration.Pa_per_FS == 1.0
    assert restored.config.mono_mix == job.config.mono_mix


def test_ensure_importable_is_a_no_op_for_a_path_that_is_not_there():
    import sys

    before = list(sys.path)
    render._ensure_importable("/no/such/directory")

    assert sys.path == before


# ── Figures whose data the parent already computed ───────────────────────────

def make_figure_job(**overrides) -> render.FigureJob:
    job = render.FigureJob(
        source_dir="",
        out_dir="/out",
        stem="thing",
        func_name="plot_waveform_pa",
        args=(),
        kwargs={},
        formats=["png"],
        artifact_prefix="thing",
        label="a figure",
    )
    for key, value in overrides.items():
        setattr(job, key, value)
    return job


def test_a_figure_that_plots_does_not_have_is_skipped_not_failed():
    """
    plots.py has been rewritten alongside this before.

    A drawing function that has not landed yet is a figure this build does not
    produce -- not an error, and not a warning the operator can act on.
    """
    result = render.render_figure(make_figure_job(func_name="plot_something_unwritten"))

    assert result.drawn is False
    assert result.warnings == []


def test_a_figure_that_raises_is_recorded_against_its_label():
    result = render.render_figure(make_figure_job(args=("not an array",)))

    assert result.drawn is False
    assert result.warnings and result.warnings[0].startswith("a figure could not be produced")


def test_pool_draws_here_when_it_has_no_workers(monkeypatch):
    """A FigurePool with nothing to justify a pool is still a working pool."""
    def explode(*_args, **_kwargs):
        raise AssertionError("a pool was started for one figure")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", explode)
    pool = render.FigurePool(expected=1).start()

    assert pool.parallel is False
    pool.submit(make_figure_job(func_name="plot_something_unwritten"))
    results = pool.gather()

    assert len(results) == 1


def test_pool_reports_each_figure_once_as_it_finishes(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("a pool was started")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", explode)
    seen = []
    pool = render.FigurePool(expected=1, on_result=seen.append).start()
    for _ in range(3):
        pool.submit(make_figure_job(func_name="plot_something_unwritten"))

    results = pool.gather()

    assert len(seen) == 3
    assert len(results) == 3


def test_gather_closes_the_pool_and_close_is_idempotent(monkeypatch):
    closed = {"n": 0}

    class Pool:
        def __init__(self, *_a, **_k):
            pass

        def submit(self, fn, job):
            raise RuntimeError("queue is full")

        def shutdown(self, wait=True):
            closed["n"] += 1

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Pool)
    pool = render.FigurePool(expected=render.MIN_SHOTS_FOR_POOL + 1, workers=4).start()
    assert pool.parallel is True

    pool.submit(make_figure_job(func_name="plot_something_unwritten"))
    pool.gather()
    pool.close()
    pool.close()

    assert closed["n"] == 1, "the workers are released once, not once per call"


def test_a_submission_that_will_not_queue_is_drawn_here(monkeypatch):
    """A payload that will not pickle costs the pool, not the figure."""
    class Pool:
        def __init__(self, *_a, **_k):
            pass

        def submit(self, _fn, _job):
            raise TypeError("cannot pickle this")

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Pool)
    pool = render.FigurePool(expected=render.MIN_SHOTS_FOR_POOL + 1, workers=4).start()
    pool.submit(make_figure_job(args=("not an array",)))

    results = pool.gather()

    assert len(results) == 1
    assert results[0].warnings, "it was drawn here, and here it failed for its own reason"


def test_render_any_sends_each_kind_to_its_own_renderer():
    shot = render.render_any(make_job(4))
    figure = render.render_any(make_figure_job(func_name="plot_something_unwritten"))

    assert isinstance(shot, ShotFigureResult)
    assert isinstance(figure, render.FigureResult)


def test_both_result_kinds_carry_a_label():
    """One caller reports either without asking which it has."""
    assert render.render_shot_figures(make_job(9)).label == "shot 9 summary"
    assert render.render_figure(make_figure_job()).label == "a figure"
