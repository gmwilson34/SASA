"""
test_packaging.py - the installed distribution matches the repository.

SASA is a flat layout: every analysis module sits at the top level and is
imported by bare name. Two places decide what actually ships -- pyproject.toml's
py-modules for `pip install`, and sasa.spec's datas for the frozen app -- and
neither is checked by importing the code, because the test suite runs with the
repository root already on sys.path and so cannot tell the difference between a
module that was installed and one that merely exists on disk.

That gap shipped a broken release. The py-modules list had gone stale by eight
modules (ahaah, anomaly, array, atmosphere, pairing, report, session,
stringstats), and because the `sasa` console script does NOT put the working
directory on sys.path, `sasa --help` died with ModuleNotFoundError on any
machine that was not sitting in the source tree. CI caught it only at the tag,
after the version had been cut, and the binary build jobs never ran -- so the
release went out with no macOS or Windows download attached at all.

These tests fail on the commit that adds a module, not at the release.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 is supported for running SASA, not for packaging it
    tomllib = pytest.importorskip("tomllib")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def top_level_modules() -> set[str]:
    """Every importable module at the repository root, by import name."""
    return {
        path.stem
        for path in _REPO_ROOT.glob("*.py")
        if not path.name.startswith("_")
    }


def pyproject() -> dict:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_py_modules_covers_every_top_level_module():
    """
    A module on disk that is not in py-modules is not installed.

    The failure is invisible in the source tree and total outside it, so the
    only honest check is against the filesystem.
    """
    declared = set(pyproject()["tool"]["setuptools"]["py-modules"])
    on_disk = top_level_modules()

    missing = sorted(on_disk - declared)
    assert not missing, (
        "these modules exist but would not be installed; add them to "
        f"py-modules in pyproject.toml: {missing}"
    )


def test_py_modules_lists_nothing_that_does_not_exist():
    """A stale entry makes the wheel build fail late, with a worse message."""
    declared = set(pyproject()["tool"]["setuptools"]["py-modules"])
    on_disk = top_level_modules()

    phantom = sorted(declared - on_disk)
    assert not phantom, (
        f"py-modules names modules that are not in the repository: {phantom}"
    )


def test_no_top_level_module_shadows_the_standard_library():
    """
    A module named after a stdlib module resolves differently per interpreter.

    array.py was the case that proved it. CPython builds `array` as a shared
    object on some platforms and links it into the binary on others; a built-in
    is served by BuiltinImporter, which sits ahead of every path entry, so
    `from array import ArrayGeometry` imported the standard library and raised
    ImportError on exactly those interpreters -- while passing on the ones CI
    happened to use. It was renamed to mic_array.

    The reverse harm is worse and quieter: with the source tree on sys.path,
    any dependency asking for the real `array` gets ours instead.
    """
    collisions = sorted(top_level_modules() & sys.stdlib_module_names)
    assert not collisions, (
        "these top-level modules share a name with the standard library and "
        f"will resolve unpredictably; rename them: {collisions}"
    )


def test_console_script_target_is_importable_and_callable():
    """
    `sasa = "main:main"` is only a string until something resolves it.

    The entry point is what a user runs first; a rename of main() would leave
    the whole install working except for the one command it advertises.
    """
    scripts = pyproject()["project"]["scripts"]
    assert "sasa" in scripts, "the sasa console script disappeared from pyproject.toml"

    module_name, _, attribute = scripts["sasa"].partition(":")
    module = __import__(module_name)
    assert callable(getattr(module, attribute)), (
        f"{scripts['sasa']} does not resolve to a callable"
    )


def test_one_version_and_it_is_the_one_in_pyproject():
    """
    A record states which version produced it. That claim has to be true.

    provenance.py kept its own constant and it went stale at 2.0.0 while
    releases went out as 2.1.x, so every analysis in between named the wrong
    producer in its own software block -- invisible in the interface, wrong in
    the archive. main.py now re-exports provenance's constant, and this holds
    both to the version that is actually packaged and shipped.
    """
    import main
    import provenance

    declared = pyproject()["project"]["version"]
    assert provenance.__version__ == declared, (
        f"provenance.__version__ is {provenance.__version__}, pyproject says {declared}"
    )
    assert main.__version__ == provenance.__version__, (
        "main.py has gone back to keeping its own version constant"
    )


def test_spec_ships_every_module_the_installer_does():
    """
    The frozen app and the wheel must agree on what SASA consists of.

    sasa.spec derives its list from disk, so this compares the two derivations:
    if they diverge, one of the two distributions is incomplete.
    """
    spec_text = (_REPO_ROOT / "sasa.spec").read_text(encoding="utf-8")
    assert "_MODULE_DIR.glob('*.py')" in spec_text, (
        "sasa.spec no longer derives its module list from disk; if it went back "
        "to a hand-written list, this test must compare against that list instead"
    )
