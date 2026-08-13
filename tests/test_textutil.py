"""
Tests for textutil.

Small module, but everything it produces is read by a person on a measurement
record, so the cases that matter are the ones English is irregular about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from textutil import count, join_list, plural


# ---------------------------------------------------------------------------
# plural
# ---------------------------------------------------------------------------

def test_one_takes_the_singular():
    assert plural(1, "shot") == "shot"
    assert plural(-1, "shot") == "shot"


def test_zero_takes_the_plural():
    # "0 shot" is not English. Zero is plural.
    assert plural(0, "shot") == "shots"


def test_many_takes_the_plural():
    assert plural(2, "shot") == "shots"
    assert plural(97, "sample") == "samples"


def test_a_fractional_count_takes_the_plural():
    assert plural(1.5, "second") == "seconds"
    assert plural(0.5, "second") == "seconds"


def test_irregular_verbs_agree():
    assert plural(1, "is") == "is"
    assert plural(3, "is") == "are"
    assert plural(1, "was") == "was"
    assert plural(3, "was") == "were"
    assert plural(1, "has") == "has"
    assert plural(3, "has") == "have"


def test_explicit_plural_wins():
    assert plural(3, "its", "their") == "their"
    assert plural(1, "its", "their") == "its"


def test_a_non_numeric_count_is_treated_as_plural():
    # Never raise while formatting a message: a message that cannot be written
    # is worse than one that reads slightly wrong.
    assert plural(None, "shot") == "shots"
    assert plural("many", "shot") == "shots"


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_writes_the_number_and_the_agreeing_noun():
    assert count(1, "shot") == "1 shot"
    assert count(6, "shot") == "6 shots"
    assert count(0, "shot") == "0 shots"


def test_count_does_not_print_a_decimal_part_on_a_whole_number():
    # numpy and division both produce floats; "6.0 shots" is not what anyone
    # counted.
    assert count(6.0, "shot") == "6 shots"
    assert count(1.0, "shot") == "1 shot"


def test_count_keeps_a_real_fraction():
    assert count(1.5, "second") == "1.5 seconds"


# ---------------------------------------------------------------------------
# join_list
# ---------------------------------------------------------------------------

def test_join_list_shapes():
    assert join_list([]) == ""
    assert join_list(["A"]) == "A"
    assert join_list(["A", "B"]) == "A and B"
    assert join_list(["A", "B", "C"]) == "A, B and C"


def test_join_list_drops_empty_pieces():
    # So a caller can pass conditional fragments straight in.
    assert join_list(["A", "", None and "x", "  ", "B"]) == "A and B"


def test_join_list_conjunction_and_oxford_comma():
    assert join_list(["A", "B", "C"], "or") == "A, B or C"
    assert join_list(["A", "B", "C"], oxford=True) == "A, B, and C"


# ---------------------------------------------------------------------------
# The reason this module exists
# ---------------------------------------------------------------------------

# A plural placeholder: a word, then "(s)". "Time (s)" is a unit — it has a
# space before the bracket — and is not matched.
_PLURAL_PLACEHOLDER = re.compile(r"\w\(s\)")

# textutil quotes the pattern in its own docstring to explain it; this test
# quotes it to test it.
_EXEMPT = {"textutil.py"}


def _string_literals(source: str):
    """Every string constant in a module, with its line number."""
    import ast

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def test_no_source_file_writes_a_plural_as_a_parenthesised_s():
    """
    "6 shot(s) were excluded" is a form, not a sentence.

    The code always knows the number before it writes the line, so it can write
    the right word. This asserts the sweep that removed them stays swept.

    Only STRING LITERALS are examined, so `math.isnan(s)` and other code that
    happens to take an argument called `s` is not accused of anything.
    """
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name in _EXEMPT:
            continue
        for number, text in _string_literals(path.read_text(encoding="utf-8")):
            if _PLURAL_PLACEHOLDER.search(text):
                offenders.append(f"{path.name}:{number}: {text.strip()[:90]}")
    assert offenders == [], (
        "These strings write a plural as \"(s)\". Use textutil.count()/plural():\n  "
        + "\n  ".join(offenders)
    )


def test_seconds_as_a_unit_is_not_what_this_forbids():
    # Guard against the check above being tightened into uselessness: a unit in
    # brackets, with a space before it, is correct and must keep passing.
    assert not _PLURAL_PLACEHOLDER.search("Time (s)")
    assert _PLURAL_PLACEHOLDER.search("6 shot(s)")


@pytest.mark.parametrize("n,expected", [(0, "0 bands"), (1, "1 band"), (31, "31 bands")])
def test_count_in_the_shape_the_captions_use(n, expected):
    assert count(n, "band") == expected
