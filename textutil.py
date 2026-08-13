#!/usr/bin/env python3
"""
textutil.py - Sentence construction for operator-facing text

Every string SASA shows an operator ends up somewhere it will be read closely:
a warning on a measurement record, a line in an analysis log, a finding in a
report that someone else has to act on. "6 shot(s) were excluded" is a form
waiting to be filled in rather than a sentence, and the code always knows the
number before it writes the line -- so it can simply write the right word.

This module is deliberately tiny and has no dependencies. It exists so that no
message anywhere in the application has to carry a "(s)", and so that lists of
things are joined the same way everywhere.

Usage:
    from textutil import count, plural, join_list

    count(6, "shot")                  -> "6 shots"
    count(1, "shot")                  -> "1 shot"
    f"{n} {plural(n, 'is')} missing"  -> "3 are missing"
    join_list(["A", "B", "C"])        -> "A, B and C"
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

# Irregular forms this application actually uses. Kept as an explicit table
# rather than a rule, because guessing at English plurals is how "analysises"
# reaches a customer.
_IRREGULAR = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "this": "these",
    "it": "they",
    "its": "their",
    "analysis": "analyses",
    "axis": "axes",
}


def plural(n: float, word: str, plural_form: Optional[str] = None) -> str:
    """
    The form of ``word`` that agrees with ``n``.

    Zero takes the plural, as it does in English ("0 shots"). A non-integer
    count takes the plural too ("1.5 seconds").

    Args:
        n: The count the word has to agree with.
        word: The singular form, or a verb in its third-person singular form.
        plural_form: An explicit plural, for anything not in the table above.
    """
    try:
        singular = abs(float(n)) == 1.0
    except (TypeError, ValueError):
        singular = False
    if singular:
        return word
    if plural_form is not None:
        return plural_form
    return _IRREGULAR.get(word, f"{word}s")


def count(n: float, word: str, plural_form: Optional[str] = None) -> str:
    """
    A count and its noun, agreeing: "1 shot", "6 shots", "0 shots".

    Integral counts are written without a decimal part, so ``count(6.0, "shot")``
    is "6 shots" rather than "6.0 shots".
    """
    try:
        value = float(n)
        text = str(int(round(value))) if value == int(value) else f"{value:g}"
    except (TypeError, ValueError):
        text = str(n)
    return f"{text} {plural(n, word, plural_form)}"


def join_list(items: Iterable[str], conjunction: str = "and", *, oxford: bool = False) -> str:
    """
    Join items into a phrase a person would read out: "A, B and C".

    Args:
        items: The pieces, in the order they should appear. Empty pieces are
               dropped, so a caller can pass conditional fragments directly.
        conjunction: "and" or "or".
        oxford: Add the serial comma before the conjunction. Off by default;
                on for lists whose items themselves contain commas.
    """
    parts: Sequence[str] = [
        str(i).strip() for i in items if i is not None and str(i).strip()
    ]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {conjunction} {parts[1]}"
    separator = ", " if not oxford else ", "
    tail = f"{',' if oxford else ''} {conjunction} {parts[-1]}"
    return separator.join(parts[:-1]) + tail
