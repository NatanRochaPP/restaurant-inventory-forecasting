"""Chart titles carry no figure numbers.

A figure number belongs to the caption of the document a chart is placed in, so each
title only says what its chart shows, and no annotation refers to another chart by
number.
"""
from __future__ import annotations

import pathlib
import re

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_report_figures.py"


def literals():
    """Every double-quoted string in the figure script, with its line number."""
    text = SOURCE.read_text(encoding="utf-8")
    return [(text[: m.start()].count("\n") + 1, m.group(1))
            for m in re.finditer(r'"([^"\n]*)"', text)]


def test_no_chart_numbers_itself():
    numbered = [(line, s) for line, s in literals() if re.match(r"\s*Figure \d", s)]
    assert not numbered, f"chart titles must not carry a figure number: {numbered}"


def test_no_chart_refers_to_another_by_number():
    cross = [(line, s) for line, s in literals() if re.search(r"\bFigure \d", s)]
    assert not cross, f"annotations must not name a figure number: {cross}"


def test_every_chart_still_has_a_title():
    text = SOURCE.read_text(encoding="utf-8")
    titles = re.findall(r"(?:set_title|suptitle)\(\s*f?\"([^\"]+)\"", text)
    assert len(titles) == 8, f"expected eight titled charts, found {len(titles)}"
    assert all(t[0].isupper() for t in titles), titles
