"""Tests for the dissertation generator: citation form, style, and a complete build."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import citations  # noqa: E402
import literature_text  # noqa: E402
import references  # noqa: E402
import reflection_text  # noqa: E402

BANNED = ("delve crucial robust comprehensive nuanced multifaceted furthermore moreover pivotal "
          "landscape tapestry underscore foster showcase intricate vibrant fundamental significant "
          "interplay").split()


def _paragraphs(module_sections):
    return [text for _, paragraphs in module_sections for text in paragraphs]


def test_chapter_text_cites_registry_sources_in_solent_form():
    prose = "\n".join([literature_text.INTRO, *_paragraphs(literature_text.SECTIONS),
                       *reflection_text.CALIBRATION, *_paragraphs(reflection_text.REFLECTION)])
    findings, _ = citations.check(prose)
    assert [f"{f.kind} {f.cited}" for f in findings if f.kind != "UNCITED"] == []


def test_reflection_text_has_no_dashes_or_banned_words():
    prose = " ".join([*reflection_text.CALIBRATION, *_paragraphs(reflection_text.REFLECTION)])
    assert not re.search("[\u2013\u2014]", prose)
    assert [w for w in BANNED if re.search(rf"\b{w}", prose, re.I)] == []


def test_reflection_word_counts_are_in_range():
    assert 300 <= len(" ".join(reflection_text.CALIBRATION).split()) <= 450
    total = sum(len(" ".join(p).split()) for _, p in reflection_text.REFLECTION)
    assert 1500 <= total <= 1900


def test_naming_legislation_cites_it():
    _, cited = citations.check("obligations under the Data Protection Act 2018 are met")
    assert "dpa" in cited


@pytest.mark.slow
def test_generated_draft_is_complete_and_matches_outputs(tmp_path):
    import generate_dissertation_draft as g

    document = g.build(tmp_path / "draft.docx")
    text = g.document_text(document)
    for marker in ("[TO WRITE", "[SOURCE NEEDED", "[insert]", "not found - run"):
        assert marker not in text, marker

    stamp = next(p.text for p in document.paragraphs if p.text.startswith(g.WORD_COUNT_PREFIX))
    assert stamp.startswith(f"{g.WORD_COUNT_PREFIX}{g.main_text_words(document):,}  |  Date: ")

    matched = pd.read_csv(g.OUT / "matched_service_level.csv")
    best = matched.loc[matched["waste_reduction_pct"].idxmax()]
    assert f"{best['waste_reduction_pct']:.1f}% less waste" in text

    body = text.split("\nReferences\n")[0]
    findings, cited = citations.check(body)
    assert [f.kind for f in findings if f.kind != "UNCITED"] == []
    for key in references.REFERENCES:
        rendered = references.plain(key)
        assert (rendered in text) == (key in cited), key
