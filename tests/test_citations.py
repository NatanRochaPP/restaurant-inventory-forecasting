"""Gate tests for the in-text citation checker. Deterministic, no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import citations as C


def kinds(text):
    findings, cited = C.check(text)
    return [(f.kind, f.cited, f.expected) for f in findings if f.kind != "UNCITED"], cited


def test_correct_parenthetical_citation_passes():
    found, cited = kinds("as shown before (Hyndman and Athanasopoulos 2021).")
    assert found == [] and "fpp3" in cited


def test_comma_before_year_is_reported():
    found, _ = kinds("as shown before (Hyndman and Athanasopoulos, 2021).")
    assert found == [("COMMA", "Hyndman and Athanasopoulos, 2021", "Hyndman and Athanasopoulos 2021")]


def test_et_al_on_a_two_author_paper_is_a_form_error():
    found, _ = kinds("Saputra et al. (2024) model holiday effects.")
    assert found == [("FORM", "Saputra et al. 2024", "Saputra and Kumar 2024")]


def test_three_authors_must_all_be_named():
    found, _ = kinds("Dietvorst et al. (2015) demonstrate algorithm aversion.")
    assert found == [("FORM", "Dietvorst et al. 2015", "Dietvorst, Simmons and Massey 2015")]


def test_year_suffix_selects_the_right_paper():
    found, _ = kinds("the M5 retail dataset (Makridakis et al., 2022b), which")
    assert found == [("FORM", "Makridakis et al., 2022b", "Makridakis, Spiliotis and Assimakopoulos 2022b")]


def test_leading_article_on_a_corporate_author_is_ignored():
    found, cited = kinds("The Information Commissioner's Office and The Alan Turing Institute (2020) set out")
    assert found == [] and "ico_ai" in cited


def test_several_sources_in_one_bracket_are_each_checked():
    found, cited = kinds("comes from supermarkets (van Donselaar et al. 2010; van Donselaar et al. 2006).")
    assert found == [] and {"vd2010", "vd2006"} <= cited


def test_unknown_source_is_reported():
    found, _ = kinds("the Data Protection Act 2018 (Great Britain, 2018) are met")
    assert [f[0] for f in found] == ["UNKNOWN"]


def test_no_date_must_be_written_n_d():
    found, _ = kinds("per outlet per year (WRAP, no date), which")
    assert found == [("COMMA", "WRAP, n.d.", "WRAP n.d.")]


def test_uncited_entries_are_listed():
    findings, _ = C.check("(Croston 1972)")
    uncited = {f.cited for f in findings if f.kind == "UNCITED"}
    assert "croston" not in uncited and "nahmias" in uncited
