"""Gate tests for the Solent Harvard renderer. Deterministic, no network, well under 2s."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import re

import pytest

import references as R


@pytest.mark.parametrize("key", sorted(R.REFERENCES))
def test_every_entry_renders_without_dashes_or_doubled_punctuation(key):
    text = R.plain(key)
    assert "—" not in text and "–" not in text
    assert ".." not in text.replace("et al..", "")
    assert ", ," not in text and "  " not in text


def test_two_authors_reverse_only_the_first():
    assert R.plain("badorf").startswith("BADORF, F. and K. HOBERG, 2020. ")


def test_three_authors_are_all_named():
    assert R.plain("prak").startswith("PRAK, D., R. TEUNTER and A.A. SYNTETOS, 2017. ")
    assert R.in_text("prak") == "Prak, Teunter and Syntetos 2017"


def test_four_or_more_authors_use_et_al_in_the_list_and_in_text():
    assert R.plain("posch").startswith("POSCH, K. et al., 2022. ")
    assert R.in_text("posch") == "Posch et al. 2022"
    segs = R.render("posch")
    assert ("et al.", True) in segs


def test_in_text_has_no_comma_before_the_year():
    for key in R.REFERENCES:
        assert not re.search(r", (19|20)\d\d[a-z]?$", R.in_text(key)), key


def test_journal_article_layout_matches_the_guide():
    assert R.plain("nahmias") == ("NAHMIAS, S., 1982. Perishable inventory theory: a review. Operations research, "
                                   "30(4), 680-708. Available from: https://doi.org/10.1287/opre.30.4.680")
    assert ("Operations research", True) in R.render("nahmias")


def test_article_numbers_use_no_pagination():
    assert ", 52, article no: 101921 [no pagination]." in R.plain("badorf")


def test_book_layout_matches_the_guide():
    assert R.plain("silver") == ("SILVER, E.A., D.F. PYKE and R. PETERSON, 1998. Inventory management and production "
                                 "planning and scheduling. 3rd ed. New York: Wiley")


def test_web_pages_carry_viewed_date_and_url():
    text = R.plain("wrap2024")
    assert text.startswith("WRAP, 2024. ")
    assert f"[viewed {R.VIEWED}]. Available from: https://www.wrap.ngo/" in text


def test_legislation_layout_matches_the_guide():
    assert R.plain("dpa") == "Data Protection Act 2018, ch.12."
    assert R.in_text("dpa") == "Data Protection Act 2018"


def test_same_authors_same_year_are_distinguished():
    assert R.in_text("m5acc") == "Makridakis, Spiliotis and Assimakopoulos 2022a"
    assert R.in_text("m5des") == "Makridakis, Spiliotis and Assimakopoulos 2022b"


def test_solo_and_two_author_works_precede_longer_author_lists_for_the_same_first_author():
    order = R.reference_list()
    assert order.index("sb2005") < order.index("sb2006") < order.index("sbc2005")
    assert order.index("fpp3") < order.index("hk2006")  # Athanasopoulos before Koehler
    assert order.index("ico_gdpr") < order.index("ico_ai")  # solo corporate work first


def test_list_is_alphabetical_by_first_author():
    firsts = [R.sort_key(k)[0] for k in R.reference_list()]
    assert firsts == sorted(firsts)


def test_papers_without_a_doi_point_to_their_stable_url():
    assert R.plain("lundberg").endswith("Available from: https://proceedings.neurips.cc/paper_files/paper/2017/hash/"
                                        "8a20a8621978632d76c43dfd28b67767-Abstract.html")


def test_every_journal_article_has_a_doi_and_every_web_item_a_url():
    for key, ref in R.REFERENCES.items():
        if ref.kind == "article":
            assert ref.doi, key
        if ref.kind in ("web", "report"):
            assert ref.url.startswith("https://"), key
