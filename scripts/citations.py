"""Check in-text citations against the reference registry, in Solent Harvard form.

Finds author-year citations in prose, both narrative ("Badorf and Hoberg (2020)") and
parenthetical ("(Prak, Teunter and Syntetos 2017; Croston 1972)"), and reports:

* COMMA   - a comma before the year: "(Hyndman and Athanasopoulos, 2021)";
* FORM    - the right source cited in the wrong author form, e.g. "Saputra et al." for a
            two-author paper, or "no date" instead of "n.d.";
* UNKNOWN - a citation that matches no registry entry;
* UNCITED - a registry entry that is never cited.

Usage: python citations.py <text-file> [<text-file> ...]
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import references as R

YEAR = r"(?:19|20)\d\d[a-z]?|n\.d\.|no date"
NAME = r"(?:[Vv]an |[Vv]on |[Dd]e |[Ee]l )?[A-Z][\w'À-ſ.\-]*(?: [A-Z][\w'À-ſ.\-]*)*"
AUTHORS = rf"{NAME}(?: et al\.)?(?:, {NAME})*(?: and {NAME})?"
NARRATIVE = re.compile(rf"({AUTHORS}) \(({YEAR})\)")
PARENS = re.compile(r"\(([^()]*?(?:(?:19|20)\d\d[a-z]?|n\.d\.|no date)[^()]*?)\)")
ITEM = re.compile(rf"^\s*({AUTHORS}),? ({YEAR})(?:, ?p+\.\s?\d+(?:-\d+)?)?\s*$")


@dataclass
class Finding:
    kind: str
    cited: str
    expected: str
    context: str


def _labels():
    labels = {}
    for key in R.REFERENCES:
        label = R.in_text(key)
        author, _, year = label.rpartition(" ")
        labels[key] = (author, year)
    return labels


LABELS = _labels()


def _first_surname(author: str) -> str:
    """Lower-case first surname with accents stripped, so 'Migueis' still finds 'Miguéis'."""
    import unicodedata
    name = re.split(r",| and | et al\.", author)[0].strip().lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch))


def _match(author: str, year: str, context: str, cited_keys: set, findings: list, comma: bool):
    author = re.sub(r"^The (?=[A-Z])", "", author.strip())  # "The Information Commissioner's Office and ..."
    year = "n.d." if year == "no date" else year
    for key, (a, y) in LABELS.items():
        if author.lower() == a.lower() and year == y:
            cited_keys.add(key)
            if comma:
                findings.append(Finding("COMMA", f"{author}, {year}", f"{a} {y}", context))
            return
    # Wrong author form: prefer an exact year (keeps 2022a and 2022b apart), then a suffix-free year.
    for exact in (True, False):
        for key, (a, y) in LABELS.items():
            same_year = year == y if exact else year.rstrip("ab") == y.rstrip("ab")
            if _first_surname(author) == _first_surname(a) and same_year:
                cited_keys.add(key)
                findings.append(Finding("FORM", f"{author}{',' if comma else ''} {year}", f"{a} {y}", context))
                return
    findings.append(Finding("UNKNOWN", f"{author} {year}", "", context))


def check(text: str):
    findings, cited = [], set()
    for m in NARRATIVE.finditer(text):
        author, year = m.group(1), m.group(2)
        author = re.sub(r"^(?:[A-Z][a-z]+ )+(?=(?:[Vv]an |[Vv]on )?[A-Z][\wÀ-ſ'\-]+(?: et al\.| and |, |$))", "", author) \
            if not any(author.lower() == a.lower() for a, _ in LABELS.values()) else author
        _match(author, year, text[max(0, m.start() - 60): m.end() + 20].replace("\n", " "), cited, findings,
               comma=False)
    for m in PARENS.finditer(text):
        for part in m.group(1).split(";"):
            im = ITEM.match(part)
            if im:
                raw = part.strip()
                comma = bool(re.search(rf",\s*(?:{YEAR})", raw))
                _match(im.group(1), im.group(2), text[max(0, m.start() - 60): m.end() + 20].replace("\n", " "),
                       cited, findings, comma=comma)
    for key, ref in R.REFERENCES.items():  # legislation is cited by naming it
        if ref.kind == "legislation" and ref.title in text:
            cited.add(key)
    for key in sorted(set(R.REFERENCES) - cited):
        findings.append(Finding("UNCITED", key, R.in_text(key), ""))
    return findings, cited


def main(paths):
    text = "\n".join(open(p, encoding="utf-8").read() for p in paths)
    findings, cited = check(text)
    counts = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    print(f"registry entries cited: {len(cited)} of {len(R.REFERENCES)} | findings: {counts}")
    for f in findings:
        print(f"  {f.kind:8} {f.cited!r:52} expected {f.expected!r:48} ...{f.context[:110]}")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
