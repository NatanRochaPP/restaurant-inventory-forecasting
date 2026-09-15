"""The dissertation's references as structured data, rendered in Solent Harvard style.

Every field below was checked against the publisher record, Crossref, a library catalogue
or the source page on 14 September 2026. Rendering follows the Harvard Solent Referencing
Guide (https://libguides.solent.ac.uk/HarvardSolent):

* only the first author is reversed (``VIDIC, P. and L. GREENE, 2024``), surnames in capitals;
* four or more authors become ``et al.`` in the reference list as well as in the text;
* in-text citations have no comma before the year: ``(Vidic and Greene 2024)``;
* journal articles: ``Title of article. Title of journal, volume(issue), pages``, and
  ``article no: N [no pagination]`` where a journal uses article numbers;
* web pages and online reports: ``Title [viewed date]. Available from: URL``;
* legislation: ``Data Protection Act 2018, ch.12.``

The guide defines no DOI element. The assessment brief asks for DOI links, so articles
and papers with a DOI end ``Available from: https://doi.org/...``.

``render()`` returns ``(text, italic)`` segments so the Word generator can italicise titles;
``plain()`` joins them for tests and search.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VIEWED = "14 September 2026"


@dataclass(frozen=True)
class Ref:
    kind: str                      # article | conference | book | web | report | legislation
    authors: tuple = ()            # ((surname, initials), ...) ; initials None for corporate authors
    year: object = "n.d."          # int or "n.d."
    suffix: str = ""               # "a"/"b" for same authors, same year
    title: str = ""
    container: str = ""            # journal, proceedings title, or book edition statement's parent
    volume: str = ""
    issue: str = ""
    pages: str = ""
    article_no: str = ""
    doi: str = ""
    url: str = ""
    edition: str = ""
    place: str = ""
    publisher: str = ""
    note: str = ""                 # e.g. "Version 8"
    editors: tuple = ()
    chapter: str = ""              # legislation chapter number
    unverified: tuple = field(default=())  # fields deliberately left out because they could not be verified


A = lambda *pairs: tuple(pairs)  # noqa: E731 - keeps the table below readable

REFERENCES = {
    "aci": Ref("article", A(("Aci", "M."), ("Yergök", "D.")), 2023, title="Demand forecasting for food production using machine learning algorithms: a case study of university refectory",
               container="Tehnički vjesnik", volume="30", issue="6", pages="1683-1691", doi="10.17559/TV-20230117000232"),
    "adida": Ref("article", A(("Nikolopoulos", "K."), ("Syntetos", "A.A."), ("Boylan", "J.E."), ("Petropoulos", "F."), ("Assimakopoulos", "V.")), 2011,
                 title="An aggregate-disaggregate intermittent demand approach (ADIDA) to forecasting: an empirical proposition and analysis",
                 container="Journal of the Operational Research Society", volume="62", issue="3", pages="544-554", doi="10.1057/jors.2010.32"),
    "babai": Ref("article", A(("Babai", "M.Z."), ("Dallery", "Y."), ("Boubaker", "S."), ("Kalai", "R.")), 2019,
                 title="A new method to forecast intermittent demand in the presence of inventory obsolescence",
                 container="International journal of production economics", volume="209", pages="30-41", doi="10.1016/j.ijpe.2018.01.026"),
    "badorf": Ref("article", A(("Badorf", "F."), ("Hoberg", "K.")), 2020, title="The impact of daily weather on retail sales: an empirical study in brick-and-mortar stores",
                  container="Journal of retailing and consumer services", volume="52", article_no="101921", doi="10.1016/j.jretconser.2019.101921"),
    "bcs": Ref("report", A(("BCS, The Chartered Institute for IT", None)), 2022, title="Code of conduct for BCS members", note="Version 8",
               place="Swindon", publisher="BCS", url="https://www.bcs.org/media/2211/bcs-code-of-conduct.pdf"),
    "chen": Ref("conference", A(("Chen", "T."), ("Guestrin", "C.")), 2016, title="XGBoost: a scalable tree boosting system",
                container="Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, "
                          "San Francisco, California, 13-17 August 2016",
                place="New York", publisher="ACM", pages="785-794", doi="10.1145/2939672.2939785"),
    "chopra": Ref("book", A(("Chopra", "S."), ("Meindl", "P.")), 2016, title="Supply chain management: strategy, planning, and operation",
                  edition="6th ed., global ed.", place="Boston", publisher="Pearson"),
    "corsten": Ref("article", A(("Corsten", "D."), ("Gruen", "T.")), 2003,
                   title="Desperately seeking shelf availability: an examination of the extent, the causes, and the efforts to address retail out-of-stocks",
                   container="International journal of retail & distribution management", volume="31", issue="12", pages="605-617", doi="10.1108/09590550310507731"),
    "croston": Ref("article", A(("Croston", "J.D.")), 1972, title="Forecasting and stock control for intermittent demands",
                   container="Operational research quarterly", volume="23", issue="3", pages="289-303", doi="10.2307/3007885"),
    "dietvorst": Ref("article", A(("Dietvorst", "B.J."), ("Simmons", "J.P."), ("Massey", "C.")), 2015,
                     title="Algorithm aversion: people erroneously avoid algorithms after seeing them err",
                     container="Journal of experimental psychology: general", volume="144", issue="1", pages="114-126", doi="10.1037/xge0000033"),
    "dpa": Ref("legislation", title="Data Protection Act 2018", year=2018, chapter="12"),
    "fpp3": Ref("book", A(("Hyndman", "R.J."), ("Athanasopoulos", "G.")), 2021, title="Forecasting: principles and practice",
                edition="3rd ed.", place="Melbourne", publisher="OTexts"),
    "huber": Ref("article", A(("Huber", "J."), ("Stuckenschmidt", "H.")), 2020,
                 title="Daily retail demand forecasting using machine learning with emphasis on calendric special days",
                 container="International journal of forecasting", volume="36", issue="4", pages="1420-1438", doi="10.1016/j.ijforecast.2020.02.005"),
    "hk2006": Ref("article", A(("Hyndman", "R.J."), ("Koehler", "A.B.")), 2006, title="Another look at measures of forecast accuracy",
                  container="International journal of forecasting", volume="22", issue="4", pages="679-688", doi="10.1016/j.ijforecast.2006.03.001"),
    "ico_ai": Ref("web", A(("Information Commissioner's Office", None), ("The Alan Turing Institute", None)), 2020,
                  title="Explaining decisions made with AI",
                  url="https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/explaining-decisions-made-with-artificial-intelligence/",
                  unverified=("the ICO page shows no date; 2020 is the first-publication year given by secondary sources",)),
    "ico_gdpr": Ref("web", A(("Information Commissioner's Office", None)), "n.d.", title="UK GDPR guidance and resources",
                    url="https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/"),
    "lundberg": Ref("conference", A(("Lundberg", "S.M."), ("Lee", "S.-I.")), 2017, title="A unified approach to interpreting model predictions",
                    editors=A(("Guyon", "I."), ("von Luxburg", "U."), ("Bengio", "S."), ("Wallach", "H."), ("Fergus", "R."), ("Vishwanathan", "S."), ("Garnett", "R.")),
                    container="Advances in neural information processing systems 30 (NIPS 2017), Long Beach, 4-9 December 2017",
                    place="Red Hook, NY", publisher="Curran Associates",
                    pages="4765-4774", url="https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html"),
    "m5acc": Ref("article", A(("Makridakis", "S."), ("Spiliotis", "E."), ("Assimakopoulos", "V.")), 2022, suffix="a",
                 title="M5 accuracy competition: results, findings, and conclusions", container="International journal of forecasting",
                 volume="38", issue="4", pages="1346-1364", doi="10.1016/j.ijforecast.2021.11.013"),
    "m5des": Ref("article", A(("Makridakis", "S."), ("Spiliotis", "E."), ("Assimakopoulos", "V.")), 2022, suffix="b",
                 title="The M5 competition: background, organization, and implementation", container="International journal of forecasting",
                 volume="38", issue="4", pages="1325-1336", doi="10.1016/j.ijforecast.2021.07.007"),
    "migueis": Ref("article", A(("Miguéis", "V.L."), ("Pereira", "A.A."), ("Pereira", "J."), ("Figueira", "G.")), 2022,
                   title="Reducing fresh fish waste while ensuring availability: demand forecast using censored data and machine learning",
                   container="Journal of cleaner production", volume="359", article_no="131852", doi="10.1016/j.jclepro.2022.131852"),
    "nahmias": Ref("article", A(("Nahmias", "S.")), 1982, title="Perishable inventory theory: a review", container="Operations research",
                   volume="30", issue="4", pages="680-708", doi="10.1287/opre.30.4.680"),
    "pierskalla": Ref("article", A(("Pierskalla", "W.P."), ("Roach", "C.D.")), 1972, title="Optimal issuing policies for perishable inventory",
                      container="Management science", volume="18", issue="11", pages="603-614", doi="10.1287/mnsc.18.11.603"),
    "posch": Ref("article", A(("Posch", "K."), ("Truden", "C."), ("Hungerländer", "P."), ("Pilz", "J.")), 2022,
                 title="A Bayesian approach for predicting food and beverage sales in staff canteens and restaurants",
                 container="International journal of forecasting", volume="38", issue="1", pages="321-338", doi="10.1016/j.ijforecast.2021.06.001"),
    "prak": Ref("article", A(("Prak", "D."), ("Teunter", "R."), ("Syntetos", "A.A.")), 2017, title="On the calculation of safety stocks when demand is forecasted",
                container="European journal of operational research", volume="256", issue="2", pages="454-461", doi="10.1016/j.ejor.2016.06.035"),
    "rodrigues": Ref("article", A(("Rodrigues", "M."), ("Miguéis", "V."), ("Freitas", "S."), ("Machado", "T.")), 2024,
                     title="Machine learning models for short-term demand forecasting in food catering services: a solution to reduce food waste",
                     container="Journal of cleaner production", volume="435", article_no="140265", doi="10.1016/j.jclepro.2023.140265"),
    "saputra": Ref("article", A(("Saputra", "J.P.B."), ("Kumar", "A.")), 2024,
                   title="Modeling the impact of holidays and events on retail demand forecasting in online marketing campaigns using intervention analysis",
                   container="Journal of digital market and digital currency", volume="1", issue="2", pages="144-164", doi="10.47738/jdmdc.v1i2.9"),
    "sb2005": Ref("article", A(("Syntetos", "A.A."), ("Boylan", "J.E.")), 2005, title="The accuracy of intermittent demand estimates",
                  container="International journal of forecasting", volume="21", issue="2", pages="303-314", doi="10.1016/j.ijforecast.2004.10.001"),
    "sb2006": Ref("article", A(("Syntetos", "A.A."), ("Boylan", "J.E.")), 2006, title="On the stock control performance of intermittent demand estimators",
                  container="International journal of production economics", volume="103", issue="1", pages="36-47", doi="10.1016/j.ijpe.2005.04.004"),
    "sbc2005": Ref("article", A(("Syntetos", "A.A."), ("Boylan", "J.E."), ("Croston", "J.D.")), 2005, title="On the categorization of demand patterns",
                   container="Journal of the Operational Research Society", volume="56", issue="5", pages="495-503", doi="10.1057/palgrave.jors.2601841"),
    "schmidt": Ref("article", A(("Schmidt", "A."), ("Kabir", "M.W.U."), ("Hoque", "M.T.")), 2022, title="Machine learning based restaurant sales forecasting",
                   container="Machine learning and knowledge extraction", volume="4", issue="1", pages="105-130", doi="10.3390/make4010006"),
    "seyam": Ref("article", A(("Seyam", "A."), ("Mathew", "S.S."), ("Du", "B."), ("El Barachi", "M."), ("Shen", "J.")), 2025,
                 title="A stacking ensemble model for food demand forecasting: a preventative approach to food waste reduction",
                 container="Cleaner logistics and supply chain", volume="15", article_no="100225", doi="10.1016/j.clscn.2025.100225"),
    "silver": Ref("book", A(("Silver", "E.A."), ("Pyke", "D.F."), ("Peterson", "R.")), 1998, title="Inventory management and production planning and scheduling",
                  edition="3rd ed.", place="New York", publisher="Wiley"),
    "slack": Ref("book", A(("Slack", "N."), ("Brandon-Jones", "A."), ("Johnston", "R.")), 2016, title="Operations management",
                 edition="8th ed.", place="Harlow", publisher="Pearson"),
    "taylor": Ref("article", A(("Taylor", "S.J."), ("Letham", "B.")), 2018, title="Forecasting at scale", container="The American statistician",
                  volume="72", issue="1", pages="37-45", doi="10.1080/00031305.2017.1380080"),
    "teunter": Ref("article", A(("Teunter", "R.H."), ("Babai", "M.Z."), ("Syntetos", "A.A.")), 2010, title="ABC classification: service levels and inventory costs",
                   container="Production and operations management", volume="19", issue="3", pages="343-352", doi="10.1111/j.1937-5956.2009.01098.x"),
    "vd2006": Ref("article", A(("van Donselaar", "K.H."), ("van Woensel", "T."), ("Broekmeulen", "R.A.C.M."), ("Fransoo", "J.C.")), 2006,
                  title="Inventory control of perishables in supermarkets", container="International journal of production economics",
                  volume="104", issue="2", pages="462-472", doi="10.1016/j.ijpe.2004.10.019"),
    "vd2010": Ref("article", A(("van Donselaar", "K.H."), ("Gaur", "V."), ("van Woensel", "T."), ("Broekmeulen", "R.A.C.M."), ("Fransoo", "J.C.")), 2010,
                  title="Ordering behavior in retail stores and implications for automated replenishment", container="Management science",
                  volume="56", issue="5", pages="766-784", doi="10.1287/mnsc.1090.1141"),
    "wrap2013": Ref("web", A(("WRAP", None)), 2013, title="Overview of waste in the UK hospitality and food service sector",
                    url="https://www.wrap.ngo/sites/default/files/2020-10/WRAP-Overview%20of%20Waste%20in%20the%20UK%20Hospitality%20and%20Food%20Service%20Sector%20FINAL.pdf",
                    unverified=("place of publication (Banbury does not appear in the document)",)),
    "wrap2024": Ref("web", A(("WRAP", None)), 2024, title="WRAP urges hospitality and food service CEOs and leaders to take a stand against food waste",
                    url="https://www.wrap.ngo/media-centre/press-releases/wrap-urges-hospitality-and-food-service-ceos-and-leaders-take-stand"),
    "wrap2025": Ref("web", A(("WRAP", None)), 2025, title="UK food waste and food surplus: key facts. Updated July 2025",
                    url="https://www.wrap.ngo/sites/default/files/2025-06/WRAP-UK-Food-Waste-and-Food-Surplus-Key-Facts-July-2025-v5.pdf"),
    "wrap_sector": Ref("web", A(("WRAP", None)), "n.d.", title="Hospitality and food service",
                       url="https://www.wrap.ngo/taking-action/food-drink/sectors/hospitality-food-service"),
    "zipkin": Ref("book", A(("Zipkin", "P.H.")), 2000, title="Foundations of inventory management", place="Boston", publisher="McGraw-Hill"),
}


def _year(ref: Ref) -> str:
    return f"{ref.year}{ref.suffix}"


def _name_list(people: tuple, italic_et_al=True):
    """Solent author block as segments: first reversed, others initials first, 4+ -> et al."""
    first_surname, first_initials = people[0]
    head = first_surname.upper() if first_initials is None else f"{first_surname.upper()}, {first_initials}"
    if len(people) >= 4:
        return [(head + " ", False), ("et al.", italic_et_al)]
    rest = [s.upper() if i is None else f"{i} {s.upper()}" for s, i in people[1:]]
    if not rest:
        return [(head, False)]
    if len(rest) == 1:
        return [(f"{head} and {rest[0]}", False)]
    return [(f"{head}, {', '.join(rest[:-1])} and {rest[-1]}", False)]


def _sentence(title: str) -> str:
    return title if title.endswith(("?", "!", ".")) else title + "."


def render(key: str) -> list:
    """The full reference-list entry for ``key`` as ``(text, italic)`` segments."""
    r = REFERENCES[key]
    if r.kind == "legislation":
        return [(r.title, True), (f", ch.{r.chapter}.", False)]
    label = _year(r)
    segs = _name_list(r.authors) + [(f", {label}{' ' if label.endswith('.') else '. '}", False)]  # "n.d." keeps one stop
    if r.kind == "article":
        segs.append((_sentence(r.title) + " ", False))
        segs.append((r.container, True))
        numbering = r.volume + (f"({r.issue})" if r.issue else "")
        if r.article_no:
            segs.append((f", {numbering}, article no: {r.article_no} [no pagination]", False))
        else:
            segs.append((f", {numbering}, {r.pages}", False))
    elif r.kind == "conference":
        segs.append((_sentence(r.title) + " In: ", False))
        if r.editors:
            segs += _name_list(r.editors) + [(", eds. ", False)]
        segs.append((r.container, True))
        segs.append((f". {r.place}: {r.publisher}, pp.{r.pages}", False))
    elif r.kind == "book":
        segs.append((_sentence(r.title), True))
        segs.append(((f" {r.edition}" if r.edition else "") + f" {r.place}: {r.publisher}", False))
    elif r.kind == "report":
        segs.append((_sentence(r.title), True))
        segs.append(((f" {r.note}." if r.note else "") + f" {r.place}: {r.publisher} [viewed {VIEWED}]. Available from: {r.url}", False))
    elif r.kind == "web":
        segs.append((r.title, True))
        segs.append((f" [viewed {VIEWED}]. Available from: {r.url}", False))
    else:
        raise ValueError(f"unknown reference kind {r.kind!r} for {key}")
    if r.doi:
        segs.append((f". Available from: https://doi.org/{r.doi}", False))
    elif r.url and r.kind in ("article", "conference"):  # no DOI: point to the stable proceedings page instead
        segs.append((f". Available from: {r.url}", False))
    return segs


def plain(key: str) -> str:
    return "".join(text for text, _ in render(key))


def in_text(key: str) -> str:
    """Author-year label without brackets: 'Badorf and Hoberg 2020', 'Posch et al. 2022'."""
    r = REFERENCES[key]
    if r.kind == "legislation":
        return r.title
    names = [s for s, _ in r.authors]
    if len(names) >= 4:
        who = f"{names[0]} et al."
    elif len(names) == 3:
        who = f"{names[0]}, {names[1]} and {names[2]}"
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = names[0]
    return f"{who} {_year(r)}"


def sort_key(key: str):
    """Solent order: first author; their solo works first; joint works by later authors; then date."""
    r = REFERENCES[key]
    if r.kind == "legislation":
        return (r.title.upper(), 0, (), 0, "")
    surnames = tuple(s.upper() for s, _ in r.authors)
    year = r.year if isinstance(r.year, int) else 9999  # undated works after dated ones
    return (surnames[0], len(surnames) > 1, surnames[1:], year, r.suffix)


def reference_list() -> list:
    return sorted(REFERENCES, key=sort_key)
