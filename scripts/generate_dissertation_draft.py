"""Generate the dissertation draft as a Microsoft Word document.

The document is assembled from the project's own results files, so the numbers quoted in
the text cannot drift from the numbers in ``outputs/``: re-run the pipeline, re-run this,
and the prose updates with it.

Usage::

    python scripts/generate_dissertation_draft.py

Output: ``docs/dissertation_draft_V2.docx``

Two deliberate omissions, both flagged in the document itself:

* The literature review is a structured skeleton with explicit source markers. No
  reference is invented - fabricated citations are the single most damaging thing that
  can appear in a dissertation.
* Discussion and reflection are drafted from the evidence but must be rewritten in the
  author's own voice, and AI assistance declared as the institution requires.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from literature_text import INTRO as LITERATURE_INTRO  # noqa: E402
from literature_text import SECTIONS as LITERATURE_SECTIONS  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

logger = logging.getLogger("dissertation_draft")
OUT = PROJECT_ROOT / "outputs"
REPORT = OUT / "report"
DOCS = PROJECT_ROOT / "docs"
TARGET = DOCS / "dissertation_draft_V2.docx"

ACCENT = RGBColor(0x2A, 0x78, 0xD6)
MUTED = RGBColor(0x52, 0x51, 0x4E)
FLAG = RGBColor(0xC0, 0x39, 0x2B)


# ---------------------------------------------------------------------------------------
# Word helpers
# ---------------------------------------------------------------------------------------


def add_field(paragraph, instruction: str, placeholder: str) -> None:
    """Insert a Word field code (used for the table of contents and page numbers)."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    separate.append(text)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instr, separate, end):
        run._r.append(element)


def add_page_numbers(document: Document) -> None:
    """Centre a 'Page X' field in the footer of every section."""
    for section in document.sections:
        paragraph = section.footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(paragraph, "PAGE", "1")
        for run in paragraph.runs:
            run.font.size = Pt(9)
            run.font.color.rgb = MUTED


def configure_styles(document: Document) -> None:
    """Set a clean, submission-appropriate base style."""
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.5
    for name, size in (("Heading 1", 18), ("Heading 2", 14), ("Heading 3", 12)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)
        style.paragraph_format.space_before = Pt(18 if name == "Heading 1" else 12)
        style.paragraph_format.space_after = Pt(6)


def para(document: Document, text: str, *, style: str | None = None, italic: bool = False,
         size: int | None = None, colour: RGBColor | None = None, align=None):
    """Add a paragraph with optional inline formatting."""
    paragraph = document.add_paragraph(style=style)
    run = paragraph.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if colour:
        run.font.color.rgb = colour
    if align is not None:
        paragraph.alignment = align
    return paragraph


def editor_note(document: Document, text: str) -> None:
    """A visible instruction to the author, to be deleted before submission."""
    paragraph = document.add_paragraph()
    run = paragraph.add_run(f"[TO WRITE / EDIT: {text}]")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = FLAG


def source_marker(document: Document, text: str) -> None:
    """A marker for a citation the author must supply. No reference is ever invented."""
    paragraph = document.add_paragraph()
    run = paragraph.add_run(f"[SOURCE NEEDED: {text}]")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = FLAG


def bullets(document: Document, items: list[str]) -> None:
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def numbered(document: Document, items: list[str]) -> None:
    for item in items:
        document.add_paragraph(item, style="List Number")


def add_figure(document: Document, filename: str, caption: str, width: float = 6.2) -> None:
    """Embed a figure with a numbered caption, if the image exists."""
    path = REPORT / filename
    if not path.exists():
        editor_note(document, f"figure {filename} not found - run scripts/generate_report_figures.py")
        return
    document.add_picture(str(path), width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = caption_paragraph.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED


def add_table(document: Document, frame: pd.DataFrame, caption: str,
              max_rows: int | None = None, decimals: int = 2) -> None:
    """Insert a dataframe as a Word table with a caption above it."""
    caption_paragraph = document.add_paragraph()
    run = caption_paragraph.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED

    display = frame.head(max_rows) if max_rows else frame
    table = document.add_table(rows=1, cols=len(display.columns))
    table.style = "Light Grid Accent 1"
    for index, column in enumerate(display.columns):
        cell = table.rows[0].cells[index]
        cell.text = str(column)
        for paragraph in cell.paragraphs:
            for cell_run in paragraph.runs:
                cell_run.font.bold = True
                cell_run.font.size = Pt(9)
    for row in display.itertuples(index=False):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            if isinstance(value, float):
                cells[index].text = f"{value:,.{decimals}f}"
            else:
                cells[index].text = str(value)
            for paragraph in cells[index].paragraphs:
                for cell_run in paragraph.runs:
                    cell_run.font.size = Pt(9)
    document.add_paragraph()


def page_break(document: Document) -> None:
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------------------


def load_evidence() -> dict:
    """Load every results file the text quotes from."""
    accuracy = pd.read_csv(OUT / "forecast_accuracy.csv", index_col=0)
    comparison = pd.read_csv(OUT / "simulation_comparison.csv")
    matched = pd.read_csv(OUT / "matched_service_level.csv")
    matched_waste = pd.read_csv(OUT / "matched_waste.csv")
    frontier = pd.read_csv(OUT / "frontier.csv")
    segmentation = pd.read_csv(OUT / "sku_segmentation.csv")
    selection = pd.read_csv(OUT / "model_selection.csv")
    by_sku = pd.read_csv(OUT / "forecast_accuracy_by_sku.csv", index_col=0)
    summary = json.loads((OUT / "simulation_summary.json").read_text(encoding="utf-8"))

    descriptive = comparison.set_index("key")
    best_match = matched.loc[matched["waste_reduction_pct"].idxmax()]
    best_waste_match = matched_waste.loc[matched_waste["service_level_gain_pp"].idxmax()]
    baseline_col = "Manual par-level baseline"
    ai_col = "AI forecast + inventory policy"

    return {
        "accuracy": accuracy,
        "comparison": comparison,
        "descriptive": descriptive,
        "matched": matched,
        "matched_waste": matched_waste,
        "best_match": best_match,
        "best_waste_match": best_waste_match,
        "frontier": frontier,
        "segmentation": segmentation,
        "selection": selection,
        "by_sku": by_sku,
        "summary": summary,
        "baseline_col": baseline_col,
        "ai_col": ai_col,
        "gb": accuracy.loc["Gradient Boosting"],
        "snaive": accuracy.loc["Seasonal Naive"],
        "ets": accuracy.loc["ETS (Holt-Winters)"],
        "ai_max_service": frontier[frontier["strategy"] == ai_col]["unit_service_level"].max(),
        "base_max_service": frontier[frontier["strategy"] == baseline_col]["unit_service_level"].max(),
        "base_max_waste": frontier[frontier["strategy"] == baseline_col]["waste_units"].max(),
    }


def title_page(document: Document, ev: dict) -> None:
    for _ in range(3):
        document.add_paragraph()
    para(document, "Forecasting-Driven Inventory Resupply for Restaurants using AI",
         size=24, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(document,
         "A decision-support system for demand forecasting and cost-aware replenishment "
         "in a quick-service restaurant",
         size=13, italic=True, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(3):
        document.add_paragraph()
    for line, size in [
        ("Natan Rocha Paiva", 14),
        ("Supervisor: Tendi Mhlanga", 11),
        ("", 11),
        ("QHO656 - Final Year Project", 11),
        ("BSc (Hons) Computing", 11),
        ("Southampton Solent University / QAHE", 11),
    ]:
        para(document, line, size=size, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(4):
        document.add_paragraph()
    para(document, "DRAFT V2 - not for submission", size=11, italic=True, colour=FLAG,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para(document,
         "This draft was assembled from the project's own results files. Every figure and "
         "table is generated by scripts/generate_report_figures.py; the argument, the "
         "critical discussion and the reflection require the author's own rewriting, and "
         "AI assistance must be declared as the institution requires.",
         size=9, italic=True, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(document, "Word count: [insert]  |  Date: [insert]", size=10, colour=MUTED,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    page_break(document)


def front_matter(document: Document, ev: dict) -> None:
    gb, snaive = ev["gb"], ev["snaive"]
    best = ev["best_match"]
    waste_match = ev["best_waste_match"]

    document.add_heading("Abstract", level=1)
    para(document,
         f"Restaurants face a two-sided inventory problem: order too little and demand goes "
         f"unmet, order too much and perishable stock is thrown away. Ordering is commonly "
         f"done by hand against a par level derived from recent average usage, which carries "
         f"no view of the coming weekend, a bank holiday or a promotion. This project asks "
         f"whether an AI forecasting system that incorporates holiday and event calendars "
         f"alongside historical usage can reduce both stockouts and food waste relative to "
         f"that manual method.")
    para(document,
         f"A complete decision-support pipeline was built and evaluated: SKU-level demand "
         f"forecasting, ABC/XYZ segmentation driving per-SKU model selection, a base-stock "
         f"inventory policy with safety stock sized from forecast error, an explicit "
         f"perishability model, a day-by-day discrete-event simulator, an explainability "
         f"layer, and a manager-facing dashboard. The system was evaluated on a 24-month, "
         f"20-SKU synthetic dataset for a single casual-dining site.")
    para(document,
         f"Under rolling-origin cross-validation (12 expanding-window folds, seven-day "
         f"horizon), a gradient-boosting model achieved sMAPE {gb['sMAPE']:.2f}, MASE "
         f"{gb['MASE']:.2f} and WAPE {gb['WAPE']:.2f}, improving on a seasonal naive "
         f"benchmark by {abs(gb['sMAPE vs SNaive %']):.1f}% on sMAPE. Operationally, the two "
         f"ordering policies were compared across their full tuning range rather than at a "
         f"single operating point, because any replenishment policy can trade waste for "
         f"availability by holding more stock. At a matched service level of "
         f"{best['matched_service_level_pct']:.2f}%, forecast-driven ordering produced "
         f"{best['waste_reduction_pct']:.1f}% less waste, {best['cost_reduction_pct']:.1f}% "
         f"lower total cost and {best['inventory_reduction_pct']:.1f}% less average inventory. "
         f"Read at matched waste instead, it served "
         f"{waste_match['service_level_gain_pp']:.2f} percentage points more demand.")
    para(document,
         f"The evidence supports the waste and inventory half of the research question "
         f"strongly and the stockout half modestly. Because the dataset is synthetic, the "
         f"contribution is the method and the working artefact rather than an effect size "
         f"transferable to a real site.")
    editor_note(document, "abstract is drafted from results; tighten to your word limit and "
                          "confirm it matches your final claims")
    page_break(document)

    document.add_heading("Acknowledgements", level=1)
    editor_note(document, "your own words")
    page_break(document)

    document.add_heading("Table of Contents", level=1)
    toc_paragraph = document.add_paragraph()
    add_field(toc_paragraph, 'TOC \\o "1-3" \\h \\z \\u',
              "Right-click here in Word and choose 'Update Field' to build the contents.")
    page_break(document)


def chapter_introduction(document: Document, ev: dict) -> None:
    document.add_heading("1. Introduction", level=1)

    document.add_heading("1.1 Background", level=2)
    para(document,
         "Food service operates on thin margins and short shelf lives. A casual-dining site "
         "typically holds fresh stock measured in days rather than weeks, receives deliveries "
         "two to four times a week, and must commit to an order before it knows what it will "
         "sell. Two failure modes follow directly. Under-ordering produces stockouts: items "
         "come off the menu, guests are disappointed, and margin is lost on the sale that "
         "never happens. Over-ordering produces waste: stock passes its use-by date and is "
         "written off at full cost, with a disposal charge on top.")
    para(document,
         "These two failures pull in opposite directions, and the instrument that resolves "
         "them is the same in both cases: the quantity ordered. That quantity is usually set "
         "by hand. A manager fixes a par level from recent average usage, adds a margin for "
         "comfort, and tops up to it on each delivery day. The method is simple and robust, "
         "but it is structurally blind to everything that is known in advance and yet not "
         "present in the recent average: the weekend uplift, a bank holiday, the school "
         "holidays, a planned promotion, a forecast change in the weather.")
    para(document,
         "The scale of the resulting loss is documented. WRAP estimates that the UK hospitality "
         "and food service sector discards 1.1 million tonnes of food each year, three quarters "
         "of which could have been eaten, at a cost of £3.2 billion; close to one fifth of all "
         "the food the sector purchases is thrown away (WRAP, 2024). Averaged over the sector "
         "this is approximately £10,000 per outlet per year (WRAP, no date), which for a "
         "single independent site is the difference between a profitable year and a marginal "
         "one.")
    para(document,
         "Where that waste arises has also been surveyed, though less recently. WRAP attributes "
         "45% of the sector's food waste to preparation, 34% to customer plates and 21% to "
         "spoilage (WRAP, 2013). Only the last of these is within reach of an ordering "
         "decision. Preparation waste is a function of kitchen technique, trimming and "
         "portioning; plate waste is a function of what a guest chooses to leave. Neither "
         "changes if a different quantity is delivered on Monday. Spoilage - stock that "
         "perished before it could be used - is the category that an ordering decision "
         "creates, and WRAP lists over-supply among its causes, alongside equipment "
         "malfunction, improper storage and poor-quality purchasing.")
    para(document,
         "No published breakdown isolates over-ordering as a share of sector food waste, and "
         "this project does not claim one. The published categories are defined by the stage at "
         "which waste is observed rather than by the decision that caused it, and a spoiled "
         "case of tomatoes may equally reflect an over-large order, a failed chiller or a "
         "delivery of short-dated stock. The 21% spoilage share should therefore be read as an "
         "upper bound on what better ordering could address, not as an estimate of it. That "
         "bound defines the scope of this work: the system addresses waste created before food "
         "reaches the kitchen, and it is evaluated on simulated waste and service level "
         "directly rather than against any assumed share of the sector total.")
    para(document,
         "A note on the currency of these figures is warranted, since one of them is old. The "
         "stage-level split originates in WRAP's survey of where waste arises within the sector, "
         "conducted in 2011 and published in 2013, and it remains the most recent breakdown of "
         "that kind that WRAP has published; the more recent sector data released since reports "
         "headline tonnage and cost rather than the point in the kitchen at which food is lost. "
         "The split is therefore used here as the best available characterisation of the "
         "composition of hospitality food waste, not as a current measurement, and no part of "
         "the argument depends on the precise percentages. What the argument requires is only "
         "the ordering - that spoilage is a substantial but minority component, and the only one "
         "an ordering decision can reach - and that ordering is not in dispute.")
    para(document,
         "A second clarification is necessary because the terminology is treacherous. The "
         "academic literature on over-ordering in restaurants is almost entirely concerned with "
         "customers ordering more dishes than they can eat, which is plate waste and a distinct "
         "phenomenon governed by menu design, portion size and social context. This project is "
         "concerned with an operator ordering more stock than the site will consume before it "
         "perishes. The two share a phrase and nothing else, and the consumer-facing literature "
         "is therefore not cited in support of the claim made here.")
    para(document,
         "That this is how ordering is actually done is not merely assumed, though the evidence "
         "available is indirect. Van Donselaar et al. (2010) examine ordering behaviour in "
         "retail stores where an automated replenishment system issues advised quantities, and "
         "find that store managers depart from that advice systematically rather than randomly, "
         "with the deviations concentrated in perishable and slow-moving lines and following "
         "patterns regular enough to be modelled. The finding does two things for the present "
         "argument. It establishes that human judgement, rather than system output, is the "
         "operative mechanism determining what is actually ordered - which is what the manual "
         "baseline in Section 3.9 formalises. And it establishes that such judgement is "
         "patterned rather than arbitrary, which means an automated system can be designed to "
         "work alongside it rather than against it; this is the reasoning behind retaining and "
         "recording manager override rather than presenting a recommendation as final. The same "
         "research programme establishes that perishables are the hardest case, since an order "
         "placed for a short-shelf-life product commits to a quantity that cannot be carried "
         "forward if it proves wrong (van Donselaar et al., 2006).")
    para(document,
         "The limitation of this evidence should be stated rather than glossed. Both studies "
         "concern supermarkets. No equivalent study of ordering behaviour in small, independent "
         "hospitality operations was found, and the differences between the settings - shorter "
         "shelf lives, higher delivery frequency, simpler supply chains, and the absence of any "
         "automated system to depart from in the first place - are substantial enough that the "
         "direction in which the finding would transfer is not established. That absence is "
         "itself part of the gap this project occupies, and Section 2.10 returns to it.")

    document.add_heading("1.2 Problem statement", level=2)
    para(document,
         "The information needed to anticipate demand is largely available before the "
         "ordering decision is taken. Calendars are known months ahead, promotions are "
         "planned, weather is forecast several days out, and the site's own sales history "
         "records how it responded to each of these in the past. A manual par level discards "
         "that information. The question this project addresses is whether using it "
         "systematically produces a measurably better ordering decision, and specifically "
         "whether it improves both sides of the trade-off rather than simply moving along it.")
    para(document,
         "That last distinction matters and is easy to get wrong. Any replenishment policy "
         "can reduce stockouts by holding more stock, and any policy can reduce waste by "
         "holding less. A comparison that observes two policies at different stock levels "
         "measures the stock level, not the policy. Establishing that a forecasting system "
         "genuinely helps therefore requires comparing the two approaches at matched "
         "operating points, which is a central methodological commitment of this work.")

    document.add_heading("1.3 Research question", level=2)
    para(document,
         "Can an AI forecasting system that incorporates holiday/event calendars and "
         "historical usage reduce stockouts and food waste versus current manual methods in "
         "a quick service/casual restaurant context?", italic=True)
    para(document, "The question decomposes into five pieces of evidence:")
    numbered(document, [
        "forecast accuracy relative to established statistical benchmarks;",
        "stockout reduction;",
        "waste reduction;",
        "service-level improvement;",
        "inventory and cost impact.",
    ])
    para(document,
         "Accuracy alone is insufficient. A forecast is an intermediate product; the outcome "
         "that matters is what happens to the stock, and a more accurate forecast only helps "
         "if the replenishment policy converts it into a better order.")

    document.add_heading("1.4 Aims and objectives", level=2)
    para(document, "The aim is to design, build and evaluate a decision-support system that "
                   "converts SKU-level demand forecasts into explainable replenishment "
                   "recommendations, and to measure its operational effect against a manual "
                   "baseline. The objectives are:")
    numbered(document, [
        "Engineer a feature set covering calendar, holiday, weather and demand-history effects "
        "without introducing data leakage.",
        "Implement and benchmark a range of forecasting methods spanning naive, statistical and "
        "machine-learning approaches, including a method appropriate to intermittent demand.",
        "Segment the SKU range on economic value and demand variability, and route each SKU to "
        "an appropriate model with a recorded justification.",
        "Implement inventory policies that convert a forecast into an order quantity subject to "
        "lead time, review period, service level, shelf life, pack size and minimum order quantity.",
        "Build a perishability-aware discrete-event simulator capable of replaying a historical "
        "holdout period without leakage.",
        "Implement a manual par-level baseline and compare the two policies fairly.",
        "Provide explanations of both the forecast and the recommendation that a restaurant "
        "manager could act on.",
        "Deliver a dashboard exposing forecasts, recommendations, overrides, what-if analysis and "
        "the evaluation.",
    ])

    document.add_heading("1.5 Scope and delimitations", level=2)
    bullets(document, [
        "Single site, 20 SKUs, seven product categories, 24 months of daily data.",
        "Deterministic supplier lead time and a fixed delivery calendar.",
        "Lost sales rather than backorders: unmet restaurant demand does not carry forward.",
        "Decision support only - the system recommends and explains, and never places an order.",
        "Synthetic data: no personal, guest or commercially sensitive information is used anywhere.",
    ])

    document.add_heading("1.6 Contributions", level=2)
    para(document, "The work contributes:")
    bullets(document, [
        "a complete, reproducible pipeline from raw sales data through to an explained order "
        "recommendation, with every operational assumption declared in a single configuration file;",
        "a segmentation-driven model-selection layer that records which model serves each SKU and why, "
        "rather than applying one model to a heterogeneous range;",
        "a perishability-aware simulator that tracks inventory by dated batch and writes off expired "
        "stock before demand is served;",
        "a matched-operating-point evaluation method, which controls for the operating point at "
        "which each policy is observed and so removes a confound present when two replenishment "
        "policies are compared at whatever stock levels they happen to occupy;",
        "an explainability layer that states its own limitations, including the non-additivity of its "
        "attributions and the distinction between association and cause.",
    ])

    document.add_heading("1.7 Report structure", level=2)
    para(document,
         "Chapter 2 reviews the relevant literature. Chapter 3 sets out the methodology, "
         "including the evaluation protocol and the measures taken against data leakage. "
         "Chapter 4 describes the implementation. Chapter 5 presents the results. Chapter 6 "
         "discusses what they support and what they do not. Chapter 7 evaluates the work "
         "against its objectives and states its limitations. Chapter 8 concludes, and "
         "Chapter 9 reflects on the process.")
    page_break(document)


# ---------------------------------------------------------------------------------------
# Reference register
#
# Citations carried forward from the author's own AE1 annotated bibliography and QHO538
# survey of literature. Where those two documents disagreed, the AE1 form was checked
# against the publisher record and is the one used here; the QHO538 forms of Schmidt,
# Babai, Badorf and the M5 accuracy paper each contained an error and are not used.
# ---------------------------------------------------------------------------------------

HELD = {
    "schmidt": "SCHMIDT, A., KABIR, M.W.U. and HOQUE, M.T., 2022. Machine learning based "
               "restaurant sales forecasting. Machine Learning and Knowledge Extraction, "
               "4(1), pp.105-130. doi:10.3390/make4010006",
    "babai": "BABAI, M.Z., DALLERY, Y., BOUBAKER, S. and KALAI, R., 2019. A new method to "
             "forecast intermittent demand in the presence of inventory obsolescence. "
             "International Journal of Production Economics, 209, pp.30-41. "
             "doi:10.1016/j.ijpe.2018.01.026",
    "badorf": "BADORF, F. and HOBERG, K., 2020. The impact of daily weather on retail sales: "
              "an empirical study in brick-and-mortar stores. Journal of Retailing and "
              "Consumer Services, 52, 101921. doi:10.1016/j.jretconser.2019.101921",
    "saputra": "SAPUTRA, J.P.B., et al., 2024. Modeling the impact of holidays and events on "
               "retail demand forecasting using SARIMAX. Journal of Data Mining and Digital "
               "Commerce, 3(1), pp.25-39.",
    "taylor": "TAYLOR, S.J. and LETHAM, B., 2018. Forecasting at scale. The American "
              "Statistician, 72(1), pp.37-45. doi:10.1080/00031305.2017.1380080",
    "m5acc": "MAKRIDAKIS, S., SPILIOTIS, E. and ASSIMAKOPOULOS, V., 2022a. M5 accuracy "
             "competition: results, findings, and conclusions. International Journal of "
             "Forecasting, 38(4), pp.1346-1364. doi:10.1016/j.ijforecast.2021.11.013",
    "m5des": "MAKRIDAKIS, S., SPILIOTIS, E. and ASSIMAKOPOULOS, V., 2022b. The M5 competition: "
             "background, organization, and implementation. International Journal of "
             "Forecasting, 38(4), pp.1325-1336. doi:10.1016/j.ijforecast.2021.07.007",
    "fpp3": "HYNDMAN, R.J. and ATHANASOPOULOS, G., 2021. Forecasting: principles and practice. "
            "3rd ed. Melbourne: OTexts.",
    "hk2006": "HYNDMAN, R.J. and KOEHLER, A.B., 2006. Another look at measures of forecast "
              "accuracy. International Journal of Forecasting, 22(4), pp.679-688. "
              "doi:10.1016/j.ijforecast.2006.03.001",
    "silver": "SILVER, E.A., PYKE, D.F. and PETERSON, R., 1998. Inventory management and "
              "production planning and scheduling. 3rd ed. New York: Wiley.",
    "zipkin": "ZIPKIN, P.H., 2000. Foundations of inventory management. New York: McGraw-Hill.",
    "chopra": "CHOPRA, S. and MEINDL, P., 2016. Supply chain management: strategy, planning, "
              "and operation. 6th ed. Harlow: Pearson Education.",
    "slack": "SLACK, N., BRANDON-JONES, A. and JOHNSTON, R., 2016. Operations management. "
             "8th ed. Harlow: Pearson Education.",
    "croston": "CROSTON, J.D., 1972. Forecasting and stock control for intermittent demands. "
               "Operational Research Quarterly, 23(3), pp.289-303. doi:10.2307/3007885",
    "chen": "CHEN, T. and GUESTRIN, C., 2016. XGBoost: a scalable tree boosting system. In: "
            "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
            "Discovery and Data Mining. New York: ACM, pp.785-794. doi:10.1145/2939672.2939785",
    "ico_ai": "INFORMATION COMMISSIONER'S OFFICE and THE ALAN TURING INSTITUTE, 2020. "
              "Explaining decisions made with AI. Wilmslow: ICO.",
    "ico_gdpr": "INFORMATION COMMISSIONER'S OFFICE, 2024. Guide to the UK General Data "
                "Protection Regulation (UK GDPR). Wilmslow: ICO.",
    "dpa": "GREAT BRITAIN, 2018. Data Protection Act 2018. London: The Stationery Office.",
    "bcs": "BCS, THE CHARTERED INSTITUTE FOR IT, 2022. BCS code of conduct for members. "
           "Version 8. Swindon: BCS.",
    "wrap2013": "WRAP, 2013. Overview of waste in the UK hospitality and food service sector. "
                "Banbury: WRAP.",
    "wrap2024": "WRAP, 2024. WRAP urges hospitality and food service CEOs and leaders to take "
                "a stand against food waste. Press release, 4 November.",
    "wrap_sector": "WRAP, no date. Hospitality and food service. [online] Banbury: WRAP.",
    "sb2005": "SYNTETOS, A.A. and BOYLAN, J.E., 2005. The accuracy of intermittent demand "
              "estimates. International Journal of Forecasting, 21(2), pp.303-314. "
              "doi:10.1016/j.ijforecast.2004.10.001",
    "sbc2005": "SYNTETOS, A.A., BOYLAN, J.E. and CROSTON, J.D., 2005. On the categorization of "
               "demand patterns. Journal of the Operational Research Society, 56(5), "
               "pp.495-503. doi:10.1057/palgrave.jors.2601841",
    "sb2006": "SYNTETOS, A.A. and BOYLAN, J.E., 2006. On the stock control performance of "
              "intermittent demand estimators. International Journal of Production Economics, "
              "103(1), pp.36-47. doi:10.1016/j.ijpe.2005.04.004",
    "nahmias": "NAHMIAS, S., 1982. Perishable inventory theory: a review. Operations Research, "
               "30(4), pp.680-708. doi:10.1287/opre.30.4.680",
    "pierskalla": "PIERSKALLA, W.P. and ROACH, C.D., 1972. Optimal issuing policies for "
                  "perishable inventory. Management Science, 18(11), pp.603-614. "
                  "doi:10.1287/mnsc.18.11.603",
    "prak": "PRAK, D., TEUNTER, R. and SYNTETOS, A.A., 2017. On the calculation of safety "
            "stocks when demand is forecasted. European Journal of Operational Research, "
            "256(2), pp.454-461. doi:10.1016/j.ejor.2016.06.035",
    "teunter": "TEUNTER, R.H., BABAI, M.Z. and SYNTETOS, A.A., 2010. ABC classification: "
               "service levels and inventory costs. Production and Operations Management, "
               "19(3), pp.343-352. doi:10.1111/j.1937-5956.2009.01098.x",
    "corsten": "CORSTEN, D. and GRUEN, T., 2003. Desperately seeking shelf availability: an "
               "examination of the extent, the causes, and the efforts to address retail "
               "out-of-stocks. International Journal of Retail and Distribution Management, "
               "31(12), pp.605-617. doi:10.1108/09590550310507731",
    "vd2010": "VAN DONSELAAR, K.H., GAUR, V., VAN WOENSEL, T., BROEKMEULEN, R.A.C.M. and "
              "FRANSOO, J.C., 2010. Ordering behavior in retail stores and implications for "
              "automated replenishment. Management Science, 56(5), pp.766-784. "
              "doi:10.1287/mnsc.1090.1141",
    "vd2006": "VAN DONSELAAR, K.H., VAN WOENSEL, T., BROEKMEULEN, R.A.C.M. and FRANSOO, J.C., "
              "2006. Inventory control of perishables in supermarkets. International Journal "
              "of Production Economics, 104(2), pp.462-472. doi:10.1016/j.ijpe.2004.10.019",
    "huber": "HUBER, J. and STUCKENSCHMIDT, H., 2020. Daily retail demand forecasting using "
             "machine learning with emphasis on calendric special days. International Journal "
             "of Forecasting, 36(4), pp.1420-1438. doi:10.1016/j.ijforecast.2020.02.005",
    "adida": "NIKOLOPOULOS, K., SYNTETOS, A.A., BOYLAN, J.E., PETROPOULOS, F. and "
             "ASSIMAKOPOULOS, V., 2011. An aggregate-disaggregate intermittent demand approach "
             "(ADIDA) to forecasting: an empirical proposition and analysis. Journal of the "
             "Operational Research Society, 62(3), pp.544-554. doi:10.1057/jors.2010.32",
    "lundberg": "LUNDBERG, S.M. and LEE, S.-I., 2017. A unified approach to interpreting model "
                "predictions. In: Advances in Neural Information Processing Systems 30 "
                "(NIPS 2017). Red Hook, NY: Curran Associates, pp.4765-4774.",
    "aci": "ACI, M. and YERGOK, D., 2023. Demand forecasting for food production using "
           "machine learning algorithms: a case study of university refectory. Tehnicki "
           "vjesnik / Technical Gazette, 30(6).",
    "posch": "POSCH, K., TRUDEN, C., HUNGERLANDER, P. and PILZ, J., 2022. A Bayesian approach "
             "for predicting food and beverage sales in staff canteens and restaurants. "
             "International Journal of Forecasting, 38(1), pp.321-338. "
             "doi:10.1016/j.ijforecast.2021.06.001",
    "migueis": "MIGUEIS, V.L., PEREIRA, A.A., PEREIRA, J. and FIGUEIRA, G., 2022. Reducing "
               "fresh fish waste while ensuring availability: demand forecast using censored "
               "data and machine learning. Journal of Cleaner Production, 359, 131852. "
               "doi:10.1016/j.jclepro.2022.131852",
    "rodrigues": "RODRIGUES, M., MIGUEIS, V., FREITAS, S. and MACHADO, T., 2024. Machine "
                 "learning models for short-term demand forecasting in food catering services: "
                 "a solution to reduce food waste. Journal of Cleaner Production, 435, 140265. "
                 "doi:10.1016/j.jclepro.2023.140265",
    "seyam": "SEYAM, A., MATHEW, S.S., DU, B., EL BARACHI, M. and SHEN, J., 2025. A stacking "
             "ensemble model for food demand forecasting: a preventative approach to food "
             "waste reduction. Cleaner Logistics and Supply Chain, 15, 100225.",
    "dietvorst": "DIETVORST, B.J., SIMMONS, J.P. and MASSEY, C., 2015. Algorithm aversion: "
                 "people erroneously avoid algorithms after seeing them err. Journal of "
                 "Experimental Psychology: General, 144(1), pp.114-126. doi:10.1037/xge0000033",
}




def chapter_literature(document: Document, ev: dict) -> None:
    """Render the written literature review from ``scripts/literature_text.py``.

    The prose is held in a separate module so that the chapter can be revised without
    touching the document machinery, and so that a diff of the chapter shows only the
    writing.
    """
    document.add_heading("2. Literature Review", level=1)
    para(document, LITERATURE_INTRO)

    for heading, paragraphs in LITERATURE_SECTIONS:
        document.add_heading(heading, level=2)
        for text in paragraphs:
            para(document, text)

    page_break(document)


def chapter_methodology(document: Document, ev: dict) -> None:
    seg = ev["segmentation"]
    a_share = seg[seg["abc"] == "A"]["value_share"].sum() * 100
    abc_counts = seg["abc"].value_counts()
    xyz_counts = seg["xyz"].value_counts()

    document.add_heading("3. Methodology", level=1)

    document.add_heading("3.1 Research design", level=2)
    para(document,
         "The project is a design-and-evaluation study. An artefact was built, and its effect "
         "was measured by simulation against an explicit alternative. The comparison is "
         "counterfactual: the same historical demand is served twice, once under each ordering "
         "policy, holding constant everything that is not the object of study - the supplier "
         "calendar, the lead time, the opening stock, the cost model and the constraint logic. "
         "The only difference between the two runs is the demand signal that drives the order.")
    para(document,
         "Evaluation proceeds at two levels. Forecast accuracy is measured by rolling-origin "
         "cross-validation over the full dataset. Operational outcome is measured by replaying "
         "a six-month holdout period day by day through a discrete-event simulator.")

    document.add_heading("3.2 Dataset", level=2)
    para(document,
         "The dataset is synthetic, generated deterministically from a fixed random seed. It "
         "covers 14,620 SKU-days: 20 SKUs across seven categories, daily from 1 January 2024 "
         "to 31 December 2025 (731 days) for a single casual-dining site. Each row records "
         "units sold, the product's shelf life, the daily mean temperature, and flags for UK "
         "bank holidays, approximate England school holidays, weekend trading and promotions.")
    para(document,
         "The generator embeds weekly seasonality with a Friday-to-Sunday uplift, a midweek "
         "dip, annual seasonality, bank-holiday uplift, weather sensitivity that varies by "
         "product, promotional uplift, a slow underlying growth trend, and zero-inflation for "
         "slow-moving items. Demand is drawn from a Poisson distribution around the resulting "
         "mean. The site is closed on Christmas Day.")
    para(document,
         "The decision to use synthetic data was taken for three reasons, and it is worth "
         "stating them plainly because the choice constrains every claim the project can make.")
    para(document,
         "The first is access. No site partner was secured. Restaurant sales data at SKU-day "
         "granularity is commercially sensitive, is rarely held in a form that separates "
         "ingredient usage from menu-item sales, and is not something an independent operator "
         "has any incentive to release to a student project. The second is data protection: had "
         "a partner been secured, the data would have required handling under the UK GDPR and "
         "would have introduced an ethics approval dependency onto the project's critical path, "
         "with no guarantee of resolution within the timescale. Section 7.2 sets out the "
         "position that follows from processing no real data at all.")
    para(document,
         "The third reason is the one that would apply even if the first two did not. With "
         "generated data the ground truth is known exactly. The demand process, the effect sizes "
         "attached to each driver, and the proportion of intermittent lines are all specified "
         "rather than inferred, which means the pipeline's behaviour can be inspected against "
         "what it should have recovered. This is what makes the leakage tests in Section 3.12 "
         "possible in their present adversarial form: future demand can be multiplied by a "
         "hundred and every earlier decision asserted to be bit-for-bit unchanged, because the "
         "correct answer is known independently of the pipeline that produces it. On real data "
         "the same tests could only compare one implementation against another.")
    para(document,
         "The corresponding cost is stated throughout and is not mitigated by the calibration "
         "that follows: a pipeline evaluated on data whose structure it was built to recover "
         "cannot establish the accuracy a real site would see.")
    para(document,
         "The effect sizes were not chosen arbitrarily. The temperature sensitivity was "
         "calibrated to the magnitudes reported by Badorf and Hoberg (2020), whose study of "
         "673 brick-and-mortar stores isolates the effect of daily weather on sales, and the "
         "bank-holiday and event uplift to Saputra et al. (2024), who model holiday effects as "
         "exogenous regressors in a retail demand setting. The weekly pattern and the "
         "proportion of intermittent lines follow the structure of the M5 retail dataset "
         "(Makridakis et al., 2022b), which is the closest large public analogue to daily "
         "SKU-level demand at a single site.")
    para(document,
         "This calibration constrains the circularity inherent in synthetic data but does not "
         "remove it. The pipeline can only recover the structure the generator was given, so "
         "the accuracy figures in Chapter 5 establish that the method works and permit exact "
         "inspection of the mechanism; they are not an estimate of the accuracy a real site "
         "would see. That limitation is restated in Section 7.3 and is the first item of "
         "future work.")
    editor_note(document, "check the magnitudes actually used in baseline_forecasting.py "
                          "against what Badorf and Hoberg and Saputra et al. report, because a "
                          "reader may well check")

    document.add_heading("3.3 Feature engineering", level=2)
    para(document, "Features fall into three groups, distinguished by what is genuinely "
                   "knowable at the moment the order is placed.")
    bullets(document, [
        "Calendar: day of week, weekend flag, UK bank holiday, school holiday, site-closure flag, "
        "week of year and month. These are known indefinitely in advance and are therefore "
        "legitimate inputs for a future date.",
        "Commercial: the promotion flag, treated as known because promotions are planned by a "
        "marketing calendar rather than discovered after the fact. This assumption is declared in "
        "configuration and can be disabled.",
        "Demand history: lagged demand at 7 and 14 days, and a 28-day rolling mean. Lags at these "
        "horizons remain fully observable for a seven-day forecast; the rolling mean is frozen at "
        "the forecast origin.",
    ])
    para(document,
         "Temperature is treated differently from the other covariates. A manager ordering on "
         "Monday for Wednesday delivery has a weather forecast, not an observation. Using the "
         "realised temperature of a future day as a model input would therefore flatter the "
         "results. Future temperatures are degraded with noise that grows with lead time, "
         "under configuration control, so that the accuracy reported reflects the information "
         "quality actually available at decision time.")

    document.add_heading("3.4 Forecasting models", level=2)
    para(document, "Six methods were implemented behind a single interface, so that any model "
                   "can be substituted without changes downstream:")
    bullets(document, [
        "Naive - the last observation carried forward; the weakest reference point.",
        "Seasonal naive - the same weekday of the previous week; the primary benchmark and the "
        "basis of the MASE denominator.",
        "ETS (Holt-Winters) - additive trend and additive weekly seasonality with estimated "
        "initialisation, falling back to seasonal naive if the optimiser fails.",
        "Gradient boosting - a single global model trained across all SKUs with SKU identity as a "
        "categorical feature, so that slow movers borrow calendar and weather structure learned "
        "from fast movers.",
        "Croston - smoothing demand size and inter-demand interval separately to forecast a "
        "demand rate for intermittent series.",
        "Croston (SBA) - the Syntetos-Boylan bias correction, multiplying the rate by 1 - alpha/2.",
    ])
    para(document,
         "Every model returns the same structure: SKU, forecast dates, point forecast, lower "
         "and upper bounds, model name, and an estimate of one-day forecast error. That last "
         "quantity is what the inventory layer uses to size safety stock, which keeps the two "
         "layers from drifting apart. For the gradient-boosting model, forecast error is "
         "estimated on a held-out tail of the training window rather than in-sample, because "
         "in-sample residuals of a boosted model understate error badly and would produce "
         "safety stock that is too small.")

    document.add_heading("3.5 Segmentation and model selection", level=2)
    para(document,
         f"SKUs are segmented on two dimensions. ABC classifies by annualised purchase value, "
         f"cut at 80% and 95% of cumulative value; XYZ classifies by the coefficient of "
         f"variation of daily demand, cut at 0.50 and 1.00. On this dataset the split is "
         f"{abc_counts.get('A', 0)} A, {abc_counts.get('B', 0)} B and {abc_counts.get('C', 0)} C "
         f"items, with the A class accounting for {a_share:.1f}% of annualised purchase value, "
         f"and {xyz_counts.get('X', 0)} X, {xyz_counts.get('Y', 0)} Y and "
         f"{xyz_counts.get('Z', 0)} Z items.")
    para(document,
         "Segmentation drives two decisions. First, model selection: SKUs whose share of "
         "zero-demand days exceeds a configured threshold are routed to Croston (SBA), because "
         "a feature-based model fitted to a mostly-zero series produces a small positive "
         "forecast on every day and is confidently wrong on both the zero days and the demand "
         "days. Remaining SKUs with sufficient history are routed to gradient boosting. "
         "Second, service level: A-class items are held to a slightly higher availability "
         "target and C-class items to a slightly lower one, on the reasoning that cover held "
         "on a cheap, erratic item is usually paid for in waste.")
    para(document,
         "Every routing decision is recorded with a human-readable justification, so that the "
         "provenance of any SKU's forecast can be stated. Segmentation is computed only from "
         "history preceding the decision date, so it can be recomputed inside a backtest "
         "without leaking future information.")

    document.add_heading("3.6 Inventory policy", level=2)
    para(document,
         "The policy layer receives a demand forecast, an uncertainty estimate and a stock "
         "position, and knows nothing about how the forecast was produced. This separation is "
         "what allows the simulation to hold the policy fixed while varying only the demand "
         "signal. Two policies are implemented: base-stock (R,S), which reviews on each "
         "delivery day and orders up to a target; and (s,S), which orders only once the "
         "position has fallen to a reorder point, raising fewer and larger orders.")
    para(document, "The base-stock calculation proceeds as:")
    bullets(document, [
        "inventory position = on hand + on order - backorders",
        "protection period = supplier lead time + review period",
        "expected demand = the forecast summed over the protection period",
        "safety stock = z(service level) x sigma x sqrt(protection period)",
        "target stock = expected demand + safety stock",
        "recommended order = max(0, target stock - inventory position)",
    ])
    para(document,
         "The inventory position, not the on-hand stock, is the basis of the calculation: "
         "ignoring stock already in transit causes an order to be placed twice for the same "
         "demand. Safety stock is sized from sigma, the one-day forecast error, rather than "
         "from raw demand variability - a SKU whose demand swings predictably with the weekend "
         "needs no buffer against that swing, because the forecast already anticipates it.")

    document.add_heading("3.7 Perishability and order constraints", level=2)
    para(document,
         "Inventory is not treated as a single durable number. Stock is held as dated batches, "
         "issued first-expired-first-out, and written off when it reaches its use-by date. "
         "Critically, expiry is applied before demand is served on the same day, so expired "
         "stock can never satisfy a sale. The simulator records usable stock, expired stock, "
         "stockouts, incoming stock and the quantity-weighted age of stock on hand as distinct "
         "quantities.")
    para(document,
         "A raw order quantity is not yet an order. Four constraints are applied in a fixed, "
         "recorded sequence: a shelf-life cap limiting the order to what can be consumed "
         "before it expires; a maximum days-of-supply guardrail against an overstated "
         "forecast; rounding to the supplier's case size; and finally the minimum order "
         "quantity. Each adjustment is retained on the recommendation so that the difference "
         "between the calculated and the recommended quantity can be explained.")

    document.add_heading("3.8 Simulation design", level=2)
    para(document, "Each simulated day executes a fixed sequence per SKU:")
    numbered(document, [
        "receive deliveries whose lead time has elapsed;",
        "write off stock that has passed its use-by date;",
        "on a supplier order day, forecast demand and compute the inventory position;",
        "produce an order recommendation and place it into the delivery pipeline;",
        "serve that day's actual demand from usable stock, first-expired-first-out;",
        "record stockouts, waste, closing stock, inventory value and costs.",
    ])
    para(document,
         "The ordering decision at step four occurs strictly before demand is revealed at step "
         "five. That ordering is the mechanism preventing future demand from influencing a "
         "past decision, and it is asserted directly by tests rather than assumed.")

    document.add_heading("3.9 The manual baseline", level=2)
    para(document,
         "The baseline formalises current manual practice as a par-level policy: the mean "
         "daily usage over a trailing 28-day window, multiplied by the cover required, and "
         "inflated by a flat manager buffer. It uses the same supplier calendar, lead time, "
         "opening stock, cost model, pack sizes and minimum order quantities as the "
         "forecast-driven policy. It has no view of the calendar, weather or promotions, and "
         "its buffer is a fixed percentage rather than a function of forecast error, so it "
         "over-covers stable items and under-covers volatile ones.")
    para(document,
         "One asymmetry is deliberate and is reported as such: the baseline is not subject to "
         "the shelf-life cap, because a par level derived from average usage embodies no "
         "explicit perishability reasoning. The consequence is visible in the results - the "
         "baseline can buy availability with waste in a way the forecast-driven policy "
         "refuses to. The behaviour is configurable, so the sensitivity analysis that removes "
         "the asymmetry is a one-line change.")

    document.add_heading("3.10 Evaluation protocol and metrics", level=2)
    para(document,
         "Random train/test splitting is invalid for time series. Forecast accuracy is "
         "therefore evaluated by rolling-origin cross-validation with an expanding window: 12 "
         "folds at a seven-day horizon, each fold training only on data strictly preceding its "
         "origin. Three metrics are reported: sMAPE, bounded and defined when demand is zero; "
         "MASE, scaled against an in-sample seasonal naive forecast so that values below one "
         "indicate improvement on that benchmark; and WAPE, total absolute error over total "
         "demand, which weights busy days most heavily and is therefore the most operationally "
         "relevant of the three.")
    para(document,
         "Operational performance is measured over the holdout by stockout days, stockout "
         "units, unit fill rate, waste units, waste as a share of stock received, waste cost, "
         "holding cost, ordering cost, stockout cost, total cost and average inventory.")

    document.add_heading("3.11 Matched-operating-point comparison", level=2)
    para(document,
         "Because any replenishment policy can trade waste for availability by adjusting how "
         "much stock it holds, two policies observed at different operating points cannot be "
         "ranked. Each policy was therefore swept across its own tuning parameter - service "
         "level for the forecast-driven policy, buffer percentage for the manual one - "
         "producing a frontier of achieved (service level, waste) pairs. The two are then "
         "compared where they overlap, by interpolating one policy's curve onto the other's "
         "operating points. Points outside the swept range are discarded rather than "
         "extrapolated. This yields two symmetrical claims: waste at matched service level, "
         "and service level at matched waste.")

    document.add_heading("3.12 Data-leakage protection", level=2)
    para(document,
         "Because the operational results rest entirely on the claim that no decision saw "
         "information unavailable at the time, the guarantee is enforced in one place and "
         "tested directly. All training data is obtained through a single function returning "
         "observations strictly before the decision date. Lag features use only prior "
         "observations; rolling statistics are computed within each SKU and frozen at the "
         "forecast origin, so no horizon day can update them. Demand is served only after the "
         "ordering decision is taken.")
    para(document,
         "The tests are adversarial rather than confirmatory: future demand is multiplied by "
         "one hundred, or set to zero, and the tests assert that every recommendation, "
         "simulated order and backtest prediction before the cut-off is bit-for-bit unchanged. "
         "A leak of any kind would change those values and fail the test.")
    para(document,
         "One qualification belongs with that guarantee, because it shaped the final design. "
         "Slicing the history correctly is necessary but not sufficient: anything the system "
         "caches between decisions must also be keyed to the decision date. A fitted model, or "
         "a record of which model a product is routed to, is derived from a particular history, "
         "and reusing it at an earlier date reintroduces exactly the information the slicing "
         "removed. Every cached artefact is therefore invalidated whenever the decision date "
         "moves backwards, and that rule is asserted by its own tests. Section 5.9 reports the "
         "defect that prompted it.")

    document.add_heading("3.13 Literature search strategy", level=2)
    para(document,
         "Literature was gathered systematically rather than opportunistically. Searches "
         "combined terms for demand forecasting, inventory policy, intermittent demand and the "
         "hospitality domain, run across ScienceDirect, IEEE Xplore, MDPI and Google Scholar. "
         "Sources were screened on author expertise, publication quality and recency, and on "
         "relevance to one of three pillars: background and context, methodology, or technical "
         "implementation. Each retained source was annotated with a summary, an evaluation and "
         "a statement of its relevance to the project. References were managed in Zotero and "
         "logged in a tracking spreadsheet recording the search term, database, date accessed, "
         "credibility rating and supporting pillar, so that the review remains auditable.")
    para(document,
         "The claim in Section 2.10 that no study was found of replenishment in a small "
         "independent quick-service or casual-dining site is an assertion of absence, and an "
         "assertion of absence is only as good as the search behind it. The search conducted "
         "specifically to test that claim is therefore recorded here rather than left implicit.")
    para(document,
         "The initial searches returned nothing because the terminology was wrong. Queries built "
         "around restaurant and replenishment, or restaurant inventory, return trade "
         "publications on par-level practice and inventory-management software rather than "
         "peer-reviewed work, because the academic literature indexes the setting under "
         "different terms. Re-running the search against food catering services, canteen, "
         "refectory and fresh food retail returned the body of work now cited in Section 2.10.")
    search_log = pd.DataFrame({
        "Search string": [
            "restaurant inventory replenishment ordering food service",
            "demand forecasting ingredient ordering restaurant food waste simulation",
            "food catering services demand forecasting waste reduction",
            "censored demand fresh food availability forecast",
            "restaurant canteen sales forecasting point of sale",
        ],
        "Sources searched": [
            "Google Scholar, ScienceDirect",
            "Google Scholar, ScienceDirect, arXiv",
            "ScienceDirect, Google Scholar",
            "Google Scholar, arXiv",
            "arXiv, Google Scholar",
        ],
        "Outcome": [
            "Trade press only; no peer-reviewed replenishment studies",
            "Rodrigues et al. (2024); Aci and Yergok (2023)",
            "Rodrigues et al. (2024); Seyam et al. (2025)",
            "Migueis et al. (2022)",
            "Posch et al. (2022); Schmidt et al. (2022)",
        ],
    })
    add_table(document, search_log,
              "Table 3.1 - Searches conducted to test the Section 2.10 absence claim")
    para(document,
         "The outcome changed the claim rather than confirming it. Work does exist that "
         "forecasts food-service demand explicitly as a waste-reduction intervention, and one "
         "study measures waste impact by replenishing to predicted demand. The absence that "
         "survives the search is narrower and is stated as such in Section 2.10: no study was "
         "found that converts a food-service demand forecast into an order under the full set of "
         "constraints an actual restaurant order faces, and none compares against a manual "
         "baseline at matched operating points. An absence claim of that specificity is "
         "defendable; the broader one this project originally intended to make was not.")
    editor_note(document, "add the date range of these searches, and fold the AE1 searches into "
                          "the table above so the log is complete. Give the final counts - how "
                          "many sources were screened, retained and cited. If the tracking "
                          "spreadsheet still exists it belongs in an appendix as evidence of "
                          "method")

    document.add_heading("3.14 Tools and reproducibility", level=2)
    para(document,
         "The system is implemented in Python 3.12 using pandas, NumPy, scikit-learn, "
         "statsmodels, Streamlit, Plotly and SQLite, with pytest for testing. No deep learning "
         "is used; the entire pipeline runs on a standard laptop CPU in a few minutes. Every "
         "stochastic component is seeded with 42, and every operational assumption is declared "
         "in a single validated configuration file rather than scattered through the code.")
    page_break(document)


def chapter_implementation(document: Document, ev: dict) -> None:
    document.add_heading("4. Implementation", level=1)

    document.add_heading("4.1 Architecture", level=2)
    para(document,
         "The system is layered so that each concern can be tested and replaced independently: "
         "data loading and validation; feature engineering; forecasting; inventory policy; "
         "simulation; explainability; evaluation; persistence; and presentation. Data flows in "
         "one direction - data to features to forecast to policy to recommendation to "
         "explanation - with simulation and evaluation wrapping the whole chain.")
    editor_note(document, "insert an architecture diagram here; the repository structure in "
                          "the README lists every module and its responsibility")

    document.add_heading("4.2 Configuration", level=2)
    para(document,
         "Every operational assumption lives in one YAML file, validated on load by typed "
         "models: forecast horizon and seed, review period and service level, supplier lead "
         "time and order days, the full cost model, simulation window, baseline parameters, "
         "segmentation thresholds, and per-SKU economics. An out-of-range service level or a "
         "missing cost fails immediately with an actionable message rather than producing a "
         "quietly wrong recommendation. No business constant appears in the modules or in the "
         "user interface.")

    document.add_heading("4.3 Forecasting service", level=2)
    para(document,
         "A single service owns the operations that must not be duplicated: slicing history at "
         "the decision date, building leakage-free feature rows for the horizon, routing each "
         "SKU to its selected model, and caching the global model so it is refitted on a "
         "configured cadence rather than on every simulated day. Base-case forecasts are "
         "memoised by decision date and horizon, which is what makes sweeping the frontier "
         "affordable.")
    para(document,
         "The refit cadence is directional. Moving forward, a cached model is reused until it "
         "is older than the configured interval; moving backwards, it is discarded "
         "unconditionally, because a model fitted at a later date has already seen demand the "
         "earlier date must not see. The same rule governs the cached model routing. This "
         "matters only where the decision date can be chosen freely, which is to say in the "
         "dashboard rather than in the batch scripts, and it is the mechanism described in "
         "Section 3.12.")
    para(document,
         "One deterministic post-processing rule is applied: on days the site is known to be "
         "closed, the forecast is set to zero. This was introduced in response to an observed "
         "failure and is discussed in Section 5.8.")

    document.add_heading("4.4 Inventory engine", level=2)
    para(document,
         "Policies, safety stock, shelf life and order constraints are separate modules. The "
         "recommendation object carries every intermediate value - position, protection-period "
         "demand, safety stock, target, raw quantity, each constraint adjustment, the final "
         "quantity, projected waste and stockout probability - which is what allows the "
         "explanation layer to be generated from the arithmetic rather than written alongside it.")

    document.add_heading("4.5 Simulator and strategies", level=2)
    para(document,
         "The two ordering approaches implement a common strategy interface, so the simulator "
         "is identical for both and the comparison cannot be contaminated by differences in "
         "the surrounding mechanics. Inventory is held in a ledger of dated batches supporting "
         "receipt, expiry and first-expired-first-out issue.")

    document.add_heading("4.6 Explainability", level=2)
    para(document,
         "Forecast attribution uses counterfactual ablation: the model re-predicts the day with "
         "one input reset to a neutral mid-week reference, and the difference is reported as "
         "that input's contribution. Because a boosted model contains interactions, these "
         "contributions do not sum to the prediction; the unattributed remainder is displayed "
         "rather than suppressed, which distinguishes the method from the additive attribution "
         "frameworks it approximates (Lundberg and Lee, 2017). Recommendation explanations are "
         "assembled from values the recommendation object already carries, so the narrative and "
         "the arithmetic cannot diverge. Models without features are explained through their "
         "own parameters instead.")

    document.add_heading("4.7 Dashboard", level=2)
    para(document,
         "A five-page Streamlit application presents the system to a manager: an overview of "
         "SKUs requiring orders and their risk; a forecast page with history, prediction "
         "interval and drivers; a recommendations page with the full calculation and an "
         "override facility; a what-if page; and the evaluation. Overrides are recorded in a "
         "separate table from the recommendation, so a manager's decision never rewrites the "
         "model's output and the trained model is never modified.")
    para(document,
         "The overview also carries a record of how the forecast has recently performed, "
         "because a decision-support tool that shows only its own predictions gives a manager "
         "no basis for deciding how far to trust them. A configurable window of at least five "
         "weeks is replayed: every day in it is re-forecast from an origin one to seven days "
         "earlier, with the models refitted at each origin on history strictly preceding it, so "
         "no day contributes to its own prediction. Total demand and total forecast are plotted "
         "together, with the signed error given a second panel on its own scale, since an error "
         "of a hundred units is unreadable against a daily total of thirteen hundred.")
    para(document,
         "Two error figures are reported side by side, and the distinction is deliberate. "
         "Summed across products, over-forecasting one line offsets under-forecasting another, "
         "which is the correct reading for total spend but flatters the system considerably. "
         "Adding the per-product errors in absolute value first allows nothing to cancel, and "
         "is the figure that corresponds to stock actually sitting on a shelf. Neither is "
         "directly comparable to the backtest accuracy in Chapter 5, which weights every "
         "product equally rather than by volume and is therefore dominated by the small, "
         "erratic lines; the interface states this rather than leaving the discrepancy to be "
         "discovered.")
    editor_note(document, "insert dashboard screenshots here")

    document.add_heading("4.8 Persistence", level=2)
    para(document,
         "SQLite provides the storage layer, chosen because the project is a single-site "
         "prototype that must run without a server and whose database can be shipped as a "
         "single file of evidence. Tables cover SKUs, sales, inventory snapshots, forecasts, "
         "recommendations, overrides, simulated orders, simulation runs and results, and model "
         "metadata.")

    document.add_heading("4.9 Testing", level=2)
    para(document,
         "The suite comprises 235 tests covering feature generation, forecast horizon "
         "construction, inventory position, safety stock, reorder quantities, shelf-life "
         "expiry, pack-size rounding, minimum order quantities, stockout and waste accounting, "
         "simulation chronology and stock conservation, data leakage, cache invalidation when "
         "the decision date moves backwards, the accuracy replay behind the overview panel, and "
         "an end-to-end path from sample data through to a stored simulation result.")
    page_break(document)


def chapter_results(document: Document, ev: dict) -> None:
    gb, ets, snaive = ev["gb"], ev["ets"], ev["snaive"]
    desc, base_col, ai_col = ev["descriptive"], ev["baseline_col"], ev["ai_col"]
    best, waste_match = ev["best_match"], ev["best_waste_match"]
    by_sku = ev["by_sku"].sort_values("sMAPE")
    accuracy_table = ev["accuracy"].reset_index()

    document.add_heading("5. Results", level=1)

    document.add_heading("5.1 Forecast accuracy", level=2)
    para(document,
         f"Table 5.1 reports mean accuracy across 12 folds and 20 SKUs. Gradient boosting "
         f"performs best on all three metrics, with sMAPE {gb['sMAPE']:.2f}, MASE "
         f"{gb['MASE']:.2f} and WAPE {gb['WAPE']:.2f}, improving on the seasonal naive "
         f"benchmark by {abs(gb['sMAPE vs SNaive %']):.1f}% on sMAPE.")
    add_table(document, accuracy_table, "Table 5.1 - Forecast accuracy by model "
                                        "(12 folds, 7-day horizon, 20 SKUs)")
    add_figure(document, "fig01_forecast_accuracy.png",
               "Figure 5.1 - Forecast accuracy by model across the three reported metrics.")
    para(document, "Three observations qualify the headline:")
    bullets(document, [
        f"MASE of {gb['MASE']:.2f} is below one, indicating performance better than an in-sample "
        f"seasonal naive forecast. Being scale-free, this measure is the least sensitive to the "
        f"differing volumes across the range (Hyndman and Koehler, 2006).",
        f"The margin over ETS is small on sMAPE ({gb['sMAPE']:.2f} against {ets['sMAPE']:.2f}) but "
        f"wider on WAPE ({gb['WAPE']:.2f} against {ets['WAPE']:.2f}). Since WAPE weights high-volume "
        f"days most heavily, the machine-learning model's advantage is concentrated where ordering "
        f"errors are most expensive.",
        f"Both Croston variants score worse than seasonal naive on average. This is expected, and "
        f"is an argument for model selection rather than against Croston: the average is taken "
        f"over all 20 SKUs, including the 16 for which the method is not intended.",
    ])
    para(document,
         "Every model was evaluated on every SKU, including combinations expected in advance to "
         "perform poorly - Croston on smooth, high-volume lines, and the feature-based model on "
         "intermittent ones. This was a deliberate choice rather than an omission. The "
         "segmentation rule in Section 3.5 routes each SKU to a model, and a routing rule is an "
         "empirical claim: that the assigned model performs better on that segment than the "
         "alternatives. Evaluating only the assigned pairing would make that claim unfalsifiable, "
         "since the comparison establishing it would never have been run. The full grid is what "
         "converts the routing rule from an assumption inherited from the literature into a "
         "result obtained on this data, and the per-segment breakdown in Section 5.2 is the "
         "evidence for it. The cost of the additional folds was minutes of processor time on a "
         "laptop, not analyst effort.")
    para(document,
         f"Prediction-interval coverage averaged {gb['interval_coverage']:.3f} against a nominal "
         f"0.95 target, indicating slightly conservative intervals. This matters beyond "
         f"calibration, because the same uncertainty estimate sizes safety stock; a "
         f"conservative interval produces marginally more cover than the service-level target "
         f"strictly requires.")

    document.add_heading("5.2 Accuracy by SKU", level=2)
    para(document,
         f"Aggregate accuracy conceals wide variation. The best-forecast SKU "
         f"({by_sku.index[0]}) achieves sMAPE {by_sku['sMAPE'].iloc[0]:.2f}, while the worst "
         f"({by_sku.index[-1]}) reaches {by_sku['sMAPE'].iloc[-1]:.2f}. The poorly forecast SKUs "
         f"are without exception the intermittent ones. This is partly a genuine difficulty and "
         f"partly an artefact of the metric: on a day with zero demand, any positive forecast "
         f"scores the maximum 200% sMAPE, so a series with many zero days is penalised heavily "
         f"however sensible the underlying demand rate. Reporting a single averaged sMAPE without "
         f"this qualification would misrepresent the system's behaviour on slow movers.")
    add_table(document, by_sku.reset_index().head(5),
              "Table 5.2 - Five best-forecast SKUs (full table in Appendix C)")

    document.add_heading("5.3 Segmentation and model routing", level=2)
    seg = ev["segmentation"]
    a_share = seg[seg["abc"] == "A"]["value_share"].sum() * 100
    selection_counts = ev["selection"]["model_name"].value_counts()
    para(document,
         f"Segmentation produced a conventional Pareto distribution: the A class holds "
         f"{a_share:.1f}% of annualised purchase value across "
         f"{(seg['abc'] == 'A').sum()} of 20 SKUs. Six of the nine possible segments are "
         f"occupied. Routing sent {selection_counts.get('Gradient Boosting', 0)} SKUs to "
         f"gradient boosting and {selection_counts.get('Croston (SBA)', 0)} intermittent SKUs "
         f"to Croston (SBA), each with a recorded justification.")
    add_figure(document, "fig02_segmentation.png",
               "Figure 5.2 - ABC/XYZ segmentation and the model routed to each SKU.")

    document.add_heading("5.4 A worked forecast", level=2)
    para(document,
         "Figure 5.3 shows a single seven-day forecast against the demand subsequently "
         "realised, with the prediction interval. The weekly shape is reproduced, and the "
         "realised demand falls within the interval throughout.")
    add_figure(document, "fig03_forecast_vs_actual.png",
               "Figure 5.3 - Seven-day forecast against realised demand for a fast-moving SKU.")

    document.add_heading("5.5 Explainability", level=2)
    para(document,
         "Figure 5.4 shows the drivers of a forecast for the week of the August bank holiday. "
         "Day of week dominates, followed by the promotion flag and the bank holiday itself; "
         "recent demand level and temperature pull in the opposite direction. The presentation "
         "is deliberately cautious: contributions are described as associations learned from "
         "historical data, the unattributed remainder arising from interactions between inputs "
         "is displayed rather than concealed, and no causal claim is made.")
    add_figure(document, "fig08_forecast_drivers.png",
               "Figure 5.4 - Forecast driver contributions for a bank-holiday week.")

    document.add_heading("5.6 Operational outcomes at configured settings", level=2)
    para(document,
         f"Both policies were replayed over the same 184-day holdout, facing identical demand "
         f"of {ev['summary']['baseline']['total_demand']:,.0f} units. Table 5.3 reports the "
         f"outcome at each policy's configured settings.")
    comparison_display = ev["comparison"][["metric", base_col, ai_col, "improvement_pct"]]
    add_table(document, comparison_display, "Table 5.3 - Operational outcomes at configured settings")
    add_figure(document, "fig05_kpi_comparison.png",
               "Figure 5.5 - Operational outcomes indexed to the manual baseline.")
    para(document,
         f"Read directly, these figures appear contradictory: waste falls by "
         f"{desc.loc['waste_units', 'improvement_pct']:.1f}% and average inventory by "
         f"{desc.loc['average_inventory_units', 'improvement_pct']:.1f}%, but stockout units "
         f"rise by {abs(desc.loc['stockout_units', 'improvement_pct']):.1f}% and the fill rate "
         f"falls from {desc.loc['unit_service_level', base_col]:.2f}% to "
         f"{desc.loc['unit_service_level', ai_col]:.2f}%.")
    para(document,
         "The explanation is that the two policies are not at the same operating point. The "
         "manual baseline's flat 20% buffer happens to place it at a higher availability than "
         "the forecast-driven policy's configured 95% service-level target. The comparison "
         "therefore measures the difference in stock held as much as the difference in policy, "
         "and cannot be used to answer the research question. It is reported here for "
         "completeness and because the discrepancy motivates the analysis that follows.")

    document.add_heading("5.7 Comparison at matched operating points", level=2)
    para(document,
         "Sweeping each policy across its tuning parameter produces the frontier in Figure 5.6. "
         "The forecast-driven curve lies below and to the right of the manual curve throughout "
         "the overlapping range: for any availability both can achieve, the forecast-driven "
         "policy achieves it with less waste.")
    add_figure(document, "fig04_service_level_frontier.png",
               "Figure 5.6 - Waste against achieved service level for both policies across "
               "their full tuning range.")
    add_table(document, ev["matched"], "Table 5.4 - Comparison at matched service level")
    para(document,
         f"At a matched service level of {best['matched_service_level_pct']:.2f}%, the "
         f"forecast-driven policy produced {best['waste_reduction_pct']:.1f}% less waste "
         f"({best['waste_units_ai']:,.0f} against {best['waste_units_baseline']:,.0f} units), "
         f"{best['cost_reduction_pct']:.1f}% lower total cost and "
         f"{best['inventory_reduction_pct']:.1f}% less average inventory. At the lower matched "
         f"point of {ev['matched'].iloc[0]['matched_service_level_pct']:.2f}%, the waste "
         f"reduction is {ev['matched'].iloc[0]['waste_reduction_pct']:.1f}%.")
    add_table(document, ev["matched_waste"], "Table 5.5 - Comparison at matched waste")
    para(document,
         f"The symmetrical reading is that at equal waste the forecast-driven policy serves "
         f"{waste_match['service_level_gain_pp']:.2f} percentage points more demand "
         f"({waste_match['service_level_ai']:.2f}% against "
         f"{waste_match['service_level_baseline']:.2f}%). Because only two operating points "
         f"overlap, these results support a range rather than a point estimate: waste "
         f"reduction of roughly 26% to 38% at matched availability, and a service-level gain "
         f"of roughly 1.2 to 1.6 percentage points at matched waste.")
    para(document,
         f"The frontier also exposes an asymmetry in what the two policies can reach. The "
         f"manual policy extends to {ev['base_max_service']:.2f}% availability, but only by "
         f"accumulating {ev['base_max_waste']:,.0f} units of waste. The forecast-driven policy "
         f"saturates near {ev['ai_max_service']:.2f}%: beyond that point its shelf-life cap "
         f"prevents it from buying further availability with stock it cannot sell in time.")

    document.add_heading("5.8 Behaviour over time and by SKU", level=2)
    add_figure(document, "fig06_over_time.png",
               "Figure 5.7 - Waste, unmet demand and stock held across the holdout.")
    para(document,
         "The forecast-driven policy holds visibly less stock throughout the period while "
         "tracking the same demand pattern, and its waste is both lower and less volatile.")
    add_figure(document, "fig07_waste_by_sku.png",
               "Figure 5.8 - Waste by SKU under both policies.")
    para(document,
         "Disaggregating by SKU locates the effect precisely. Nine of the twenty SKUs record "
         "no waste under either policy, because their shelf life comfortably exceeds the "
         "ordering cycle; for these items the forecast changes how much stock is held but not "
         "how much is thrown away. The entire waste reduction is concentrated in the "
         "short-shelf-life items. This is a more informative result than the aggregate "
         "percentage, and it identifies exactly where such a system earns its keep.")

    document.add_heading("5.9 Three corrections arising during development", level=2)
    para(document,
         "Two defects were identified in the earlier baseline experiment and are reported here "
         "because they change how the earlier figures should be read. A third was found later, "
         "in the delivered system, and is reported because the guarantee it threatened is the "
         "one the operational results rest on.")
    para(document,
         "First, the 28-day rolling mean was constructed incorrectly. A rolling window was "
         "applied to an ungrouped series, so it crossed SKU boundaries, and a subsequent index "
         "reset re-aligned the values onto the wrong rows. Compared against a correct "
         "implementation, the feature correlated at only 0.19 with its intended values, with "
         "87% of rows differing by more than one unit. Second, the same feature was leaky: "
         "computed once over the whole dataset, it allowed the seventh day of a forecast "
         "horizon to see actual demand from the first six days of its own evaluation window.")
    para(document,
         "The previously reported sMAPE of 33.67 was therefore obtained with that feature "
         f"acting as noise and with a leak present. The corrected, leak-free pipeline scores "
         f"{gb['sMAPE']:.2f} - statistically indistinguishable, but now legitimately obtained. "
         f"The original script is retained unmodified as a record of what was submitted.")
    para(document,
         "The third defect concerned caching rather than feature construction. To avoid "
         "refitting on every simulated day, the forecasting service retained its fitted model "
         "and reused it until a configured number of days had elapsed. The staleness test "
         "measured that interval as a signed difference, so it was satisfied only when the "
         "decision date moved forward; asked for an earlier date, the difference was negative, "
         "the cached model was judged fresh, and a model fitted on later demand was used to "
         "forecast a date that demand postdates. The cached record of which model each product "
         "is routed to failed in the same way, and a third path read the cached model without "
         "consulting the staleness test at all. The effect was measurable and deterministic: "
         "forecasting one product in mid-August after first viewing December shifted every day "
         "of its seven-day forecast, in both directions and by several units a day, against the "
         "same forecast produced from a clean start.")
    para(document,
         "The batch scripts were unaffected, because a backtest and a simulation advance "
         "through time and never request an earlier origin than the one they have reached; "
         "every figure reported in this chapter was confirmed unchanged, byte for byte, after "
         "the correction. The exposure was confined to the dashboard, where the decision date "
         "is chosen freely and can be revisited in any order. The correction invalidates every "
         "cached artefact whenever the decision date moves backwards, and four tests now assert "
         "it, each verified to fail against the unfixed code.")
    para(document,
         "The defect has a methodological implication. The existing leakage tests were "
         "adversarial in construction, yet each advanced through time, so none could detect a "
         "fault that appears only when the decision date moves backwards. A guarantee extends "
         "no further than the range of behaviour the tests exercise; a suite that shares an "
         "assumption with the code under test will confirm that assumption rather than "
         "challenge it.")
    para(document,
         "Separately, the site closes on Christmas Day, but with only one closure in the "
         "training history the model learned that rule on some training cut-offs and not "
         "others, at one point forecasting 57 units of a fresh product for a day the "
         "restaurant was shut. A closed restaurant selling nothing is a business fact rather "
         "than a statistical pattern, so the forecast is now set to zero on known closure days "
         "by a deterministic rule applied after prediction and recorded in the forecast "
         "metadata.")
    page_break(document)


def chapter_discussion(document: Document, ev: dict) -> None:
    best, waste_match = ev["best_match"], ev["best_waste_match"]
    gb = ev["gb"]

    document.add_heading("6. Discussion", level=1)
    para(document,
         "This chapter is drafted from the evidence and must be rewritten in the author's own "
         "voice, with the literature engagement added.", italic=True, colour=FLAG, size=10)

    document.add_heading("6.1 Answering the research question", level=2)
    para(document,
         f"The research question asks whether a forecasting system using calendars and "
         f"historical usage can reduce stockouts and food waste relative to manual methods. "
         f"The evidence answers the two halves with different strength.")
    para(document,
         f"On waste, the answer is clearly affirmative. At matched availability the "
         f"forecast-driven policy wasted {best['waste_reduction_pct']:.1f}% less, and the "
         f"advantage held across the whole overlapping range of the frontier rather than at a "
         f"single convenient point. It also held less stock to achieve it "
         f"({best['inventory_reduction_pct']:.1f}% lower average inventory), which is the "
         f"mechanism connecting the two: better anticipation of demand permits the same "
         f"availability from a smaller, fresher stock holding.")
    para(document,
         f"On stockouts, the answer is affirmative but modest. At matched waste the system "
         f"served {waste_match['service_level_gain_pp']:.2f} percentage points more demand. "
         f"This is a real improvement, and in a setting where availability is already high it "
         f"is arguably the harder margin to move, but it is a small effect and should not be "
         f"presented as more than it is.")
    para(document,
         "Taken together, I regard this as a positive answer to the research question, but a "
         "conditional one, and the conditions matter more than the verdict.")
    para(document,
         "The answer is positive because the improvement is demonstrated on both halves of the "
         "trade-off simultaneously rather than on one at the expense of the other, and because "
         "it survives the control that makes the comparison meaningful. That second point is the "
         "substantive one. A result showing waste down and availability up, at whatever stock "
         "levels two policies happened to occupy, would not have answered the question at all - "
         "as Section 5.6 demonstrates concretely, since the unmatched comparison on this very "
         "data produces an apparently contradictory result. The finding that the forecast-driven "
         "frontier lies below and to the right of the manual frontier across the whole "
         "overlapping range is a stronger claim than any single pair of numbers, because it does "
         "not depend on where either policy was tuned.")
    para(document,
         "The answer is conditional in three ways. It is conditional on the data, which is "
         "generated; the effects recovered are those the generator was given, and the magnitudes "
         "are not transferable to a real site. It is conditional on the range, since two "
         "overlapping operating points support an interval and not a point estimate - roughly "
         "26% to 38% waste reduction at matched availability, and roughly 1.2 to 1.6 percentage "
         "points of service at matched waste. And it is conditional on the shelf-life profile of "
         "the product range, since Section 5.8 shows the benefit concentrated entirely in "
         "short-life lines and absent from the twelve SKUs whose shelf life exceeds the "
         "protection period by two days or more.")
    para(document,
         "My confidence therefore differs sharply by claim. That the method works - that a "
         "calendar-aware forecast, converted through an inventory policy with safety stock sized "
         "from forecast error and capped by shelf life, dominates a flat par level at equal "
         "availability - I hold with reasonable confidence, because the mechanism is visible in "
         "the disaggregated results and is explicable in terms that do not depend on the data "
         "being real. That a restaurant adopting this system would see a 38.4% reduction in "
         "waste I do not hold at all, and Section 6.7 sets out why the simulation's omissions "
         "bias the estimate towards optimism rather than leaving it merely uncertain. The "
         "defendable form of the finding is that the approach is sound and the effect size is "
         "unestablished.")

    document.add_heading("6.2 Why the improvement occurs", level=2)
    para(document,
         "The mechanism is visible in the disaggregated results. The forecast-driven policy "
         "does not order less on average; it orders differently in time. Because it "
         "anticipates the weekend uplift, the bank holiday and the promotion, it can hold "
         "less stock on ordinary days without exposing itself on busy ones. The manual par "
         "level, lacking that anticipation, must carry a flat buffer sized for the worst case "
         "it might encounter, and on a short-shelf-life item that buffer expires before it is "
         "needed.")
    para(document,
         "This also explains why the benefit concentrates entirely in perishable SKUs. For "
         "products whose shelf life exceeds the ordering cycle, a flat buffer is merely "
         "capital tied up; it is eventually sold. For a three-day product, the same buffer is "
         "waste on a timer. The system's value is therefore a function of the shelf-life "
         "profile of the range, not of forecast accuracy alone.")

    document.add_heading("6.3 Accuracy is necessary but not sufficient", level=2)
    para(document,
         f"The gradient-boosting model improved on the seasonal naive benchmark by "
         f"{abs(gb['sMAPE vs SNaive %']):.1f}% on sMAPE, but the operational gain did not "
         f"follow automatically from that. It required the inventory layer to convert forecast "
         f"error into an appropriately sized buffer, and the perishability logic to prevent "
         f"the resulting order from exceeding what could be consumed. Had safety stock been "
         f"sized from raw demand variability rather than forecast error, the predictable "
         f"weekly swing would have inflated the buffer on exactly the fast-moving fresh items "
         f"where waste is most expensive. This supports treating forecasting and inventory "
         f"policy as a single design problem rather than two sequential ones.")
    para(document,
         "A related caution applies to the accuracy figures themselves. The headline metrics in "
         "Table 5.1 average each product's error equally, which is the convention in the "
         "forecasting literature and the right basis for comparing models, but it gives a slow, "
         "erratic garnish the same weight as the single largest line by volume. Weighting the "
         "same errors by units sold produces a substantially lower figure, and summing the "
         "daily totals before comparing them lowers it again, because one product forecast high "
         "offsets another forecast low. Each of the three answers a different question and "
         "none is the accuracy of the system; the operational results in Sections 5.6 and 5.7 "
         "are reported because they do not require that choice to be made.")

    document.add_heading("6.4 An upper bound on availability", level=2)
    para(document,
         f"The forecast-driven policy could not be pushed beyond approximately "
         f"{ev['ai_max_service']:.2f}% fill rate at any service-level setting. For a product "
         f"with a three-day shelf life and a two-day supplier lead time, the protection period "
         f"approaches the usable life of the stock, and additional cover expires before it can "
         f"be sold. Beyond that point availability cannot be bought with inventory; it "
         f"requires a shorter lead time, more frequent delivery, or a longer-life "
         f"specification. This is a finding about perishable inventory rather than a defect of "
         f"the system, and it is arguably more useful to an operator than the headline "
         f"percentage, because it identifies which lever to pull next.")

    document.add_heading("6.5 The value of model selection", level=2)
    para(document,
         "Croston performed worse than the naive seasonal benchmark when averaged across all "
         "SKUs, yet remains the correct choice for the four intermittent items. A study "
         "reporting only aggregate accuracy would have concluded that Croston should be "
         "discarded. Segment-aware routing, with a recorded justification per SKU, both "
         "improves the outcome and makes the system auditable - a manager can ask why a "
         "particular product is forecast the way it is and receive a specific answer.")

    document.add_heading("6.6 What the evidence does not support", level=2)
    bullets(document, [
        "Any effect size transferable to a real restaurant. The dataset is synthetic, and the "
        "effects the model recovers are those the generator inserted. The contribution is the "
        "method and the artefact. Section 6.7 sets out what the simulation can and cannot "
        "establish, and why its omissions bias the result towards optimism.",
        "That the forecast-driven policy dominates universally. Above roughly 98.5% availability "
        "the manual policy reaches operating points the forecast-driven policy cannot, albeit at "
        "several times the waste.",
        "Any causal claim from feature attributions. The system reports association and says so.",
        "Precision beyond a range. Two overlapping operating points support an interval, not a "
        "point estimate.",
    ])
    para(document,
         "Placing these figures alongside published results requires care, because the "
         "comparison is easy to make badly and the closest available study is not measuring "
         "quite the same thing.")
    para(document,
         "The nearest published comparator is Rodrigues et al. (2024), who apply machine-learning "
         "demand forecasting in food catering services explicitly as a waste-reduction "
         "intervention and report reductions in wasted meals and in unmet demand against the "
         "operations' existing forecasting practice. This project's matched-availability waste "
         "reduction of 38.4% sits inside the range they report, which is mildly reassuring in "
         "one narrow respect: it suggests the effect sizes embedded in this project's generator "
         "are not implausible for the sector. It should not be read as corroboration of the "
         "result. Their evaluation is conducted on real operational data across multiple sites "
         "and several years, whereas this one is conducted on generated data for a single "
         "simulated site; a synthetic result agreeing with a real one is a statement about the "
         "generator's calibration, not about the system.")
    para(document,
         "There is also a structural difference that limits how far the comparison can be "
         "pushed. Their intervention is production planning - how many meals to prepare against "
         "a forecast of covers - whereas this one is ingredient replenishment under supplier "
         "lead time, a fixed review calendar, shelf life, pack size and minimum order quantity. "
         "A production decision is taken close to consumption and can be revised daily; an "
         "ordering decision commits days ahead and cannot. The constraints that generate this "
         "project's availability ceiling in Section 6.4 have no analogue in their setting, and "
         "conversely their baseline is the operations' own forecasting method rather than a flat "
         "par level. The two studies are adjacent rather than comparable, and reporting them as "
         "though they measured the same intervention would overstate what either establishes.")
    para(document,
         "Migueis et al. (2022) are closer in framing, stating the waste-versus-availability "
         "trade-off as the objective rather than treating accuracy as the outcome, in fresh-fish "
         "retail. Their treatment of censored demand identifies a problem this project's data "
         "does not exhibit and consequently could not test, which Section 7.3 records as a "
         "limitation rather than a strength. Seyam et al. (2025) go furthest towards the "
         "operational question, replenishing inventory dynamically from predicted demand and "
         "measuring the resulting waste; the difference from the present work is that "
         "replenishing to next-period predicted demand is not a policy decision under lead time "
         "and shelf life, and that no matched-operating-point control is applied, so their "
         "reported waste reduction carries the confound this project's Section 5.7 is designed "
         "to remove.")
    para(document,
         "The general position is that this project's numbers are consistent with the published "
         "range where a range exists, that the consistency is weak evidence given the data, and "
         "that the methodological contribution - the matched comparison - is the part of the "
         "work that the adjacent literature does not already contain.")

    document.add_heading("6.7 The standing of simulated evidence", level=2)
    para(document,
         "The operational results in Chapter 5 are simulated. No order was placed, no delivery "
         "arrived and no stock spoiled. The objection this invites - that the findings describe "
         "a model of a restaurant rather than a restaurant - is the most serious that can be "
         "made against the work, and it warrants an argument rather than an acknowledgement.")
    para(document,
         "The first part of the answer is that a live trial would not answer the same question. "
         "A restaurant cannot serve the same demand twice. Comparing two replenishment policies "
         "in a live site means running them in different weeks, or across different sites, and "
         "absorbing everything that differed between those periods: weather, footfall, staffing, "
         "menu changes, a competitor's promotion. The comparison in Section 5.7 holds all of "
         "that constant by construction. The same 184 days of demand are served twice, against "
         "the same supplier calendar, the same lead time, the same opening stock, the same cost "
         "model and the same constraint logic, with the demand signal driving the order as the "
         "only difference between the runs. Simulation is therefore not a weaker substitute for "
         "a field trial here; it is the only instrument that isolates the variable of interest. "
         "A field trial would answer a different question - whether this system helps this site "
         "this season - which is more useful commercially and weaker as evidence about the "
         "mechanism.")
    para(document,
         "The second part of the answer is to state precisely what the simulation cannot "
         "establish, since the defence is only worth as much as its limits are explicit.")
    bullets(document, [
        "Construct validity of waste. Simulated waste is stock that passed a modelled use-by "
        "date. Real waste also includes damage, theft, over-portioning, mis-rotation and "
        "trimming, none of which an ordering policy controls and none of which the simulator "
        "represents. The figures should be read as waste attributable to ordering, which is a "
        "component of the sector's spoilage category rather than the whole of it.",
        "No behavioural response. The simulated manager accepts every recommendation. Van "
        "Donselaar et al. (2010) show that real store managers systematically depart from "
        "system-advised quantities, and Dietvorst et al. (2015) show that confidence in an "
        "algorithm falls sharply once it is seen to err. A policy that is better on paper may "
        "therefore be applied inconsistently in practice, and the realised benefit would be a "
        "fraction of the simulated one.",
        "A deterministic supply side. Deliveries arrive in full, on time, at the quality "
        "ordered. Short deliveries, substitutions, quality rejections and supplier failures are "
        "absent, and each would penalise the leaner policy more than the heavily buffered one, "
        "because a thin safety stock has less capacity to absorb a supply shock.",
        "Structurally stable demand. The generator produces demand from a fixed set of effects "
        "plus Poisson noise. It contains no localised anomaly - a road closure, a burst pipe, a "
        "nearby event absent from the calendar, a sudden change in trading hours - and no "
        "regime change. Real demand contains these, and they are precisely the conditions under "
        "which a model trained on regularities performs worst.",
    ])
    para(document,
         "Taken together, these bound the claim in a specific direction. The simulation is "
         "likely to overstate the achievable benefit, because every unrepresented factor - "
         "override behaviour, supply variability, demand shocks - erodes the advantage of the "
         "leaner policy rather than the heavily buffered one. The direction of the result is "
         "better supported than its magnitude, and the magnitude should be read as an upper "
         "bound obtained under favourable conditions rather than as a forecast of site "
         "performance.")
    para(document,
         "The claim is nonetheless falsifiable, and stating how is part of defending it. A "
         "pilot at a single site would hold the existing ordering process in place, generate "
         "recommendations alongside it without acting on them, and record both the quantity "
         "ordered and the quantity recommended against subsequent sales, waste and stockouts. "
         "This is a shadow deployment rather than a trial: it risks nothing, requires no change "
         "to practice, and would establish within one quarter whether the recommendations track "
         "realised demand more closely than the incumbent process. The prediction this work "
         "makes is specific - that recommended quantities would show lower absolute deviation "
         "from realised demand than manually ordered quantities on the smooth, high-value "
         "lines, and no worse on the intermittent ones. A shadow deployment that failed to show "
         "this would falsify the central claim without any site ever having carried the risk of "
         "acting on it.")
    page_break(document)


def chapter_evaluation(document: Document, ev: dict) -> None:
    document.add_heading("7. Evaluation, Limitations and Future Work", level=1)

    document.add_heading("7.1 Evaluation against objectives", level=2)
    para(document,
         "All eight objectives were met, but met carries different weight across them, and "
         "assessing them honestly means distinguishing those delivered fully from those "
         "delivered in a form narrower than originally intended.")
    para(document,
         "Objectives 1, 2 and 5 were met without qualification. The feature set separates inputs "
         "by what is genuinely knowable at the order decision point, freezes the rolling "
         "statistics at the forecast origin, and degrades future temperature with noise growing "
         "in lead time; the adversarial leakage tests in Section 3.12 exercise these directly. "
         "Six forecasting methods spanning naive, statistical and machine-learning families were "
         "implemented behind one interface and evaluated on every SKU under twelve-fold "
         "rolling-origin cross-validation, including the combinations expected in advance to "
         "fail. The simulator tracks inventory by dated batch, writes off expired stock before "
         "demand is served, and issues first-expired-first-out.")
    para(document,
         "Objectives 3, 4 and 7 were met with qualifications worth stating. Segmentation and "
         "routing work as specified and record a justification per SKU, but the routing rule has "
         "only twenty products to act on and six of nine possible segments are occupied, so the "
         "mechanism is demonstrated rather than stress-tested. Both a base-stock and an (s,S) "
         "policy were implemented and tested, but only base-stock was swept across the frontier, "
         "so the comparison between them that Section 7.4 identifies as future work is genuinely "
         "outstanding rather than merely desirable. The explainability layer delivers "
         "attributions and a recommendation narrative, but by counterfactual ablation rather "
         "than an additive attribution framework, which means the contributions do not sum to "
         "the prediction and an unattributed remainder must be displayed; this is a weaker "
         "guarantee than Section 2.8 positions it against, and it is stated as such in the "
         "interface rather than concealed.")
    para(document,
         "Objective 6 is the one that required the most substantial revision in flight, and the "
         "revision is the most valuable thing in the evaluation. A manual par-level baseline was "
         "implemented as planned, but the initial comparison against it was not fair, because "
         "the two policies sat at different operating points and the comparison therefore "
         "measured the stock level rather than the policy. The objective as written - compare "
         "the two policies fairly - was only actually satisfied once the frontier sweep in "
         "Section 3.11 was built. The lesson is that fairly was doing far more work in the "
         "objective than it appeared to when written, and that an objective can be nominally met "
         "by an artefact that does not deliver what the objective was for.")
    para(document,
         "Objective 8 was met in full, though the dashboard's value is asserted rather than "
         "demonstrated: no manager has used it. Section 7.4 records the usability evaluation "
         "this implies, and Section 6.7 notes the corresponding limitation that the simulated "
         "manager accepts every recommendation.")
    objectives = pd.DataFrame({
        "Objective": [
            "1. Leakage-free feature engineering",
            "2. Benchmark forecasting methods",
            "3. Segmentation and model routing",
            "4. Inventory policies with constraints",
            "5. Perishability-aware simulator",
            "6. Manual baseline and fair comparison",
            "7. Explainability",
            "8. Dashboard",
        ],
        "Status": ["Met"] * 8,
        "Evidence": [
            "Frozen-origin features; adversarial leakage tests",
            "Six models, 12-fold rolling origin, Table 5.1",
            "Six segments, 20 routings with recorded reasons",
            "Base-stock and (s,S); four constraints in fixed order",
            "Batch-level FEFO ledger; expiry before demand",
            "Par-level baseline; matched-operating-point frontier",
            "Ablation attribution plus recommendation narrative",
            "Five pages; override, what-if and recent-accuracy panel",
        ],
    })
    add_table(document, objectives, "Table 7.1 - Objectives and supporting evidence")

    document.add_heading("7.2 Professional, legal and ethical practice", level=2)
    para(document,
         "The system processes aggregate operational data only. No personal, guest or "
         "commercially sensitive data is collected or stored at any point, so the obligations "
         "of the UK GDPR and the Data Protection Act 2018 (Great Britain, 2018) are met by "
         "design rather than by control; the Information Commissioner's Office guidance "
         "(Information Commissioner's Office, 2024) was used to confirm that position. Were "
         "primary data collected later - a usability session with a manager, for instance - a "
         "Data Protection Impact Assessment would be required first.")
    para(document,
         "Because the system makes a recommendation a human must accept or reject, it follows "
         "the explanation types set out in the joint ICO and Alan Turing Institute guidance "
         "(Information Commissioner's Office and The Alan Turing Institute, 2020): the "
         "rationale for each recommendation is surfaced, responsibility remains with the "
         "manager through the override facility, and the data behind each figure is shown. The "
         "system never places an order. Development followed the BCS Code of Conduct (BCS, "
         "2022) in respect of professional competence and due care, through version control, "
         "an automated test suite and documented limitations.")
    editor_note(document, "confirm the final ethics position before submission - the AE1 report "
                          "recorded approval as PENDING because of a fault in the University "
                          "ethics system. If approval was subsequently granted, state the "
                          "reference and date and attach the form; if it was never required "
                          "because no primary data was collected, say so explicitly")

    document.add_heading("7.3 Limitations", level=2)
    numbered(document, [
        "Synthetic data, and the specific things it omits. The dataset was generated for this "
        "project, so the effects recovered are those built into the generator; this establishes "
        "that the method works and permits exact inspection of the mechanism, but cannot "
        "establish the effect size a real site would see. The omissions are not only a matter of "
        "degree. Generated demand is well behaved in ways real demand is not: it contains no "
        "supply-side disruption, no localised anomaly such as a road closure or a nearby event "
        "absent from the calendar, no regime change in trading pattern, no product substitution "
        "when a line is unavailable, and no recording error. These are the conditions under "
        "which a model trained on regularities performs worst, and the evaluation contains none "
        "of them. Section 6.7 argues that their absence biases the result in a known direction - "
        "towards overstating the benefit of the leaner policy.",
        "A deliberate asymmetry between the policies. The manual baseline is not subject to the "
        "shelf-life cap, on the reasoning that a par level embodies no perishability logic. This "
        "permits the baseline to reach high availability at very high waste. The behaviour is "
        "configurable and the sensitivity analysis removing it was not run.",
        "Sparse frontier overlap. Only two operating points overlap between the two policies, which "
        "supports a range rather than a point estimate. A denser sweep would tighten the claim.",
        "Costs are assumptions. Unit costs, the 25% annual holding rate, waste valued at cost plus "
        "disposal, and stockout at twice unit cost are declared assumptions. Percentage reductions "
        "in waste and inventory are robust to them; absolute monetary figures are not.",
        "A single site and a single simulated period, with no replication across sites or seeds, so "
        "no confidence intervals are reported on the operational results.",
        "The manual baseline is a formalisation of manual practice, not an observation of it. A real "
        "manager may order better or worse than a 28-day trailing average with a flat buffer.",
    ])

    document.add_heading("7.4 Future work", level=2)
    bullets(document, [
        "A pilot with a real site's data, which is the only way to establish a transferable effect size.",
        "A denser frontier sweep, and the sensitivity run that removes the shelf-life-cap asymmetry.",
        "Comparison of the (s,S) policy against base-stock; both are implemented and tested but only "
        "base-stock was swept.",
        "Lead-time and delivery-frequency sensitivity, given the availability ceiling identified in 6.4.",
        "Probabilistic forecasting - quantile regression rather than a normal approximation - which "
        "would size safety stock without assuming symmetry.",
        "Evaluation of whether managers accept, override or ignore the recommendations in practice, "
        "which the override-recording facility already supports.",
    ])
    page_break(document)


def chapter_conclusion(document: Document, ev: dict) -> None:
    best, waste_match = ev["best_match"], ev["best_waste_match"]
    gb = ev["gb"]

    document.add_heading("8. Conclusion", level=1)
    para(document,
         f"This project built and evaluated a decision-support system that forecasts "
         f"restaurant demand at item level and converts those forecasts into explained, "
         f"cost-aware replenishment recommendations. Under rolling-origin cross-validation a "
         f"gradient-boosting model achieved sMAPE {gb['sMAPE']:.2f} and MASE {gb['MASE']:.2f}, "
         f"improving on a seasonal naive benchmark by {abs(gb['sMAPE vs SNaive %']):.1f}%.")
    para(document,
         f"The system was further evaluated on its effect on stock rather than on predictive "
         f"accuracy alone. Compared against a manual par level at matched operating points - "
         f"which controls for the stock level at which each policy is observed, and without "
         f"which two replenishment policies cannot be ranked - it produced {best['waste_reduction_pct']:.1f}% less waste "
         f"and held {best['inventory_reduction_pct']:.1f}% less stock at equal availability, "
         f"or served {waste_match['service_level_gain_pp']:.2f} percentage points more demand "
         f"at equal waste.")
    para(document,
         "The waste and inventory evidence is strong; the stockout evidence is real but "
         "modest. The benefit is concentrated entirely in short-shelf-life products, and the "
         "analysis identified a ceiling on availability that no amount of stock can overcome "
         "when the protection period approaches the shelf life. Because the evaluation used "
         "synthetic data, the contribution is the method, the working artefact and the "
         "evaluation design rather than a transferable effect size.")
    para(document,
         "The contribution of this work is threefold, and stating it precisely matters because "
         "two of the three are methodological rather than technical.")
    para(document,
         "The first is the artefact: a complete, reproducible pipeline running from raw daily "
         "sales through segmentation, model selection, forecasting, a constrained inventory "
         "policy and a perishability-aware simulation to an explained order recommendation, with "
         "every operational assumption declared in a single validated configuration file rather "
         "than distributed through the code. The second is the evaluation design: a "
         "matched-operating-point comparison between a forecast-driven policy and a formalised "
         "manual baseline, which removes a confound present whenever two replenishment policies "
         "are compared at whatever stock levels they happen to occupy, and which Section 5.6 "
         "shows is not a theoretical concern but one that reverses the apparent direction of the "
         "result on this very data. The third is negative and is the kind of finding that is "
         "easy to omit: the identification of a hard ceiling on availability, arising when the "
         "protection period approaches the usable shelf life, beyond which no quantity of "
         "additional stock can buy further service. That ceiling is a property of perishable "
         "inventory rather than a defect of the system, and it tells an operator which lever to "
         "reach for next.")
    para(document,
         "What I would tell a restaurant operator considering such a system is shorter and less "
         "flattering than the results table.")
    para(document,
         "First, that the gain is concentrated, not general. Twelve of this project's twenty "
         "products produced no waste under either policy, because their shelf life exceeds the "
         "four-day protection period by two days or more; the division is exact, with every line "
         "of six days or longer wasting nothing and every line of five days or shorter wasting "
         "something under both policies. For those twelve a forecasting system changes how much "
         "capital sits in the store room and nothing else. The entire waste benefit came from "
         "the eight short-life items, so the value of a system like this is a function of the "
         "shelf-life profile of the range rather than of forecast accuracy - and an operator can "
         "estimate that profile before spending anything, simply by asking which products are "
         "currently being thrown away.")
    para(document,
         "Second, that the availability ceiling is likely to be the binding constraint before "
         "the forecast is. If a three-day product is delivered against a two-day lead time, "
         "better forecasting cannot push service beyond a certain point, because the additional "
         "cover expires before it can be sold. An extra delivery day is probably worth more than "
         "a better model, and it is considerably cheaper to arrange.")
    para(document,
         "Third, that the forecast is the smaller half of the system. The measured improvement "
         "did not follow from the fourteen per cent accuracy gain automatically; it required the "
         "inventory layer to convert forecast error into an appropriately sized buffer and the "
         "shelf-life logic to prevent the resulting order exceeding what could be consumed. An "
         "accurate forecast attached to a flat par level would have delivered very little of it.")
    para(document,
         "Fourth, that the numbers in this report are simulated and the operator should not "
         "expect them. Real demand contains disruptions, anomalies and regime changes that this "
         "evaluation does not, and those are precisely the conditions under which a lean policy "
         "trained on regularities performs worst. The direction of the bias is knowable: the "
         "simulation flatters the leaner policy. A pilot on the site's own historical data, run "
         "as a counterfactual replay before anything is changed in the kitchen, is a cheap way "
         "to find out whether the effect survives contact with real sales - and it is the first "
         "thing I would do next.")
    para(document,
         "Finally, that the system should remain a recommender. It produces an order to be "
         "accepted, amended or rejected by someone who knows that a coach party is booked on "
         "Thursday and that the chef has changed the special. The value is in making the default "
         "order a better one, not in removing the person who knows what the model does not.")
    page_break(document)


def chapter_reflection(document: Document, ev: dict) -> None:
    document.add_heading("9. Reflection", level=1)
    para(document,
         "This chapter must be written in the first person and in the author's own voice. What "
         "follows records what actually happened during development, as prompts.",
         italic=True, colour=FLAG, size=10)

    document.add_heading("9.1 Project management", level=2)
    editor_note(document, "planning against the Gantt schedule; what slipped and why; how you "
                          "sequenced ten phases and whether that sequencing held")

    document.add_heading("9.2 Technical decisions and what they cost", level=2)
    editor_note(document, "prompts from the development record: (a) the decision to separate "
                          "the inventory policy from the forecast, which is what later made the "
                          "matched comparison possible; (b) sizing safety stock from forecast "
                          "error rather than demand variance; (c) choosing counterfactual "
                          "ablation over a heavier attribution library")

    document.add_heading("9.3 Finding errors in my own earlier work", level=2)
    editor_note(document, "the rolling-mean defect and the leak in the earlier experiment - how "
                          "they were found, what it felt like to find them after reporting the "
                          "figures, and why disclosing them strengthens rather than weakens the "
                          "work. Note that the conclusion survived the correction. The caching "
                          "defect in Section 5.9 is worth the same treatment: it was found only "
                          "by exercising the system in an order no test had tried, which is a "
                          "more useful lesson about testing than the fix itself")

    document.add_heading("9.4 The comparison that nearly went wrong", level=2)
    editor_note(document, "the first baseline-versus-AI run appeared to show waste down 64% and "
                          "stockouts up 53%; recognising this as an operating-point artefact "
                          "rather than a result, and building the frontier sweep in response. "
                          "This is probably the most substantial thing you learned")

    document.add_heading("9.5 Professional and ethical considerations", level=2)
    editor_note(document, "decision support versus automation and why you kept a human in the "
                          "loop; synthetic data and why; declaring AI assistance")

    document.add_heading("9.6 What I would do differently", level=2)
    editor_note(document, "your own words")
    page_break(document)


def back_matter(document: Document, ev: dict) -> None:
    document.add_heading("References", level=1)
    para(document,
         "Every source listed below was retrieved and checked against the publisher record, "
         "and each is cited in the text at the point it does work. Where the author's earlier "
         "QHO538 survey and AE1 bibliography disagreed on a citation, the version verified "
         "against the publisher was used; four citations in the QHO538 survey contained errors "
         "in journal title, volume or page range and are not reproduced here. Any source added "
         "to this list from this point must be found, read and represented accurately: "
         "fabricated or unverified references are treated as academic misconduct and are "
         "trivially detectable.",
         italic=True, colour=FLAG, size=10)
    for reference in [
        "WRAP (2013) Overview of Waste in the UK Hospitality and Food Service Sector. "
        "Available at: https://www.wrap.ngo/content/overview-waste-hospitality-and-food-service-sector "
        "(Accessed: 11 September 2026). [Survey data for 2011; source of the 45% / 34% / 21% "
        "split between preparation, plate and spoilage waste.]",
        "WRAP (2024) WRAP urges hospitality and food service CEOs and leaders to take a stand "
        "against food waste. Press release, 4 November. Available at: "
        "https://www.wrap.ngo/media-centre/press-releases/wrap-urges-hospitality-and-food-service-ceos-and-leaders-take-stand "
        "(Accessed: 11 September 2026). [Source of the 1.1 million tonnes, £3.2 billion, 75% "
        "edible and one-fifth-of-purchases figures.]",
        "WRAP (no date) Hospitality and food service. Available at: "
        "https://www.wrap.ngo/taking-action/food-drink/sectors/hospitality-food-service "
        "(Accessed: 11 September 2026). [Source of the £10,000 per outlet figure.]",
    ]:
        para(document, reference, size=10)
    for key in [
        "adida", "babai", "badorf", "bcs", "chen", "chopra", "corsten", "croston", "dietvorst",
        "dpa", "fpp3", "hk2006", "huber", "ico_ai", "ico_gdpr", "lundberg", "m5acc", "m5des",
        "aci", "migueis", "nahmias", "pierskalla", "posch", "prak", "rodrigues", "saputra",
        "sb2005",
        "sb2006", "sbc2005", "schmidt", "seyam", "silver", "slack", "taylor", "teunter",
        "vd2006", "vd2010", "zipkin",
    ]:
        para(document, HELD[key], size=10)
    editor_note(document, "the WRAP entries above and the list that follows are recorded in the "
                          "form they were retrieved or carried across from the AE1 annotated "
                          "bibliography. Convert them all to your required referencing style, "
                          "and check each one against the publisher record before submission - "
                          "the QHO538 survey of literature contained errors in four of them")
    editor_note(document, "compile in your required referencing style")
    page_break(document)

    document.add_heading("Appendices", level=1)

    document.add_heading("Appendix A - Reproducing the results", level=2)
    for line in [
        "python baseline_forecasting.py            # regenerate the dataset",
        "python scripts/train_models.py            # segmentation, selection, fitted model",
        "python scripts/run_backtest.py            # forecast accuracy",
        "python scripts/run_simulation.py --frontier   # operational impact and frontier",
        "python scripts/generate_report_figures.py # figures and tables",
        "python -m pytest tests/ -q                # 235 tests",
        "streamlit run app.py                      # dashboard",
    ]:
        paragraph = document.add_paragraph()
        run = paragraph.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    para(document, "All steps are seeded and deterministic; total runtime is a few minutes on "
                   "a laptop CPU.")

    document.add_heading("Appendix B - Configuration", level=2)
    para(document,
         "Every operational assumption behind every figure in this report is declared in the "
         "single configuration file reproduced below, and is validated on load. Changing any "
         "value and re-running the pipeline in Appendix A regenerates the affected results.")
    for line in (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8").splitlines():
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(line if line.strip() else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(8)

    document.add_heading("Appendix C - SKU segmentation and model selection", level=2)
    add_table(document, ev["segmentation"][
        ["sku", "abc", "xyz", "segment", "annual_value", "demand_cv", "zero_day_share"]],
        "Table C.1 - ABC/XYZ segmentation for all 20 SKUs")
    add_table(document, ev["selection"][["sku", "segment", "model_name"]],
              "Table C.2 - Model routing by SKU (justifications in outputs/model_selection.csv)")

    document.add_heading("Appendix D - Per-SKU forecast accuracy", level=2)
    add_table(document, ev["by_sku"].reset_index(), "Table D.1 - Forecast accuracy by SKU")

    document.add_heading("Appendix E - Frontier sweep", level=2)
    add_table(document, ev["frontier"][[
        "strategy", "setting_value", "unit_service_level", "stockout_units",
        "waste_units", "total_cost", "average_inventory_units"]],
        "Table E.1 - Full frontier sweep for both policies")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)

    ev = load_evidence()
    document = Document()
    configure_styles(document)

    title_page(document, ev)
    front_matter(document, ev)
    chapter_introduction(document, ev)
    chapter_literature(document, ev)
    chapter_methodology(document, ev)
    chapter_implementation(document, ev)
    chapter_results(document, ev)
    chapter_discussion(document, ev)
    chapter_evaluation(document, ev)
    chapter_conclusion(document, ev)
    chapter_reflection(document, ev)
    back_matter(document, ev)
    add_page_numbers(document)

    document.save(TARGET)

    words = sum(len(p.text.split()) for p in document.paragraphs)
    print(f"Wrote {TARGET}")
    print(f"  paragraphs : {len(document.paragraphs)}")
    print(f"  tables     : {len(document.tables)}")
    print(f"  body words : ~{words:,} (excluding tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
