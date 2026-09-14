# Report evidence map

A section-by-section map of which evidence supports which part of the write-up, so that
drafting is a matter of explaining results you already have rather than hunting for them.

**What this is not:** it is not draft text. The argument, the critical discussion, the
literature engagement and the reflection are the assessed contribution and need to be in
your own words. What follows tells you which figure, table and number belongs where, and
flags the places where the evidence is thin so you can qualify the claim rather than
overstate it.

Numbers, sources and caveats: [`docs/evidence_pack.md`](evidence_pack.md).

---

## Introduction

Evidence available: research question (README), the manual-ordering problem the baseline
policy formalises (`src/inventory/policies.py::ParLevelPolicy`).

You write: motivation, scope, aims and objectives, contribution statement.

---

## Literature review

Evidence available: none — this section is entirely yours.

Where the implementation makes a claim that needs a citation, the code comments say so
explicitly. Specifically, you will need sources for:

- Croston's method and the Syntetos–Boylan bias correction (`src/forecasting/croston.py`
  states the `1 − α/2` correction and why classic Croston is positively biased).
- ABC/XYZ segmentation as a stock-policy differentiator.
- The safety-stock formula `z × σ × √(L+R)` and its normality assumption.
- sMAPE / MASE / WAPE selection, and the known instability of sMAPE on intermittent series.
- Rolling-origin evaluation as the correct protocol for time series.
- Food-waste magnitudes in hospitality, to contextualise the 0.9–2.4% of stock received.

---

## Methodology

| Sub-section | Evidence | Notes |
|---|---|---|
| Data | `data/README.md`, `baseline_forecasting.py` | State synthetic provenance up front |
| Features | `src/features/` | Calendar known in advance; weather deliberately *not* |
| Segmentation | Figure 2, Table 5 | A class = 80.7% of value |
| Model selection | Table 6 | The `reason` column is quotable per SKU |
| Forecasting models | `src/forecasting/` | One interface, six models |
| Inventory policy | README "How the system works" | The five formulas in sequence |
| Perishability | `src/inventory/shelf_life.py` | FEFO, expiry before demand is served |
| Simulator | `src/inventory/simulator.py` docstring | The 6-step daily sequence |
| Evaluation protocol | §1 and §6 of the evidence pack | Rolling origin + leakage guarantees |

The leakage section (§6) is the strongest methodological material you have. Give it real
space: the guarantee, the mechanism, and the test that proves it.

---

## Results

Suggested order, because it moves from intermediate to operational outcome:

1. **Forecast accuracy** — Figure 1, Table 1. Lead with MASE 0.65 (scale-free, beats the
   benchmark). Note the ETS gap is small and that WAPE is the operationally relevant metric.
2. **Per-SKU accuracy** — `forecast_accuracy_by_sku.csv`. Range 15.6 to 119.0 sMAPE.
   Explain *why* intermittent SKUs score badly rather than just reporting it.
3. **Worked forecast** — Figure 3.
4. **Explainability** — Figure 8, with the three limitations from evidence pack §5.
5. **Operational outcomes, descriptive** — Figure 5, Table 2. Immediately flag that the
   two runs sit at different service levels.
6. **Operational outcomes, matched** — Figure 4, Tables 3 and 4. This is the result.
7. **Behaviour over time and by SKU** — Figures 6 and 7.

RQ coverage checklist — the research question asks for five things:

| Required evidence | Where | Status |
|---|---|---|
| Forecast accuracy | Table 1, Figure 1 | Complete |
| Stockout reduction | Table 4 (matched waste): +1.2–1.6 pp | Complete, but modest — do not overstate |
| Waste reduction | Table 3: −25.8% to −38.4% | Complete, strongest result |
| Service-level improvement | Table 4 | Complete |
| Inventory / cost impact | Table 3: inventory −24.0%, cost −10.4% to −13.9% | Complete |

---

## Discussion

Points the evidence genuinely supports:

- The gain concentrates in short-shelf-life SKUs (Figure 7) — the system helps where
  perishability binds, which is a mechanism, not just a number.
- Better forecasts convert into *less stock for the same availability* (inventory −24%),
  which is the operational route by which accuracy becomes waste reduction.
- Availability saturates near 98.5% for 3-day-shelf-life items: beyond that point
  availability cannot be bought with stock, only with shorter lead times or more frequent
  delivery. This is a finding about perishable inventory, worth discussing.
- Model selection matters: Croston is worse on average yet correct for 4 SKUs.

Points the evidence does **not** support — avoid claiming these:

- Any effect size transferable to a real restaurant (synthetic data).
- That the AI policy dominates on every metric at every setting (it does not above ~98.5%
  availability).
- Causal statements from feature contributions.

---

## Evaluation, limitations and future work

Take all six threats to validity from evidence pack §7. The two that most need your own
judgement:

- the deliberate shelf-life-cap asymmetry between the policies, and whether to run the
  sensitivity check that removes it;
- the synthetic-data circularity, and what a pilot with real data would need to establish.

Future work that follows directly from what exists: a denser frontier sweep, lead-time
sensitivity, a real-site pilot, and the `(s,S)` policy comparison (implemented and tested
but not yet swept).

---

## Ethics, legal, social, professional

Evidence available: no personal or guest data anywhere in the project; the dataset is
synthetic and regenerable; the system is decision support and never places an order; the
interface states this on every page and in every recommendation explanation; overrides are
stored separately from recommendations so a manager's decision never rewrites the model's
output, and the trained model is never modified by an override.

You write: the analysis itself, plus anything your ethics submission requires.

---

## Appendices

- Appendix: repository structure and how to reproduce (README "Getting started").
- Appendix: full configuration (`config/config.yaml`) — every assumption in one place.
- Appendix: test suite summary (227 tests) — `python -m pytest tests/ -q`.
- Appendix: full per-SKU results (Table 7), segmentation (Table 5), model selection (Table 6).

---

## Before you submit

- [ ] Every quoted number traced to a file in `outputs/` (evidence pack has the mapping).
- [ ] Synthetic-data caveat stated once, early, and not repeated defensively.
- [ ] The AE1 `roll28` correction disclosed, with the 33.67 → 34.11 change explained.
- [ ] The descriptive comparison (Table 2) never presented as the headline result.
- [ ] Waste and stockout claims given as ranges across the matched operating points.
- [ ] Feature contributions described as association, never cause.
- [ ] Cost figures labelled as configured assumptions.
- [ ] Figures legible in greyscale if the submission may be printed mono (marker shape
      and direct labels already carry the information, so this should hold).
- [ ] Declare AI assistance as your institution requires.
