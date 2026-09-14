# Evidence pack

Every result the system produces, with the file it came from, the command that
regenerates it, and the caveat that has to travel with it. Nothing here is a draft of the
dissertation — it is the raw material to write from, so that any figure quoted in the
report can be traced back to a command.

**Regenerate everything:**

```bash
python baseline_forecasting.py                 # dataset + original AE1 figures
python scripts/train_models.py                 # segmentation, model selection, fitted model
python scripts/run_backtest.py                 # forecast accuracy
python scripts/run_simulation.py --frontier    # operational impact + frontier
python scripts/generate_report_figures.py      # report figures and tables
```

All steps are seeded (`random_seed: 42`) and deterministic. Total runtime is roughly four
minutes on a laptop CPU.

---

## 1. Study design

| Item | Value | Source |
|---|---|---|
| Dataset | 14,620 rows, 20 SKUs, 7 categories, 1 Jan 2024 – 31 Dec 2025 (731 days) | `data/sales_data.csv` |
| Data type | Synthetic, self-generated, no personal or guest data | `baseline_forecasting.py` |
| Forecast horizon | 7 days (one delivery cycle) | `config/config.yaml` |
| Validation | Rolling-origin, 12 expanding-window folds, 1,440 SKU-fold-model scores | `outputs/backtest_metrics.csv` |
| Holdout simulation | 1 Jul – 31 Dec 2025 (184 days), 3,680 SKU-days per policy | `outputs/simulation_daily_*.csv` |
| Supplier | 2-day lead time, orders Mon/Wed/Fri (3 deliveries per week) | `config/config.yaml` |
| Inventory policy | Base-stock (R,S), 95% service level, (s,S) also implemented | `src/inventory/policies.py` |

**Caveat to state once, early, and not repeat:** the dataset is synthetic. It demonstrates
that the *method* works and lets the mechanism be inspected exactly, but it cannot
establish the effect size a real restaurant would see. The effect sizes built into the
generator (weekend uplift, holiday spikes, weather sensitivity) are the ones the forecast
then rediscovers, which is a circularity a reader will spot if it is not acknowledged.

---

## 2. Forecast accuracy (RQ evidence 1)

**Source:** `outputs/forecast_accuracy.csv` · **Figure:** `outputs/report/fig01_forecast_accuracy.png` · **Table:** `table1_forecast_accuracy`

| Model | sMAPE | MASE | WAPE | vs Seasonal Naïve |
|---|---|---|---|---|
| **Gradient Boosting** | **34.11** | **0.65** | **29.25** | **−14.4%** |
| ETS (Holt–Winters) | 34.22 | 0.71 | 31.02 | −14.1% |
| Seasonal Naïve | 39.84 | 0.93 | 39.41 | — |
| Croston | 41.28 | 1.06 | 39.01 | +3.6% |
| Croston (SBA) | 41.73 | 1.06 | 38.76 | +4.7% |
| Naïve | 46.65 | 1.20 | 42.68 | +17.1% |

Points worth making:

- MASE 0.65 < 1 means the model beats an in-sample seasonal naïve benchmark; this is the
  scale-free claim, and it is the one to lead with.
- The gap to ETS is small (0.11 sMAPE points). Gradient boosting's real advantage is
  WAPE (29.25 vs 31.02) — it is more accurate on high-volume days, which is where
  ordering errors actually cost money.
- Croston looks *worse* on average because it is averaged over all 20 SKUs, including 16
  it was never intended for. This is an argument for model selection, not against
  Croston; per-SKU numbers are in `outputs/forecast_accuracy_by_sku.csv`.
- Prediction-interval coverage is 96.9% against a nominal 95% — the intervals are
  slightly conservative, which is worth one sentence since safety stock depends on them.
- Accuracy varies enormously by SKU: Milk 15.64 sMAPE, Truffle Oil 118.97. Intermittent
  SKUs inflate sMAPE by construction (a zero-demand day with a small positive forecast
  scores 200%). Do not present a single average without this qualification.

---

## 3. Operational impact (RQ evidence 2, 3, 4, 5)

### 3a. The two policies at their configured settings

**Source:** `outputs/simulation_comparison.csv` · **Figure:** `fig05_kpi_comparison.png` · **Table:** `table2_operational_outcomes`

| Measure | Manual baseline | AI policy | Change |
|---|---|---|---|
| Waste (units) | 6,574 | 2,364 | −64.0% |
| Waste (% of stock received) | 2.43% | 0.90% | −63.1% |
| Waste cost | £6,903 | £3,972 | −42.5% |
| Stockout units | 2,959 | 4,540 | **+53.4%** |
| Service level (fill rate) | 98.89% | 98.30% | −0.6% |
| Average inventory (units) | 120.9 | 78.6 | −35.0% |
| Holding cost | £221 | £170 | −23.2% |
| Total cost | £16,386 | £15,455 | −5.7% |

**Do not report this table on its own as the headline result.** The two policies are at
different operating points, so the comparison is not like-for-like — see §3b. Present it
as descriptive, then explain why a fair comparison requires the frontier. Doing this
deliberately is a methodological strength; being caught not doing it is a serious weakness.

### 3b. Compared at matched service level (the defensible result)

**Source:** `outputs/matched_service_level.csv` · **Figure:** `fig04_service_level_frontier.png` (key figure) · **Table:** `table3_matched_service_level`

| Matched service level | Waste (baseline → AI) | Total cost | Average inventory |
|---|---|---|---|
| 96.76% | 2,499 → 1,854 units (**−25.8%**) | £22,502 → £20,162 (−10.4%) | −24.0% |
| 97.90% | 3,374 → 2,078 units (**−38.4%**) | £18,202 → £15,679 (−13.9%) | −24.0% |

**Source:** `outputs/matched_waste.csv` · **Table:** `table4_matched_waste`

| Matched waste | Service level (baseline → AI) | Gain |
|---|---|---|
| 1,638 units | 94.80% → 95.94% | +1.15 pp |
| 2,499 units | 96.76% → 98.38% | +1.62 pp |

How to phrase the claim: *at equal availability the forecast-driven policy wastes 26–38%
less; at equal waste it serves 1.2–1.6 percentage points more demand.* Give the range, not
one number — two overlapping operating points is thin evidence for a point estimate.

### 3c. Where the gain comes from

**Source:** `outputs/simulation_by_sku_*.csv` · **Figure:** `fig07_waste_by_sku.png` · **Table:** `table7_per_sku_outcomes`

Only short-shelf-life SKUs can waste at all: 9 of 20 SKUs record zero waste under both
policies because their shelf life exceeds the ordering cycle. The waste reduction is
concentrated in Burger Buns, Chicken Breast and Lettuce. This is the honest mechanism —
the system helps most where perishability actually binds — and it is a better finding than
a single aggregate percentage.

---

## 4. Segmentation and model selection

**Source:** `outputs/sku_segmentation.csv`, `outputs/model_selection.csv` · **Figure:** `fig02_segmentation.png` · **Tables:** `table5_segmentation`, `table6_model_selection`

- ABC: 10 A / 5 B / 5 C. The A class holds **80.7%** of annualised purchase value — a
  textbook Pareto distribution, worth one sentence because it justifies differentiating
  service levels by class.
- XYZ: 16 X / 2 Y / 2 Z.
- Six segments occur: AX (10), BX (4), CX (2), CY (2), BZ (1), CZ (1).
- Routing: 16 SKUs → gradient boosting, 4 → Croston (SBA), each with a recorded
  justification in `table6_model_selection` (the `reason` column is written for a human
  reader and can be quoted directly).

---

## 5. Explainability

**Figure:** `fig08_forecast_drivers.png` — Burger Buns, week of the August 2025 bank holiday.

The mechanism: counterfactual ablation. The model re-predicts each day with one input
reset to a typical mid-week value; the difference is that input's contribution.

Three limitations that must be stated whenever the figure is used:

1. Contributions are **not additive** — the model contains interactions, so they do not
   sum to the forecast. The unattributed remainder is reported in the interface rather
   than hidden.
2. They show **association learned from historical data, not cause**. The wording
   throughout the system is "contributed to the prediction"; keep that wording in the report.
3. Croston and the naïve family expose no feature contributions and are explained through
   their own parameters instead (average demand size and inter-demand interval).

---

## 6. Data-leakage protection (methodology defence)

This is likely the most examinable part of the work. The guarantee, the mechanism, and
the test:

| Guarantee | Mechanism | Test |
|---|---|---|
| A decision on day *t* sees only data before *t* | All training data flows through `history_as_of()` | `tests/test_features.py::TestFutureFeatures` |
| Rolling statistics cannot update mid-horizon | Frozen at the forecast origin | `test_rolling_features_are_frozen_at_the_origin` |
| Lag features never cross SKU boundaries | `groupby.transform` | `test_rolling_mean_does_not_cross_sku_boundaries` |
| Weather is forecast, not observed | `WeatherProvider` adds lead-time-dependent noise | `TestWeatherProvider` |
| Demand is served only after ordering | Fixed daily sequence in the simulator | `test_the_order_decision_precedes_the_days_demand` |
| Future demand cannot change a past order | — | `TestNoDataLeakage`, `TestSystemLevelLeakage` |

The strongest tests multiply future demand by 100 (or zero it) and assert that every
recommendation, simulated order and backtest prediction *before* the cut-off is unchanged.

### Two defects found in the AE1 baseline — disclose these

1. **`roll28` was corrupted.** `baseline_forecasting.py:233` applied `.rolling()` to an
   ungrouped Series (so the window crossed SKU boundaries) and then `.reset_index(drop=True)`
   re-aligned the values onto the wrong rows. Measured against a correct implementation:
   correlation **0.19**, with **87%** of rows differing by more than one unit.
2. **`roll28` was also leaky** — computed once over the whole frame, so horizon day 7 saw
   actual demand from days 1–6 of its own evaluation window.

Consequence: the published **33.67%** sMAPE was obtained with that feature acting as
noise. The corrected, leak-free pipeline scores **34.11%** — statistically the same
performance, now obtained legitimately. `baseline_forecasting.py` is deliberately left
unmodified as a record of what was submitted.

This is a genuine strength if you present it as one: you audited your own earlier work,
quantified the error, and the conclusion survived.

### One limitation handled as a business rule

The site closes on Christmas Day, but with a single closure in the training history the
tree model learned that rule on some training cut-offs and not others — leaving a manager
ordering fresh stock for a day the restaurant is shut. The forecast is now set to zero on
known closure days by a deterministic post-processing rule, recorded in the forecast
metadata (`features.zero_demand_on_closed_days`). Worth a paragraph: it illustrates where
a business rule is more reliable than a learned pattern.

---

## 7. Threats to validity — say these before an examiner does

1. **Synthetic data.** The effect sizes recovered are those built into the generator. The
   contribution is the method and the artefact, not the effect size.
2. **The one asymmetry between policies.** The baseline runs without the shelf-life cap
   (`baseline.apply_shelf_life_cap: false`), because a par level from average usage
   embodies no perishability logic. This lets the baseline reach 99.3% availability at the
   cost of 25,726 wasted units. It is configurable, so a sensitivity run removing the
   asymmetry is a one-line change and would strengthen the chapter.
3. **The AI policy saturates near 98.5% availability.** For a 3-day-shelf-life product
   with a 2-day lead time, more availability cannot be bought with stock alone. Report
   this as a finding about perishable inventory, not as a defect.
4. **Only two overlapping operating points** on the frontier. A denser sweep would support
   a firmer claim; the current evidence supports a range, not a point estimate.
5. **Costs are assumptions, not observations.** Unit costs, the 25% annual holding rate,
   waste at cost plus £0.05 disposal, and stockout at 2× unit cost are all declared in
   `config/config.yaml`. The percentage reductions in waste and inventory are robust to
   them; the absolute £ figures are not.
6. **Single site, single simulated period.** No confidence intervals across replications —
   the simulation is deterministic given the data, so uncertainty comes from the data, not
   the method.

---

## 8. Figures and tables index

| File | Content | Suggested use |
|---|---|---|
| `fig01_forecast_accuracy.png` | Six models × three metrics | Results — forecasting |
| `fig02_segmentation.png` | ABC/XYZ with model routing | Method — model selection |
| `fig03_forecast_vs_actual.png` | 7-day forecast, interval, actuals | Results — worked example |
| `fig04_service_level_frontier.png` | **Key figure** — waste vs service level | Results — operational impact |
| `fig05_kpi_comparison.png` | KPIs indexed to baseline | Results — descriptive comparison |
| `fig06_over_time.png` | Waste / unmet demand / stock over 184 days | Results — behaviour over time |
| `fig07_waste_by_sku.png` | Per-SKU waste, baseline → AI | Discussion — where the gain arises |
| `fig08_forecast_drivers.png` | Driver contributions, bank-holiday week | Results — explainability |
| `table1`–`table7` (`.csv` and `.md`) | Every table above, paste-ready | Throughout |
| `headline_numbers.json` | Machine-readable headline figures | Abstract / conclusion |

Existing AE1 figures (`outputs/fig1`–`fig4`) remain valid for the dataset description and
project-plan sections.
