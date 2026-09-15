# Forecasting-Driven Inventory Resupply for Restaurants using AI

Final-year dissertation project (**QHO656**, BSc (Hons) Computing, Southampton Solent University / QAHE).
A decision-support tool that forecasts item-level (SKU) demand for a restaurant and turns those
forecasts into cost-aware order recommendations, explicitly modelling UK bank-holiday, weather and
seasonality effects.

**Research question:** *Can an AI forecasting system that incorporates holiday/event calendars and
historical usage reduce stockouts and food waste versus current manual methods in a
quick-service/casual restaurant context?*

**Author:** Natan Rocha Paiva &nbsp;·&nbsp; **Supervisor:** Tendi Mhlanga

The system is **decision support only**. It recommends order quantities and explains them; it never
places an order.

---

## Headline findings

**Forecast accuracy** — rolling-origin backtest, 12 expanding-window folds, 7-day horizon, 20 SKUs:

| Model | sMAPE | MASE | WAPE |
|---|---|---|---|
| Naïve | 46.65 | 1.20 | 42.68 |
| Seasonal Naïve | 39.84 | 0.93 | 39.41 |
| Croston (SBA) | 41.73 | 1.06 | 38.76 |
| ETS (Holt–Winters) | 34.22 | 0.71 | 31.02 |
| **Gradient Boosting** | **34.11** | **0.65** | **29.25** |

Averaged over all SKUs. Croston is worse on average precisely because it is only appropriate for the
four intermittent SKUs; the model-selection layer routes it there rather than applying one model
everywhere.

**Operational impact** — six-month holdout replay (Jul–Dec 2025), manual par-level baseline versus
forecast-driven ordering, **compared at a matched service level of 97.90%**:

| Measure | Manual baseline | AI policy | Change |
|---|---|---|---|
| Waste (units) | 3,374 | 2,078 | **−38.4%** |
| Total cost | £18,202 | £15,679 | **−13.9%** |
| Average inventory (units) | 92.7 | 70.4 | **−24.0%** |

Read at matched *waste* instead, the forecast-driven policy serves **1.6 percentage points** more
demand. Both framings come from `outputs/frontier.csv`; see *Comparing the policies fairly* below
for why a single pair of runs is not sufficient evidence.

---

## The pipeline

```
DATA → FEATURES → FORECAST → INVENTORY POLICY → ORDER RECOMMENDATION
                                  ↓                      ↓
                              SIMULATION            EXPLANATION
                                  ↓                      ↓
                              EVALUATION            DASHBOARD
```

## Getting started

Requires **Python 3.12+**. Everything runs on a normal laptop CPU; there is no deep learning and no
GPU dependency.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Reproduce every result, in order:

```bash
python baseline_forecasting.py           # regenerate the dataset + the original AE1 figures
python scripts/train_models.py           # segment SKUs, select and fit models, write the registry
python scripts/run_backtest.py           # forecast accuracy  → outputs/forecast_accuracy.csv
python scripts/run_simulation.py --frontier   # operational impact → outputs/simulation_comparison.csv
streamlit run app.py                     # the manager-facing dashboard
```

Every step is seeded with `random_seed: 42` and is deterministic: re-running regenerates identical
data, metrics and simulation results.

## Dashboard

`streamlit run app.py` opens five pages. The page and decision date are held in the URL, so a view
can be bookmarked or shared (`?page=Forecast&as_of=2025-12-01`).

| Page | What it shows |
|---|---|
| **Overview** | SKUs requiring orders, stockout and waste risk, projected costs, total order value |
| **Forecast** | Historical demand, the 7-day forecast with its uncertainty interval, and the drivers behind it |
| **Recommendations** | The order table with a reason per SKU, the full calculation, and manager override |
| **What-if** | Controlled changes to demand, temperature, promotion, service level, lead time and stock |
| **Simulation & Evaluation** | Baseline versus AI, the service-level frontier, and per-SKU results |

## Writing up

Report assets are generated, not hand-made, so a figure can never disagree with the
numbers behind it:

```bash
python scripts/generate_report_figures.py    # -> outputs/report/
```

This produces eight 300 dpi figures and seven tables (CSV plus paste-ready Markdown).

- [`docs/evidence_pack.md`](docs/evidence_pack.md) - every result with the file it came
  from, the command that regenerates it, and the caveat that must travel with it.
- [`docs/report_evidence_map.md`](docs/report_evidence_map.md) - which evidence supports
  which section, and where the evidence is thin enough to need qualifying.

## Repository structure

```
config/config.yaml              every operational assumption (horizons, costs, supplier, policy)
src/
  config.py                     Pydantic models validating that configuration
  segmentation.py               ABC/XYZ segmentation
  scenarios.py                  what-if scenario definitions
  app_services.py               orchestration shared by the dashboard and the scripts
  viz.py                        Plotly figures
  data/                         loader, validation, time-aware slicing (the leakage guard)
  features/                     calendar, weather, demand features (lags, rolling statistics)
  forecasting/                  base interface, naive, seasonal naive, ETS, Croston/SBA,
                                gradient boosting, model selection, forecasting service
  inventory/                    safety stock, shelf life, policies, ordering, strategies, simulator
  explainability/               forecast drivers, recommendation reasons
  evaluation/                   forecast metrics, inventory KPIs, backtest, frontier
  persistence/                  SQLite schema and repository
app.py                          Streamlit dashboard
scripts/                        train_models.py, run_backtest.py, run_simulation.py
tests/                          312 tests, including dedicated leakage tests
baseline_forecasting.py         the original AE1 experiment, preserved unchanged
```

## How the system works

### Forecasting

Every model implements one interface — `fit(history)` / `predict(future)` — and returns a
`ForecastResult` carrying the SKU, forecast dates, point forecast, prediction interval, model name
and `sigma`, the estimated one-day forecast error. `sigma` is what sizes safety stock, so the
forecasting and inventory layers cannot drift apart.

### SKU segmentation and model selection

SKUs are segmented **ABC** by annualised purchase value and **XYZ** by demand variability, then
routed to a model with a recorded reason (`outputs/model_selection.csv`):

* 16 smooth SKUs → gradient boosting, which can use day of week, holidays, weather and recent demand;
* 4 intermittent SKUs (≥25% zero-demand days) → Croston with the Syntetos–Boylan bias correction.

A/B/C class also differentiates the service-level target, since cover on a cheap, erratic item is
usually paid for in waste.

### Inventory policy

The policy layer takes a demand forecast and a stock position and knows nothing about how the
forecast was produced — which is what allows the simulation to hold the policy fixed and vary only
the demand signal. Base-stock `(R,S)` and `(s,S)` are both implemented.

```
inventory position = on hand + on order − backorders
protection period  = lead time + review period
safety stock       = z(service level) × σ × √(protection period)
target stock       = expected demand over the protection period + safety stock
recommended order  = max(0, target stock − inventory position)
```

The raw quantity is then constrained, in a fixed and recorded order: shelf-life cap → maximum days
of supply → pack-size rounding → minimum order quantity. Every adjustment is retained on the
recommendation and shown to the manager.

### Perishability

Stock is held as dated batches, issued first-expired-first-out, and written off when it passes its
use-by date — *before* demand is served, so expired stock can never satisfy a sale. The simulator
records usable stock, expired stock, stockouts, incoming stock and inventory age separately.

### Explainability

The dashboard explains both the forecast and the order:

* **Why this much demand** — feature contributions estimated by counterfactual ablation: the model
  re-predicts the day with one input reset to a typical mid-week value, and the difference is that
  input's contribution. These are *not* additive, so the unattributed remainder is shown rather
  than hidden.
* **Why this much stock** — every figure in the narrative is read from the recommendation itself, so
  the text cannot drift from the arithmetic.

Language is deliberately associative ("contributed to the prediction"), never causal. Croston and
the naive family have no features to attribute and are explained through their own parameters.

## Comparing the policies fairly

Any replenishment policy can trade waste for availability by holding more stock, so **two runs at
different service levels cannot be ranked**. The manual baseline's flat 20% buffer happens to land
at a 98.9% fill rate while the AI policy targets 95%; comparing those two runs directly would say
the AI wastes 65% less *and* stocks out 55% more, which is an artefact of the operating point, not
a finding.

`scripts/run_simulation.py --frontier` therefore sweeps both policies across their tuning parameter
(service level for the AI policy, buffer percentage for the manual one) and compares them where
they meet. The AI frontier lies below and to the right of the baseline's throughout the overlapping
range: less waste at equal availability, more availability at equal waste.

**Two limitations are reported rather than hidden.** First, the baseline runs without the
shelf-life cap (`baseline.apply_shelf_life_cap: false`), because a par level set from average usage
embodies no perishability logic — this is the one deliberate asymmetry between the policies, and it
is what lets the baseline reach 99.3% availability at the cost of 25,726 wasted units. Second, the
AI policy saturates near 98.5% availability: for a three-day-shelf-life product with a two-day lead
time, availability beyond that point cannot be bought with stock alone.

## Data-leakage protection

This is the guarantee the operational results rest on, so it is enforced in one place and tested
directly.

* All training data is obtained through `history_as_of(sales, as_of)`, which returns observations
  **strictly before** the decision date. A decision on day *t* cannot see day *t*'s demand.
* Lag features use only prior observations; rolling statistics are computed **within** each SKU and
  **frozen at the forecast origin**, so no horizon day can update them.
* Calendar effects and planned promotions are treated as known in advance, which is realistic.
* Temperature is **not**: `WeatherProvider` degrades future temperatures with noise that grows with
  lead time, because a manager ordering on Monday has a weather forecast for Wednesday, not the
  outcome. Set `weather.perfect_foresight: true` to quantify the difference.
* The simulator serves demand only *after* the ordering decision for that day is made.

The leakage tests multiply future demand by 100 (or zero it) and assert that recommendations,
simulated orders and backtest predictions before the cut-off are bit-for-bit unchanged. See
`tests/test_simulator.py::TestNoDataLeakage`, `tests/test_evaluation.py::TestBacktest` and
`tests/test_end_to_end.py::TestSystemLevelLeakage`.

### Two defects found in the AE1 baseline

Both are documented here because they change how the original numbers should be read.

1. **`roll28` was corrupted.** In `baseline_forecasting.py:233`, `.rolling()` was applied to an
   ungrouped Series so the window crossed SKU boundaries, and `.reset_index(drop=True)` then
   re-aligned the values onto the wrong rows. Measured against a correct `groupby.transform`
   version: correlation 0.19, and 87% of rows differing by more than one unit. The published 33.67%
   sMAPE was achieved with that feature acting as noise.
2. **`roll28` was also leaky.** Features were computed once over the whole frame, so for horizon day
   7 the 28-day window included actual demand from days 1–6 of the evaluation window.

`src/features/demand_features.py` fixes both. The corrected, leak-free pipeline scores 34.11 sMAPE
versus the published 33.67 — statistically the same performance, now obtained legitimately.
`baseline_forecasting.py` is deliberately left untouched as a record of the AE1 submission.

### One model limitation handled as a business rule

The site closes on Christmas Day, but with only one closure in the training history a tree model
learns that rule on some training cut-offs and not others — leaving a manager ordering fresh stock
for a day the restaurant is shut. A closed restaurant selling nothing is a business fact, not a
statistical pattern, so the forecast is set to zero on known closure days by a deterministic rule
applied after prediction, recorded in the forecast metadata and switchable via
`features.zero_demand_on_closed_days`.

## Configuration

Every assumption lives in `config/config.yaml` and is validated by Pydantic on load, so an
out-of-range service level or a missing cost fails immediately with a useful message rather than
producing a quietly wrong recommendation. Nothing operational is hard-coded in `src/` or in the
dashboard.

Cost figures (unit costs, 25% annual holding rate, waste at cost plus disposal, stockout at twice
unit cost) are **stated assumptions**, not observed data, and should be cited as such.

## Testing

```bash
python -m pytest tests/ -q            # 312 tests, about a minute
python -m pytest -m "not slow" -q     # skip the full-dataset end-to-end tests
```

Coverage includes feature generation and horizon construction, inventory position, safety stock,
reorder quantities, shelf-life expiry, pack-size rounding, minimum order quantities, stockout and
waste accounting, simulation chronology and stock conservation, data leakage, and one end-to-end
test running sample data → features → forecast → recommendation → simulation → stored result.

## Data and ethics

The dataset is **synthetic** and self-generated (14,620 rows; 20 SKUs; 1 Jan 2024 – 31 Dec 2025). It
contains no personal, guest or commercially sensitive data. Effect sizes (weekend uplift, bank
holiday spikes, weather sensitivity, promotions, intermittent slow movers) are calibrated to
magnitudes reported in the demand-forecasting literature. `baseline_forecasting.py` regenerates it
deterministically from seed 42.

## Roadmap

- [x] Synthetic dataset + feature engineering (calendar, holidays, weather, lags)
- [x] Baseline back-test: Naïve, Seasonal Naïve, ETS, gradient boosting (rolling-origin CV)
- [x] Croston / SBA for intermittent SKUs, with ABC/XYZ-driven model selection
- [x] Inventory simulator: base-stock and (s,S) with lead time, shelf life and a cost model
- [x] Streamlit dashboard: forecasts, recommendations, driver explanations, what-if scenarios
- [x] Full evaluation: accuracy metrics plus simulated operational KPIs, compared at matched service level
- [ ] Pilot testing with a real site's data
