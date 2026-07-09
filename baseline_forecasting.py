"""
QHO656 Dissertation - Forecasting-Driven Inventory Resupply for Restaurants
Baseline forecasting experiment (feasibility evidence for AE1 Progress Report, Section 3.3-3.4).

Generates a reproducible 24-month synthetic single-site restaurant sales dataset with
weekly seasonality, UK bank-holiday effects, weather sensitivity and promotions, then
back-tests four forecasting approaches with rolling-origin cross-validation:
    - Naive            (last observed value carried forward)
    - Seasonal Naive   (previous week repeated)
    - ETS              (Holt-Winters additive, weekly seasonality)
    - Gradient Boosting (feature-based global ML model, LightGBM stand-in)

Metrics: sMAPE, MASE, WAPE. Outputs Figures 1-4 and results_table.csv for Appendix C.

Author: Natan Rocha Paiva
Reproducible: fixed random seed. Run:  python baseline_forecasting.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings, json, os
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data"); os.makedirs(DATA, exist_ok=True)
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)

# Clean, colour-blind-friendly palette (Okabe-Ito subset)
C = {"naive": "#999999", "snaive": "#E69F00", "ets": "#0072B2",
     "gb": "#009E73", "actual": "#333333", "accent": "#D55E00"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# --------------------------------------------------------------------------------------
# 1. UK bank holidays (England & Wales), 2024-2025
# --------------------------------------------------------------------------------------
BANK_HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
    "2024-08-26", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
    "2025-08-25", "2025-12-25", "2025-12-26",
])
CLOSED_DAYS = pd.to_datetime(["2024-12-25", "2025-12-25"])  # restaurant closed Christmas Day

def school_holiday(d):
    # Approximate England school holidays: summer, Christmas, Easter, half-terms
    md = (d.month, d.day)
    if d.month in (7, 8):                     return 1            # summer
    if d.month == 12 and d.day >= 20:         return 1            # Christmas
    if d.month == 1 and d.day <= 2:           return 1
    if d.month == 4 and 1 <= d.day <= 14:     return 1            # Easter
    if (d.month, d.day) >= (10, 26) and (d.month, d.day) <= (10, 31): return 1  # Oct half-term
    if d.month == 2 and 12 <= d.day <= 18:    return 1            # Feb half-term
    if d.month == 5 and 26 <= d.day <= 31:    return 1            # May half-term
    return 0

# --------------------------------------------------------------------------------------
# 2. SKU catalogue (single casual-dining site)
# --------------------------------------------------------------------------------------
# base = mean daily units; wknd = Fri/Sat/Sun uplift; temp = weather sensitivity;
# hol = bank-holiday uplift; inter = intermittency (prob of a zero-demand day); shelf_life days
SKUS = [
    # fast movers (smooth, high volume)
    ("Burger Buns",       "Bakery",    120, 1.55, 0.05,  0.45, 0.00, 3),
    ("Fries (Potatoes)",  "Produce",   140, 1.50, 0.10,  0.40, 0.00, 7),
    ("Chicken Breast",    "Meat",       95, 1.45, 0.05,  0.40, 0.00, 4),
    ("Milk",              "Dairy",      80, 1.20, 0.05,  0.20, 0.00, 6),
    ("Lettuce",           "Produce",    70, 1.40, 0.30,  0.35, 0.00, 4),
    ("Tomatoes",          "Produce",    75, 1.40, 0.25,  0.35, 0.00, 5),
    ("Cheese Slices",     "Dairy",      85, 1.45, 0.05,  0.40, 0.00, 10),
    ("Soft Drinks",       "Beverage",  110, 1.60, 0.55,  0.50, 0.00, 90),
    ("Coffee Beans",      "Beverage",   45, 1.10, -0.20, 0.15, 0.00, 120),
    ("Cooking Oil",       "Ambient",    30, 1.15, 0.00,  0.20, 0.00, 180),
    # medium movers
    ("Bacon",             "Meat",       55, 1.40, 0.05,  0.35, 0.02, 7),
    ("Salmon Fillet",     "Fish",       35, 1.55, 0.05,  0.45, 0.05, 3),
    ("Mushrooms",         "Produce",    40, 1.35, 0.05,  0.30, 0.03, 5),
    ("Onions",            "Produce",    60, 1.30, 0.05,  0.25, 0.00, 21),
    ("Butter",            "Dairy",      38, 1.25, 0.05,  0.25, 0.02, 30),
    ("Eggs",              "Dairy",      65, 1.30, 0.05,  0.25, 0.00, 21),
    # slow / intermittent (condiments, seasonal, specials)
    ("Truffle Oil",       "Ambient",     6, 1.60, 0.00,  0.55, 0.45, 240),
    ("Vegan Cheese",      "Dairy",       9, 1.50, 0.05,  0.40, 0.35, 14),
    ("Gluten-free Buns",  "Bakery",     12, 1.50, 0.05,  0.45, 0.30, 3),
    ("Seasonal Berries",  "Produce",     8, 1.55, 0.40,  0.50, 0.40, 3),
]

# --------------------------------------------------------------------------------------
# 3. Generate the dataset
# --------------------------------------------------------------------------------------
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
n = len(dates)
doy = dates.dayofyear.values
# synthetic daily mean temperature (deg C): annual cycle + noise
temp = 11 + 8 * np.sin(2 * np.pi * (doy - 110) / 365.25) + RNG.normal(0, 2.2, n)
is_bh = dates.isin(BANK_HOLIDAYS).astype(int)
is_closed = dates.isin(CLOSED_DAYS).astype(int)
sch = np.array([school_holiday(d) for d in dates])
dow = dates.dayofweek.values                      # 0=Mon
weekend = np.isin(dow, [4, 5, 6]).astype(int)     # Fri/Sat/Sun busy
# promotions: ~8% of days per SKU, uplift ~35%
rows = []
for (name, cat, base, wknd, tsens, hol, inter, shelf) in SKUS:
    trend = np.linspace(1.0, 1.12, n)             # slow 12% growth over 2 years
    seasonal_year = 1 + 0.12 * np.sin(2 * np.pi * (doy - 40) / 365.25)
    dow_factor = np.where(weekend == 1, wknd, 1.0)
    # midweek dip
    dow_factor = dow_factor * np.where(np.isin(dow, [0, 1]), 0.9, 1.0)
    temp_factor = 1 + tsens * (temp - temp.mean()) / 10.0
    hol_factor = 1 + hol * is_bh
    promo = (RNG.random(n) < 0.08).astype(int)
    promo_factor = 1 + 0.35 * promo
    mu = base * trend * seasonal_year * dow_factor * temp_factor * hol_factor * promo_factor
    mu = mu * (1 - is_closed)                      # closed on Christmas Day
    mu = np.clip(mu, 0.01, None)
    # counts: Poisson for smooth SKUs; zero-inflation for intermittent ones
    y = RNG.poisson(mu)
    if inter > 0:
        zero_mask = RNG.random(n) < inter
        y = np.where(zero_mask & (is_bh == 0), 0, y)
    for i in range(n):
        rows.append((dates[i], name, cat, shelf, int(y[i]), round(float(temp[i]), 1),
                     int(is_bh[i]), int(sch[i]), int(weekend[i]), int(dow[i]), int(promo[i])))

df = pd.DataFrame(rows, columns=["date", "sku", "category", "shelf_life_days", "units_sold",
                                 "temp_c", "bank_holiday", "school_holiday", "weekend",
                                 "dow", "promo"])
df.to_csv(os.path.join(DATA, "sales_data.csv"), index=False)
print(f"Dataset: {len(df):,} rows | {df.sku.nunique()} SKUs | "
      f"{df.date.nunique()} days ({df.date.min().date()} to {df.date.max().date()})")

# --------------------------------------------------------------------------------------
# 4. EDA figures
# --------------------------------------------------------------------------------------
DAYNAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Figure 1: weekly seasonality
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
tot = df.groupby("date")["units_sold"].sum()
by_dow = df.groupby("dow")["units_sold"].sum().reindex(range(7))
by_dow = by_dow / df.groupby("dow")["date"].nunique().reindex(range(7))
ax[0].bar(DAYNAMES, by_dow.values, color=[C["accent"] if i >= 4 else C["ets"] for i in range(7)])
ax[0].set_title("(a) Mean total units by day of week")
ax[0].set_ylabel("Mean units/day")
sample = tot.loc["2025-06-01":"2025-07-13"]
ax[1].plot(sample.index, sample.values, marker="o", ms=3, color=C["ets"])
for d in sample.index:
    if d.dayofweek in (4, 5, 6):
        ax[1].axvspan(d - pd.Timedelta("0.5D"), d + pd.Timedelta("0.5D"), color=C["accent"], alpha=0.08)
ax[1].set_title("(b) Six-week window (weekends shaded)")
ax[1].set_ylabel("Total units/day")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")
fig.suptitle("Figure 1 - Weekly seasonality in daily sales", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_weekly_seasonality.png"), bbox_inches="tight")
plt.close(fig)

# Figure 2: holiday / event spikes
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
d2025 = tot.loc["2025-01-01":"2025-12-31"]
ax[0].plot(d2025.index, d2025.rolling(7, center=True).mean(), color=C["ets"], lw=1.3, label="7-day mean")
bh25 = BANK_HOLIDAYS[BANK_HOLIDAYS.year == 2025]
ax[0].scatter(bh25, tot.reindex(bh25).values, color=C["accent"], zorder=5, s=30, label="Bank holiday")
ax[0].set_title("(a) 2025 daily sales with bank holidays")
ax[0].set_ylabel("Total units/day")
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax[0].legend(fontsize=8)
hol_mean = tot.reindex(BANK_HOLIDAYS).mean()
open_bh = [d for d in BANK_HOLIDAYS if d not in CLOSED_DAYS]
normal_mean = tot[~tot.index.isin(BANK_HOLIDAYS)].mean()
openbh_mean = tot.reindex(open_bh).mean()
ax[1].bar(["Normal\nday", "Bank holiday\n(open)"], [normal_mean, openbh_mean],
          color=[C["ets"], C["accent"]])
ax[1].set_title("(b) Mean demand: normal vs holiday")
ax[1].set_ylabel("Total units/day")
upl = 100 * (openbh_mean - normal_mean) / normal_mean
ax[1].text(1, openbh_mean, f"+{upl:.0f}%", ha="center", va="bottom", fontweight="bold")
fig.suptitle("Figure 2 - Holiday and event demand spikes", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_holiday_spikes.png"), bbox_inches="tight")
plt.close(fig)
print(f"Bank-holiday (open) demand uplift vs normal day: +{upl:.0f}%")

# --------------------------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------------------------
def smape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    d = np.abs(y) + np.abs(f)
    return np.mean(np.where(d == 0, 0.0, 2 * np.abs(f - y) / d)) * 100

def wape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    s = np.abs(y).sum()
    return np.abs(f - y).sum() / s * 100 if s > 0 else np.nan

def mase(y, f, insample, m=7):
    y, f = np.asarray(y, float), np.asarray(f, float)
    ins = np.asarray(insample, float)
    denom = np.mean(np.abs(ins[m:] - ins[:-m])) if len(ins) > m else np.nan
    return np.mean(np.abs(f - y)) / denom if denom and denom > 0 else np.nan

# --------------------------------------------------------------------------------------
# 6. Rolling-origin cross-validation, horizon = 7 days (one delivery cycle)
# --------------------------------------------------------------------------------------
H = 7
N_FOLDS = 12
FEATS = ["dow", "weekend", "bank_holiday", "school_holiday", "promo", "temp_c",
         "week_of_year", "month", "lag7", "lag14", "roll28", "sku_code"]

wide = df.pivot(index="date", columns="sku", values="units_sold").sort_index()
skus = list(wide.columns)
sku_code = {s: i for i, s in enumerate(skus)}
all_dates = wide.index
origins = [len(all_dates) - H * (N_FOLDS - k) for k in range(N_FOLDS)]  # expanding window

# precompute feature frame for the ML model
feat_df = df.copy()
feat_df["week_of_year"] = feat_df["date"].dt.isocalendar().week.astype(int)
feat_df["month"] = feat_df["date"].dt.month
feat_df["sku_code"] = feat_df["sku"].map(sku_code)
feat_df = feat_df.sort_values(["sku", "date"])
feat_df["lag7"] = feat_df.groupby("sku")["units_sold"].shift(7)
feat_df["lag14"] = feat_df.groupby("sku")["units_sold"].shift(14)
feat_df["roll28"] = feat_df.groupby("sku")["units_sold"].shift(1).rolling(28).mean().reset_index(0, drop=True)
feat_df = feat_df.set_index(["date", "sku"]).sort_index()

records = []
for k, o in enumerate(origins):
    tr_dates = all_dates[:o]
    te_dates = all_dates[o:o + H]
    if len(te_dates) < H:
        continue
    for s in skus:
        hist = wide[s].loc[tr_dates]
        actual = wide[s].loc[te_dates].values
        insample = hist.values
        # Naive
        f_naive = np.repeat(hist.iloc[-1], H)
        # Seasonal naive: repeat last 7 days
        last_week = hist.iloc[-7:].values
        f_snaive = last_week[:H]
        # ETS (Holt-Winters, weekly)
        try:
            ets = ExponentialSmoothing(hist.values, trend="add", seasonal="add",
                                       seasonal_periods=7, initialization_method="estimated").fit()
            f_ets = np.clip(ets.forecast(H), 0, None)
        except Exception:
            f_ets = f_snaive.astype(float)
        for name, fc in [("Naive", f_naive), ("Seasonal Naive", f_snaive), ("ETS (Holt-Winters)", f_ets)]:
            records.append((k, s, name, smape(actual, fc), wape(actual, fc),
                            mase(actual, fc, insample)))

    # Gradient boosting: one global model per fold, trained on all SKU-days < origin
    train = feat_df[feat_df.index.get_level_values(0).isin(tr_dates)].dropna(subset=["lag7", "lag14", "roll28"])
    test = feat_df[feat_df.index.get_level_values(0).isin(te_dates)]
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6,
                                       random_state=42, categorical_features=[FEATS.index("sku_code")])
    gb.fit(train[FEATS], train["units_sold"])
    for s in skus:
        sub = test[test.index.get_level_values(1) == s].sort_index()
        if len(sub) < H or sub[["lag7", "lag14", "roll28"]].isna().any().any():
            continue
        pred = np.clip(gb.predict(sub[FEATS]), 0, None)
        actual = sub["units_sold"].values
        insample = wide[s].loc[tr_dates].values
        records.append((k, s, "Gradient Boosting", smape(actual, pred), wape(actual, pred),
                        mase(actual, pred, insample)))

res = pd.DataFrame(records, columns=["fold", "sku", "model", "sMAPE", "WAPE", "MASE"])

# --------------------------------------------------------------------------------------
# 7. Results table (mean across folds x SKUs)
# --------------------------------------------------------------------------------------
order = ["Naive", "Seasonal Naive", "ETS (Holt-Winters)", "Gradient Boosting"]
table = res.groupby("model")[["sMAPE", "MASE", "WAPE"]].mean().reindex(order).round(2)
base = table.loc["Seasonal Naive", "sMAPE"]
table["sMAPE vs SNaive"] = ((table["sMAPE"] - base) / base * 100).round(1)
table.to_csv(os.path.join(OUT, "results_table.csv"))
print("\n=== Table 1: Rolling-origin back-test (7-day horizon, 12 folds x 20 SKUs) ===")
print(table.to_string())

best = table["sMAPE"].idxmin()
gb_impr = -table.loc["Gradient Boosting", "sMAPE vs SNaive"]
ets_impr = -table.loc["ETS (Holt-Winters)", "sMAPE vs SNaive"]

# --------------------------------------------------------------------------------------
# 8. Figure 3: forecast vs actual (fast mover, last fold)
# --------------------------------------------------------------------------------------
demo_sku = "Fries (Potatoes)"
o = origins[-1]
tr_dates, te_dates = all_dates[:o], all_dates[o:o + H]
hist = wide[demo_sku].loc[tr_dates]
ctx = hist.iloc[-21:]
actual = wide[demo_sku].loc[te_dates]
f_snaive = hist.iloc[-7:].values[:H]
try:
    ets = ExponentialSmoothing(hist.values, trend="add", seasonal="add", seasonal_periods=7,
                               initialization_method="estimated").fit()
    f_ets = np.clip(ets.forecast(H), 0, None)
except Exception:
    f_ets = f_snaive.astype(float)
train = feat_df[feat_df.index.get_level_values(0).isin(tr_dates)].dropna(subset=["lag7", "lag14", "roll28"])
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42,
                                   categorical_features=[FEATS.index("sku_code")]).fit(train[FEATS], train["units_sold"])
sub = feat_df[(feat_df.index.get_level_values(0).isin(te_dates)) &
              (feat_df.index.get_level_values(1) == demo_sku)].sort_index()
f_gb = np.clip(gb.predict(sub[FEATS]), 0, None)
fig, ax = plt.subplots(figsize=(9.5, 3.8))
ax.plot(ctx.index, ctx.values, color=C["actual"], lw=1.2, label="History")
ax.plot(actual.index, actual.values, color=C["actual"], lw=2, marker="o", ms=4, label="Actual")
ax.plot(te_dates, f_snaive, color=C["snaive"], ls="--", marker="s", ms=3, label="Seasonal Naive")
ax.plot(te_dates, f_ets, color=C["ets"], ls="--", marker="^", ms=3, label="ETS")
ax.plot(te_dates, f_gb, color=C["gb"], ls="--", marker="D", ms=3, label="Gradient Boosting")
ax.axvline(te_dates[0], color="grey", ls=":", lw=1)
ax.set_title(f"Figure 3 - Sample 7-day forecast vs actual ({demo_sku})", fontweight="bold")
ax.set_ylabel("Units/day"); ax.legend(fontsize=8, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_forecast_vs_actual.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------------------
# 9. Figure 4: Gantt chart matching the Section 4 milestone table (June-Nov 2026)
# --------------------------------------------------------------------------------------
tasks = [
    ("Scoping, ethics, background reading", "2026-06-01", "2026-06-21", "done"),
    ("Data ingestion, cleaning, features", "2026-06-15", "2026-06-30", "done"),
    ("Baseline & initial ML modelling",     "2026-07-01", "2026-07-31", "prog"),
    ("Inventory policy & simulator",        "2026-08-01", "2026-08-31", "todo"),
    ("Prototype dashboard (Streamlit)",     "2026-09-01", "2026-09-21", "todo"),
    ("Evaluation & pilot testing",          "2026-09-22", "2026-10-19", "todo"),
    ("Final write-up & refinement",         "2026-10-20", "2026-11-16", "todo"),
    ("Submission & buffer",                 "2026-11-17", "2026-11-30", "todo"),
]
colmap = {"done": C["gb"], "prog": C["ets"], "todo": "#CCCCCC"}
fig, ax = plt.subplots(figsize=(9.5, 4.2))
for i, (t, s, e, st) in enumerate(tasks):
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    ax.barh(i, (e - s).days, left=s, color=colmap[st], edgecolor="white")
ax.axvline(pd.Timestamp("2026-07-07"), color=C["accent"], lw=1.6, ls="--")
ax.text(pd.Timestamp("2026-07-08"), len(tasks) - 0.3, "now (7 Jul)", color=C["accent"], fontsize=8)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in tasks], fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["gb"], label="Complete"), Patch(color=C["ets"], label="In progress"),
                   Patch(color="#CCCCCC", label="Planned")], fontsize=8, loc="lower right")
ax.set_title("Figure 4 - Project timeline / Gantt (400 hours, Jun-Nov 2026)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_gantt.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------------------
# 10. Save headline numbers for the report
# --------------------------------------------------------------------------------------
summary = {
    "rows": int(len(df)), "skus": int(df.sku.nunique()), "days": int(df.date.nunique()),
    "date_start": str(df.date.min().date()), "date_end": str(df.date.max().date()),
    "n_features_engineered": 12, "n_folds": N_FOLDS, "horizon_days": H,
    "holiday_uplift_pct": round(float(upl)),
    "best_model": best,
    "gb_smape": float(table.loc["Gradient Boosting", "sMAPE"]),
    "ets_smape": float(table.loc["ETS (Holt-Winters)", "sMAPE"]),
    "snaive_smape": float(table.loc["Seasonal Naive", "sMAPE"]),
    "naive_smape": float(table.loc["Naive", "sMAPE"]),
    "gb_improvement_vs_snaive_pct": round(float(gb_impr), 1),
    "ets_improvement_vs_snaive_pct": round(float(ets_impr), 1),
    "table": table.reset_index().to_dict(orient="records"),
}
with open(os.path.join(OUT, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nBest model: {best} | GB sMAPE {summary['gb_smape']:.1f}% "
      f"({summary['gb_improvement_vs_snaive_pct']:.1f}% better than Seasonal Naive)")
print("Saved: sales_data.csv, results_table.csv, results_summary.json, fig1-4 PNGs")
