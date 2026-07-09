# -*- coding: utf-8 -*-
"""Regenerate Figure 4 (Gantt) for a June -> September 2026 timeline."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Patch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
C = {"gb": "#009E73", "ets": "#0072B2", "accent": "#D55E00"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

tasks = [
    ("Scoping & background reading",         "2026-06-01", "2026-06-18", "done"),
    ("Data ingestion, cleaning, features",   "2026-06-12", "2026-06-30", "done"),
    ("Baseline & initial ML modelling",      "2026-06-25", "2026-07-20", "prog"),
    ("Inventory policy & simulator",         "2026-07-15", "2026-07-31", "todo"),
    ("Prototype dashboard (Streamlit)",      "2026-08-01", "2026-08-15", "todo"),
    ("Evaluation & pilot testing",           "2026-08-10", "2026-08-25", "todo"),
    ("Final write-up & refinement",          "2026-08-20", "2026-09-12", "todo"),
    ("Submission & buffer",                  "2026-09-13", "2026-09-20", "todo"),
]
colmap = {"done": C["gb"], "prog": C["ets"], "todo": "#CCCCCC"}
fig, ax = plt.subplots(figsize=(9.5, 4.2))
for i, (t, s, e, st) in enumerate(tasks):
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    ax.barh(i, (e - s).days, left=s, color=colmap[st], edgecolor="white")
ax.axvline(pd.Timestamp("2026-07-09"), color=C["accent"], lw=1.6, ls="--")
ax.text(pd.Timestamp("2026-07-10"), len(tasks) - 0.3, "now (9 Jul)", color=C["accent"], fontsize=8)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in tasks], fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(handles=[Patch(color=C["gb"], label="Complete"), Patch(color=C["ets"], label="In progress"),
                   Patch(color="#CCCCCC", label="Planned")], fontsize=8, loc="lower right")
ax.set_title("Figure 4 - Project timeline / Gantt (400 hours, Jun-Sep 2026)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_gantt.png"), bbox_inches="tight")
plt.close(fig)
print("Regenerated fig4_gantt.png (Jun-Sep 2026)")
