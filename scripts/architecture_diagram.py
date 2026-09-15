"""Section 4.1 architecture diagram, drawn from the src/ tree itself.

Each layer box lists the modules actually present on disk, so the diagram cannot drift from
the code. Row heights come from the tallest box in each row, boxes in a row share a top
edge, and connectors are orthogonal paths through the gutters between boxes, so no arrow
crosses a box or its text. Colours reuse the design tokens in src/viz.py.

Usage: python scripts/architecture_diagram.py [repo_root] [output_png]
"""
import pathlib
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1])
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "outputs" / "report" / "fig_architecture.png")

SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

COL_X = [0.3, 4.55, 8.8]
BOX_W = 3.75
HEADER_H = 0.55
LINE_H = 0.34
PAD_TOP, PAD_BOTTOM = 0.22, 0.28
ROW_GAP = 0.95
ENTER = HEADER_H + 0.32  # connectors meet a side edge just below the header band


def modules(*parts):
    """Python module names in a src/ folder, or the single file if the path is a file."""
    path = ROOT.joinpath(*parts)
    if path.is_file():
        return [path.name]
    return sorted(p.name for p in path.glob("*.py") if p.name != "__init__.py")


@dataclass
class Box:
    key: str
    title: str
    mods: list
    col: int
    row: int
    colour: str
    top: float = 0.0

    @property
    def x(self):
        return COL_X[self.col]

    @property
    def h(self):
        return HEADER_H + PAD_TOP + LINE_H * max(len(self.mods), 1) + PAD_BOTTOM

    @property
    def bottom(self):
        return self.top - self.h

    @property
    def cx(self):
        return self.x + BOX_W / 2

    @property
    def right(self):
        return self.x + BOX_W


BOXES = [
    Box("config", "Configuration", ["config/config.yaml", *modules("src", "config.py")], 0, 0, MUTED),
    Box("data", "Data", modules("src", "data"), 1, 0, BLUE),
    Box("features", "Features", modules("src", "features"), 1, 1, BLUE),
    Box("forecasting", "Forecasting and model selection",
        modules("src", "forecasting") + modules("src", "segmentation.py"), 1, 2, BLUE),
    Box("evaluation", "Evaluation", modules("src", "evaluation"), 0, 3, MUTED),
    Box("inventory", "Inventory policy and ordering", modules("src", "inventory"), 1, 3, ORANGE),
    Box("explain", "Explainability", modules("src", "explainability"), 2, 3, AQUA),
    Box("scripts", "Batch scripts", modules("scripts"), 0, 4, MUTED),
    Box("app", "Dashboard and services", ["app.py", *modules("src", "app_services.py"),
                                          *modules("src", "viz.py"), *modules("src", "scenarios.py")], 1, 4, AQUA),
    Box("persistence", "Persistence (SQLite)", modules("src", "persistence"), 1, 5, MUTED),
]


def layout():
    rows = sorted({b.row for b in BOXES})
    heights = {r: max(b.h for b in BOXES if b.row == r) for r in rows}
    top = 0.0
    for r in rows:
        for b in BOXES:
            if b.row == r:
                b.top = top
        top -= heights[r] + ROW_GAP
    return {b.key: b for b in BOXES}


def connector(ax, *points):
    xs, ys = zip(*points)
    ax.add_line(Line2D(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=MUTED, linewidth=1.2, solid_capstyle="butt"))
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, mutation_scale=12, shrinkA=0, shrinkB=0))


def main():
    b = layout()
    connections = [
        (b["config"].right, b["config"].top - ENTER, b["data"].x, b["data"].top - ENTER),
        ("v", "data", "features"), ("v", "features", "forecasting"), ("v", "forecasting", "inventory"),
        (b["inventory"].right, b["inventory"].top - ENTER, b["explain"].x, b["explain"].top - ENTER),
        (b["inventory"].x, b["inventory"].top - ENTER, b["evaluation"].right, b["evaluation"].top - ENTER),
        ("v", "inventory", "app"), ("v", "app", "persistence"),
    ]
    bottom = min(box.bottom for box in BOXES)
    width, height = COL_X[-1] + BOX_W + 0.3, 2.0 - bottom + 0.3
    fig, ax = plt.subplots(figsize=(11.0, 11.0 * height / width), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for box in BOXES:
        ax.add_patch(FancyBboxPatch((box.x, box.bottom), BOX_W, box.h, boxstyle="round,pad=0,rounding_size=0.12",
                                    linewidth=1.4, edgecolor=box.colour, facecolor="white", zorder=2))
        ax.add_patch(FancyBboxPatch((box.x, box.top - HEADER_H), BOX_W, HEADER_H,
                                    boxstyle="round,pad=0,rounding_size=0.12", linewidth=0, facecolor=box.colour,
                                    zorder=3))
        ax.text(box.x + 0.15, box.top - HEADER_H / 2, box.title, color="white", fontsize=10, fontweight="bold",
                va="center", zorder=4)
        for i, name in enumerate(box.mods):
            ax.text(box.x + 0.18, box.top - HEADER_H - PAD_TOP - LINE_H * (i + 0.5), name, color=TEXT, fontsize=8.4,
                    va="center", family="monospace", zorder=4)

    for spec in connections:
        if spec[0] == "v":
            upper, lower = b[spec[1]], b[spec[2]]
            connector(ax, (upper.cx, upper.bottom), (lower.cx, lower.top))
        else:
            x0, y0, x1, y1 = spec
            connector(ax, (x0, y0), (x1, y1))
    # Forecasting feeds the explainer: out of its right edge, along the empty right column, down.
    fc, ex = b["forecasting"], b["explain"]
    connector(ax, (fc.right, fc.bottom + 0.35), (ex.cx, fc.bottom + 0.35), (ex.cx, ex.top))
    # Explanations reach the dashboard: down the right column, then left into its right edge.
    ap = b["app"]
    connector(ax, (ex.cx, ex.bottom), (ex.cx, ap.top - ENTER), (ap.right, ap.top - ENTER))
    # Batch scripts drive the evaluation.
    sc, ev = b["scripts"], b["evaluation"]
    connector(ax, (sc.cx, sc.top), (ev.cx, ev.bottom))

    ax.text(COL_X[0], 1.45, "System architecture", fontsize=14, fontweight="bold", color=TEXT)
    ax.text(COL_X[0], 0.85, "Modules as they exist in the repository. Arrows show the direction of data flow.",
            fontsize=9.5, color=MUTED)
    ax.set_xlim(0, width)
    ax.set_ylim(bottom - 0.3, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")
    for box in BOXES:
        print(f"  {box.title}: {len(box.mods)} modules, top {box.top:.2f}, bottom {box.bottom:.2f}")
    overlaps = [(p.key, q.key) for i, p in enumerate(BOXES) for q in BOXES[i + 1:]
                if p.col == q.col and not (p.bottom >= q.top or q.bottom >= p.top)]
    print("overlapping boxes:", overlaps or "none")
    return 1 if overlaps else 0


if __name__ == "__main__":
    sys.exit(main())
