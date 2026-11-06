"""One consistent, quiet figure style for every chart in results/figures.

Rules followed (so the figures stay readable for colour-blind readers and in
print): fixed categorical colour order, thin marks, hairline solid grid, text
in grey ink (never in the series colour), a legend whenever there are two
or more series, and every figure has a CSV table with the same numbers in
results/tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots, always used in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

FUEL_COLOURS = {"Jet A-1": SERIES[0], "SAF50": SERIES[1], "SAF100": SERIES[2]}
AIRCRAFT_COLOURS = {"Concorde": SERIES[0], "Overture-like": SERIES[1], "A350-900": SERIES[2]}

FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
TABLE_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_2,
        "font.family": "sans-serif",
        "font.size": 9.5,
        "text.color": INK,
    })


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(df, name: str, float_format: str = "%.4g"):
    """Write a table as CSV (machine-readable) and Markdown (for the docs)."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, float_format=float_format)
    (TABLE_DIR / f"{name}.md").write_text(_to_markdown(df), encoding="utf-8")
    return TABLE_DIR / f"{name}.csv"


def _fmt(v):
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        if abs(v) >= 1e5 or (abs(v) < 1e-2 and v != 0):
            return f"{v:.3g}"
        return f"{v:.3g}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)


def _to_markdown(df) -> str:
    """Tiny Markdown writer (avoids the optional 'tabulate' dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"
