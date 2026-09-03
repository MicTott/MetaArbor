"""Publication style for MetaArbor figures.

One rc context (Helvetica/Arial, editable-text PDF output, disciplined size
hierarchy), the Okabe-Ito colorblind-safe palette, journal-width presets,
and a save helper writing vector PDF + 600 dpi PNG.
"""
from __future__ import annotations

import contextlib

MM = 1 / 25.4
SINGLE_COL = 89 * MM
ONEHALF_COL = 120 * MM
DOUBLE_COL = 180 * MM

OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
QUERY_ACCENT = OKABE_ITO["vermillion"]
QUERY_ACCENT2 = OKABE_ITO["blue"]
TREE_GRAY = "#b9b4ae"
TEXT_DARK = "#2b2b2b"
TEXT_MID = "#5a5a5a"
RULE_GRAY = "#d9d5d0"

SIZES = {"title": 10, "axis": 7.5, "tick": 6.5, "tip": 5.8, "annot": 6.8,
         "caption": 6.2, "legend": 6.5}


@contextlib.contextmanager
def pub_style():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue",
                            "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": 7,
        "axes.titlesize": SIZES["title"], "axes.labelsize": SIZES["axis"],
        "xtick.labelsize": SIZES["tick"], "ytick.labelsize": SIZES["tick"],
        "legend.fontsize": SIZES["legend"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.dpi": 110,
    }
    with plt.rc_context(rc):
        yield plt


def save_pub(fig, path_base, formats=("pdf", "png"), dpi=600):
    """Vector PDF (editable text) + high-resolution PNG."""
    paths = []
    for ext in formats:
        p = f"{path_base}.{ext}"
        fig.savefig(p, dpi=dpi if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        paths.append(p)
    return paths
