"""Left-to-right rectangular phylogram — the presentation view of the
alignment (molecular distance or annotation depth on x, taxonomy order on
y, query labels in a collision-free aligned margin with leader lines).

Also exports the fitted tree as Newick with branch lengths, so external
renderers (ete3, ggtree, iTOL) can be used without adding a Qt dependency
to this package.
"""
from __future__ import annotations

import numpy as np

from .plots import AGREEMENT_COLORS, _mpl, _short
from .tree import ancestors


def _phylo_layout(tree, node_positions=None):
    """x = cumulative distance (node_positions) or annotation depth;
    y = leaves evenly spaced in taxonomy (DFS) order, internal nodes at the
    mean of their children."""
    ys, order = {}, []

    def dfs(v):
        kids = tree["children"][v]
        if not kids:
            order.append(v)
            ys[v] = float(len(order))
            return ys[v]
        ys[v] = float(np.mean([dfs(c) for c in kids]))
        return ys[v]

    dfs("root")
    if node_positions is None:
        xs = {v: float(len(ancestors(tree, v))) for v in ys}
    else:
        xs = {v: float(node_positions[v]) for v in ys}
    return {v: (xs[v], ys[v]) for v in ys}, order


def _draw_backbone(ax, tree, coords, color="#b0b0b0", lw=0.8):
    for v, kids in tree["children"].items():
        if not kids:
            continue
        xv = coords[v][0]
        kid_ys = [coords[c][1] for c in kids]
        ax.plot([xv, xv], [min(kid_ys), max(kid_ys)], color=color, lw=lw,
                zorder=1)
        for c in kids:
            ax.plot([xv, coords[c][0]], [coords[c][1], coords[c][1]],
                    color=color, lw=lw, zorder=1)


def to_newick(tree, edge_lengths=None, node=None):
    """Newick string for `tree`; `edge_lengths` (node -> length, chain-
    collapsed nodes may be absent = 0) attaches branch lengths for external
    renderers (ete3, ggtree, iTOL)."""
    node = node or "root"
    kids = tree["children"][node]
    ln = ""
    if edge_lengths is not None and node != "root":
        ln = f":{edge_lengths.get(node, 0.0):.6g}"
    safe = str(node).replace("(", "").replace(")", "").replace(",", "|") \
                    .replace(":", "_").replace(";", "")
    if not kids:
        return f"'{safe}'{ln}"
    inner = ",".join(to_newick(tree, edge_lengths, k) for k in kids)
    return (f"({inner})'{safe}'{ln};" if node == "root"
            else f"({inner})'{safe}'{ln}")


def plot_alignment_phylogram(summary_rows, tree, node_positions=None,
                             label_full=True, figsize=None):
    """Presentation phylogram of the alignment. x = cumulative molecular
    distance when `node_positions` is given (from
    branch_fit.fit_branch_lengths — report its fit quality alongside),
    otherwise annotation depth. Queries sit at their selected nodes with the
    same agreement symbology as plot_alignment_tree; labels occupy evenly
    spaced slots in an aligned right margin (collision-free by
    construction), connected by dotted leaders."""
    plt = _mpl()
    coords, leaf_order = _phylo_layout(tree, node_positions)
    n = len(leaf_order)
    figsize = figsize or (10, max(6, 0.16 * n))
    fig, ax = plt.subplots(figsize=figsize)
    _draw_backbone(ax, tree, coords)
    xmax = max(x for x, _ in coords.values())
    margin_x = xmax * 1.06 + (0.4 if node_positions is None else 0.02 * xmax)

    placed = []
    for r in summary_rows:
        cat = r.get("agreement", "walk_only")
        col = AGREEMENT_COLORS.get(cat, "#000000")
        w, t = r.get("walk_selected"), r.get("transport_node")
        wa = r.get("walk_decision_support")
        wa = 1.0 if wa is None or (isinstance(wa, float) and np.isnan(wa)) else wa
        ta = r.get("transport_mass", 1.0) or 0.0
        combined = w and t and (w == t or cat == "topologically_equivalent")
        anchor = w or t
        if anchor is None:
            continue
        if combined:
            ax.scatter([coords[w][0]], [coords[w][1]], marker="s", s=44,
                       color=col, alpha=max(0.35, wa), zorder=3,
                       edgecolor="#08306b"
                       if cat == "topologically_equivalent" else "none",
                       linewidth=1.1)
        else:
            if w and t:
                ax.plot([coords[w][0], coords[t][0]],
                        [coords[w][1], coords[t][1]], color=col, lw=0.8,
                        alpha=0.65, zorder=2)
            if w:
                ax.scatter([coords[w][0]], [coords[w][1]], marker="o", s=36,
                           color=col, alpha=max(0.35, wa), zorder=3)
            if t:
                ax.scatter([coords[t][0]], [coords[t][1]], marker="D", s=32,
                           color=col, alpha=max(0.35, ta), zorder=3)
        placed.append((coords[anchor], r["query"], col))

    # aligned margin labels: evenly spaced slots ordered by anchor y
    placed.sort(key=lambda p: p[0][1])
    if placed:
        slot_y = np.linspace(1, n, len(placed)) if len(placed) > 1 \
            else [placed[0][0][1]]
        for ((xa, ya), q, col), sy in zip(placed, slot_y):
            ax.plot([xa, margin_x * 0.995], [ya, sy], color=col, lw=0.5,
                    ls=":", alpha=0.7, zorder=2)
            ax.text(margin_x, sy, q if label_full else _short(q),
                    fontsize=6.5, va="center", ha="left", color="#111111")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c,
                          markeredgecolor="#08306b"
                          if k == "topologically_equivalent" else "none",
                          label=k)
               for k, c in AGREEMENT_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color="grey",
                           label="Walk"),
                plt.Line2D([], [], marker="D", ls="", color="grey",
                           label="Transport")]
    ax.legend(handles=handles, fontsize=6, loc="lower left", frameon=False)
    ax.set_xlim(-0.02 * xmax, margin_x + 0.45 * xmax)
    ax.set_ylim(n + 1, 0)
    ax.set_yticks([])
    ax.set_xlabel("cumulative fitted transcriptomic distance "
                  "(median leaf pair = 1)" if node_positions is not None
                  else "annotation depth (curatorial level, not molecular "
                       "distance)", fontsize=8)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_title("MetaArbor alignment phylogram")
    fig.tight_layout()
    return fig, ax
