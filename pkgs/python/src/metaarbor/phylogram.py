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
    """METHOD-COMPARISON phylogram: Walk vs Transport agreement (circle vs
    diamond, merged when equivalent; orange connectors for same-branch depth
    differences; red only for genuine branch conflicts). Query labels only —
    method agreement is the point here. For the primary biological result
    with every reference leaf labeled, use `plot_harmonized_phylogram`.

    Presentation phylogram of the alignment. x = cumulative molecular
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
    fig.legend(handles=handles, fontsize=6.5, ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.text(0.02, -0.045,
             "Horizontal position = root-to-node distance; the patristic "
             "distance between two nodes is the full path through their "
             "common ancestor (tips at similar x need not be similar).",
             fontsize=6.5, style="italic", color="#333333")
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


def to_ete4(tree, edge_lengths=None):
    """The tree as an `ete4.Tree` (optional dependency: install the `ete`
    extra — ete4 + PyQt6; static `render()` works headlessly with
    QT_QPA_PLATFORM=offscreen, and `explore()` opens the Qt-free
    interactive web explorer)."""
    try:
        from ete4 import Tree as EteTree
    except ImportError as e:
        raise ImportError("to_ete4 needs the `ete` extra: "
                          "pip install metaarbor[ete]") from e
    return EteTree(to_newick(tree, edge_lengths), parser=1)


RELATION_SHAPES = {"equal": "o", "parent_child": "^", "broader_ancestor": "P"}


def _relation_class(tree, node):
    """Tree-intrinsic assignment class on the CHAIN-COLLAPSED topology
    (unary annotation levels do not count as depth): terminal = equal;
    node whose collapsed children are all terminal = parent_child;
    higher = broader_ancestor."""
    from .branch_fit import _collapse_chains
    red_parent, node_map = _collapse_chains(tree)
    r = node_map[node]
    kids_of = {}
    for c, p in red_parent.items():
        kids_of.setdefault(p, []).append(c)
    kids = kids_of.get(r, [])
    if not kids:
        return "equal"
    if all(k not in kids_of or not kids_of[k] for k in kids):
        return "parent_child"
    return "broader_ancestor"


def plot_harmonized_phylogram(summary_rows, tree, method="walk",
                              node_positions=None,
                              ref_name="reference", query_name="query",
                              ref_color="#555555", query_color="#c2410c",
                              figsize=None):
    """HARMONIZED phylogram — the primary biological result, one method at a
    time (`method` = "walk" or "transport").

    Visual grammar (fixed):
      label/attachment COLOR  = dataset (reference vs query)
      marker SHAPE            = relation class (equal / parent-child /
                                broader ancestor; tree-intrinsic)
      line STYLE              = confidence class (solid confident, dashed
                                underconfident/discordant, dotted stub
                                ending before the tree = unmatched)
      line WIDTH/OPACITY      = quantitative support (Walk decision support
                                or vote; Transport row-normalized mass)
    Every reference leaf is labeled at its terminal branch (aligned, thin
    gray leaders); every query label attaches at its inferred node; a query
    mapping to an internal node lightly shades its descendant clade."""
    plt = _mpl()
    coords, leaf_order = _phylo_layout(tree, node_positions)
    n = len(leaf_order)
    figsize = figsize or (13, max(7, 0.13 * n))
    fig, ax = plt.subplots(figsize=figsize)
    _draw_backbone(ax, tree, coords)
    xmax = max(x for x, _ in coords.values())
    ref_margin = xmax * 1.04
    q_margin = xmax * 1.42

    # reference leaf labels, aligned with thin leaders
    for l in leaf_order:
        x, y = coords[l]
        ax.plot([x, ref_margin * 0.997], [y, y], color="#cccccc", lw=0.4,
                ls=":", zorder=1)
        ax.text(ref_margin, y, l, fontsize=4.6, va="center", ha="left",
                color=ref_color)

    def q_fields(r):
        if method == "walk":
            node = r.get("walk_selected")
            rel = r.get("walk_relation")
            sup = r.get("walk_decision_support")
            if sup is None or (isinstance(sup, float) and np.isnan(sup)):
                sup = r.get("walk_vote", 1.0) or 1.0
            style = ("solid" if rel in ("leaf", "family")
                     else "dashed" if rel == "discordant" else "unmatched")
        else:
            node = r.get("transport_node")
            b = r.get("transport_bin", "unmatched")
            sup = r.get("transport_mass", 0.0) or 0.0
            style = ("solid" if b == "confident"
                     else "dashed" if b in ("moderate", "diffuse")
                     else "unmatched")
        return node, style, float(sup)

    placed = []
    for r in summary_rows:
        node, style, sup = q_fields(r)
        placed.append((r["query"], node, style, sup))
    order_y = {q: (coords[nd][1] if nd else n) for q, nd, _, _ in placed}
    placed.sort(key=lambda p: order_y[p[0]])
    slot_y = np.linspace(1, n, len(placed)) if len(placed) > 1 else [n / 2]

    for (q, node, style, sup), sy in zip(placed, slot_y):
        lw = 0.6 + 1.8 * sup
        alpha = 0.35 + 0.6 * sup
        if node is None or style == "unmatched":
            # dotted stub ending before the tree
            ax.plot([q_margin * 0.995, ref_margin + 0.12 * xmax], [sy, sy],
                    color=query_color, lw=0.8, ls=":", alpha=0.8)
            ax.text(q_margin, sy, q, fontsize=6.2, va="center", ha="left",
                    color=query_color, style="italic")
            continue
        x, y = coords[node]
        rel_cls = _relation_class(tree, node)
        if tree["children"][node]:  # internal: shade the descendant clade
            from .tree import leaves_under
            lv = leaves_under(tree, node)
            ys = [coords[l][1] for l in lv]
            xt = max(coords[l][0] for l in lv)
            ax.add_patch(plt.Rectangle((x, min(ys) - 0.45), xt - x + 0.02 * xmax,
                                       max(ys) - min(ys) + 0.9,
                                       facecolor=query_color, alpha=0.07,
                                       edgecolor="none", zorder=0.5))
        ax.scatter([x], [y], marker=RELATION_SHAPES[rel_cls], s=42,
                   color=query_color, alpha=max(0.4, alpha), zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.plot([x, q_margin * 0.995], [y, sy], color=query_color, lw=lw,
                ls="-" if style == "solid" else (0, (4, 2)),
                alpha=alpha, zorder=2)
        ax.text(q_margin, sy, q, fontsize=6.2, va="center", ha="left",
                color=query_color)

    handles = [
        plt.Line2D([], [], color=ref_color, marker="s", ls="",
                   label=f"{ref_name} (reference tree)"),
        plt.Line2D([], [], color=query_color, marker="s", ls="",
                   label=f"{query_name} (queries, {method})"),
        plt.Line2D([], [], color=query_color, marker="o", ls="",
                   label="equal (terminal)"),
        plt.Line2D([], [], color=query_color, marker="^", ls="",
                   label="parent-child (family)"),
        plt.Line2D([], [], color=query_color, marker="P", ls="",
                   label="broader ancestor"),
        plt.Line2D([], [], color=query_color, ls="-", label="confident"),
        plt.Line2D([], [], color=query_color, ls=(0, (4, 2)),
                   label="underconfident / discordant"),
        plt.Line2D([], [], color=query_color, ls=":",
                   label="unmatched (stub)"),
    ]
    fig.legend(handles=handles, fontsize=6.5, ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.text(0.02, -0.05,
             "Line width/opacity = quantitative support. Shaded band = the "
             "descendant clade of an internal-node assignment. Horizontal "
             "position = root-to-node distance; pairwise patristic distance "
             "is the complete path between two nodes.",
             fontsize=6.5, style="italic", color="#333333")
    ax.set_xlim(-0.02 * xmax, q_margin + 0.4 * xmax)
    ax.set_ylim(n + 1, 0)
    ax.set_yticks([])
    ax.set_xlabel("cumulative fitted transcriptomic distance "
                  "(median leaf pair = 1)" if node_positions is not None
                  else "annotation depth (curatorial level, not molecular "
                       "distance)", fontsize=8)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_title(f"MetaArbor harmonized taxonomy ({method}): where every "
                 f"type from both datasets fits", fontsize=10)
    fig.tight_layout()
    return fig, ax
