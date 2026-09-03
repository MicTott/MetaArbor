"""Publication phylograms — gutter-track architecture.

Annotations live in aligned vertical tracks BESIDE the tree, never across
it: tree | tip labels | structure strip | annotation lanes. Two views share
the chassis:

- `plot_harmonized_phylogram` (primary biological result, one method at a
  time): each query is a clade BRACKET spanning exactly its assigned rows,
  stacked into non-overlapping lanes; dataset = the only accent color;
  relation class = glyph on the bracket spine; confidence class = line
  style; support = line weight and glyph opacity; unmatched queries are
  listed in an "unplaced" box, not drawn as floating lines.
- `plot_alignment_phylogram` (method comparison): per query a DUMBBELL in
  the gutter — Walk dot and Transport diamond at their assigned rows on
  one vertical connector; merged square when the selections are equivalent;
  orange connector for same-branch depth differences; red reserved for
  genuine branch conflicts.

`to_newick` / `to_ete4` export the (fitted) tree for external renderers.
"""
from __future__ import annotations

import numpy as np

from .style import (OKABE_ITO, QUERY_ACCENT, RULE_GRAY, SIZES, TEXT_DARK,
                    TEXT_MID, TREE_GRAY, pub_style)
from .tree import ancestors, leaves_under

RELATION_SHAPES = {"equal": "o", "parent_child": "^", "broader_ancestor": "P"}


# ---------------------------------------------------------------------------
# layout chassis
# ---------------------------------------------------------------------------

def _phylo_layout(tree, node_positions=None):
    """x = cumulative distance (or annotation depth); y = leaves evenly
    spaced in taxonomy (DFS) order, internal nodes at their children's
    mean."""
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


def _relation_class(tree, node):
    """Assignment class on the CHAIN-COLLAPSED topology (unary annotation
    levels do not count as depth): terminal = equal; collapsed children all
    terminal = parent_child; higher = broader_ancestor."""
    from .branch_fit import _collapse_chains
    red_parent, node_map = _collapse_chains(tree)
    kids_of = {}
    for c, p in red_parent.items():
        kids_of.setdefault(p, []).append(c)
    r = node_map[node]
    kids = kids_of.get(r, [])
    if not kids:
        return "equal"
    if all(not kids_of.get(k) for k in kids):
        return "parent_child"
    return "broader_ancestor"


def _lane_assign(spans, gap=0.6):
    """Greedy non-overlapping lane assignment for row spans."""
    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], spans[i][1]))
    lane_end, lanes = [], [0] * len(spans)
    for i in order:
        s, e = spans[i]
        for L in range(len(lane_end)):
            if s > lane_end[L] + gap:
                lanes[i], lane_end[L] = L, e
                break
        else:
            lane_end.append(e)
            lanes[i] = len(lane_end) - 1
    return lanes, max(1, len(lane_end))


def _leaf_blocks(tree, leaf_order):
    """(top-level ancestor, immediate collapsed parent) per leaf, for the
    structure strip."""
    from .branch_fit import _collapse_chains
    red_parent, node_map = _collapse_chains(tree)
    out = []
    for l in leaf_order:
        par = red_parent.get(node_map[l], "root")
        top = l
        while tree["parent"][top] not in (None, "root"):
            top = tree["parent"][top]
        out.append((top, par))
    return out


def _base_axes(plt, tree, coords, leaf_order, figsize=None):
    """Tree + aligned tip labels + structure strip. Returns geometry."""
    n = len(leaf_order)
    figsize = figsize or (11, max(6.5, 0.105 * n + 1.3))
    fig, ax = plt.subplots(figsize=figsize)
    T = max(x for x, _ in coords.values()) or 1.0
    for v, kids in tree["children"].items():
        if not kids:
            continue
        xv = coords[v][0]
        kid_ys = [coords[c][1] for c in kids]
        ax.plot([xv, xv], [min(kid_ys), max(kid_ys)], color=TREE_GRAY,
                lw=0.7, zorder=1, solid_capstyle="round")
        for c in kids:
            ax.plot([xv, coords[c][0]], [coords[c][1], coords[c][1]],
                    color=TREE_GRAY, lw=0.7, zorder=1,
                    solid_capstyle="round")
    lab_x = 1.03 * T
    for l in leaf_order:
        x, y = coords[l]
        ax.plot([x + 0.004 * T, lab_x - 0.008 * T], [y, y], color=RULE_GRAY,
                lw=0.35, ls=(0, (1, 2.2)), zorder=1)
        ax.text(lab_x, y, l, fontsize=SIZES["tip"], va="center", ha="left",
                color=TEXT_DARK)
    strip_x, strip_w = 1.66 * T, 0.022 * T
    blocks = _leaf_blocks(tree, leaf_order)
    shade, prev = 0, None
    for i, l in enumerate(leaf_order):
        top, par = blocks[i]
        if prev is not None and par != prev[1]:
            shade = 1 - shade
        ax.add_patch(plt.Rectangle((strip_x, i + 0.5), strip_w, 1.0,
                                   facecolor="#e9e5e0" if shade
                                   else "#d6d0c9",
                                   edgecolor="none", zorder=1))
        if prev is not None and top != prev[0]:
            ax.plot([strip_x - 0.012 * T, strip_x + strip_w + 0.012 * T],
                    [i + 0.5] * 2, color="#9c968e", lw=0.6, zorder=2)
        prev = (top, par)
    ax.set_ylim(n + 1, 0)
    ax.set_yticks([])
    return {"fig": fig, "ax": ax, "T": T, "n": n,
            "track_x0": strip_x + strip_w + 0.05 * T}


def _finish(plt, geom, node_positions, handles, caption, title):
    ax, fig, T = geom["ax"], geom["fig"], geom["T"]
    ax.set_xlim(-0.02 * T, geom["xmax_used"])
    ax.set_xticks(np.round(np.linspace(0, T, 4), 2))
    ax.set_xlabel("cumulative fitted transcriptomic distance "
                  "(median leaf pair = 1)" if node_positions is not None
                  else "annotation depth", fontsize=SIZES["axis"],
                  loc="left")
    ax.set_title(title, fontsize=SIZES["title"], loc="left", pad=10,
                 fontweight="bold", color=TEXT_DARK)
    if handles:
        fig.legend(handles=handles, fontsize=SIZES["legend"], ncol=4,
                   frameon=False, loc="upper left",
                   bbox_to_anchor=(0.055, 0.005), handlelength=1.6,
                   columnspacing=1.2)
    import textwrap as _tw
    fig.text(0.055, -0.062, "\n".join(_tw.wrap(caption, 150)),
             fontsize=SIZES["caption"], style="italic", color=TEXT_MID,
             va="top")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# view 1: harmonized (primary biological result)
# ---------------------------------------------------------------------------

def plot_harmonized_phylogram(summary_rows, tree, method="walk",
                              node_positions=None, ref_name="reference",
                              query_name="query", accent=QUERY_ACCENT,
                              figsize=None):
    """Primary biological result, one method at a time: where every type
    from both datasets fits. See module docstring for the visual grammar."""
    with pub_style() as plt:
        coords, leaf_order = _phylo_layout(tree, node_positions)
        geom = _base_axes(plt, tree, coords, leaf_order, figsize)
        ax, T, n = geom["ax"], geom["T"], geom["n"]

        entries, unplaced = [], []
        for r in summary_rows:
            if method == "walk":
                node, rel = r.get("walk_selected"), r.get("walk_relation")
                sup = r.get("walk_decision_support")
                if sup is None or (isinstance(sup, float) and np.isnan(sup)):
                    sup = r.get("walk_vote") or 1.0
                conf = ("solid" if rel in ("leaf", "family")
                        else "dashed" if rel == "discordant" else "un")
            else:
                node = r.get("transport_node")
                b = r.get("transport_bin", "unmatched")
                sup = r.get("transport_mass") or 0.0
                conf = ("solid" if b == "confident"
                        else "dashed" if b in ("moderate", "diffuse")
                        else "un")
            if node is None or conf == "un":
                unplaced.append(r["query"])
                continue
            lv = leaves_under(tree, node)
            rows_ = [coords[l][1] for l in lv]
            entries.append({"q": r["query"], "node": node,
                            "span": (min(rows_), max(rows_)),
                            "sup": float(sup), "style": conf,
                            "rel": _relation_class(tree, node)})

        lanes, n_lanes = _lane_assign([e["span"] for e in entries])
        pitch = 0.11 * T
        x0 = geom["track_x0"]
        lab_col = x0 + n_lanes * pitch + 0.03 * T
        # aligned label column with a minimal-overlap nudge
        mids = [(e["span"][0] + e["span"][1]) / 2 for e in entries]
        order = np.argsort(mids)
        lab_y = np.array(mids, float)
        min_gap = 0.95
        for k in range(1, len(order)):
            a, b = order[k - 1], order[k]
            if lab_y[b] - lab_y[a] < min_gap:
                lab_y[b] = lab_y[a] + min_gap
        for idx, (e, L) in enumerate(zip(entries, lanes)):
            xs = x0 + L * pitch
            y0, y1 = e["span"][0] - 0.18, e["span"][1] + 0.18
            lw = 0.9 + 1.6 * min(e["sup"], 1.0)
            al = 0.55 + 0.45 * min(e["sup"], 1.0)
            ls = "-" if e["style"] == "solid" else (0, (4, 2.2))
            ax.plot([xs, xs], [y0, y1], color=accent, lw=lw, ls=ls,
                    alpha=al, zorder=3, solid_capstyle="butt")
            for yy in (y0, y1):
                ax.plot([xs - 0.016 * T, xs], [yy, yy], color=accent,
                        lw=lw * 0.9, alpha=al, zorder=3)
            ym = (y0 + y1) / 2
            ax.scatter([xs], [ym], marker=RELATION_SHAPES[e["rel"]], s=16,
                       facecolor="white", edgecolor=accent, linewidth=0.9,
                       alpha=max(al, 0.75), zorder=4)
            if abs(lab_y[idx] - ym) > 0.55 or xs < lab_col - pitch:
                ax.plot([xs + 0.008 * T, lab_col - 0.008 * T],
                        [ym, lab_y[idx]], color=accent, lw=0.45,
                        ls=(0, (1, 2)), alpha=0.65, zorder=2)
            ax.text(lab_col, lab_y[idx], e["q"], fontsize=SIZES["annot"],
                    va="center", ha="left", color=accent,
                    fontweight="bold")
        geom["xmax_used"] = lab_col + 0.85 * T
        if unplaced:
            ax.text(geom["xmax_used"] * 0.99, n - 0.5,
                    "unplaced:\n" + "\n".join(unplaced),
                    fontsize=SIZES["annot"], va="bottom", ha="right",
                    color=TEXT_MID, style="italic",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f5f3f0",
                              ec=RULE_GRAY, lw=0.6))

        handles = [
            plt.Line2D([], [], color=TEXT_MID, marker="s", ls="",
                       label=f"{ref_name} (tree, tips)"),
            plt.Line2D([], [], color=accent, lw=1.6,
                       label=f"{query_name} ({method} assignment)"),
            plt.Line2D([], [], color=accent, marker="o", ls="",
                       markerfacecolor="white", label="equal (terminal)"),
            plt.Line2D([], [], color=accent, marker="^", ls="",
                       markerfacecolor="white", label="parent–child"),
            plt.Line2D([], [], color=accent, marker="P", ls="",
                       markerfacecolor="white", label="broader ancestor"),
            plt.Line2D([], [], color=accent, lw=1.4, ls=(0, (4, 2.2)),
                       label="underconfident"),
        ]
        caption = ("Each bracket spans exactly the reference clade assigned "
                   "to that query; bracket weight and glyph opacity encode "
                   "support. Horizontal position is root-to-node distance; "
                   "the distance between two nodes is the full path through "
                   "their common ancestor.")
        title = (f"Harmonized taxonomy — {query_name} onto {ref_name} "
                 f"(MetaArbor-{'Walk' if method == 'walk' else 'Transport'})")
        return _finish(plt, geom, node_positions, handles, caption, title)


# ---------------------------------------------------------------------------
# view 2: method comparison (dumbbell track)
# ---------------------------------------------------------------------------

def plot_alignment_phylogram(summary_rows, tree, node_positions=None,
                             figsize=None, **_legacy):
    """Method-comparison view: Walk vs Transport as gutter dumbbells.
    Merged navy square when the two selections are node-identical or
    topologically equivalent; orange connector for same-branch depth
    differences; red reserved for genuine branch conflicts."""
    with pub_style() as plt:
        coords, leaf_order = _phylo_layout(tree, node_positions)
        geom = _base_axes(plt, tree, coords, leaf_order, figsize)
        ax, T, n = geom["ax"], geom["T"], geom["n"]

        entries = []
        for r in summary_rows:
            w, t = r.get("walk_selected"), r.get("transport_node")
            cat = r.get("agreement", "walk_only")
            yw = coords[w][1] if w else None
            yt = coords[t][1] if t else None
            ys = [y for y in (yw, yt) if y is not None]
            if not ys:
                continue
            entries.append({"q": r["query"], "yw": yw, "yt": yt, "cat": cat,
                            "span": (min(ys), max(ys))})
        lanes, n_lanes = _lane_assign([e["span"] for e in entries])
        pitch = 0.13 * T
        x0 = geom["track_x0"]
        navy = OKABE_ITO["blue"]
        for e, L in zip(entries, lanes):
            xs = x0 + L * pitch
            merged = (e["yw"] is not None and e["yt"] is not None and
                      e["cat"] in ("agree", "topologically_equivalent"))
            if merged:
                ax.scatter([xs], [e["yw"]], marker="s", s=22, color=navy,
                           edgecolor="white", linewidth=0.5, zorder=4)
                lab_col = navy
            else:
                ccol = ("#d7191c" if e["cat"] == "conflicting_branch"
                        else OKABE_ITO["orange"])
                if e["yw"] is not None and e["yt"] is not None:
                    ax.plot([xs, xs], [e["yw"], e["yt"]], color=ccol,
                            lw=1.3, zorder=3,
                            ls="-" if e["cat"] == "conflicting_branch"
                            else (0, (4, 2)))
                if e["yw"] is not None:
                    ax.scatter([xs], [e["yw"]], marker="o", s=18,
                               color="#4a4a4a", edgecolor="white",
                               linewidth=0.5, zorder=4)
                if e["yt"] is not None:
                    ax.scatter([xs], [e["yt"]], marker="D", s=16,
                               color="#4a4a4a", edgecolor="white",
                               linewidth=0.5, zorder=4)
                lab_col = ccol
            e["_ym"], e["_labcol"] = (np.mean([y for y in (e["yw"], e["yt"])
                                               if y is not None]), lab_col)
        lab_x = x0 + n_lanes * pitch + 0.03 * T
        mids = [e["_ym"] for e in entries]
        order = np.argsort(mids)
        lab_y = np.array(mids, float)
        for k in range(1, len(order)):
            a, b = order[k - 1], order[k]
            if lab_y[b] - lab_y[a] < 0.95:
                lab_y[b] = lab_y[a] + 0.95
        for idx, (e, L) in enumerate(zip(entries, lanes)):
            xs = x0 + L * pitch
            if abs(lab_y[idx] - e["_ym"]) > 0.55 or xs < lab_x - pitch:
                ax.plot([xs + 0.008 * T, lab_x - 0.008 * T],
                        [e["_ym"], lab_y[idx]], color=e["_labcol"], lw=0.45,
                        ls=(0, (1, 2)), alpha=0.65, zorder=2)
            ax.text(lab_x, lab_y[idx], e["q"], fontsize=SIZES["annot"],
                    va="center", ha="left", color=e["_labcol"])
        geom["xmax_used"] = lab_x + 0.85 * T

        handles = [
            plt.Line2D([], [], marker="s", ls="", color=navy,
                       label="Walk ≡ Transport (incl. equivalent)"),
            plt.Line2D([], [], marker="o", ls="", color="#4a4a4a",
                       label="Walk"),
            plt.Line2D([], [], marker="D", ls="", color="#4a4a4a",
                       label="Transport"),
            plt.Line2D([], [], color=OKABE_ITO["orange"], lw=1.3,
                       ls=(0, (4, 2)), label="same branch, different depth"),
            plt.Line2D([], [], color="#d7191c", lw=1.3,
                       label="branch conflict"),
        ]
        caption = ("Each dumbbell connects the two methods' selections for "
                   "one query; a single navy square means they chose the "
                   "same or topologically equivalent nodes. Horizontal "
                   "position is root-to-node distance.")
        return _finish(plt, geom, node_positions, handles, caption,
                       "MetaArbor-Walk vs MetaArbor-Transport")


# ---------------------------------------------------------------------------
# exports for external renderers
# ---------------------------------------------------------------------------

def to_newick(tree, edge_lengths=None, node=None):
    """Newick string; `edge_lengths` (node -> length) attaches branch
    lengths for external renderers (ete4, ggtree, iTOL)."""
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


def to_ete4(tree, edge_lengths=None):
    """The tree as an `ete4.Tree` (install the `ete` extra; static render()
    works headlessly with QT_QPA_PLATFORM=offscreen)."""
    try:
        from ete4 import Tree as EteTree
    except ImportError as e:
        raise ImportError("to_ete4 needs the `ete` extra: "
                          "pip install metaarbor[ete]") from e
    return EteTree(to_newick(tree, edge_lengths), parser=1)
