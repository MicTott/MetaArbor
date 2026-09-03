"""MetaArbor packaged plots (matplotlib; requires the `viz` extra).

All functions return (fig, axes) so callers can restyle; nothing here
recomputes or alters estimator decisions. Taxonomy order is always the
target tree's own order — reclustering exists only as an explicitly
labeled exploratory option on the heatmaps.
"""
from __future__ import annotations

import numpy as np

from .tree import ancestors, leaves_under

AGREEMENT_COLORS = {
    "agree": "#2b8cbe",
    "topologically_equivalent": "#2b8cbe",   # same evidence, distinct outline
    "same_branch_different_depth": "#fdae61",
    "conflicting_branch": "#d7191c",
    "walk_only": "#984ea3",
    "transport_only": "#1a9850",
    "both_unmatched": "#666666",
}
ERROR_COLORS = {
    "correct": "#bbbbbb", "premature_stop": "#984ea3", "too_deep": "#fdae61",
    "adjacent_same_class": "#e6ab02", "wrong_branch": "#d7191c",
    "unmatched": "#666666",
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _short(x, n=22):
    x = str(x).split(":", 1)[-1]
    return x if len(x) <= n else x[: n - 1] + "…"


def tree_layout(tree):
    """Node -> (x, y): leaves at DFS/taxonomy order on x, depth on y.
    Children are stored sorted, so DFS order is the curated level order."""
    xs, order = {}, []

    def dfs(v):
        kids = tree["children"][v]
        if not kids:
            order.append(v)
            xs[v] = float(len(order))
            return xs[v]
        pos = [dfs(c) for c in kids]
        xs[v] = float(np.mean(pos))
        return xs[v]

    dfs("root")
    return {v: (xs[v], float(len(ancestors(tree, v)))) for v in xs}, order


def _backbone(ax, tree, coords, nodes=None, color="#cccccc", lw=0.6):
    keep = set(coords if nodes is None else nodes)
    for v in keep:
        p = tree["parent"].get(v)
        if p is not None and p in keep:
            ax.plot([coords[v][0], coords[p][0]], [coords[v][1], coords[p][1]],
                    color=color, lw=lw, zorder=1)


def plot_alignment_tree(summary_rows, tree, collapse=True, label=True,
                        orient="vertical", figsize=None):
    """The primary biological result plot: query labels attached to their
    selected nodes on the gray target hierarchy. Walk (circle) and
    Transport (diamond) side by side, combined into one square when they
    agree; fill opacity encodes Walk decision support / Transport mass;
    colors encode the agreement category. `collapse=True` drops terminal
    leaves that carry no selection (ancestor spine and internal structure
    are kept); taxonomy order is never changed."""
    plt = _mpl()
    coords0, leaf_order = tree_layout(tree)
    if orient == "horizontal":
        coords = {k: (y, x) for k, (x, y) in coords0.items()}
    else:
        coords = coords0
    placed = set()
    for r in summary_rows:
        for k in ("walk_selected", "transport_node"):
            if r.get(k):
                placed.add(r[k])
                placed.update(ancestors(tree, r[k]))
    if collapse:
        keep = {v for v in coords
                if tree["children"][v] or v in placed}
    else:
        keep = set(coords)
    figsize = figsize or ((max(8, 0.16 * len(leaf_order)), 6.5)
                          if orient == "vertical"
                          else (8.5, max(6, 0.16 * len(leaf_order))))
    fig, ax = plt.subplots(figsize=figsize)
    _backbone(ax, tree, coords, keep)
    labels_todo = []
    for r in summary_rows:
        cat = r.get("agreement", "walk_only")
        col = AGREEMENT_COLORS.get(cat, "#000000")
        w, t = r.get("walk_selected"), r.get("transport_node")
        wa = r.get("walk_decision_support")
        wa = 1.0 if wa is None or (isinstance(wa, float) and np.isnan(wa)) else wa
        ta = r.get("transport_mass", 1.0) or 0.0
        combined = w and t and (w == t or cat == "topologically_equivalent")
        if combined:
            x, y = coords[w]
            ax.scatter([x], [y], marker="s", s=46, color=col,
                       alpha=max(0.35, wa), zorder=3,
                       edgecolor="#08306b" if cat == "topologically_equivalent"
                       else "none",
                       linewidth=1.2,
                       linestyle="--" if cat == "topologically_equivalent"
                       else "-")
        else:
            dx = (0.18, 0) if orient == "vertical" else (0, 0.18)
            if w and t and coords.get(w) and coords.get(t):
                ax.plot([coords[w][0], coords[t][0]],
                        [coords[w][1], coords[t][1]],
                        color=col, lw=0.7, alpha=0.6, zorder=2)
            if w:
                ax.scatter([coords[w][0] - dx[0]], [coords[w][1] - dx[1]],
                           marker="o", s=38, color=col,
                           alpha=max(0.35, wa), zorder=3)
            if t:
                ax.scatter([coords[t][0] + dx[0]], [coords[t][1] + dx[1]],
                           marker="D", s=34, color=col,
                           alpha=max(0.35, ta), zorder=3)
        if label and w:
            labels_todo.append((coords[w], _short(r["query"])))
    # greedy label repulsion: sort along the taxonomy axis; when neighbors
    # crowd, cycle through offset tiers instead of overplotting
    axis0 = 0 if orient == "vertical" else 1
    labels_todo.sort(key=lambda p: (p[0][axis0], p[0][1 - axis0]))
    tiers = [(3, 6), (3, 14), (3, -10), (3, 22)] if orient == "vertical" \
        else [(6, 2), (34, 2), (62, 2), (90, 2)]
    min_gap = max(1.5, 0.02 * len(leaf_order))
    prev_pos, tier = None, 0
    for (xy, txt) in labels_todo:
        if prev_pos is not None and abs(xy[axis0] - prev_pos) < min_gap:
            tier = (tier + 1) % len(tiers)
        else:
            tier = 0
        prev_pos = xy[axis0]
        ax.annotate(txt, xy, textcoords="offset points",
                    xytext=tiers[tier], fontsize=6, color="#222222")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c,
                          markeredgecolor="#08306b" if k == "topologically_equivalent" else "none",
                          label=k)
               for k, c in AGREEMENT_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color="grey",
                           label="Walk"),
                plt.Line2D([], [], marker="D", ls="", color="grey",
                           label="Transport")]
    ax.legend(handles=handles, fontsize=6, loc="upper right", frameon=False)
    depth_max = max(xy[1 if orient == "vertical" else 0]
                    for xy in coords.values())
    if orient == "vertical":
        ax.set_ylim(depth_max + 0.6, -0.4)
        ax.set_xticks([])
        ax.set_ylabel("tree depth")
    else:
        ax.set_xlim(-0.4, depth_max + 2.5)
        ax.set_yticks([])
        ax.set_xlabel("tree depth")
    ax.set_title("MetaArbor alignment: queries at their selected target nodes")
    fig.tight_layout()
    return fig, ax


def _heatmap_core(M, row_names, col_names, plt, family_of_leaf=None,
                  annot_colors=None, figsize=None, cmap="Greys",
                  vmax=None, cbar_label=""):
    figsize = figsize or (max(7, 0.10 * len(col_names) + 2),
                          max(4, 0.14 * len(row_names) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0,
                   vmax=vmax if vmax is not None else np.nanmax(M),
                   interpolation="nearest")
    if family_of_leaf is not None:
        fams = [family_of_leaf.get(c) for c in col_names]
        for j in range(1, len(col_names)):
            if fams[j] != fams[j - 1]:
                ax.axvline(j - 0.5, color="#888888", lw=0.4)
    if annot_colors is not None:
        for j, c in enumerate(annot_colors):
            ax.add_patch(plt.Rectangle((j - 0.5, -1.4), 1, 0.8, color=c,
                                       clip_on=False))
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels([_short(r) for r in row_names], fontsize=5)
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, shrink=0.6, label=cbar_label)
    return fig, ax


def plot_evidence_heatmap(cache, test_labels, tree, metric="vote_fraction",
                          family_of_leaf=None, walk_selected=None,
                          transport_node=None, annotations=None,
                          recluster=False, figsize=None):
    """What the measurement layer sees: query x target-leaf vote fractions
    (default) or one-vs-all AUROCs, in taxonomy order, with family
    boundaries, optional annotation colors (list per column), Walk selection
    marks (green rectangle over the selected node's leaves) and optional
    Transport argmax marks (blue dots). `recluster=True` reorders by
    hierarchical clustering and is labeled as exploratory."""
    plt = _mpl()
    from .accessors import node_auroc_matrix, vote_fraction_matrix
    if metric == "vote_fraction":
        M, rows, cols = vote_fraction_matrix(cache, test_labels)
        vmax, lab = 1.0, "vote fraction"
    elif metric == "auroc":
        M, rows, cols = node_auroc_matrix(cache, test_labels, tree)
        vmax, lab = 1.0, "one-vs-all AUROC (interpretability view)"
    else:
        raise ValueError(metric)
    _, leaf_order = tree_layout(tree)
    ord_cols = [c for c in leaf_order if c in cols]
    ci = [cols.index(c) for c in ord_cols]
    M = M[:, ci]
    title = f"MetaArbor evidence heatmap ({metric})"
    if recluster:
        from scipy.cluster.hierarchy import leaves_list, linkage
        ri = leaves_list(linkage(M, "average"))
        M, rows = M[ri], [rows[i] for i in ri]
        title += "  [EXPLORATORY ORDER — reclustered, not taxonomy]"
    fig, ax = _heatmap_core(M, rows, ord_cols, plt,
                            family_of_leaf=family_of_leaf,
                            annot_colors=annotations, figsize=figsize,
                            vmax=vmax, cbar_label=lab)
    if walk_selected:
        for i, q in enumerate(rows):
            sel = walk_selected.get(q)
            if not sel:
                continue
            lv = [ord_cols.index(l) for l in leaves_under(tree, sel)
                  if l in ord_cols]
            if lv:
                ax.add_patch(plt.Rectangle((min(lv) - 0.5, i - 0.5),
                                           max(lv) - min(lv) + 1, 1,
                                           fill=False, edgecolor="#1a9850",
                                           lw=0.7))
    if transport_node:
        for i, q in enumerate(rows):
            sel = transport_node.get(q)
            if not sel:
                continue
            lv = [ord_cols.index(l) for l in leaves_under(tree, sel)
                  if l in ord_cols]
            if lv:
                ax.scatter([np.mean(lv)], [i], marker="D", s=8,
                           color="#2b8cbe", zorder=4)
    ax.set_title(title, fontsize=9)
    return fig, ax


def plot_transport_heatmap(pi, row_names, col_names, tree,
                           family_of_leaf=None, view="leaf", normalize=True,
                           annotations=None, figsize=None):
    """How Transport distributes each query. Row-normalized by default so
    queries are comparable (`normalize=False` shows raw coupling mass, and
    the legend says so). `view='family'` aggregates columns by
    `family_of_leaf`. Same taxonomy ordering and annotation system as the
    evidence heatmap; argmax family marked per row; right-side annotation
    shows effective target leaves."""
    plt = _mpl()
    P = np.asarray(pi, dtype=float)
    if normalize:
        tot = P.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        P = P / tot
        lab = "row-normalized mass"
    else:
        lab = "raw coupling mass (not comparable across queries)"
    _, leaf_order = tree_layout(tree)
    ord_cols = [c for c in leaf_order if c in col_names]
    ci = [list(col_names).index(c) for c in ord_cols]
    P = P[:, ci]
    if view == "family":
        fams = sorted(set(family_of_leaf[c] for c in ord_cols))
        P = np.column_stack([
            P[:, [j for j, c in enumerate(ord_cols)
                  if family_of_leaf[c] == f]].sum(axis=1) for f in fams])
        cols_show, fol = fams, None
    else:
        cols_show, fol = ord_cols, family_of_leaf
    fig, ax = _heatmap_core(P, list(row_names), cols_show, plt,
                            family_of_leaf=fol, annot_colors=annotations,
                            figsize=figsize, cmap="Blues",
                            vmax=1.0 if normalize else None, cbar_label=lab)
    # argmax family mark + effective-leaves side annotation
    Pn = np.asarray(pi, dtype=float)
    tot = Pn.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    Pn = Pn / tot
    for i in range(P.shape[0]):
        ax.scatter([int(np.argmax(P[i]))], [i], marker="|", s=40,
                   color="#d7191c", zorder=4)
        p = Pn[i][Pn[i] > 0]
        eff = np.exp(-(p * np.log(p)).sum()) if p.size else np.nan
        ax.annotate(f"{eff:.1f}", (P.shape[1] - 0.2, i), fontsize=4,
                    color="#555555", annotation_clip=False)
    ax.set_title(f"MetaArbor-Transport mass ({view} view); right margin = effective leaves",
                 fontsize=9)
    return fig, ax


def plot_query_path(trace_rows, query, transport_row=None, col_names=None,
                    tree=None, margin=0.01, figsize=None):
    """Diagnostic for one query: every Walk split with child vote fractions,
    child AUROCs, the sibling contrast, and which frozen rule fired; when a
    transport coupling row is given, per-child transport mass is overlaid
    as blue diamonds."""
    plt = _mpl()
    tr = [t for t in trace_rows if t["query"] == query]
    splits = list(dict.fromkeys(t["split_at"] for t in tr))
    figsize = figsize or (8, 2.4 * max(1, len(splits)))
    fig, axes = plt.subplots(len(splits), 1, figsize=figsize, squeeze=False)
    for si, sp in enumerate(splits):
        ax = axes[si][0]
        d = [t for t in tr if t["split_at"] == sp]
        xs = np.arange(len(d))
        ax.bar(xs, [t["vote"] for t in d],
               color=["#2b8cbe" if t["is_best"] else "#cccccc" for t in d])
        for k, t in enumerate(d):
            ax.text(k, t["vote"] + 0.02, f"{t['child_auroc']:.3f}",
                    rotation=90, ha="center", va="bottom", fontsize=6,
                    color="#333333")
        if transport_row is not None and tree is not None:
            for k, t in enumerate(d):
                lv = [list(col_names).index(l)
                      for l in leaves_under(tree, t["child"])
                      if l in col_names]
                m = float(np.asarray(transport_row)[lv].sum() /
                          max(np.asarray(transport_row).sum(), 1e-300))
                ax.scatter([k], [m], marker="D", s=18, color="#08519c",
                           zorder=4)
        t0 = d[0]
        if t0["override"]:
            verdict = f"vote override (top vote >= 0.90) -> DESCEND"
        elif t0["decision"] == "descend":
            verdict = f"sibling dAUROC lower bound {t0['sib_lo']:.3f} > margin {margin} -> DESCEND"
        elif not np.isnan(t0.get("par_lo", np.nan)) and t0["par_lo"] > 0:
            verdict = f"parent significantly better (lb {t0['par_lo']:.3f}) -> STOP"
        else:
            verdict = f"sibling dAUROC lower bound {t0['sib_lo']:.3f} <= margin {margin} -> STOP"
        ax.set_title(f"split at {_short(sp)} — {verdict}", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([_short(t["child"], 14) for t in d], rotation=45,
                           ha="right", fontsize=6)
        ax.set_ylabel("vote / mass", fontsize=7)
        ax.set_ylim(0, 1.15)
    fig.suptitle(f"MetaArbor-Walk path: {query}", fontsize=10)
    fig.tight_layout()
    return fig, axes


def classify_outcome(selected, true_node, tree, class_level=1):
    """Benchmark-only outcome classes. `class_level`: tree depth at which
    'same-class' is judged (default 1 = the root's children)."""
    if selected is None:
        return "unmatched"
    if selected == true_node:
        return "correct"
    anc_t = [true_node] + ancestors(tree, true_node)
    anc_s = [selected] + ancestors(tree, selected)
    if selected in anc_t:
        return "premature_stop"
    if true_node in anc_s:
        return "too_deep"

    def at_level(chain):
        below = [n for n in chain if len(ancestors(tree, n)) == class_level]
        return below[0] if below else chain[-1]

    return ("adjacent_same_class" if at_level(anc_t) == at_level(anc_s)
            else "wrong_branch")


def plot_error_tree(summary_rows, true_node_of, tree, node_key="walk_selected",
                    class_level=1, figsize=None):
    """Benchmark-only: dot at the true node, diamond at the selection, an
    arc only when they differ. Correct results stay visually quiet."""
    plt = _mpl()
    coords, leaf_order = tree_layout(tree)
    figsize = figsize or (max(8, 0.16 * len(leaf_order)), 6)
    fig, ax = plt.subplots(figsize=figsize)
    _backbone(ax, tree, coords)
    counts = {}
    for r in summary_rows:
        q = r["query"]
        t = true_node_of[q]
        s = r.get(node_key)
        cat = classify_outcome(s, t, tree, class_level)
        counts[cat] = counts.get(cat, 0) + 1
        col = ERROR_COLORS[cat]
        ax.scatter([coords[t][0]], [coords[t][1]], s=14, color=col, zorder=3)
        if cat not in ("correct",) and s is not None:
            ax.scatter([coords[s][0]], [coords[s][1]], s=26, marker="D",
                       facecolor="none", edgecolor=col, zorder=3)
            ax.annotate("", coords[s], coords[t],
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8,
                                        connectionstyle="arc3,rad=-0.15"))
    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=f"{k} ({counts.get(k, 0)})")
               for k, c in ERROR_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper right")
    depth_max = max(y for _, y in coords.values())
    ax.set_ylim(depth_max + 0.6, -0.4)
    ax.set_xticks([])
    ax.set_ylabel("tree depth")
    ax.set_title("MetaArbor benchmark errors on the target hierarchy")
    fig.tight_layout()
    return fig, ax
