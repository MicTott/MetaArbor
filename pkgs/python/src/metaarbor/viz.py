"""The MetaArbor audit-view renderer — the package's default way to
draw reconciled trees and cross-method comparisons.

This codifies the figure conventions developed on the Allen and
retina benchmarks (the 'audit figure' style), so every MetaArbor
figure carries the same six encodings and is readable without a
caption archaeologist:

  1. GROUP BANDS   light background band behind each label row,
                   colored by any grouping (curated truth, a
                   consensus-cut tier, a class annotation) —
                   OPTIONAL: with no grouping the bands are simply
                   absent and everything else stands.
  2. DATASET COLOR label text and marker colored by source dataset.
  3. MERGES        reciprocal/explicit merges are black squares with
                   black bold '&'-joined text.
  4. EDGE KINDS    'certified' = solid black; 'containment' = dashed
                   gray (evidence-supported placement);
                   'ancestry'  = light gray (inherited input-tree
                   parentage). Unknown kinds fall back to ancestry.
  5. SHARED ORDER  all panels of a comparison use one canonical
                   vertical label order (children sorted by minimum
                   canonical rank), so the eye can compare across
                   panels.
  6. DUMPED REGION labels attached flat at a tree's root (typical of
                   competitor outputs) are drawn in a shaded
                   'root-dumped' region below the tree instead of
                   masquerading as structure.

Numbers policy: subtitles should carry RAW counts
('correct/denominator'), never proportions alone — computed by the
caller (scoring lives outside the package; see the retina audit).

The nested-tree input format is {'label': str, 'children': [...],
'edge': kind}: 'label' is ''/anonymous or one-or-more real labels
joined by '&'; 'edge' describes the edge INTO the node.

Builders are provided for the common sources: harmonize()/cut()
node maps, input trees, and plain nested label trees (competitor
output). This module supersedes consensus.plot_reconciled (now
deprecated) and the older plots.plot_alignment_tree generation.
"""
from __future__ import annotations

DEFAULT_DATASET_COLORS = ("#8b1a1a", "#14477d", "#1d6b4f", "#7b4fa6",
                          "#8a6d00")
DEFAULT_GROUP_BANDS = ("#ede2f6", "#fcebdd", "#e2ecf9", "#e5f2e2",
                       "#f2e2ee")
EDGE_STYLES = {"certified": ("k", "solid", 1.5),
               "containment": ("#999999", "dashed", 1.0),
               "ancestry": ("#bbbbbb", "solid", 1.0)}


def canonical_order(groups):
    """Label order from a grouping dict {label: group}: groups in
    first-appearance order of sorted group names, labels sorted
    within each group. Pass any explicit list instead if you have
    a better order."""
    by = {}
    for l, g in groups.items():
        by.setdefault(g, []).append(l)
    out = []
    for g in sorted(by):
        out.extend(sorted(by[g]))
    return out


def nested_from_harmonize(nodes, real_labels, matched_calls=None):
    """Harmonize()/cut()/interleave node map -> nested audit tree.
    Edge kinds: backbone children get 'certified'; single-member
    nodes whose label has a matched one-way call get 'containment';
    everything else 'ancestry'."""
    matched_calls = matched_calls or set()
    kids, roots = {}, []
    for i in sorted(nodes):
        p = nodes[i].get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)

    def build(i):
        nd = nodes[i]
        labs = sorted(m for m in
                      (mm for _d, mm in nd.get("members", {}).items())
                      if m in real_labels)
        if nd.get("status") == "backbone":
            edge = "certified"
        elif len(labs) == 1 and labs[0] in matched_calls:
            edge = "containment"
        else:
            edge = "ancestry"
        return {"label": "&".join(labs), "edge": edge,
                "children": [build(c) for c in sorted(kids.get(i, []))]}
    return {"label": "", "edge": "ancestry",
            "children": [build(r) for r in sorted(roots)]}


def nested_from_input_tree(tree, real_labels):
    """{'parent','children','leaves'} input tree -> nested audit
    tree; all edges 'ancestry'."""
    def build(n):
        return {"label": n if n in real_labels else "",
                "edge": "ancestry",
                "children": [build(c)
                             for c in tree["children"].get(n, [])]}
    return {"label": "", "edge": "ancestry",
            "children": [build(c)
                         for c in tree["children"].get("root", [])]}


def split_root_dumped(nested, real_labels):
    """Remove root-attached single-label leaves; return (tree,
    dumped label list). Use for competitor outputs that dump
    unplaced labels at the root."""
    dumped = [c["label"] for c in nested["children"]
              if not c["children"] and c["label"]
              and "&" not in c["label"] and c["label"] in real_labels]
    nested = dict(nested)
    nested["children"] = [c for c in nested["children"]
                          if c["children"] or "&" in c["label"]]
    return nested, dumped


def plot_audit_tree(ax, nested, *, order, groups=None,
                    dataset_of=None, dataset_colors=None,
                    group_colors=None, title="", subtitle="",
                    dumped=None, max_depth=None):
    """Draw one audit-view tree on `ax`. `order`: canonical label
    list (shared across panels of a comparison). `groups`: optional
    {label: group} for background bands. `dataset_of`: callable
    label -> dataset key (default: prefix before first '-' or '|')."""
    rank = {l: i for i, l in enumerate(order)}
    groups = groups or {}
    dataset_of = dataset_of or (
        lambda l: l.split("|")[0] if "|" in l else l.split("-")[0])
    ds_keys = sorted({dataset_of(l) for l in order})
    dcols = dict(zip(ds_keys, dataset_colors or
                     DEFAULT_DATASET_COLORS))
    gkeys = sorted(set(groups.values()))
    gcols = dict(zip(gkeys, group_colors or DEFAULT_GROUP_BANDS))

    def leaf_rank(n):
        if n["label"]:
            rs = [rank[p] for p in n["label"].split("&") if p in rank]
            if rs:
                return min(rs)
        rs = [leaf_rank(c) for c in n["children"]]
        return min(rs) if rs else 10 ** 6

    counter = [0]

    def layout(n, depth):
        n["_x"] = depth
        n["children"] = sorted(n["children"], key=leaf_rank)
        if not n["children"]:
            n["_y"] = counter[0]
            counter[0] += 1
        else:
            for c in n["children"]:
                layout(c, depth + 1)
            n["_y"] = sum(c["_y"] for c in n["children"]) / \
                len(n["children"])
    layout(nested, 0)
    nrows = counter[0]

    def leaves(n, out):
        if not n["children"]:
            out.append(n)
        for c in n["children"]:
            leaves(c, out)
    lv = []
    leaves(nested, lv)
    for n in lv:
        parts = [p for p in n["label"].split("&") if p in rank]
        gs = {groups[p] for p in parts if p in groups}
        if gs:
            col = gcols[next(iter(gs))] if len(gs) == 1 else "#ffd6d6"
            ax.axhspan(n["_y"] - 0.42, n["_y"] + 0.42, color=col,
                       zorder=0)

    def walk(n):
        for c in n["children"]:
            col, ls, lw = EDGE_STYLES.get(c.get("edge"),
                                          EDGE_STYLES["ancestry"])
            ax.plot([n["_x"], n["_x"], c["_x"]],
                    [n["_y"], c["_y"], c["_y"]],
                    color=col, ls=ls, lw=lw, zorder=1)
            walk(c)
    walk(nested)

    def annotate(n):
        parts = [p for p in n["label"].split("&") if p in rank]
        if len(parts) >= 2:
            ax.scatter([n["_x"]], [n["_y"]], s=46, color="k",
                       marker="s", zorder=3)
            ax.text(n["_x"] + 0.12, n["_y"], " & ".join(parts),
                    va="center", fontsize=7.4, color="k",
                    fontweight="bold")
        elif parts:
            p = parts[0]
            col = dcols.get(dataset_of(p), "k")
            ax.scatter([n["_x"]], [n["_y"]], s=20, color=col,
                       zorder=3)
            if n["children"]:
                ax.text(n["_x"], n["_y"] - 0.48, p, va="bottom",
                        ha="left", fontsize=7.4, color=col)
            else:
                ax.text(n["_x"] + 0.1, n["_y"], p, va="center",
                        fontsize=7.4, color=col)
        else:
            ax.scatter([n["_x"]], [n["_y"]], s=8, color="#555",
                       zorder=3)
        for c in n["children"]:
            annotate(c)
    annotate(nested)

    if dumped:
        y0 = nrows + 0.6
        ax.axhspan(y0 - 0.4, y0 + len(dumped) - 0.5, color="#eeeeee",
                   zorder=0)
        ax.text(0, y0 - 1.0, "root-dumped:", fontsize=7.2,
                color="#666", style="italic")
        for k, p in enumerate(sorted(dumped,
                                     key=lambda l: rank.get(l, 99))):
            col = dcols.get(dataset_of(p), "k")
            ax.scatter([0.25], [y0 + k], s=18, color=col, zorder=3)
            ax.text(0.4, y0 + k, p, va="center", fontsize=7.4,
                    color=col)
            g = groups.get(p)
            if g:
                ax.axhspan(y0 + k - 0.42, y0 + k + 0.42,
                           color=gcols[g], zorder=0)
        nrows = y0 + len(dumped)
    if title or subtitle:
        ax.set_title(f"{title}\n{subtitle}".strip(), fontsize=8.6,
                     loc="left")
    depth = max_depth or (max(n["_x"] for n in lv) if lv else 3)
    ax.set_xlim(-0.4, depth + 3.0)
    ax.set_ylim(nrows + 0.2, -1.6)
    ax.axis("off")
    return ax


def audit_legend(fig, groups=None, group_colors=None,
                 dataset_colors=None, dataset_names=None, **kw):
    """The standardized six-encoding legend, attached to `fig`."""
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    handles = []
    gkeys = sorted(set((groups or {}).values()))
    gcols = dict(zip(gkeys, group_colors or DEFAULT_GROUP_BANDS))
    for g in gkeys:
        handles.append(mp.Patch(color=gcols[g], label=f"{g} group"))
    handles += [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="k", markersize=8,
                   label="reciprocal / explicit merge"),
        plt.Line2D([0], [0], color="k", lw=1.6,
                   label="certified backbone edge"),
        plt.Line2D([0], [0], color="#999", ls="dashed",
                   label="containment-supported placement"),
        plt.Line2D([0], [0], color="#bbb",
                   label="inherited input ancestry")]
    for ds, col in zip(dataset_names or [],
                       dataset_colors or DEFAULT_DATASET_COLORS):
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=col, markersize=7,
                                  label=f"{ds} label"))
    fig.legend(handles=handles, loc="lower center",
               ncol=kw.get("ncol", 5), fontsize=kw.get("fontsize", 8.6),
               frameon=False,
               bbox_to_anchor=kw.get("bbox_to_anchor", (0.5, 0.0)))
