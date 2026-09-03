"""The four consensus diagnostic figures (pub_style throughout):
candidate-by-dataset membership, decision-edge graph, eligibility power
curves, and true-vs-reconstructed consensus comparison."""
from __future__ import annotations

import numpy as np

import contextlib

from metaarbor.style import (OKABE_ITO, RULE_GRAY, SIZES, TEXT_DARK,
                             TEXT_MID, TREE_GRAY, pub_style)


@contextlib.contextmanager
def _style():
    """pub_style with matplotlib's bundled DejaVu forced first: diagnostic
    figures use very small tick text, where some system Helvetica variants
    fail glyph loading."""
    with pub_style() as plt:
        with plt.rc_context({"font.sans-serif": ["DejaVu Sans"]}):
            yield plt

CALL_COLORS = {"supported": OKABE_ITO["blue"],
               "unresolved_in_dataset": OKABE_ITO["sky"],
               "private_or_absent": OKABE_ITO["vermillion"],
               "unknown": "#bdbdbd"}


def plot_membership(backbone_out, keys, figsize=None):
    """Candidate x dataset matrix, colored by the raw eligibility call."""
    with _style() as plt:
        rows = {}
        for r in backbone_out["eligibility_table"]:
            rows.setdefault(r["candidate"], {})[r["dataset"]] = r["call"]
        cands = sorted(rows)
        fig, ax = plt.subplots(
            figsize=figsize or (2 + 0.5 * len(keys),
                                1.5 + 0.18 * len(cands)))
        for yi, c in enumerate(cands):
            for xi, k in enumerate(keys):
                call = rows[c].get(k, "unknown")
                ax.add_patch(plt.Rectangle((xi, yi), 0.92, 0.92,
                                           color=CALL_COLORS[call]))
        ax.set_xlim(0, len(keys))
        ax.set_ylim(len(cands), 0)
        ax.set_xticks([i + 0.46 for i in range(len(keys))], keys,
                      fontsize=SIZES["tick"])
        ax.set_yticks([i + 0.46 for i in range(len(cands))], cands,
                      fontsize=4.5)
        handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=k)
                   for k, c in CALL_COLORS.items()]
        fig.legend(handles=handles, fontsize=SIZES["legend"], ncol=2,
                   frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0))
        ax.set_title("Eligibility calls: candidate x dataset", loc="left",
                     fontsize=SIZES["title"], fontweight="bold")
        fig.tight_layout()
        return fig, ax


def plot_edge_graph(dec, figsize=None):
    """Reciprocal / asymmetric / refused edges between datasets' canonical
    nodes (datasets as columns)."""
    with _style() as plt:
        keys = sorted({k for pair in dec["matches"] for k in pair})
        nodes = {k: sorted({n for (ki, kj), sel in dec["selections"].items()
                            if ki == k for n in sel}) for k in keys}
        ypos = {k: {n: i for i, n in enumerate(nodes[k])} for k in keys}
        xpos = {k: i for i, k in enumerate(keys)}
        fig, ax = plt.subplots(figsize=figsize or (
            2.2 * len(keys), 1 + 0.16 * max(len(v) for v in nodes.values())))
        for k in keys:
            for n_, y in ypos[k].items():
                ax.scatter([xpos[k]], [y], s=6, color=TEXT_MID, zorder=3)
        # asymmetric (all one-way selections, light)
        for (ki, kj), sel in dec["selections"].items():
            for a, ra in sel.items():
                if ra["selected"] is not None and \
                        ra["selected"] in ypos[kj]:
                    ax.plot([xpos[ki], xpos[kj]],
                            [ypos[ki][a], ypos[kj][ra["selected"]]],
                            color=RULE_GRAY, lw=0.4, ls=(0, (2, 2)),
                            zorder=1)
        # reciprocal (solid)
        for (ki, kj), pair in dec["matches"].items():
            for m in pair:
                ax.plot([xpos[ki], xpos[kj]],
                        [ypos[ki][m["node_i"]], ypos[kj][m["node_j"]]],
                        color=OKABE_ITO["blue"], lw=0.9, zorder=2)
        ax.set_xticks(list(xpos.values()), keys)
        ax.set_yticks([])
        handles = [plt.Line2D([], [], color=OKABE_ITO["blue"], lw=1.2,
                              label="reciprocal"),
                   plt.Line2D([], [], color=RULE_GRAY, lw=0.8,
                              ls=(0, (2, 2)), label="one-way")]
        fig.legend(handles=handles, fontsize=SIZES["legend"], ncol=2,
                   frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0))
        ax.set_title("Pairwise decision graph", loc="left",
                     fontsize=SIZES["title"], fontweight="bold")
        fig.tight_layout()
        return fig, ax


def plot_power_curves(backbone_out, power_line=0.95, figsize=None):
    """P(detect) vs parent-context cells for every candidate with a
    prevalence posterior; evaluated datasets marked at their n."""
    from .eligibility import p_detect_posterior
    with _style() as plt:
        fig, ax = plt.subplots(figsize=figsize or (5.2, 3.6))
        seen = {}
        for r in backbone_out["eligibility_table"]:
            if r.get("posterior"):
                seen.setdefault(r["candidate"], []).append(r)
        ns = np.unique(np.geomspace(1, 3000, 60).astype(int))
        for ci, (cand_id, rows) in enumerate(sorted(seen.items())):
            post = rows[0]["posterior"]
            if post[0] == "beta":
                curve = [p_detect_posterior(post[1], post[2], n)
                         for n in ns]
            else:
                p_lo = post[1]
                curve = [1 - (1 - p_lo) ** n for n in ns]
            col = list(OKABE_ITO.values())[ci % 7]
            ax.plot(ns, curve, lw=1.0, color=col,
                    label=cand_id if ci < 8 else None)
            for r in rows:
                ax.scatter([max(r["n_parent"], 1)], [r["power"]], s=14,
                           color=col, zorder=3, edgecolor="white",
                           linewidth=0.4)
        ax.axhline(power_line, color="#888888", lw=0.7, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_xlabel("parent-context cells in evaluated dataset")
        ax.set_ylabel("P(detect)")
        ax.legend(fontsize=5.5, frameon=False, ncol=2)
        ax.set_title("Eligibility power curves", loc="left",
                     fontsize=SIZES["title"], fontweight="bold")
        fig.tight_layout()
        return fig, ax


def plot_consensus_comparison(backbone_out, truth_parent, figsize=None):
    """True (left) vs reconstructed (right) consensus, private in
    vermillion; unknown and conflict counts annotated.
    `truth_parent`: {true node: parent or None}."""
    with _style() as plt:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (9, 5))

        def draw(ax, parent_map, colors, title):
            kids = {}
            for c, p in parent_map.items():
                kids.setdefault(p, []).append(c)
            order, depth = [], {}

            def dfs(v, d):
                for c in sorted(kids.get(v, [])):
                    depth[c] = d
                    dfs(c, d + 1)
                    if c not in kids or not kids.get(c):
                        order.append(c)
            dfs(None, 0)
            ys = {}
            for v in parent_map:
                if v in order:
                    ys[v] = order.index(v)
            def yof(v):
                if v in ys:
                    return ys[v]
                ch = kids.get(v, [])
                return np.mean([yof(c) for c in ch]) if ch else 0
            for c, p in parent_map.items():
                x1, y1 = depth.get(c, 0), yof(c)
                if p is not None:
                    x0, y0 = depth.get(p, 0), yof(p)
                    ax.plot([x0, x0, x1], [y0, y1, y1], color=TREE_GRAY,
                            lw=0.8)
                ax.scatter([x1], [y1], s=18,
                           color=colors.get(c, OKABE_ITO["blue"]),
                           zorder=3)
                ax.text(x1 + 0.06, y1, str(c).split("|")[-1], fontsize=5.5,
                        va="center", color=TEXT_DARK)
            ax.set_title(title, loc="left", fontsize=SIZES["axis"],
                         fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
            for s_ in ("top", "right", "left", "bottom"):
                ax.spines[s_].set_visible(False)

        draw(axes[0], truth_parent, {}, "truth")
        rec_parent = {nd["id"]: nd["parent"]
                      for nd in backbone_out["nodes"]}
        colors = {nd["id"]: (OKABE_ITO["vermillion"]
                             if nd["status"] == "private"
                             else OKABE_ITO["blue"])
                  for nd in backbone_out["nodes"]}
        draw(axes[1], rec_parent, colors, "reconstructed consensus")
        fig.text(0.55, 0.02,
                 f"unknown: {len(backbone_out['unknown'])}   conflicts: "
                 f"{len(backbone_out['conflicts'])}   private: "
                 f"{sum(nd['status'] == 'private' for nd in backbone_out['nodes'])}",
                 fontsize=SIZES["caption"], color=TEXT_MID)
        fig.tight_layout()
        return fig, axes
