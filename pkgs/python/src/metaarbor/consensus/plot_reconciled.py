"""Reconciled-hierarchy figure: the two (or K) input trees beside the
merged output, every original label shown, colored by dataset. The merged
panel is a genuinely new tree — meta-clade nodes carry all their aliases,
coarse labels sit above finer descendants, private branches in accent,
single-atlas (correspondence-unresolved) nodes open-marked, conflict
count annotated."""
from __future__ import annotations

import numpy as np

from metaarbor.style import (OKABE_ITO, RULE_GRAY, SIZES, TEXT_DARK,
                             TEXT_MID, TREE_GRAY)
from .diagnostics import _style as pub_style

DS_COLORS = [OKABE_ITO["blue"], OKABE_ITO["vermillion"],
             OKABE_ITO["green"], OKABE_ITO["purple"]]
STATUS_MARK = {"backbone": ("o", "full"), "private": ("D", "full"),
               "single_atlas": ("o", "open"),
               "unplaced_single_atlas": ("s", "open")}


def _layout(parent, children, roots):
    order, depth = [], {}

    def dfs(v, d):
        depth[v] = d
        kids = children.get(v, [])
        if not kids:
            order.append(v)
            return
        for c in kids:
            dfs(c, d + 1)
    for r in roots:
        dfs(r, 1)
    ys = {v: i for i, v in enumerate(order)}

    def yof(v):
        if v in ys:
            return float(ys[v])
        return float(np.mean([yof(c) for c in children[v]]))
    return depth, yof, order


def _draw_input_tree(ax, tree, color, title):
    from metaarbor.tree import ancestors
    parent = tree["parent"]
    children = tree["children"]
    roots = [c for c in children["root"]]
    depth = {v: len(ancestors(tree, v)) for v in parent if v != "root"}
    leaves = tree["leaves"]
    ys = {l: i for i, l in enumerate(leaves)}

    def yof(v):
        if v in ys:
            return float(ys[v])
        return float(np.mean([yof(c) for c in children[v]]))
    for v, p in parent.items():
        if v == "root" or p is None:
            continue
        x1, y1 = depth[v], yof(v)
        x0 = depth.get(p, 0)
        y0 = yof(p) if p != "root" else y1
        ax.plot([x0, x0, x1], [y0, y1, y1], color=TREE_GRAY, lw=0.7)
        if not children[v]:
            ax.text(x1 + 0.08, y1, str(v).split("|", 1)[-1],
                    fontsize=5.2, va="center", color=color)
        else:
            ax.text(x1, y1 - 0.35, str(v).split("|", 1)[-1].split(":")[-1],
                    fontsize=4.6, va="bottom", ha="center", color=color,
                    alpha=0.8)
        ax.scatter([x1], [y1], s=6, color=color, zorder=3)
    ax.set_title(title, loc="left", fontsize=SIZES["axis"],
                 fontweight="bold", color=color)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_reconciled_tree(harm, trees, dataset_names=None, figsize=None):
    """K input trees beside the merged reconciled hierarchy."""
    with pub_style() as plt:
        keys = sorted(trees)
        colors = {k: DS_COLORS[i % len(DS_COLORS)]
                  for i, k in enumerate(keys)}
        names = dataset_names or {k: k for k in keys}
        n_nodes = len(harm["tree"])
        row_h = 0.30 if n_nodes <= 70 else 0.17
        figsize = figsize or (5 + 4.5 * len(keys),
                              max(6, row_h * n_nodes + 1.5))
        fig, axes = plt.subplots(
            1, len(keys) + 1, figsize=figsize,
            width_ratios=[1] * len(keys) + [1.8])
        for ax, k in zip(axes, keys):
            _draw_input_tree(ax, trees[k], colors[k],
                             f"input: {names[k]}")

        ax = axes[-1]
        nodes = harm["tree"]
        children = {i: nd["children"] for i, nd in nodes.items()}
        parent = {i: nd["parent"] for i, nd in nodes.items()}
        depth, yof, order = _layout(parent, children, harm["roots"])
        for i, nd in nodes.items():
            x1, y1 = depth[i], yof(i)
            p = parent[i]
            if p is not None:
                ax.plot([depth[p], depth[p], x1], [yof(p), y1, y1],
                        color=TREE_GRAY, lw=0.8)
            else:
                ax.plot([0.35, 0.35, x1], [y1, y1, y1], color=TREE_GRAY,
                        lw=0.8)
            mark, fill = STATUS_MARK.get(nd["status"], ("s", "full"))
            mcol = (OKABE_ITO["vermillion"] if nd["status"] == "private"
                    else "#4a4a4a")
            if nd["status"] in ("single_atlas", "unplaced_single_atlas"):
                ds = next(iter(nd["members"]))
                mk = "s" if nd["status"] == "unplaced_single_atlas" else mark
                ax.scatter([x1], [y1], s=26, marker=mk,
                           facecolor="white", edgecolor=colors[ds],
                           linewidth=1.1, zorder=3,
                           linestyle=(":" if nd.get("assembly_repair")
                                      or nd["status"] ==
                                      "unplaced_single_atlas" else "-"))
            else:
                ax.scatter([x1], [y1], s=30, marker=mark, color=mcol,
                           zorder=3)
            # every original label, colored by its dataset, stacked
            texts = []
            for ds in keys:
                for lab in nd["aliases"]:
                    base = str(lab)
                    if base.startswith(f"{ds}|") or \
                            base.split("|", 1)[0] == ds or \
                            (ds in nd["members"] and
                             nd["members"][ds] == lab):
                        texts.append((base.split("|", 1)[-1], colors[ds]))
            seen = set()
            texts = [t for t in texts
                     if not (t in seen or seen.add(t))]
            for j, (txt, col) in enumerate(texts):
                ax.text(x1 + 0.10, y1 + (j - (len(texts) - 1) / 2) * 0.42,
                        txt.split(":")[-1], fontsize=5.2, va="center",
                        color=col,
                        fontweight=("bold" if not children[i] or True
                                    else "normal"))
        n_conf = len([c for c in harm["conflicts"]
                      if c.get("class") == "genuine_conflict"])
        ax.set_title("reconciled hierarchy (new tree, no atlas "
                     "privileged)", loc="left", fontsize=SIZES["axis"],
                     fontweight="bold")
        ax.text(0.0, -0.06,
                f"backbone: {sum(nd['status'] == 'backbone' for nd in nodes.values())}   "
                f"private: {sum(nd['status'] == 'private' for nd in nodes.values())}   "
                f"single-atlas: {sum(nd['status'] == 'single_atlas' for nd in nodes.values())}   "
                f"unplaced (assembly repair): "
                f"{sum(nd['status'] == 'unplaced_single_atlas' for nd in nodes.values())}   "
                f"affiliate aliases: {len(harm['affiliates'])}   "
                f"conflicts: {n_conf}   "
                f"unplaced internal layers: "
                f"{sum(len(v) for v in harm.get('unplaced_internals', {}).values())}",
                transform=ax.transAxes, fontsize=SIZES["caption"],
                color=TEXT_MID)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        handles = [plt.Line2D([], [], marker="s", ls="", color=colors[k],
                              label=names[k]) for k in keys]
        handles += [
            plt.Line2D([], [], marker="o", ls="", color="#4a4a4a",
                       label="meta-clade (multi-atlas)"),
            plt.Line2D([], [], marker="D", ls="",
                       color=OKABE_ITO["vermillion"], label="private"),
            plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                       color="#4a4a4a",
                       label="single-atlas (correspondence unresolved)"),
        ]
        fig.legend(handles=handles, fontsize=SIZES["legend"], ncol=3,
                   frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 0.02))
        fig.tight_layout()
        return fig, axes
