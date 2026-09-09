"""Consensus-cut CLI: derive the usable taxonomy views from a saved
harmonize_k3 run directory (tree.json + provenance.csv) — offline, no
recomputation, no evidence rules touched.

Usage:
    python consensus_cut.py <run_dir> [--level strict|default|complete]
                            [--out subdir]

Reads  <run_dir>/tree.json and <run_dir>/provenance.csv (the latter
supplies the exact original-label sets and rejection reasons).
Writes, into <run_dir>/<out> (default 'cut_<level>'):
    cut.json          the full cut structure
    cut_table.csv     one row per kept node: id, tier, display,
                      n_datasets, parent, members, aliases
    unresolved.csv    the side ledger with reasons
    cut.nwk           Newick export of the cut topology
    cut.png           compact tiered rendering (certified solid black,
                      atlas_specific diamonds, provisional open)
    prints the summary (top-level count, tier counts, per-atlas
    label accounting)
"""
import argparse
import csv
import json
import os

import numpy as np

from metaarbor import to_newick
from metaarbor.consensus.cut import consensus_cut, cut_to_tree
from metaarbor.consensus.diagnostics import _style as pub_style
from metaarbor.style import OKABE_ITO, SIZES, TEXT_MID, TREE_GRAY, save_pub

TIER_STYLE = {"certified": ("o", "#1a1a1a", "full"),
              "atlas_specific": ("D", OKABE_ITO["vermillion"], "full"),
              "provisional": ("o", "#4a4a4a", "open")}
DS_COLORS = [OKABE_ITO["blue"], OKABE_ITO["vermillion"],
             OKABE_ITO["green"], OKABE_ITO["purple"]]


def draw_cut(cut, path):
    nodes = cut["nodes"]
    atlases = sorted({ds for nd in nodes.values() for ds in nd["members"]})
    colors = {ds: DS_COLORS[i % len(DS_COLORS)]
              for i, ds in enumerate(atlases)}
    kids = {i: nd["children"] for i, nd in nodes.items()}
    order, depth = [], {}

    def dfs(v, d):
        depth[v] = d
        if not kids[v]:
            order.append(v)
        for c in kids[v]:
            dfs(c, d + 1)
    for r in cut["roots"]:
        dfs(r, 0)
    ys = {v: i for i, v in enumerate(order)}

    def yof(v):
        if v in ys:
            return float(ys[v])
        return float(np.mean([yof(c) for c in kids[v]]))
    with pub_style() as plt:
        fig, ax = plt.subplots(
            figsize=(8, max(3.5, 0.28 * len(nodes) + 1)))
        for i, nd in nodes.items():
            x1, y1 = depth[i], yof(i)
            p = nd["parent"]
            if p is not None:
                ax.plot([depth[p], depth[p], x1], [yof(p), y1, y1],
                        color=TREE_GRAY, lw=0.8)
            mark, col, fill = TIER_STYLE.get(nd["tier"],
                                             ("s", "#4a4a4a", "open"))
            if fill == "open":
                ds1 = next(iter(nd["members"]), None)
                ax.scatter([x1], [y1], s=26, marker=mark,
                           facecolor="white",
                           edgecolor=colors.get(ds1, col),
                           linewidth=1.1, zorder=3)
            else:
                ax.scatter([x1], [y1], s=30, marker=mark, color=col,
                           zorder=3)
            texts = []
            for a in nd["aliases"]:
                base = str(a)
                aff = base.startswith("≈ ")
                core = base[2:] if aff else base
                ds1 = core.split("|", 1)[0] if "|" in core else None
                texts.append((("≈ " if aff else "") +
                              core.split("|", 1)[-1],
                              colors.get(ds1, "#4a4a4a")))
            if not texts:
                texts = [(nd["display"], "#4a4a4a")]
            for j, (txt, c) in enumerate(texts):
                ax.text(x1 + 0.08, y1 + (j - (len(texts) - 1) / 2) * .4,
                        txt, fontsize=5.4, va="center", color=c,
                        fontweight=("bold" if nd["tier"] == "certified"
                                    else "normal"))
        s = cut["summary"]
        ax.set_title(f"consensus cut — {cut['level']}: "
                     f"{s['tier_counts'].get('certified', 0)} certified"
                     f" / {s['tier_counts'].get('atlas_specific', 0)} "
                     f"atlas-specific / "
                     f"{s['tier_counts'].get('provisional', 0)} "
                     f"provisional; {s['n_unresolved']} unresolved "
                     "(side ledger)", loc="left",
                     fontsize=SIZES["axis"], fontweight="bold")
        ax.text(0.0, -0.04, "solid black = cross-atlas certified; "
                "diamond = atlas-specific (powered absence elsewhere); "
                "open = provisional single-atlas",
                transform=ax.transAxes, fontsize=SIZES["caption"],
                color=TEXT_MID)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        save_pub(fig, path, formats=("png",), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--level", default="default",
                    choices=["strict", "default", "complete"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "tree.json")) as fh:
        tree_nodes = json.load(fh)
    prov_path = os.path.join(args.run_dir, "provenance.csv")
    prov, labels = [], {}
    if os.path.exists(prov_path):
        with open(prov_path) as fh:
            prov = list(csv.DictReader(fh))
        for r in prov:
            labels.setdefault(r["atlas"], set()).add(
                f"{r['atlas']}|{r['label']}")
    cut = consensus_cut(tree_nodes, level=args.level,
                        leaf_labels=labels or None,
                        provenance_rows=prov)
    out = os.path.join(args.run_dir,
                       args.out or f"cut_{args.level}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cut.json"), "w") as fh:
        json.dump(cut, fh, indent=1)
    with open(os.path.join(out, "cut_table.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "tier", "display", "n_datasets",
                    "parent", "members", "aliases"])
        for i, nd in sorted(cut["nodes"].items()):
            w.writerow([i, nd["tier"], nd["display"],
                        nd["n_datasets"], nd["parent"] or "",
                        "; ".join(f"{d}:{m}" for d, m in
                                  sorted(nd["members"].items())),
                        "; ".join(nd["aliases"])])
    with open(os.path.join(out, "unresolved.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "display", "members", "reason"])
        for r in cut["unresolved"]:
            w.writerow([r["node_id"], r["display"],
                        "; ".join(f"{d}:{m}" for d, m in
                                  sorted(r["members"].items())),
                        r["reason"]])
    with open(os.path.join(out, "cut.nwk"), "w") as fh:
        fh.write(to_newick(cut_to_tree(cut)))
    draw_cut(cut, os.path.join(out, "cut"))
    print(json.dumps(cut["summary"], indent=1))
    print("wrote cut.json, cut_table.csv, unresolved.csv, cut.nwk, "
          "cut.png ->", out)


if __name__ == "__main__":
    main()
