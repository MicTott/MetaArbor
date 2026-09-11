"""Regenerates the retina K=2 seven-panel audit figure THROUGH the
package renderer (metaarbor.viz) — the regression check for the viz
promotion. Encodings, panel set, shared order, and raw denominators
as documented in retina_k2/AUDIT_FIGURE.md.

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/retina_k2_audit_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metaarbor.consensus.interleave import interleave_core_ancestry
from metaarbor.viz import (audit_legend, nested_from_harmonize,
                           nested_from_input_tree, plot_audit_tree,
                           split_root_dumped)

HERE = os.path.dirname(os.path.abspath(__file__))
RK = os.path.join(HERE, "retina_k2")
OTH = os.path.join(HERE, "..", "..", "..", "comparison",
                   "otharmonizer", "retina_oth_out")

truth = json.load(open(os.path.join(RK, "retina_k2_truth.json")))
groups = {l: {"RBC_group": "RBC", "OFF": "OFF",
              "ON": "ON"}[g]
          for g, ls in truth.items() for l in ls}
LABELS = set(groups)
ORDER = ([l for l in ("mac|c26", "she|RBC")] +
         ["mac|c27", "mac|c28", "mac|c29", "she|BC1A", "she|BC1B",
          "she|BC2", "she|BC3A", "she|BC3B", "she|BC4"] +
         ["mac|c30", "mac|c31", "mac|c32", "mac|c33", "she|BC5A",
          "she|BC5B", "she|BC5C", "she|BC5D", "she|BC6", "she|BC7",
          "she|BC8_9"])
dec = json.load(open(os.path.join(RK, "retina_k2_decisions.json")))
matched = {n for recs in dec.values() for n, r in recs.items()
           if r["matched"]}
it = json.load(open(os.path.join(RK, "retina_k2_input_trees.json")))
itrees = {k: {"parent": it[k]["parent"],
              "children": it[k]["children"],
              "leaves": it[k]["leaves"]} for k in ("mac", "she")}
dump = json.load(open(os.path.join(RK, "retina_k2_tree.json")))
canon = json.load(open(os.path.join(RK, "retina_k2_canonical.json")))
inter, _ = interleave_core_ancestry(dump, itrees, canon)


def truth_nested():
    def grp(labs):
        return {"label": "", "edge": "ancestry",
                "children": [{"label": l, "children": [],
                              "edge": "ancestry"} for l in labs]}
    return {"label": "", "edge": "ancestry",
            "children": [grp(truth["RBC_group"]),
                         {"label": "", "edge": "ancestry",
                          "children": [grp(truth["OFF"]),
                                       grp(truth["ON"])]}]}


def oth_nested(tag):
    d = json.load(open(os.path.join(OTH, f"oth_retina_{tag}.json")))

    def norm(l):
        parts = []
        for p in l.split("&"):
            for b in ("mac-", "she-"):
                if p.startswith(b) and "|" in p:
                    p = p[len(b):]
                    break
            parts.append(p)
        return "&".join(parts)

    def build(x):
        lab = norm(x["label"])
        return {"label": lab if any(p in LABELS
                                    for p in lab.split("&")) else "",
                "children": [build(c) for c in x["children"]],
                "edge": "ancestry"}
    return split_root_dumped(build(d), LABELS)


SUB = {"mac_input": "truth 51/51  she-ret 0/0  mac-ret 51/51",
       "she_input": "truth 309/309  she-ret 309/309  mac-ret 0/0",
       "MA": "truth 1291/1291  she-ret 309/309  mac-ret 51/51",
       "OTH_default": "truth 177/1291  she-ret 23/309  mac-ret 0/51",
       "OTH_mac_first": "truth 203/1291  she-ret 31/309  mac-ret 0/51",
       "OTH_she_first": "truth 103/1291  she-ret 0/309  mac-ret 0/51"}

fig = plt.figure(figsize=(22, 13.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.14,
                      wspace=0.06)
common = dict(order=ORDER, groups=groups)
plot_audit_tree(fig.add_subplot(gs[0, 0]),
                nested_from_input_tree(itrees["mac"], LABELS),
                title="1. Macosko input tree (8 coarse clusters)",
                subtitle=SUB["mac_input"], **common)
plot_audit_tree(fig.add_subplot(gs[0, 1]),
                nested_from_input_tree(itrees["she"], LABELS),
                title="2. Shekhar input tree (14 types)",
                subtitle=SUB["she_input"], **common)
plot_audit_tree(fig.add_subplot(gs[0, 2]), truth_nested(),
                title="3. Group-level truth (RBC | OFF | ON)",
                subtitle="reference for all scoring", **common)
plot_audit_tree(fig.add_subplot(gs[0, 3]),
                nested_from_harmonize(inter, LABELS, matched),
                title="4. MetaArbor interleaved",
                subtitle=SUB["MA"], max_depth=8, **common)
for slot, tag, name in ((gs[1, 0:2], "default",
                         "5. OTHarmonizer default"),
                        (gs[1, 2], "mac_first", "6. OTH mac-first"),
                        (gs[1, 3], "she_first", "7. OTH she-first")):
    t, dumped = oth_nested(tag)
    plot_audit_tree(fig.add_subplot(slot), t, title=name,
                    subtitle=SUB[f"OTH_{tag}"], dumped=dumped,
                    max_depth=5, **common)
audit_legend(fig, groups=groups, dataset_names=["mac", "she"])
fig.suptitle("Retina bipolar K=2 audit — all 22 labels, shared "
             "vertical order, raw triplet counts "
             "(correct/reference-resolved)", fontsize=13)
out = os.path.join(RK, "retina_k2_audit_figure.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
