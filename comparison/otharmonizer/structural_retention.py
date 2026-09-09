"""Structural retention: does harmonization PRESERVE (MetaArbor) or
RECOVER (OTHarmonizer) the fine v3 topology, measured on v3-only
triplets against the curated cluster->subclass truth?

Wording matters and is deliberate. Both pipelines start from IDENTICAL
inputs: two count matrices + flat label columns (v2 subclass, v3
cluster). MetaArbor's pipeline includes an explicit tree-inference
stage (infer_tree) whose v3 tree it then harmonizes — so for MetaArbor
the question is whether harmonization DAMAGES structure its own
earlier stage built. OTHarmonizer constructs its hierarchy directly
from the scVI latent (it is never handed the inferred v3 tree — see
oth_allen.py), so for OTHarmonizer the question is whether its
construction RECOVERS the same relationships. "Destroys structure it
was handed" would be unfair to OTHarmonizer and is not claimed.

Comparable universes: pairwise shared-label scoring can silently score
different trees on different triplet sets if any tree drops labels.
This script reports each tree's v3-label coverage AND rescores
everything on the COMMON intersection of v3 labels present in every
scored tree, so the retention comparison is over one identical triplet
universe. All scoring uses the structure-kept convention (labeled
leaves, anonymous internals — see rescore_current.py docstring).

Run: PYTHONPATH=../../pkgs/python/src python structural_retention.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pkgs", "python",
                                "src"))
from metrics import (MyNode, _label_nodes_depths,  # noqa: E402
                     cophenetic_spearman, from_nested, triplet_scores)

D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
MA = os.path.join(HERE, "..", "..", "pkgs", "python", "examples",
                  "harmonize_demo")

with open(os.path.join(D, "cells_10Xv3.csv")) as fh:
    truth = {c["cluster"]: c["subclass"] for c in csv.DictReader(fh)}
v3_labels = {f"v3|{c}" for c in truth}
itrees = json.load(open(os.path.join(MA, "allen_input_trees.json")))
tree_nodes = json.load(open(os.path.join(MA,
                                         "metaarbor_tree_current.json")))


def ref3(clusters):
    """v3-only reference: subclass families over the given clusters.
    Internal family nodes carry non-label names so the shared universe
    is exactly the cluster labels."""
    root = MyNode("root")
    by = {}
    for cl in sorted(clusters):
        by.setdefault(truth[cl], []).append(cl)
    for s in sorted(by):
        fam = MyNode(f"__fam_{s}__")
        root.addkid(fam)
        for cl in by[s]:
            fam.addkid(MyNode(f"v3-{cl}"))
    return root


def v3_input_tree():
    par = itrees["v3"]["parent"]
    kids, roots = {}, []
    for n, p in par.items():
        (roots if p in (None, "root") else
         kids.setdefault(p, [])).append(n)
    root = MyNode("root")

    def build(n, pn):
        node = MyNode(f"v3-{n.split('|', 1)[-1]}"
                      if n in v3_labels else f"__v3_{n}__")
        pn.addkid(node)
        for c in sorted(kids.get(n, [])):
            build(c, node)
    for r in sorted(roots):
        build(r, root)
    return root


def ma_assembly_tree():
    kids, roots = {}, []
    for i, nd in tree_nodes.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)
    root = MyNode("root")

    def build(i, pn):
        nd = tree_nodes[i]
        pp = [f"v3-{m.split('|', 1)[-1]}"
              for ds, m in sorted(nd.get("members", {}).items())
              if ds == "v3" and m in v3_labels]
        node = MyNode("&".join(pp)) if pp else MyNode(f"__a{i}__")
        pn.addkid(node)
        for c in sorted(kids.get(i, [])):
            build(c, node)
    for r in sorted(roots):
        build(r, root)
    return root


trees = [("baseline", "v3_input_tree", v3_input_tree()),
         ("MetaArbor", "current_assembly", ma_assembly_tree())]
for fn in sorted(os.listdir(HERE)):
    if fn.startswith("oth_tree_") and fn.endswith(".json"):
        trees.append(("OTHarmonizer", fn[len("oth_tree_"):-len(".json")],
                      from_nested(json.load(open(os.path.join(HERE,
                                                              fn))))))

# per-tree v3-label coverage + the common universe
want = {f"v3-{c}" for c in truth}
covers = {}
for _m, tag, tr in trees:
    covers[tag] = set(_label_nodes_depths(tr)[0]) & want
common = set.intersection(*covers.values())
print(f"curated v3 clusters: {len(want)} | common to all "
      f"{len(trees)} trees: {len(common)}")
for _m, tag, _t in trees:
    missing = len(want) - len(covers[tag])
    if missing:
        print(f"  {tag}: MISSING {missing} v3 labels")

REF_FULL = ref3(sorted(truth))
REF_COMMON = ref3(sorted(c for c in truth if f"v3-{c}" in common))

rows = []
for method, tag, tr in trees:
    rec_f, _ = triplet_scores(tr, REF_FULL)
    coph_f = cophenetic_spearman(tr, REF_FULL)
    rec_c, _ = triplet_scores(tr, REF_COMMON)
    coph_c = cophenetic_spearman(tr, REF_COMMON)
    rows.append({"method": method, "run": tag,
                 "n_v3_labels": len(covers[tag]),
                 "TRIP_REC_v3": round(rec_f, 4),
                 "COPH_v3": round(coph_f, 4),
                 "TRIP_REC_v3_common": round(rec_c, 4),
                 "COPH_v3_common": round(coph_c, 4)})
    print(f"{method:12s} {tag:16s} n={len(covers[tag]):3d} "
          f"TRIP_REC={rec_f:.4f} (common: {rec_c:.4f}) "
          f"COPH={coph_f:.4f} (common: {coph_c:.4f})")
with open(os.path.join(HERE, "structural_retention.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote structural_retention.csv")
