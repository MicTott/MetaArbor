"""Retina K=2 — the pending seed-977 stability replicate (PRESPEC
deviation closure): re-infer BOTH input trees at seed 977, rerun the
frozen harmonize, and compare against the committed seed-0 result:
directed decision agreement, backbone label pairs, and the truth
score of the interleaved assembly. Confirms (or refutes) that the
corrected structure is a property of the data, not of seed 0.

Run: python examples/retina_k2_seed977.py
"""
import json
import os
import sys

import numpy as np
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "comparison",
                                "otharmonizer"))
from metrics import MyNode, triplet_scores  # noqa: E402
from metaarbor.consensus.harmonize import harmonize  # noqa: E402
from metaarbor.consensus.interleave import (  # noqa: E402
    interleave_core_ancestry)
from metaarbor.consensus.candidates import canonical_nodes  # noqa
from metaarbor.infer_tree import infer_tree  # noqa: E402

OUT = os.path.join(HERE, "retina_k2")
SEED = 977
truth = json.load(open(os.path.join(OUT, "retina_k2_truth.json")))
group_of = {l: g for g, ls in truth.items() for l in ls}
genes = [g for g in open(os.path.join(
    OUT, "shared_genes.txt")).read().split("\n") if g]
mac_X = np.asarray(mmread(os.path.join(OUT, "mac_bipolar.mtx"))
                   .todense())
she_X = np.asarray(mmread(os.path.join(OUT, "she_bipolar.mtx"))
                   .todense())
mac_lab = np.loadtxt(os.path.join(OUT, "mac_labels.txt"), dtype=str)
she_lab = np.loadtxt(os.path.join(OUT, "she_labels.txt"), dtype=str)

print(f"infer_tree x2 at seed {SEED}...")
inf_m = infer_tree(mac_X, mac_lab, lib=mac_X.sum(axis=1),
                   n_hvg=2000, n_boot=50, seed=SEED)
inf_s = infer_tree(she_X, she_lab, lib=she_X.sum(axis=1),
                   n_hvg=2000, n_boot=50, seed=SEED)
trees = {"mac": inf_m["tree"], "she": inf_s["tree"]}
stability = {("mac", n): float(v)
             for n, v in inf_m["support"].items()}
stability.update({("she", n): float(v)
                  for n, v in inf_s["support"].items()})
print("harmonize...")
harm = harmonize({"mac": {"counts": mac_X, "labels": mac_lab,
                          "gene_names": genes,
                          "lib": mac_X.sum(axis=1)},
                  "she": {"counts": she_X, "labels": she_lab,
                          "gene_names": genes,
                          "lib": she_X.sum(axis=1)}},
                 trees, n_hvg=1000, n_boot=200, stability=stability)
nodes = {i: {"parent": nd["parent"], "status": nd["status"],
             "members": nd["members"], "aliases": nd["aliases"],
             "display": nd["display"]}
         for i, nd in harm["tree"].items()}
json.dump(nodes, open(os.path.join(OUT, "retina_k2_tree_s977.json"),
                      "w"), indent=1)

# backbone label pairs vs committed seed-0
base = json.load(open(os.path.join(OUT, "retina_k2_tree.json")))


def label_pairs(nds):
    out = set()
    for nd in nds.values():
        labs = [m for _d, m in nd["members"].items() if m in group_of]
        if nd["status"] == "backbone" and len(labs) >= 2:
            out.add(tuple(sorted(labs)))
    return out


p0, p977 = label_pairs(base), label_pairs(nodes)
print(f"backbone label pairs: seed0={sorted(p0)} "
      f"seed977={sorted(p977)} identical={p0 == p977}")

# decision agreement on shared canonical nodes
d977 = {}
for (ki, kj), recs in harm["decisions"]["selections"].items():
    for n, r in recs.items():
        d977[(ki, kj, n)] = r["selected"] if r["matched"] else None
d0raw = json.load(open(os.path.join(OUT,
                                    "retina_k2_decisions.json")))
d0 = {}
for direc, recs in d0raw.items():
    ki, kj = direc.split(">")
    for n, r in recs.items():
        d0[(ki, kj, n)] = r["selected"] if r["matched"] else None
common = set(d0) & set(d977)
agree = (sum(d0[k] == d977[k] for k in common) / len(common)
         if common else float("nan"))
print(f"decision agreement on {len(common)} shared canonical "
      f"decisions: {agree:.4f} (label-level nodes shared; internal "
      f"node ids differ across seeds and drop out)")

# truth score of the interleaved seed-977 assembly
canon = {k: canonical_nodes(trees[k])[1] for k in trees}
itrees = {k: {"parent": trees[k]["parent"],
              "children": trees[k]["children"],
              "leaves": list(trees[k]["leaves"])} for k in trees}
new, ledger = interleave_core_ancestry(nodes, itrees, canon)


def project(nds):
    kids, roots = {}, []
    for i, nd in nds.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)
    root = MyNode("root")

    def build(i, pn):
        nd = nds[i]
        pp = [m.replace("|", "-") for _d, m in
              sorted(nd.get("members", {}).items()) if m in group_of]
        node = MyNode("&".join(pp)) if pp else MyNode(f"__a{i}__")
        pn.addkid(node)
        for c in sorted(kids.get(i, [])):
            build(c, node)
    for r in sorted(roots):
        build(r, root)
    return root


def truth_tree():
    root = MyNode("root")
    rbc = MyNode("__rbc__")
    root.addkid(rbc)
    for l in truth["RBC_group"]:
        rbc.addkid(MyNode(l.replace("|", "-")))
    cbc = MyNode("__cbc__")
    root.addkid(cbc)
    for grp in ("OFF", "ON"):
        g = MyNode(f"__{grp}__")
        cbc.addkid(g)
        for l in truth[grp]:
            g.addkid(MyNode(l.replace("|", "-")))
    return root


REF = truth_tree()
r0, _ = triplet_scores(project(nodes), REF)
r1, _ = triplet_scores(project(new), REF)
print(f"seed-977 truth TRIP_REC: assembly={r0:.4f} "
      f"interleaved={r1:.4f} (interleaved added "
      f"{sum(1 for nd in new.values() if nd.get('interleaved'))}, "
      f"ledger={ledger})")
print(f"seed-0 reference: assembly 0.9318, interleaved 1.0000")
