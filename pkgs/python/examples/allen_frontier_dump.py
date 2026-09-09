"""Allen dump for the containment-frontier experiment: identical
construction to allen_tree_dump.py (strict holdout, frozen settings,
stability propagated) but ALSO serializes the one-way Walk decisions
and canonical maps — the evidence the frontier attaches labels with.
Determinism check: the assembled tree must equal the committed
metaarbor_tree_current.json exactly.
"""
import csv
import json
import os

import numpy as np
from scipy.io import mmread

from metaarbor.consensus.candidates import canonical_nodes
from metaarbor.consensus.harmonize import harmonize
from metaarbor.infer_tree import infer_tree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
if not os.path.isdir(D):
    D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "harmonize_demo")


def load(tag):
    counts = np.asarray(mmread(os.path.join(D, f"counts_{tag}.mtx"))
                        .todense()).T
    lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
    with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells


genes = open(os.path.join(D, "genes.txt")).read().split()
cA, lA, cellsA = load("10Xv2")
cB, lB, cellsB = load("10Xv3")
labA = np.asarray([f"v2|{c['subclass']}" for c in cellsA])
labB = np.asarray([f"v3|{c['cluster']}" for c in cellsB])
inf_a = infer_tree(cA, labA, lib=lA, n_hvg=2000, n_boot=50, seed=0)
inf_b = infer_tree(cB, labB, lib=lB, n_hvg=2000, n_boot=50, seed=0)
trees = {"v2": inf_a["tree"], "v3": inf_b["tree"]}
stability = {("v2", n): float(v) for n, v in inf_a["support"].items()}
stability.update({("v3", n): float(v)
                  for n, v in inf_b["support"].items()})
datasets = {
    "v2": {"counts": cA, "labels": labA, "gene_names": genes, "lib": lA},
    "v3": {"counts": cB, "labels": labB, "gene_names": genes, "lib": lB},
}
harm = harmonize(datasets, trees, n_hvg=1000, n_boot=200,
                 stability=stability)

dump = {i: {"parent": nd["parent"], "status": nd["status"],
            "members": nd["members"], "aliases": nd["aliases"],
            "display": nd["display"],
            "assembly_repair": bool(nd.get("assembly_repair"))}
        for i, nd in harm["tree"].items()}
prev = json.load(open(os.path.join(OUT, "metaarbor_tree_current.json")))
assert dump == prev, "determinism check failed vs committed dump"
print("determinism check: assembled tree identical to committed dump")


def _num(x):
    return None if x is None or (isinstance(x, float) and
                                 np.isnan(x)) else float(x)


sel_out = {}
for (ki, kj), recs in harm["decisions"]["selections"].items():
    sel_out[f"{ki}>{kj}"] = {
        n: {"selected": r["selected"], "matched": bool(r["matched"]),
            "support": _num(r["support"]),
            "compactness": r.get("compactness"),
            "relation": r.get("relation"),
            "gated_selected": r.get("gated_selected")}
        for n, r in recs.items()}
with open(os.path.join(OUT, "allen_decisions.json"), "w") as fh:
    json.dump(sel_out, fh)
with open(os.path.join(OUT, "allen_canonical.json"), "w") as fh:
    json.dump({k: canonical_nodes(trees[k])[1] for k in trees}, fh)
with open(os.path.join(OUT, "allen_input_trees.json"), "w") as fh:
    json.dump({k: {"parent": trees[k]["parent"],
                   "children": trees[k]["children"],
                   "leaves": list(trees[k]["leaves"])}
               for k in trees}, fh)
n_matched = sum(sum(1 for r in recs.values() if r["matched"])
                for recs in sel_out.values())
print(f"wrote allen_decisions.json ({n_matched} matched one-way calls),"
      " allen_canonical.json, allen_input_trees.json")
