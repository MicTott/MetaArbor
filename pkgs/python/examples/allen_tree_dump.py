"""Current-era Allen assembly dump for the refreshed OTHarmonizer
comparison: identical construction to examples/allen_harmonize.py
(strict holdout, both trees inferred, frozen settings, stability
propagated) — but serializes the assembled tree for offline scoring.
"""
import csv
import json
import os

import numpy as np
from scipy.io import mmread

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
path = os.path.join(OUT, "metaarbor_tree_current.json")
with open(path, "w") as fh:
    json.dump({i: {"parent": nd["parent"], "status": nd["status"],
                   "members": nd["members"], "aliases": nd["aliases"],
                   "display": nd["display"],
                   "assembly_repair": bool(nd.get("assembly_repair"))}
               for i, nd in harm["tree"].items()}, fh)
by = {}
for nd in harm["tree"].values():
    by[nd["status"]] = by.get(nd["status"], 0) + 1
print("statuses:", by, "| repairs:", len(harm["repairs"]),
      "->", path)
