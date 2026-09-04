"""MetaArbor seed-stability sweep for the OTHarmonizer comparison:
run harmonize() at frozen settings for each base_seed given on the
command line, dumping each assembled tree as JSON.

Usage: python allen_seed_sweep.py 313 499 631
"""
import csv
import json
import os
import sys

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
datasets = {
    "v2": {"counts": cA, "labels": labA, "gene_names": genes, "lib": lA},
    "v3": {"counts": cB, "labels": labB, "gene_names": genes, "lib": lB},
}
for seed in [int(s) for s in sys.argv[1:]]:
    harm = harmonize(datasets, trees, n_hvg=1000, n_boot=200,
                     base_seed=seed)
    path = os.path.join(OUT, f"metaarbor_tree_seed{seed}.json")
    with open(path, "w") as fh:
        json.dump({i: {"parent": nd["parent"], "status": nd["status"],
                       "members": nd["members"], "aliases": nd["aliases"]}
                   for i, nd in harm["tree"].items()}, fh)
    print(f"seed {seed}: {len(harm['tree'])} nodes -> {path}")
