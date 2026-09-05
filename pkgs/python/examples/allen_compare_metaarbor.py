"""MetaArbor arm of the frozen matched comparison vs OTHarmonizer.

Runs harmonize() at the FROZEN v0.8 settings on the identical inputs
(same counts, same own-label columns, both trees inferred from
expression) and serializes the assembled tree to JSON for the shared
scorer. Two runs: the primary (base_seed=211, the frozen default) and a
seed-perturbation run (base_seed=977) for the stability arm.

Dataset-order invariance is structural, not empirical: harmonize()
sorts dataset keys before all pairwise work (see pairwise_decisions),
so insertion order cannot change the result. Asserted here.
"""
import csv
import inspect
import json
import os

import numpy as np
from scipy.io import mmread

from metaarbor.consensus import candidates as _cand_mod
from metaarbor.consensus.harmonize import harmonize
from metaarbor.infer_tree import infer_tree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
if not os.path.isdir(D):
    D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "harmonize_demo")

# order invariance: the pairwise layer iterates sorted(datasets)
src = inspect.getsource(_cand_mod.pairwise_decisions)
assert "sorted(datasets)" in src, "order-invariance premise broken"
print("dataset-order invariance: structural (sorted keys) — confirmed")


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

for tag, seed in (("primary", 211), ("seed977", 977)):
    harm = harmonize(datasets, trees, n_hvg=1000, n_boot=200,
                     base_seed=seed, stability=stability)
    dump = {i: {"parent": nd["parent"], "status": nd["status"],
                "members": nd["members"], "aliases": nd["aliases"]}
            for i, nd in harm["tree"].items()}
    path = os.path.join(OUT, f"metaarbor_tree_{tag}.json")
    with open(path, "w") as fh:
        json.dump(dump, fh)
    print(f"{tag} (base_seed={seed}): {len(dump)} nodes -> {path}")
