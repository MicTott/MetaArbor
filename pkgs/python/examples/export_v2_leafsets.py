"""Export the v2 inferred tree's node -> subclass-leaf-set map (and the
v3 map likewise) for the external comparison scorer. Deterministic:
same infer_tree call as the comparison runs (seed 0)."""
import csv
import json
import os

import numpy as np
from scipy.io import mmread

from metaarbor.infer_tree import infer_tree
from metaarbor.tree import leaves_under

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
if not os.path.isdir(D):
    D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "harmonize_demo")

genes = open(os.path.join(D, "genes.txt")).read().split()
out = {}
for tag, level in (("10Xv2", "subclass"), ("10Xv3", "cluster")):
    counts = np.asarray(mmread(os.path.join(D, f"counts_{tag}.mtx"))
                        .todense()).T
    lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
    with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    key = tag.replace("10X", "")
    labs = np.asarray([f"{key}|{c[level]}" for c in cells])
    inf = infer_tree(counts, labs, lib=lib, n_hvg=2000, n_boot=50, seed=0)
    tree = inf["tree"]
    m = {}
    for n in list(tree["parent"]):
        if n == "root":
            continue
        m[n] = sorted(leaves_under(tree, n)) if tree["children"][n] \
            else [n]
    out[key] = m
with open(os.path.join(OUT, "input_tree_leafsets.json"), "w") as fh:
    json.dump(out, fh)
print("wrote input_tree_leafsets.json:",
      {k: len(v) for k, v in out.items()})
