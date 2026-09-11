"""Retina bipolar K=3 — stage 1 (prep): extract the MRCA subset per
PRESPEC section 3 (accession GSE243413 only; per-type cap 2000 at
rng(0)), build the common three-atlas gene panel, infer the MRCA
input tree (frozen settings), and write the 4e truth tree. The K=2
Macosko/Shekhar subsets and input trees are REUSED verbatim.

Run: PYTHONPATH=src python examples/retina_k3_prep.py
"""
import json
import os

import anndata as ad
import numpy as np
from scipy.io import mmread, mmwrite

from metaarbor.infer_tree import infer_tree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "retina_bipolar")
K2 = os.path.join(HERE, "retina_k2")
OUT = os.path.join(HERE, "retina_k3")
os.makedirs(OUT, exist_ok=True)

NAME_PAIRS = ["RBC", "BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4",
              "BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7"]
OFF = {"BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4"}
ON = {"BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7", "BC8", "BC9",
      "BC8_9"}
CAP, SEED = 2000, 0

# ---- MRCA subset (PRESPEC 3) ----------------------------------------------
print("loading mrca h5ad (backed)...")
A = ad.read_h5ad(os.path.join(D, "mrca_bipolar.h5ad"), backed="r")
keep = np.asarray(A.obs["accession"] == "GSE243413")
lab = np.asarray(A.obs["author_cell_type"].astype(str))
rng = np.random.default_rng(SEED)
sel = np.zeros(A.n_obs, dtype=bool)
for t in sorted(set(lab)):
    idx = np.where(keep & (lab == t))[0]
    if len(idx) > CAP:
        idx = np.sort(rng.choice(idx, CAP, replace=False))
    sel[idx] = True
    print(f"  mrca {t}: kept {sel[idx].sum()} of "
          f"{(keep & (lab == t)).sum()}")
print(f"mrca subset: {int(sel.sum())} cells")
sub = A[np.where(sel)[0]].to_memory()
mrca_lab = np.asarray([f"mrca|{t}" for t in
                       sub.obs["author_cell_type"].astype(str)])
mrca_genes = [g.upper() for g in sub.var["feature_name"].astype(str)]
counts = sub.raw.X.tocsc()

# ---- common gene panel (PRESPEC 3) ----------------------------------------
k2_shared = [g for g in open(os.path.join(K2, "shared_genes.txt"))
             .read().split("\n") if g]


def first_index(genes):
    out = {}
    for i, g in enumerate(genes):
        out.setdefault(g, i)
    return out


gi = first_index(mrca_genes)
common = sorted(set(k2_shared) & set(gi))
print(f"common panel: {len(common)} genes "
      f"(K=2 shared {len(k2_shared)})")
open(os.path.join(OUT, "common_genes.txt"), "w").write(
    "\n".join(common))

mrca_X = counts[:, [gi[g] for g in common]].tocsr()
k2_pos = {g: i for i, g in enumerate(k2_shared)}
cols = [k2_pos[g] for g in common]
mac_X = mmread(os.path.join(K2, "mac_bipolar.mtx")).tocsc()[:, cols]
she_X = mmread(os.path.join(K2, "she_bipolar.mtx")).tocsc()[:, cols]
mac_lab = np.asarray([l for l in open(os.path.join(
    K2, "mac_labels.txt")).read().split("\n") if l])
she_lab = np.asarray([l for l in open(os.path.join(
    K2, "she_labels.txt")).read().split("\n") if l])
print(f"matrices: mac {mac_X.shape} she {she_X.shape} "
      f"mrca {mrca_X.shape}")

# ---- truth tree (PRESPEC 4e) ----------------------------------------------
mac_groups = {}
import csv  # noqa: E402
with open(os.path.join(K2, "retina_k2_macosko_groups.csv")) as fh:
    for r in csv.DictReader(fh):
        mac_groups[f"mac|c{r['cluster']}"] = r["group"]
truth = {
    "RBC_group": {"cherries": [["she|RBC", "mrca|RBC"]],
                  "mac": [m for m, g in mac_groups.items()
                          if g == "RBC_group"]},
    "OFF": {"cherries": [[f"she|{t}", f"mrca|{t}"]
                         for t in sorted(OFF & set(NAME_PAIRS))],
            "mac": [m for m, g in mac_groups.items() if g == "OFF"]},
    "ON": {"cherries": [[f"she|{t}", f"mrca|{t}"] for t in
                        sorted({"BC5A", "BC5B", "BC5C", "BC5D",
                                "BC6", "BC7"})] +
                       [["she|BC8_9", "mrca|BC8", "mrca|BC9"]],
           "mac": [m for m, g in mac_groups.items() if g == "ON"]}}
json.dump(truth, open(os.path.join(OUT, "retina_k3_truth.json"),
                      "w"), indent=1)

# ---- MRCA input tree (frozen settings, PRESPEC 5) -------------------------
lib = np.asarray(mrca_X.sum(axis=1)).ravel()
print("infer_tree mrca...")
inf = infer_tree(np.asarray(mrca_X.todense()), mrca_lab, lib=lib,
                 n_hvg=2000, n_boot=50, seed=0)
k2_trees = json.load(open(os.path.join(K2,
                                       "retina_k2_input_trees.json")))
json.dump({"mac": k2_trees["mac"], "she": k2_trees["she"],
           "mrca": {"parent": inf["tree"]["parent"],
                    "children": inf["tree"]["children"],
                    "leaves": list(inf["tree"]["leaves"]),
                    "support": {k: float(v) for k, v in
                                inf["support"].items()}}},
          open(os.path.join(OUT, "retina_k3_input_trees.json"), "w"),
          indent=1)

# ---- save subsets for stage 2 ---------------------------------------------
mmwrite(os.path.join(OUT, "mrca_bipolar.mtx"), mrca_X.tocoo())
mmwrite(os.path.join(OUT, "mac_bipolar.mtx"), mac_X.tocoo())
mmwrite(os.path.join(OUT, "she_bipolar.mtx"), she_X.tocoo())
np.savetxt(os.path.join(OUT, "mrca_labels.txt"), mrca_lab, fmt="%s")
np.savetxt(os.path.join(OUT, "mac_labels.txt"), mac_lab, fmt="%s")
np.savetxt(os.path.join(OUT, "she_labels.txt"), she_lab, fmt="%s")
print("stage 1 complete.")
