"""Retina bipolar K=2 — stage 1 (prep): subset per PRESPEC section 3,
derive the Macosko group table per the frozen rule (section 4b),
build the truth tree (4c), and infer both input trees (section 5).
Stops BEFORE harmonize: the group table is committed as the truth
annex first; stage 2 (retina_k2_run.py) then harmonizes.

Run: python examples/retina_k2_prep.py
"""
import csv
import json
import os

import numpy as np
from scipy import sparse
from scipy.io import mmread, mmwrite

from metaarbor.infer_tree import infer_tree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "retina_bipolar")
OUT = os.path.join(HERE, "retina_k2")
os.makedirs(OUT, exist_ok=True)

BC_TYPES = {"RBC (Rod Bipolar cell)": "RBC",
            "BC1A": "BC1A", "BC1B": "BC1B", "BC2": "BC2",
            "BC3A": "BC3A", "BC3B": "BC3B", "BC4": "BC4",
            "BC5A (Cone Bipolar cell 5A)": "BC5A", "BC5B": "BC5B",
            "BC5C": "BC5C", "BC5D": "BC5D", "BC6": "BC6",
            "BC7 (Cone Bipolar cell 7)": "BC7",
            "BC8/9 (mixture of BC8 and BC9)": "BC8_9"}
OFF = {"BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4"}
ON = {"BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7", "BC8_9"}
MAC_CLUSTERS = list(range(26, 34))
MARKERS = ["Prkca", "Scgn", "Grm6", "Isl1", "Grik1"]

print("loading matrices (sparse)...")
mac = mmread(os.path.join(D, "macosko_counts.mtx")).tocsc()
she = mmread(os.path.join(D, "shekhar_counts.mtx")).tocsc()
mac_genes = open(os.path.join(D, "macosko_genes.txt")).read().split("\n")
she_genes = open(os.path.join(D, "shekhar_genes.txt")).read().split("\n")
mac_genes = [g for g in mac_genes if g]
she_genes = [g for g in she_genes if g]
mac_ids = [l for l in open(os.path.join(
    D, "macosko_cellids.txt")).read().split("\n") if l]
she_ids = [l for l in open(os.path.join(
    D, "shekhar_cellids.txt")).read().split("\n") if l]
with open(os.path.join(D, "macosko_cells.csv")) as fh:
    mac_cd = {r["cell.id"]: int(float(r["cluster"]))
              for r in csv.DictReader(fh) if r["cluster"] not in
              ("", "NA")}
with open(os.path.join(D, "shekhar_cells.csv")) as fh:
    she_cd = {r["NAME"]: r["CLUSTER"] for r in csv.DictReader(fh)}
print(f"macosko {mac.shape}, shekhar {she.shape}")

# ---- inclusion (PRESPEC 3) ------------------------------------------------
mac_keep = [i for i, cid in enumerate(mac_ids)
            if mac_cd.get(cid) in MAC_CLUSTERS]
mac_lab = np.asarray([f"mac|c{mac_cd[mac_ids[i]]}" for i in mac_keep])
she_keep = [i for i, cid in enumerate(she_ids)
            if she_cd.get(cid) in BC_TYPES]
she_lab = np.asarray([f"she|{BC_TYPES[she_cd[she_ids[i]]]}"
                      for i in she_keep])
print(f"bipolar subsets: macosko {len(mac_keep)} cells "
      f"({len(set(mac_lab))} clusters), shekhar {len(she_keep)} cells "
      f"({len(set(she_lab))} types)")

# ---- gene intersection (upper-cased; first occurrence wins) ---------------
def first_index(genes):
    out = {}
    for i, g in enumerate(genes):
        out.setdefault(g.upper(), i)
    return out


mi, si = first_index(mac_genes), first_index(she_genes)
shared = sorted(set(mi) & set(si))
print(f"shared genes: {len(shared)}")
mac_X = mac[[mi[g] for g in shared], :][:, mac_keep].T.tocsr()
she_X = she[[si[g] for g in shared], :][:, she_keep].T.tocsr()

# ---- Macosko group table (frozen rule, PRESPEC 4b) ------------------------
lib_m = np.asarray(mac_X.sum(axis=1)).ravel()
midx = {g: shared.index(g.upper()) for g in MARKERS}
rows = []
pb = {}
for cl in MAC_CLUSTERS:
    sel = mac_lab == f"mac|c{cl}"
    cpm = np.asarray(mac_X[sel].sum(axis=0)).ravel()
    cpm = np.log1p(cpm / cpm.sum() * 1e6)
    pb[cl] = {g: cpm[midx[g]] for g in MARKERS}
z = {}
for g in MARKERS:
    v = np.asarray([pb[cl][g] for cl in MAC_CLUSTERS])
    z[g] = (v - v.mean()) / v.std()
groups = {}
for k, cl in enumerate(MAC_CLUSTERS):
    if z["Prkca"][k] - z["Scgn"][k] > 1:
        grp = "RBC_group"
    elif (z["Grm6"][k] + z["Isl1"][k]) / 2 > z["Grik1"][k]:
        grp = "ON"
    else:
        grp = "OFF"
    groups[cl] = grp
    rows.append({"cluster": cl, "n_cells": int((mac_lab ==
                                                f"mac|c{cl}").sum()),
                 **{f"z_{g}": round(float(z[g][k]), 3)
                    for g in MARKERS},
                 "group": grp})
    print(f"  mac c{cl}: n={rows[-1]['n_cells']:5d} " +
          " ".join(f"{g}={rows[-1]['z_' + g]:+.2f}" for g in MARKERS)
          + f"  -> {grp}")
with open(os.path.join(OUT, "retina_k2_macosko_groups.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---- truth tree (PRESPEC 4c) ----------------------------------------------
truth = {"RBC_group": ["she|RBC"] + [f"mac|c{c}" for c in MAC_CLUSTERS
                                     if groups[c] == "RBC_group"],
         "OFF": [f"she|{t}" for t in sorted(OFF)] +
                [f"mac|c{c}" for c in MAC_CLUSTERS
                 if groups[c] == "OFF"],
         "ON": [f"she|{t}" for t in sorted(ON)] +
               [f"mac|c{c}" for c in MAC_CLUSTERS
                if groups[c] == "ON"]}
json.dump(truth, open(os.path.join(OUT, "retina_k2_truth.json"), "w"),
          indent=1)

# ---- inferred input trees (frozen settings, PRESPEC 5) --------------------
np.random.seed(0)
print("infer_tree macosko...")
inf_m = infer_tree(np.asarray(mac_X.todense()), mac_lab, lib=lib_m,
                   n_hvg=2000, n_boot=50, seed=0)
print("infer_tree shekhar...")
lib_s = np.asarray(she_X.sum(axis=1)).ravel()
inf_s = infer_tree(np.asarray(she_X.todense()), she_lab, lib=lib_s,
                   n_hvg=2000, n_boot=50, seed=0)
json.dump({"mac": {"parent": inf_m["tree"]["parent"],
                   "children": inf_m["tree"]["children"],
                   "leaves": list(inf_m["tree"]["leaves"]),
                   "support": {k: float(v) for k, v in
                               inf_m["support"].items()}},
           "she": {"parent": inf_s["tree"]["parent"],
                   "children": inf_s["tree"]["children"],
                   "leaves": list(inf_s["tree"]["leaves"]),
                   "support": {k: float(v) for k, v in
                               inf_s["support"].items()}}},
          open(os.path.join(OUT, "retina_k2_input_trees.json"), "w"),
          indent=1)

# ---- save subsets for stage 2 ---------------------------------------------
mmwrite(os.path.join(OUT, "mac_bipolar.mtx"), mac_X.tocoo())
mmwrite(os.path.join(OUT, "she_bipolar.mtx"), she_X.tocoo())
np.savetxt(os.path.join(OUT, "mac_labels.txt"), mac_lab, fmt="%s")
np.savetxt(os.path.join(OUT, "she_labels.txt"), she_lab, fmt="%s")
open(os.path.join(OUT, "shared_genes.txt"), "w").write(
    "\n".join(shared))
print("stage 1 complete — commit the group table before stage 2.")
