"""Locked four-arm Transport comparison (NOTES item 19 protocol), executed
on the Allen benchmark where ground truth exists.

Arms — identical intrinsic marginals, optimization settings (rho = 0.3,
alpha = 0.9 where a GW term exists, epsilon = 0) and readouts:
  1. hop        — the frozen model: tree hop distances as C_A / C_B
  2. raw_metric — within-atlas chord distances between terminal
                  populations, no imposed tree metric
  3. patristic  — chord distances NNLS-projected onto each atlas's OWN
                  topology (fitted branch lengths -> patristic matrix)
  4. uot        — molecular-only unbalanced OT, no GW term

Each atlas selects its own HVGs and builds its own distance matrix
independently (GW needs no shared feature space); every C is normalized by
its max before solving (the same convention the frozen solver applies to
hop distances).

This is the PRESPECIFIED EXTENSION, not a revision: the frozen hop model
remains MetaArbor-Transport's configuration for the amygdala primary
analysis regardless of this outcome.

Usage: python examples/metric_fugw_experiment.py
"""
import csv
import gzip
import os

import numpy as np
from scipy.io import mmread

from metaarbor import tree_from_levels, tree_weights, write_csv
from metaarbor.branch_fit import fit_branch_lengths, pseudobulk_distances
from metaarbor.fugw import decompose, solve
from metaarbor.tree import ancestors, leaf_path_dist

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "wmb_plilaorb")
FX = os.path.join(REPO, "pkgs", "fixtures")
OUT = os.path.join(HERE, "allen_demo", "metric_fugw")
os.makedirs(OUT, exist_ok=True)
RHO, ALPHA = 0.3, 0.9


def rcsv(name):
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        return list(csv.reader(fh))


def load_atlas(tag):
    counts = np.asarray(mmread(os.path.join(
        DATA, f"counts_{tag}.mtx")).todense()).T
    lib = np.loadtxt(os.path.join(DATA, f"lib_{tag}.txt"))
    with open(os.path.join(DATA, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells


def patristic_matrix(tree, positions, leaves):
    P = np.zeros((len(leaves), len(leaves)))
    anc = {l: [l] + ancestors(tree, l) for l in leaves}
    pos = positions
    for a in range(len(leaves)):
        sa = anc[leaves[a]]
        seta = set(sa)
        for b in range(a):
            lca = next(x for x in anc[leaves[b]] if x in seta)
            P[a, b] = P[b, a] = (pos[leaves[a]] + pos[leaves[b]]
                                 - 2 * pos[lca])
    return P


## shared inputs (identical across arms) -------------------------------------
srows = rcsv("S_sub.csv.gz")
cols = srows[0][1:]
qn = [r[0] for r in srows[1:]]
S = np.asarray([[float(x) for x in r[1:]] for r in srows[1:]])
M = 1 - S
lv = rcsv("tree_levels_b.csv.gz")
tree_b = tree_from_levels([tuple(r) for r in lv[1:]], lv[0])
cls_of = {r[1]: r[0] for r in lv[1:]}
tree_a = tree_from_levels(sorted({(cls_of[q], q) for q in qn}),
                          ["class", "subclass"])
wA_map, wB_map = tree_weights(tree_a), tree_weights(tree_b)
wA = np.asarray([wA_map[q] for q in qn])
wB = np.asarray([wB_map[c] for c in cols])
family_of_leaf = {r[3]: r[1] for r in lv[1:]}
family_leaves = {q: [c for c in cols if family_of_leaf[c] == q] for q in qn}
singletons = [q for q in qn if len(family_leaves[q]) == 1]

## per-atlas independent molecular distances ---------------------------------
counts_b, lib_b, cells_b = load_atlas("10Xv3")
D_B, leaves_B = pseudobulk_distances(
    counts_b, [c["cluster"] for c in cells_b], lib=lib_b)
ix_b = [leaves_B.index(c) for c in cols]
D_B = D_B[np.ix_(ix_b, ix_b)]
counts_a, lib_a, cells_a = load_atlas("10Xv2")
D_A, leaves_A = pseudobulk_distances(
    counts_a, [c["subclass"] for c in cells_a], lib=lib_a)
ix_a = [leaves_A.index(q) for q in qn]
D_A = D_A[np.ix_(ix_a, ix_a)]
print(f"independent chord distances: A {D_A.shape} (10Xv2 subclass "
      f"pseudobulks), B {D_B.shape} (10Xv3 cluster pseudobulks)")

## patristic projections onto each atlas's own topology ----------------------
fit_A = fit_branch_lengths(tree_a, D_A, qn)
fit_B = fit_branch_lengths(tree_b, D_B, cols)
print(f"patristic fits: A r={fit_A['pearson_r']:.3f} "
      f"stress={fit_A['stress']:.3f} | B r={fit_B['pearson_r']:.3f} "
      f"stress={fit_B['stress']:.3f}")
P_A = patristic_matrix(tree_a, fit_A["positions"], qn) * fit_A["scale"]
P_B = patristic_matrix(tree_b, fit_B["positions"], cols) * fit_B["scale"]

CA_hop, la = leaf_path_dist(tree_a)
CB_hop, lb = leaf_path_dist(tree_b)
CA_hop = CA_hop[np.ix_([la.index(q) for q in qn],
                       [la.index(q) for q in qn])]
CB_hop = CB_hop[np.ix_([lb.index(c) for c in cols],
                       [lb.index(c) for c in cols])]

## solve the four arms --------------------------------------------------------
arms = {}
for name, CA, CB in (("hop", CA_hop, CB_hop),
                     ("raw_metric", D_A, D_B),
                     ("patristic", P_A, P_B)):
    pi, gap = solve(M, CA, CB, wA, wB, alpha=ALPHA, rho=RHO, epsilon=0.0)
    arms[name] = (pi, gap)
import ot
pi_uot = ot.unbalanced.mm_unbalanced(wA, wB, M, reg_m=RHO, div="kl")
arms["uot"] = (pi_uot, 0.0)

rows = []
for name, (pi, gap) in arms.items():
    d = decompose(pi, qn, cols, family_of_leaf, family_leaves)
    argmax = sum(x["argmax_family"] == x["query"] for x in d)
    conf = sum(x["category"] == "confident_correct" for x in d)
    single_ok = sum(x["argmax_family"] == x["query"]
                    for x in d if x["query"] in singletons)
    eff = float(np.median([x.get("eff_leaves", np.nan) for x in d
                           if x.get("eff_leaves") is not None]))
    rows.append({"arm": name, "argmax_family": f"{argmax}/23",
                 "confident": conf, "singletons": f"{single_ok}/{len(singletons)}",
                 "median_eff_leaves": round(eff, 2),
                 "pq_gap": f"{gap:.1e}"})
    print(f"{name:11s} argmax {argmax}/23 | confident {conf:2d} | "
          f"singletons {single_ok}/{len(singletons)} | "
          f"median eff leaves {eff:5.2f} | pq_gap {gap:.1e}")
write_csv(rows, os.path.join(OUT, "four_arm_comparison.csv"))
for name, (pi, _) in arms.items():
    write_csv(pi, os.path.join(OUT, f"pi_{name}.csv.gz"),
              row_names=qn, col_names=cols)
print("written to", OUT)
print("NOTE: prespecified extension — the frozen hop model remains the "
      "amygdala primary configuration regardless of this outcome")
