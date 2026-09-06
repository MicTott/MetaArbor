"""Projection prototype — real-data held-out-PLATFORM evaluation + speed
benchmark on the Allen WMB PL-ILA-ORB data.

Test 1 (fine reference, coarse-labeled query):
  reference = 10Xv3 cells, CLUSTER labels, curated 4-level tree
  (class -> subclass -> supertype -> cluster);
  query = every 10Xv2 cell (different platform = the batch axis).
  Truth = each v2 cell's own subclass annotation (used ONLY to score).
  Score best_leaf via its curated subclass; measure subclass/class
  accuracy, wrong-class rate, resolution-depth distribution, and the
  abstention value (accuracy among cells that RESOLVED to subclass
  depth or deeper vs all cells). Baseline: flat argmax over the global
  leaf vote matrix (same evidence, no hierarchy, no local re-ranking).

Test 2 (coarse reference, fine-labeled query):
  reference = 10Xv2 cells, SUBCLASS labels, 2-level class -> subclass
  tree; query = every 10Xv3 cell; truth = its curated subclass.

Speed: build + project wall time at reference caps 25/50 and query
sizes 2k / 5k / full.
"""
import csv
import json
import os
import time

import numpy as np
from scipy.io import mmread

from metaarbor import tree_from_levels
from metaarbor.projection import build_projector, project

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
if not os.path.isdir(D):
    D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "projection_demo")
os.makedirs(OUT, exist_ok=True)


def load(tag):
    counts = np.asarray(mmread(os.path.join(D, f"counts_{tag}.mtx"))
                        .todense()).T
    lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
    with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells


genes = open(os.path.join(D, "genes.txt")).read().split()
c2, l2, cells2 = load("10Xv2")
c3, l3, cells3 = load("10Xv3")
sub2 = np.asarray([c["subclass"] for c in cells2])
cls2 = np.asarray([c["class"] for c in cells2])
clu3 = np.asarray([c["cluster"] for c in cells3])
sub3 = np.asarray([c["subclass"] for c in cells3])
cls3 = np.asarray([c["class"] for c in cells3])
clu_to_sub = {c["cluster"]: c["subclass"] for c in cells3}
clu_to_cls = {c["cluster"]: c["class"] for c in cells3}
sub_to_cls = {c["subclass"]: c["class"] for c in cells3}
sub_to_cls.update({c["subclass"]: c["class"] for c in cells2})

results = {}

# ---- Test 1: v3 cluster reference, v2 query -------------------------------
tree3 = tree_from_levels(
    sorted({(c["class"], c["subclass"], c["supertype"], c["cluster"])
            for c in cells3}),
    ["class", "subclass", "supertype", "cluster"])
t0 = time.time()
proj3 = build_projector([{"counts": c3, "labels": clu3,
                          "gene_names": genes, "lib": l3,
                          "name": "10Xv3"}], tree3, cap_per_leaf=50)
t_build = time.time() - t0
t0 = time.time()
out = project(proj3, c2, genes, lib=l2)
t_proj = time.time() - t0

pred_sub = np.asarray([clu_to_sub[b] for b in out["best_leaf"]])
pred_cls = np.asarray([clu_to_cls[b] for b in out["best_leaf"]])
depth = out["resolved_depth"]
acc_sub = float(np.mean(pred_sub == sub2))
acc_cls = float(np.mean(pred_cls == cls2))
deep = depth >= 2                                   # subclass or deeper
acc_sub_deep = float(np.mean(pred_sub[deep] == sub2[deep])) \
    if deep.any() else float("nan")
# flat baseline: argmax over global leaf votes (same evidence, no tree)
leaves = np.asarray(out["leaves"])
flat_leaf = leaves[np.argmax(out["leaf_mean"], axis=1)]
flat_sub = np.asarray([clu_to_sub[b] for b in flat_leaf])
flat_cls = np.asarray([clu_to_cls[b] for b in flat_leaf])
results["t1"] = {
    "n_query": int(len(sub2)), "n_ref_cells":
        int(sum(len(r["labels"]) for r in proj3["refs"])),
    "build_s": round(t_build, 1), "project_s": round(t_proj, 1),
    "cells_per_s": round(len(sub2) / t_proj, 1),
    "subclass_acc": round(acc_sub, 4),
    "class_acc": round(acc_cls, 4),
    "wrong_class_rate": round(1 - acc_cls, 4),
    "frac_resolved_subclass_or_deeper": round(float(np.mean(deep)), 4),
    "subclass_acc_when_resolved_deep": round(acc_sub_deep, 4),
    "median_resolved_depth": float(np.median(depth)),
    "flat_subclass_acc": round(float(np.mean(flat_sub == sub2)), 4),
    "flat_class_acc": round(float(np.mean(flat_cls == cls2)), 4),
}
print("Test 1 (v3 cluster ref -> v2 query):",
      json.dumps(results["t1"], indent=1))

# per-cell table for auditing
with open(os.path.join(OUT, "t1_per_cell.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["true_subclass", "best_leaf", "pred_subclass",
                "resolved_node", "resolved_depth", "stop_margin",
                "path_score", "top_leaf_mean"])
    for i in range(len(sub2)):
        w.writerow([sub2[i], out["best_leaf"][i], pred_sub[i],
                    out["resolved_node"][i], depth[i],
                    round(float(out["stop_margin"][i]), 3),
                    round(float(out["path_score"][i]), 4),
                    round(float(out["top_leaf_mean"][i]), 3)])

# ---- Test 2: v2 subclass reference, v3 query ------------------------------
tree2 = tree_from_levels(
    sorted({(c["class"], c["subclass"]) for c in cells2}),
    ["class", "subclass"])
t0 = time.time()
proj2 = build_projector([{"counts": c2, "labels": sub2,
                          "gene_names": genes, "lib": l2,
                          "name": "10Xv2"}], tree2, cap_per_leaf=50)
out2 = project(proj2, c3, genes, lib=l3)
t_all2 = time.time() - t0
pred2 = out2["best_leaf"]
pred2_cls = np.asarray([sub_to_cls.get(b, "?") for b in pred2])
results["t2"] = {
    "n_query": int(len(sub3)),
    "total_s": round(t_all2, 1),
    "subclass_acc": round(float(np.mean(pred2 == sub3)), 4),
    "class_acc": round(float(np.mean(pred2_cls == cls3)), 4),
    "wrong_class_rate": round(float(np.mean(pred2_cls != cls3)), 4),
    "frac_resolved_to_subclass":
        round(float(np.mean(out2["resolved_depth"] >= 2)), 4),
}
print("Test 2 (v2 subclass ref -> v3 query):",
      json.dumps(results["t2"], indent=1))

# ---- Speed scaling --------------------------------------------------------
speed = []
rs = np.random.RandomState(0)
for cap in (25, 50):
    pr = build_projector([{"counts": c3, "labels": clu3,
                           "gene_names": genes, "lib": l3}],
                         tree3, cap_per_leaf=cap)
    n_ref = int(sum(len(r["labels"]) for r in pr["refs"]))
    for nq in (2000, 5000, len(sub2)):
        idx = (rs.choice(len(sub2), nq, replace=False)
               if nq < len(sub2) else np.arange(len(sub2)))
        t0 = time.time()
        project(pr, c2[idx], genes, lib=l2[idx])
        dt = time.time() - t0
        speed.append({"cap_per_leaf": cap, "n_ref_cells": n_ref,
                      "n_query": int(nq), "seconds": round(dt, 1),
                      "cells_per_s": round(nq / dt, 1)})
        print(f"speed cap={cap} ref={n_ref} n={nq}: {dt:.1f}s "
              f"({nq/dt:.0f} cells/s)")
results["speed"] = speed

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("wrote", os.path.join(OUT, "results.json"))
