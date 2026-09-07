"""Projection prototype — Allen held-out-PLATFORM evaluation + speed,
under the review-round fixes (fitted projector, refinement-calibrated
statistic, tie-aware ranks). min_margin = 0.98 was fixed on synthetic
operating curves (three prespecified constraints: family-only deep
leakage <=10%, novel deep leakage <=5%, pure-null root pass <=15%)
BEFORE this script ran; nothing here tunes it.

Reports: selective (coverage-risk) curve from per-split margins, micro
and MACRO subclass accuracy, per-subclass coverage, wrong-class rates,
cap/seed sensitivity, and speed scaling.
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

tree3 = tree_from_levels(
    sorted({(c["class"], c["subclass"], c["supertype"], c["cluster"])
            for c in cells3}),
    ["class", "subclass", "supertype", "cluster"])
results = {}


def depth_at(path_margins, t):
    ok = np.ones(len(path_margins), dtype=bool)
    d = np.zeros(len(path_margins), dtype=int)
    for lvl in range(path_margins.shape[1]):
        m = path_margins[:, lvl]
        passed = ok & ~np.isnan(m) & (m >= t)
        d[passed] = lvl + 1
        ok = passed
    return d


# ---- Test 1: v3 cluster reference -> all v2 cells -------------------------
t0 = time.time()
proj3 = build_projector([{"counts": c3, "labels": clu3,
                          "gene_names": genes, "lib": l3,
                          "name": "10Xv3"}], tree3, cap_per_label=50)
t_build = time.time() - t0
t0 = time.time()
out = project(proj3, c2, genes, lib=l2)
t_proj = time.time() - t0

pred_sub = np.asarray([clu_to_sub[b] for b in out["best_label"]])
pred_cls = np.asarray([clu_to_cls[b] for b in out["best_label"]])
subs_all = sorted(set(sub2))
per_sub_acc = {s: float(np.mean(pred_sub[sub2 == s] == s))
               for s in subs_all}
macro = float(np.mean(list(per_sub_acc.values())))
deep = out["resolved_depth"] >= 2
per_sub_cov = {s: float(np.mean(deep[sub2 == s])) for s in subs_all}

# selective (coverage-risk) curve: threshold the recorded per-split
# margins offline; risk = subclass error among cells resolving >= depth2
curve = []
for t in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
    d = depth_at(out["path_margins"], t)
    cov = float(np.mean(d >= 2))
    sel = d >= 2
    risk = float(np.mean(pred_sub[sel] != sub2[sel])) if sel.any() \
        else float("nan")
    curve.append({"threshold": t, "coverage": round(cov, 4),
                  "risk": round(risk, 4)})

results["t1"] = {
    "n_query": int(len(sub2)),
    "n_ref_cells": int(sum(len(r["labels"]) for r in proj3["refs"])),
    "build_s": round(t_build, 1), "project_s": round(t_proj, 1),
    "cells_per_s": round(len(sub2) / t_proj, 1),
    "subclass_acc_micro": round(float(np.mean(pred_sub == sub2)), 4),
    "subclass_acc_macro": round(macro, 4),
    "class_acc": round(float(np.mean(pred_cls == cls2)), 4),
    "wrong_class_rate": round(float(np.mean(pred_cls != cls2)), 4),
    "coverage_subclass_or_deeper": round(float(np.mean(deep)), 4),
    "selective_subclass_acc": round(
        float(np.mean(pred_sub[deep] == sub2[deep])), 4)
        if deep.any() else None,
    "flat_subclass_acc": round(float(np.mean(np.asarray(
        [clu_to_sub[l] for l in np.asarray(out["labels"])[
            np.argmax(out["label_vote"], axis=1)]]) == sub2)), 4),
    "coverage_risk_curve": curve,
    "per_subclass_coverage_min": round(min(per_sub_cov.values()), 3),
    "per_subclass_coverage_median": round(
        float(np.median(list(per_sub_cov.values()))), 3),
}
print("Test 1:", json.dumps(
    {k: v for k, v in results["t1"].items()
     if k != "coverage_risk_curve"}, indent=1))
print("coverage-risk:", curve)
with open(os.path.join(OUT, "t1_per_subclass.csv"), "w",
          newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["subclass", "n_cells", "accuracy", "coverage"])
    for s in subs_all:
        w.writerow([s, int((sub2 == s).sum()),
                    round(per_sub_acc[s], 4), round(per_sub_cov[s], 4)])

# ---- Test 2: v2 subclass reference -> all v3 cells ------------------------
tree2 = tree_from_levels(
    sorted({(c["class"], c["subclass"]) for c in cells2}),
    ["class", "subclass"])
proj2 = build_projector([{"counts": c2, "labels": sub2,
                          "gene_names": genes, "lib": l2,
                          "name": "10Xv2"}], tree2, cap_per_label=50)
out2 = project(proj2, c3, genes, lib=l3)
pred2 = out2["best_label"]
pred2_cls = np.asarray([sub_to_cls.get(b, "?") for b in pred2])
results["t2"] = {
    "n_query": int(len(sub3)),
    "subclass_acc": round(float(np.mean(pred2 == sub3)), 4),
    "class_acc": round(float(np.mean(pred2_cls == cls3)), 4),
    "coverage_to_subclass":
        round(float(np.mean(out2["resolved_depth"] >= 2)), 4),
}
print("Test 2:", json.dumps(results["t2"], indent=1))

# ---- cap/seed sensitivity -------------------------------------------------
sens = []
for cap in (25, 50):
    for seed in (0, 1, 2):
        pr = build_projector([{"counts": c3, "labels": clu3,
                               "gene_names": genes, "lib": l3,
                               "name": "10Xv3"}], tree3,
                             cap_per_label=cap, seed=seed)
        o = project(pr, c2, genes, lib=l2)
        ps = np.asarray([clu_to_sub[b] for b in o["best_label"]])
        dd = o["resolved_depth"] >= 2
        sens.append({"cap": cap, "seed": seed,
                     "subclass_acc": round(float(np.mean(ps == sub2)), 4),
                     "coverage": round(float(np.mean(dd)), 4),
                     "selective_acc": round(
                         float(np.mean(ps[dd] == sub2[dd])), 4)})
        print("sensitivity", sens[-1])
results["sensitivity"] = sens

# ---- speed scaling --------------------------------------------------------
speed = []
rs = np.random.RandomState(0)
for cap in (25, 50):
    pr = build_projector([{"counts": c3, "labels": clu3,
                           "gene_names": genes, "lib": l3,
                           "name": "10Xv3"}], tree3, cap_per_label=cap)
    n_ref = int(sum(len(r["labels"]) for r in pr["refs"]))
    for nq in (2000, 22067):
        idx = (rs.choice(len(sub2), nq, replace=False)
               if nq < len(sub2) else np.arange(len(sub2)))
        t0 = time.time()
        project(pr, c2[idx], genes, lib=l2[idx])
        dt = time.time() - t0
        speed.append({"cap": cap, "n_ref": n_ref, "n_query": int(nq),
                      "seconds": round(dt, 1),
                      "cells_per_s": round(nq / dt, 1)})
        print("speed", speed[-1])
results["speed"] = speed

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("wrote", os.path.join(OUT, "results.json"))
