"""Molecular branch-length experiment on the Allen target atlas (10Xv3).

Prespecified (user, 2026-09-02): keep the curated topology fixed; fit
nonnegative branch lengths from target-only rank-based pseudobulk
distances; normalize (median leaf pair = 1); report fit quality; bootstrap
cells for length stability; confirm inference untouched (structural: this
module never calls Walk or Transport). Visualization decision rule: good
fit -> length_by='molecular' becomes an offered option; poor fit -> keep
the depth cladogram and report that the taxonomy is not additively
representable.

Usage: python examples/branch_length_experiment.py
Needs: data/wmb_plilaorb (benchmark inputs) + fixtures.
"""
import csv
import gzip
import os

import numpy as np
from scipy.io import mmread

from metaarbor import (fit_branch_lengths, plot_alignment_tree,
                       pseudobulk_distances, tree_from_levels, write_csv)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "wmb_plilaorb")
FX = os.path.join(REPO, "pkgs", "fixtures")
OUT = os.path.join(HERE, "allen_demo", "branch_lengths")
os.makedirs(OUT, exist_ok=True)


def rcsv(name):
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        return list(csv.reader(fh))


counts = np.asarray(mmread(os.path.join(DATA, "counts_10Xv3.mtx")).todense()).T
lib = np.loadtxt(os.path.join(DATA, "lib_10Xv3.txt"))
with open(os.path.join(DATA, "cells_10Xv3.csv")) as fh:
    cells = list(csv.DictReader(fh))
labels = np.asarray([c["cluster"] for c in cells])
lv = rcsv("tree_levels_b.csv.gz")
tree_b = tree_from_levels([tuple(r) for r in lv[1:]], lv[0])
print(f"target atlas: {counts.shape[0]} cells x {counts.shape[1]} genes, "
      f"{len(set(labels))} clusters")

D, leaves = pseudobulk_distances(counts, labels, lib=lib)
fit = fit_branch_lengths(tree_b, D, leaves)
print(f"fit: pearson r = {fit['pearson_r']:.3f} | normalized stress = "
      f"{fit['stress']:.3f} | scale (median leaf-pair 1-rho) = "
      f"{fit['scale']:.3f}")

## bootstrap cells within cluster: are fitted lengths stable? ----------------
rs = np.random.RandomState(20260902)
boot_r, boot_len = [], []
idx_by = {l: np.flatnonzero(labels == l) for l in leaves}
for b in range(20):
    take = np.concatenate([rs.choice(ix, len(ix), replace=True)
                           for ix in idx_by.values()])
    Db, _ = pseudobulk_distances(counts[take], labels[take], lib=lib[take])
    fb = fit_branch_lengths(tree_b, Db, leaves)
    boot_r.append(np.corrcoef(
        [fit["edge_lengths"][e] for e in fit["edge_lengths"]],
        [fb["edge_lengths"][e] for e in fit["edge_lengths"]])[0, 1])
    boot_len.append([fb["edge_lengths"][e] for e in fit["edge_lengths"]])
boot_len = np.asarray(boot_len)
base = np.asarray([fit["edge_lengths"][e] for e in fit["edge_lengths"]])
big = base > np.median(base[base > 0])
cv = np.std(boot_len[:, big], axis=0) / np.maximum(
    np.mean(boot_len[:, big], axis=0), 1e-12)
print(f"bootstrap (20 cell resamples): edge-length correlation to point fit "
      f"median {np.median(boot_r):.3f} [{np.min(boot_r):.3f}-"
      f"{np.max(boot_r):.3f}]; median CV of larger edges {np.median(cv):.3f}")

## render the phylogram with the frozen Allen summary ------------------------
al = []
hdr = rcsv("interp_py.csv.gz")[0]
for r in rcsv("interp_py.csv.gz")[1:]:
    d = dict(zip(hdr, r))
    for k in ("walk_selected", "transport_node"):
        if d[k] in ("", "None", "NA"):
            d[k] = None
    d["walk_decision_support"] = (float(d["walk_decision_support"])
                                  if d["walk_decision_support"] not in
                                  ("", "None", "nan") else float("nan"))
    d["transport_mass"] = float(d["transport_mass"] or 0)
    al.append(d)
fig, ax = plot_alignment_tree(al, tree_b, length_by="molecular",
                              node_positions=fit["positions"])
ax.set_title(f"MetaArbor alignment, molecular branch lengths "
             f"(fit r={fit['pearson_r']:.2f}, stress={fit['stress']:.2f})")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"alignment_tree_molecular.{ext}"),
                dpi=200, bbox_inches="tight")
write_csv([{"edge": e, "length": l} for e, l in fit["edge_lengths"].items()],
          os.path.join(OUT, "fitted_branch_lengths.csv"))
print("phylogram + lengths written to", OUT)
print("inference untouched: branch_fit never calls Walk or Transport (structural)")
