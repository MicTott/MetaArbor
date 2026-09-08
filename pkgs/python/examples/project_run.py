"""Manifest-driven SINGLE-REFERENCE projection runner (frozen
prototype; built for the amygdala evaluation, dataset-agnostic).

Thresholds are frozen (min_margin = 0.98, chosen on synthetic
operating curves); this runner never tunes them — it reports the
coverage-risk curve instead.

Usage:
    python project_run.py manifest.json

Manifest (paths relative to the manifest file):
{
  "genes": "genes.txt",
  "out": "projection_out",
  "reference": {
    "counts": "counts_ref.mtx",          // MatrixMarket, genes x cells
    "cells": "cells_ref.csv",            // per-cell CSV
    "label_col": "cluster",
    "lib": "lib_ref.txt",                // optional
    "tree_levels": ["class", "subclass", "cluster"],
                                          // curated levels, coarse->fine
                                          // (last = label_col), OR
    "infer_tree": true                    // infer from expression
  },
  "queries": [
    {"name": "yu", "counts": "counts_yu.mtx", "cells": "cells_yu.csv",
     "lib": "lib_yu.txt",
     "truth_col": "cluster"}              // optional: scoring only
  ],
  "cap_per_label": 50, "n_hvg": 1000      // optional
}

Outputs per query in <out>/:
  <name>_per_cell.csv    best_label / best_node / resolved_node /
                         depth / stop_margin / candidates / OOR
  <name>_summary.json    counts by resolved depth, coverage-risk curve,
                         and (when truth_col given) label accuracy
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.io import mmread

from metaarbor import tree_from_levels
from metaarbor.infer_tree import infer_tree
from metaarbor.projection import build_projector, project


def rp(base, p):
    return p if os.path.isabs(p) else os.path.join(base, p)


def load(base, spec):
    counts = np.asarray(mmread(rp(base, spec["counts"])).todense()).T
    with open(rp(base, spec["cells"])) as fh:
        cells = list(csv.DictReader(fh))
    if counts.shape[0] != len(cells):
        raise ValueError(f"{spec['counts']}: {counts.shape[0]} cells "
                         f"vs {len(cells)} CSV rows")
    lib = (np.loadtxt(rp(base, spec["lib"]))
           if spec.get("lib") else None)
    return counts, cells, lib


def main(manifest_path):
    with open(manifest_path) as fh:
        mf = json.load(fh)
    base = os.path.dirname(os.path.abspath(manifest_path))
    genes = open(rp(base, mf["genes"])).read().split()
    out = rp(base, mf.get("out", "projection_out"))
    os.makedirs(out, exist_ok=True)

    rspec = mf["reference"]
    rc, rcells, rlib = load(base, rspec)
    labels = np.asarray([c[rspec["label_col"]] for c in rcells])
    if rspec.get("infer_tree"):
        inf = infer_tree(rc, labels, lib=rlib,
                         n_hvg=mf.get("n_hvg", 2000), n_boot=50, seed=0)
        tree = inf["tree"]
        print(f"[ref] inferred tree: "
              f"{inf['provenance']['n_internal_kept']} internals")
    else:
        lv = rspec["tree_levels"]
        tree = tree_from_levels(
            sorted({tuple(c[x] for x in lv) for c in rcells}), lv)
        print(f"[ref] curated tree over levels {lv}")
    proj = build_projector(
        [{"counts": rc, "labels": labels, "gene_names": genes,
          "lib": rlib, "name": rspec.get("name", "reference")}],
        tree, cap_per_label=mf.get("cap_per_label", 50),
        n_hvg=mf.get("n_hvg", 1000))
    print(f"[ref] projector fitted: "
          f"{sum(len(r['labels']) for r in proj['refs'])} cells, "
          f"{len(set(labels))} labels")

    for q in mf["queries"]:
        qc, qcells, qlib = load(base, q)
        o = project(proj, qc, genes, lib=qlib)
        name = q.get("name", "query")
        with open(os.path.join(out, f"{name}_per_cell.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            hdr = ["best_label", "best_node", "resolved_node",
                   "resolved_depth", "stop_margin", "stop_candidates",
                   "max_label_vote"]
            tc = q.get("truth_col")
            w.writerow((["truth"] if tc else []) + hdr)
            for i in range(len(qcells)):
                row = ([qcells[i][tc]] if tc else []) + [
                    o["best_label"][i], o["best_node"][i],
                    o["resolved_node"][i], int(o["resolved_depth"][i]),
                    ("" if np.isnan(o["stop_margin"][i])
                     else round(float(o["stop_margin"][i]), 4)),
                    o["stop_candidates"][i],
                    round(float(o["max_label_vote"][i]), 4)]
                w.writerow(row)
        depth_counts = {int(d): int((o["resolved_depth"] == d).sum())
                        for d in sorted(set(o["resolved_depth"]))}
        curve = []
        pm = o["path_margins"]
        for t in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
            ok = np.ones(len(pm), dtype=bool)
            d = np.zeros(len(pm), dtype=int)
            for lvl in range(pm.shape[1]):
                passed = ok & ~np.isnan(pm[:, lvl]) & (pm[:, lvl] >= t)
                d[passed] = lvl + 1
                ok = passed
            curve.append({"threshold": t,
                          "mean_depth": round(float(d.mean()), 3)})
        summ = {"n_cells": int(len(qcells)),
                "depth_counts": depth_counts,
                "coverage_curve": curve,
                "params": o["params"]}
        if q.get("truth_col"):
            truth = np.asarray([c[q["truth_col"]] for c in qcells])
            summ["best_label_acc"] = round(
                float(np.mean(o["best_label"] == truth)), 4)
        with open(os.path.join(out, f"{name}_summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
        print(f"[{name}] {len(qcells)} cells -> per_cell.csv + "
              f"summary.json", summ.get("best_label_acc", ""))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python project_run.py manifest.json")
    main(sys.argv[1])
