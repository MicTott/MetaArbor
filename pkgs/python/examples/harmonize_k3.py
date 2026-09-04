"""Generic K-atlas harmonize() runner (frozen v0.8 rules) — built for
the K=3 amygdala reconciliation (Yu, Allen, Hochgerner) but atlas-count
agnostic.

No atlas is a backbone; no curated crosswalk constrains the result; no
thresholds or rules are changed. Each input tree is inferred
independently from that atlas's own expression and labels; all
bidirectional pairwise Walk comparisons feed one consensus assembly.

Usage:
    python harmonize_k3.py manifest.json

Manifest (paths relative to the manifest file):
{
  "genes": "genes.txt",                 // shared gene panel, one per line
  "out": "amygdala_excit",              // output directory
  "atlases": {
    "yu":   {"counts": "counts_yu.mtx",   // MatrixMarket, genes x cells
             "cells":  "cells_yu.csv",    // per-cell CSV with label_col
             "label_col": "cluster",
             "lib": "lib_yu.txt"},        // optional full-gene totals
    "allen": {...}, "hoch": {...}
  },
  "infer_hvg": 2000, "infer_boot": 50,   // optional (defaults shown)
  "n_hvg": 1000, "n_boot": 200
}

Outputs in <out>/:
  reconciled.png        K input trees beside the reconciled hierarchy,
                        every original label colored by atlas; meta-
                        clades / private / single-atlas / affiliates
                        (approx-prefixed) / conflicts marked
  provenance.csv        every original leaf label -> consensus node,
                        status, how placed (member/affiliate), parent
  summary.csv           counts by atlas x status (+ conflicts,
                        unplaced internal layers)
  tree.json             the assembled hierarchy (id -> node record)
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.io import mmread

from metaarbor.consensus.harmonize import harmonize
from metaarbor.consensus.plot_reconciled import plot_reconciled_tree
from metaarbor.infer_tree import infer_tree
from metaarbor.style import save_pub


def main(manifest_path):
    with open(manifest_path) as fh:
        mf = json.load(fh)
    base = os.path.dirname(os.path.abspath(manifest_path))

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    genes = open(rp(mf["genes"])).read().split()
    out = rp(mf.get("out", "harmonize_k"))
    os.makedirs(out, exist_ok=True)

    datasets, trees, label_col = {}, {}, {}
    for key, spec in mf["atlases"].items():
        counts = np.asarray(mmread(rp(spec["counts"])).todense()).T
        with open(rp(spec["cells"])) as fh:
            cells = list(csv.DictReader(fh))
        if counts.shape[0] != len(cells):
            raise ValueError(
                f"{key}: {counts.shape[0]} cells in counts vs "
                f"{len(cells)} rows in cells CSV (counts must be "
                "genes x cells in the .mtx)")
        lib = (np.loadtxt(rp(spec["lib"]))
               if spec.get("lib") else None)
        col = spec["label_col"]
        labels = np.asarray([f"{key}|{c[col]}" for c in cells])
        label_col[key] = col
        print(f"[{key}] {counts.shape[0]} cells, "
              f"{len(set(labels))} labels ({col}); inferring tree "
              "from expression only")
        inf = infer_tree(counts, labels, lib=lib,
                         n_hvg=mf.get("infer_hvg", 2000),
                         n_boot=mf.get("infer_boot", 50), seed=0)
        print(f"[{key}] {inf['provenance']['n_internal_kept']} supported "
              "internal clades")
        trees[key] = inf["tree"]
        datasets[key] = {"counts": counts, "labels": labels,
                         "gene_names": genes,
                         **({"lib": lib} if lib is not None else {})}

    harm = harmonize(datasets, trees, n_hvg=mf.get("n_hvg", 1000),
                     n_boot=mf.get("n_boot", 200))
    nodes = harm["tree"]

    # ---- provenance: every original leaf label -> consensus node ----------
    AFF = "≈ "
    rows = []
    for key in sorted(trees):
        placed = {}
        for mid, nd in nodes.items():
            m = nd["members"].get(key)
            if m in trees[key]["leaves"] and m not in placed:
                placed[m] = ("member", mid)
        for mid, nd in nodes.items():
            for a in nd["aliases"]:
                if isinstance(a, str) and a.startswith(AFF):
                    b = a[len(AFF):]
                    if b in trees[key]["leaves"] and b not in placed:
                        placed[b] = ("affiliate", mid)
        for leaf in trees[key]["leaves"]:
            how, mid = placed.get(leaf, ("absent", None))
            nd = nodes.get(mid, {})
            rows.append({
                "atlas": key,
                "label": leaf.split("|", 1)[-1],
                "consensus_node": mid or "",
                "node_display": nd.get("display", ""),
                "status": nd.get("status", "MISSING"),
                "placed_as": how,
                "parent": (nd.get("parent") or "ROOT") if mid else "",
                "co_members": "; ".join(
                    v for k2, v in nd.get("members", {}).items()
                    if k2 != key),
            })
    with open(os.path.join(out, "provenance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_absent = sum(1 for r in rows if r["placed_as"] == "absent")
    if n_absent:
        print(f"WARNING: {n_absent} labels absent from assembly — "
              "completeness violation, report this")

    # ---- summary counts by atlas x status --------------------------------
    summ = []
    for key in sorted(trees):
        counts_s = {}
        for r in rows:
            if r["atlas"] == key:
                shared = ("shared" if r["co_members"] else r["status"])
                tag = ("shared" if shared == "shared" and
                       r["status"] == "backbone" else r["status"])
                if r["placed_as"] == "affiliate":
                    tag = "affiliate"
                counts_s[tag] = counts_s.get(tag, 0) + 1
        summ.append({"atlas": key, **counts_s})
    n_conf = len([c for c in harm["conflicts"]
                  if c.get("class") == "genuine_conflict"])
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        keys_all = sorted({k for s in summ for k in s} - {"atlas"})
        w = csv.DictWriter(fh, fieldnames=["atlas"] + keys_all)
        w.writeheader()
        for s in summ:
            w.writerow({k: s.get(k, 0) for k in ["atlas"] + keys_all})
        fh.write(f"\n# genuine_conflicts,{n_conf}\n")
        fh.write("# unplaced_internal_layers,"
                 f"{sum(len(v) for v in harm['unplaced_internals'].values())}\n")
        fh.write(f"# affiliates_total,{len(harm['affiliates'])}\n")
    print("summary:", summ, "| conflicts:", n_conf)

    # ---- audit inputs: one-way Walk decisions, canonical maps, input
    # trees, per-label cell counts (everything the offline unanchored-
    # label audit consumes) ------------------------------------------------
    from metaarbor.consensus.candidates import canonical_nodes

    def _num(x):
        return None if x is None or (isinstance(x, float) and
                                     np.isnan(x)) else float(x)

    sel_out = {}
    for (ki, kj), recs in harm["decisions"]["selections"].items():
        sel_out[f"{ki}>{kj}"] = {
            n: {"selected": r["selected"], "matched": bool(r["matched"]),
                "support": _num(r["support"])}
            for n, r in recs.items()}
    with open(os.path.join(out, "decisions.json"), "w") as fh:
        json.dump(sel_out, fh)
    with open(os.path.join(out, "canonical.json"), "w") as fh:
        json.dump({k: canonical_nodes(trees[k])[1] for k in trees}, fh)
    with open(os.path.join(out, "input_trees.json"), "w") as fh:
        json.dump({k: {"parent": trees[k]["parent"],
                       "children": trees[k]["children"],
                       "leaves": list(trees[k]["leaves"])}
                   for k in trees}, fh)
    with open(os.path.join(out, "cell_counts.json"), "w") as fh:
        json.dump({k: {str(l): int(n) for l, n in
                       zip(*np.unique(datasets[k]["labels"],
                                      return_counts=True))}
                   for k in datasets}, fh)
    if harm["repairs"]:
        print(f"ASSEMBLY REPAIRS: {len(harm['repairs'])} labels "
              "reinstated as unplaced_single_atlas (upstream loss — "
              "flagged, excluded from support counts)")

    # ---- tree + figure ---------------------------------------------------
    with open(os.path.join(out, "tree.json"), "w") as fh:
        json.dump({i: {"parent": nd["parent"], "status": nd["status"],
                       "members": nd["members"], "aliases": nd["aliases"],
                       "display": nd["display"],
                       "assembly_repair": bool(nd.get("assembly_repair"))}
                   for i, nd in nodes.items()}, fh, indent=1)
    fig, _ = plot_reconciled_tree(
        harm, trees,
        dataset_names={k: k for k in sorted(trees)})
    save_pub(fig, os.path.join(out, "reconciled"), formats=("png", "pdf"),
             dpi=150)
    print("wrote reconciled.png/.pdf, provenance.csv, summary.csv, "
          "tree.json ->", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python harmonize_k3.py manifest.json")
    main(sys.argv[1])
