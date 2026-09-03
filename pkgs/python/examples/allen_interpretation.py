"""MetaArbor interpretation workflow on the Allen PL-ILA-ORB benchmark.

This is the vignette AND the render gate: it runs the full frozen
interpretation layer on the real fixtures — forward (subclasses onto the
cluster tree, with Transport) and reverse (clusters onto the subclass list,
Walk-only) — and writes the default result bundle plus the benchmark-only
error tree.

Usage: python examples/allen_interpretation.py [out_dir]
Requires: fixtures at pkgs/fixtures (built by pkgs/parity/01_export_and_gate.R)
"""
import csv
import gzip
import os
import sys

import numpy as np

from metaarbor import (plot_error_tree, result_bundle, tree_from_levels,
                       write_csv)
from metaarbor.fugw import fugw_map

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "..", "fixtures")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "allen_demo")


def rcsv(name):
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        return list(csv.reader(fh))


def load_cache(v, l):
    lr = rcsv(l)
    return {"V": np.asarray([[float(x) for x in r] for r in rcsv(v)[1:]]),
            "leaves": [r[0] for r in lr[1:]],
            "leaf_sizes": np.asarray([float(r[1]) for r in lr[1:]])}


## ---- forward: 23 subclasses onto the 103-cluster tree, with Transport -----
cache = load_cache("cacheA_V.csv.gz", "cacheA_leaves.csv.gz")
labels = np.asarray([r[0] for r in rcsv("labelsA_subclass.csv.gz")[1:]])
lv = rcsv("tree_levels_b.csv.gz")
tree_b = tree_from_levels([tuple(r) for r in lv[1:]], lv[0])
srows = rcsv("S_sub.csv.gz")
cols = srows[0][1:]
qn = [r[0] for r in srows[1:]]
S = np.asarray([[float(x) for x in r[1:]] for r in srows[1:]])
S_dir = {q: {c: S[i, j] for j, c in enumerate(cols)} for i, q in enumerate(qn)}
family_of_leaf = {r[3]: r[1] for r in lv[1:]}

cls_of = {r[1]: r[0] for r in lv[1:]}
tree_a = tree_from_levels(sorted({(cls_of[q], q) for q in qn}),
                          ["class", "subclass"])
fit = fugw_map(S, tree_a, tree_b, qn, cols)   # frozen MetaArbor-Transport

fwd = result_bundle(cache, labels, tree_b, S_dir,
                    out_dir=os.path.join(OUT, "forward"), prefix="fwd",
                    pi=fit["pi"], pi_rows=qn, pi_cols=cols,
                    family_of_leaf=family_of_leaf)
print(f"forward bundle: {len(fwd['files'])} files; agreement:",
      {a: sum(r['agreement'] == a for r in fwd['summary'])
       for a in set(r['agreement'] for r in fwd['summary'])})

# benchmark-only error tree (truth = each subclass's own node)
truth = {q: f"subclass:{q}" for q in qn}
fig, ax = plot_error_tree(fwd["summary"], truth, tree_b)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "forward", f"fwd_error_tree.{ext}"),
                dpi=200, bbox_inches="tight")
print("forward error tree written")

## ---- reverse: 103 clusters onto the subclass list, Walk-only --------------
cache_r = load_cache("cacheBsub_V.csv.gz", "cacheBsub_leaves.csv.gz")
labels_r = np.asarray([r[0] for r in rcsv("labelsB_cluster.csv.gz")[1:]])
subclasses = sorted(r[0] for r in rcsv("subclasses.csv.gz")[1:])
tree_flat = tree_from_levels([(s,) for s in subclasses], ["leaf"])
S_rev = {c: {q: S[qn.index(q), cols.index(c)] for q in qn}
         for c in sorted(set(labels_r))}
rev = result_bundle(cache_r, labels_r, tree_flat, S_rev,
                    out_dir=os.path.join(OUT, "reverse"), prefix="rev",
                    query_paths=None)
print(f"reverse bundle: {len(rev['files'])} files; relations:",
      {a: sum(r['walk_relation'] == a for r in rev['summary'])
       for a in set(r['walk_relation'] for r in rev['summary'])})
print("done; outputs in", OUT)
