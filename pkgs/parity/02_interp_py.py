"""Interpretation-layer gate, python side.

From the Allen fixtures: build the frozen Transport coupling, run the walk
summary, transport summary and alignment summary; assert the summarized
Walk selections equal the saved frozen benchmark; write the coupling and
the summary for the R-side parity comparison.

Run: <python-with-metaarbor+pot> pkgs/parity/02_interp_py.py
"""
import csv
import gzip
import os
import sys

import numpy as np

from metaarbor import (alignment_summary, family_mass, transport_summary,
                       tree_from_levels, walk_summary, write_csv)
from metaarbor.fugw import fugw_map

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "fixtures")


def rcsv(name):
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        return list(csv.reader(fh))


lr = rcsv("cacheA_leaves.csv.gz")
leaves = [r[0] for r in lr[1:]]
sizes = np.asarray([float(r[1]) for r in lr[1:]])
V = np.asarray([[float(x) for x in r] for r in rcsv("cacheA_V.csv.gz")[1:]])
cache = {"V": V, "leaves": leaves, "leaf_sizes": sizes}
labels = np.asarray([r[0] for r in rcsv("labelsA_subclass.csv.gz")[1:]])
lv = rcsv("tree_levels_b.csv.gz")
tree_b = tree_from_levels([tuple(r) for r in lv[1:]], lv[0])
srows = rcsv("S_sub.csv.gz")
cols = srows[0][1:]
qn = [r[0] for r in srows[1:]]
S = np.asarray([[float(x) for x in r[1:]] for r in srows[1:]])
S_dir = {q: {c: S[i, j] for j, c in enumerate(cols)} for i, q in enumerate(qn)}
family_of_leaf = {r[3]: r[1] for r in lv[1:]}

## walk summary + identity assertion -----------------------------------------
wk = walk_summary(cache, labels, tree_b, S_dir)
saved = {r[0]: (r[1] if r[1] != "NA" else None, r[4])
         for r in rcsv("walkR_forward.csv.gz")[1:]}
for r in wk:
    s_sel, s_rel = saved[r["query"]]
    assert r["walk_selected"] == s_sel and r["walk_relation"] == s_rel, r
print("walk summary: selections identical to saved frozen benchmark")

## frozen transport + summary -------------------------------------------------
cls_of = {r[1]: r[0] for r in lv[1:]}
tree_a = tree_from_levels(sorted({(cls_of[q], q) for q in qn}),
                          ["class", "subclass"])
fit = fugw_map(S, tree_a, tree_b, qn, cols)
ts = transport_summary(fit["pi"], qn, cols, family_of_leaf, tree=tree_b)
fm, fams = family_mass(fit["pi"], cols, family_of_leaf, normalize=True)
assert np.allclose(fm.sum(axis=1), 1, atol=1e-9), "row-normalized mass != 1"
print("transport: row-normalized family mass sums to 1 for all queries")

## joint summary + determinism ------------------------------------------------
al1 = alignment_summary(wk, ts, tree_b)
al2 = alignment_summary(walk_summary(cache, labels, tree_b, S_dir),
                        transport_summary(fit["pi"], qn, cols,
                                          family_of_leaf, tree=tree_b), tree_b)
assert al1 == al2, "alignment summary is not deterministic"
print("alignment summary: deterministic; categories:",
      sorted(set(r["agreement"] for r in al1)))

write_csv(al1, os.path.join(FX, "interp_py.csv.gz"))
write_csv(fit["pi"], os.path.join(FX, "transport_pi.csv.gz"),
          row_names=qn, col_names=cols)
print("wrote interp_py.csv.gz and transport_pi.csv.gz")
