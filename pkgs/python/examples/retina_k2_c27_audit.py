"""c27<->BC3B focused evidence audit (tree-blind, frozen kernel).

Question: did input-ancestry preservation block a legitimate
c27<->BC3B equivalence (which OTHarmonizer asserts), or is
MetaArbor's clade-level abstention correct?

Layer 1 - committed Walk decisions (no computation needed):
  mac|c27 -> she|BC3B   matched, support 1.00   (forward)
  she|BC3B -> n04       matched, support 1.00   (reverse: the mac
                         OFF CLADE {c27,c28,c29}, not c27)
  Rejection reason: NONRECIPROCITY. No ancestry test was involved;
  the input-tree prior never entered this decision.

Layer 2 - reciprocal per-cell assignment matrix (this script):
per-cell argmax of frozen leaf-size-normalized votes, label level,
tree-blind by construction. Controls sharp (c26->RBC 1.00/1.00,
c32<->BC7 0.98/0.98). Disputed pair diffuse BOTH ways:
  c27 cells:  BC3B 0.44 | BC4 0.34 | scatter   (margin 0.10, H=1.53)
  BC3B cells: c28 0.50  | c27 0.40             (margin 0.10, H=1.08)
And the wider matrix shows Macosko's OFF clusters CROSSCUT Shekhar's
OFF types (BC2 0.97, BC3A 0.98, BC4 0.81, BC1A 0.76 of cells all
argmax to c28) - a resolution mismatch, not hidden 1:1 equivalences.

VERDICT: MetaArbor's abstention is evidence-based - c27 is a
{BC3B+BC4}-leaning mixture and BC3B straddles c27/c28; the honest
relationship is clade-level containment (exactly what the one-way
support-1.0 call expresses under the contained_in vocabulary).
OTHarmonizer's c27&BC3B is a forced 1:1 on a 0.44-vs-0.34 split -
consistent with its own repeat-run flip of this pair (c28&BC2).

Run: python examples/retina_k2_c27_audit.py
"""
import numpy as np
from scipy.io import mmread
import os

from metaarbor import measure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "retina_k2")
genes = [g for g in open(os.path.join(
    OUT, "shared_genes.txt")).read().split("\n") if g]
mac_X = np.asarray(mmread(os.path.join(OUT, "mac_bipolar.mtx"))
                   .todense())
she_X = np.asarray(mmread(os.path.join(OUT, "she_bipolar.mtx"))
                   .todense())
mac_lab = np.loadtxt(os.path.join(OUT, "mac_labels.txt"), dtype=str)
she_lab = np.loadtxt(os.path.join(OUT, "she_labels.txt"), dtype=str)
m = measure(mac_X, mac_lab, she_X, she_lab, genes,
            lib_a=mac_X.sum(axis=1), lib_b=she_X.sum(axis=1))


def assign(cache, src_lab):
    V = cache["V"] / cache["leaf_sizes"]
    tgt = np.asarray(cache["leaves"])
    top = tgt[V.argmax(axis=1)]
    return {q: {t: float((top[src_lab == q] == t).mean())
                for t in tgt} for q in sorted(set(src_lab))}


fwd = assign(m["cache_a"], mac_lab)
rev = assign(m["cache_b"], she_lab)
import csv  # noqa: E402
rows = []
for direc, mat in (("mac->she", fwd), ("she->mac", rev)):
    for q, dist in mat.items():
        for t, p in dist.items():
            if p >= 0.01:
                rows.append({"direction": direc, "source": q,
                             "target": t, "fraction": round(p, 4)})
with open(os.path.join(OUT, "retina_k2_transfer_matrix.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for q in ("mac|c26", "mac|c27", "mac|c28", "mac|c29", "mac|c32"):
    d = sorted(fwd[q].items(), key=lambda x: -x[1])[:4]
    print(q, " ".join(f"{t}:{p:.2f}" for t, p in d))
for q in ("she|RBC", "she|BC3B", "she|BC4", "she|BC7"):
    d = sorted(rev[q].items(), key=lambda x: -x[1])[:4]
    print(q, " ".join(f"{t}:{p:.2f}" for t, p in d))
