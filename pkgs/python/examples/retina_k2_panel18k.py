"""c27 feature-panel sensitivity (review item 3a+3b): the K=3 run
changed a K=2 relationship — with the 20,808-gene K=2 panel
she|BC3B -> n04 (clade; c27 NOT reciprocal), while on the
18,128-gene three-way panel she|BC3B -> mac|c27 (reciprocal,
merging {c27, BC3B, BC3B}). Macosko and Shekhar cells and trees are
IDENTICAL in both runs; only the gene panel differs.

This script isolates the panel variable: (a) frozen mac<->she
harmonize alone on the 18,128-gene panel — if BC3B -> c27 here too,
the flip is a pure panel effect, not an atlas-count effect; (b) the
tree-blind reciprocal per-cell assignment matrix (frozen kernel) on
the same panel, to compare against the committed 20,808-gene matrix
(c27 cells: BC3B 0.44 / BC4 0.34; BC3B cells: c28 0.50 / c27 0.40).

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/retina_k2_panel18k.py
"""
import json
import os

import numpy as np
from scipy.io import mmread

from metaarbor import measure
from metaarbor.consensus.candidates import canonical_nodes
from metaarbor.consensus.harmonize import harmonize

HERE = os.path.dirname(os.path.abspath(__file__))
O2 = os.path.join(HERE, "retina_k2")
O3 = os.path.join(HERE, "retina_k3")

genes = [g for g in open(os.path.join(
    O3, "common_genes.txt")).read().split("\n") if g]
X = {k: np.asarray(mmread(os.path.join(
    O3, f"{k}_bipolar.mtx")).todense()) for k in ("mac", "she")}
lab = {k: np.loadtxt(os.path.join(O3, f"{k}_labels.txt"), dtype=str)
       for k in ("mac", "she")}
it = json.load(open(os.path.join(O3, "retina_k3_input_trees.json")))
trees = {k: {"parent": it[k]["parent"], "children": it[k]["children"],
             "leaves": it[k]["leaves"]} for k in ("mac", "she")}
stab = {(k, n): float(v) for k in ("mac", "she")
        for n, v in it[k]["support"].items()}

# ---- (a) frozen mac<->she harmonize on the 18,128-gene panel -------------
print("harmonize mac<->she on 18,128-gene panel...")
harm = harmonize({k: {"counts": X[k], "labels": lab[k],
                      "gene_names": genes, "lib": X[k].sum(axis=1)}
                  for k in ("mac", "she")}, trees, n_hvg=1000,
                 n_boot=200, stability=stab)
sel = harm["decisions"]["selections"]
focus = [("she", "mac", "she|BC3B"), ("mac", "she", "mac|c27"),
         ("she", "mac", "she|BC4"), ("mac", "she", "mac|c28")]
out = {}
for si, ti, n in focus:
    r = sel.get((si, ti), {}).get(n)
    out[f"{si}>{ti}:{n}"] = r and {
        "selected": r["selected"], "matched": bool(r["matched"]),
        "support": None if r["support"] != r["support"]
        else float(r["support"])}
    print(f"  {si}>{ti} {n}: {out[f'{si}>{ti}:{n}']}")
cert = [{"members": dict(nd["members"]),
         "support": float(nd["mean_boot_support"])}
        for nd in harm["backbone"]["nodes"]
        if nd["status"] == "backbone" and len(nd["members"]) >= 2]
print("certified merges on this panel:")
for m in cert:
    print("  ", sorted(m["members"].items()),
          f"support={m['support']:.3f}")
c27_pair = any(set(m["members"].values()) >= {"mac|c27", "she|BC3B"}
               for m in cert)
print(f"c27<->BC3B CERTIFIED on 18k panel (K=2 alone): {c27_pair}")

# ---- (b) tree-blind per-cell assignment matrix on this panel -------------
m = measure(X["mac"], lab["mac"], X["she"], lab["she"], genes,
            lib_a=X["mac"].sum(axis=1), lib_b=X["she"].sum(axis=1))


def assign(cache, src_lab):
    V = cache["V"] / cache["leaf_sizes"]
    tgt = np.asarray(cache["leaves"])
    top = tgt[V.argmax(axis=1)]
    return {q: {t: float((top[src_lab == q] == t).mean())
                for t in tgt} for q in sorted(set(src_lab))}


fwd = assign(m["cache_a"], lab["mac"])    # mac cells -> she types
rev = assign(m["cache_b"], lab["she"])    # she cells -> mac clusters


def top3(d):
    return sorted(d.items(), key=lambda kv: -kv[1])[:3]


print("\nper-cell matrix (18,128-gene panel), key rows:")
for q in ("mac|c26", "mac|c27", "mac|c28", "mac|c32"):
    print(f"  {q}: {[(t, round(v, 3)) for t, v in top3(fwd[q])]}")
for q in ("she|RBC", "she|BC3B", "she|BC4", "she|BC7"):
    print(f"  {q}: {[(t, round(v, 3)) for t, v in top3(rev[q])]}")
json.dump({"walk_calls": out, "certified": cert,
           "c27_bc3b_certified": bool(c27_pair),
           "percell_mac": {q: fwd[q] for q in fwd},
           "percell_she": {q: rev[q] for q in rev}},
          open(os.path.join(O3, "retina_k2_panel18k_audit.json"),
               "w"), indent=1)
print("done")
