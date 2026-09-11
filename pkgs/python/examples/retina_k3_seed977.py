"""Retina K=3 — the prespecified seed-977 replicate (PRESPEC section
7, flagged absent by review): re-infer ALL THREE input trees at seed
977 on the common 18,128-gene panel, rerun frozen harmonize, and
compare against the committed seed-0 run: directed decision
agreement, certified merge set, and certified-quotient structure.

Run: PYTHONPATH=src python examples/retina_k3_seed977.py
"""
import json
import os

import numpy as np
from scipy.io import mmread

from metaarbor.consensus.candidates import canonical_nodes
from metaarbor.consensus.harmonize import harmonize
from metaarbor.consensus.quotient import quotient_assemble
from metaarbor.infer_tree import infer_tree

HERE = os.path.dirname(os.path.abspath(__file__))
O3 = os.path.join(HERE, "retina_k3")
SEED = 977
DS = ("mac", "she", "mrca")
genes = [g for g in open(os.path.join(
    O3, "common_genes.txt")).read().split("\n") if g]
X = {k: np.asarray(mmread(os.path.join(
    O3, f"{k}_bipolar.mtx")).todense()) for k in DS}
lab = {k: np.loadtxt(os.path.join(O3, f"{k}_labels.txt"), dtype=str)
       for k in DS}

trees, stab = {}, {}
for k in DS:
    print(f"infer_tree {k} at seed {SEED}...")
    inf = infer_tree(X[k], lab[k], lib=X[k].sum(axis=1),
                     n_hvg=2000, n_boot=50, seed=SEED)
    trees[k] = inf["tree"]
    stab.update({(k, n): float(v)
                 for n, v in inf["support"].items()})
print("harmonize...")
harm = harmonize({k: {"counts": X[k], "labels": lab[k],
                      "gene_names": genes, "lib": X[k].sum(axis=1)}
                  for k in DS}, trees, n_hvg=1000, n_boot=200,
                 stability=stab)

base = json.load(open(os.path.join(O3, "retina_k3_decisions.json")))
rep = {}
for (ki, kj), recs in harm["decisions"]["selections"].items():
    for n, r in recs.items():
        rep[(f"{ki}>{kj}", n)] = (r["selected"] if r["matched"]
                                  else None)
bd = {(d_, n): (r["selected"] if r["matched"] else None)
      for d_, recs in base.items() for n, r in recs.items()}
common = set(bd) & set(rep)
agree = sum(bd[k] == rep[k] for k in common) / len(common)
print(f"decision agreement vs seed-0 (comparable node names): "
      f"{agree:.4f} ({len(common)} decisions; NOTE anonymous "
      f"internal-node names may differ between seed trees — "
      f"agreement over shared names only)")

cert = [{"members": dict(nd["members"]),
         "support": float(nd["mean_boot_support"])}
        for nd in harm["backbone"]["nodes"]
        if nd["status"] == "backbone" and len(nd["members"]) >= 2]
print("certified merges at seed 977:")
for m in cert:
    print("  ", sorted(m["members"].items()),
          f"support={m['support']:.3f}")
canon = {k: canonical_nodes(trees[k])[1] for k in trees}
sel_out = {}
for (ki, kj), recs in harm["decisions"]["selections"].items():
    sel_out[f"{ki}>{kj}"] = {
        n: {"selected": r["selected"], "matched": bool(r["matched"]),
            "support": (None if r["support"] is None or
                        r["support"] != r["support"]
                        else float(r["support"]))}
        for n, r in recs.items()}
g = quotient_assemble({k: {"parent": trees[k]["parent"],
                           "children": trees[k]["children"],
                           "leaves": list(trees[k]["leaves"])}
                       for k in DS}, canon, sel_out, certified=cert)
sh = sum(1 for v in g["vertices"].values() if v["shared"])
tri = sum(1 for v in g["vertices"].values()
          if len(v["members"]) == 3)
NAME_PAIRS = ["RBC", "BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4",
              "BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7"]
vid_of = {}
for r, v in g["vertices"].items():
    for d_, m in v["members"].items():
        vid_of[(d_, m)] = r
hits = [t for t in NAME_PAIRS
        if vid_of.get(("she", f"she|{t}")) ==
        vid_of.get(("mrca", f"mrca|{t}"))
        and vid_of.get(("she", f"she|{t}")) is not None]
print(f"seed-977 certified quotient: shared={sh} three_atlas={tri} "
      f"forest={g['is_forest']} certs={len(g['certificates'])} "
      f"name_pairs={len(hits)}/13 {sorted(hits)}")
json.dump({"agreement_shared_names": agree,
           "certified": cert,
           "quotient": {"shared": sh, "three_atlas": tri,
                        "forest": g["is_forest"],
                        "certs": len(g["certificates"]),
                        "name_pairs": sorted(hits)}},
          open(os.path.join(O3, "retina_k3_seed977.json"), "w"),
          indent=1)
print("done")
