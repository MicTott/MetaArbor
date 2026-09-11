"""K=3 with PAIR-SPECIFIC gene panels (review item 3c): instead of
forcing every pair onto the three-way 18,128-gene intersection, each
pairwise Walk runs on that pair's own gene intersection (mac<->she on
their 20,808-gene panel — the committed K=2 evidence basis — and
mac<->mrca / she<->mrca on their respective pairwise intersections).
Input trees are the committed K=3 trees (taxonomies are fixed; only
the evidence panel varies). Decisions from the three runs combine
into one decision set; quotient assembly runs raw and certified
(union of the pairwise certification outputs). The question: does
atlas addition change mac<->she relationships ONLY through the
shared-panel restriction (removable by pair-specific panels), and
what happens to c27?

Run: PYTHONPATH=src python examples/retina_k3_pairwise_panels.py
"""
import csv
import gc
import json
import os

import anndata as ad
import numpy as np
from scipy.io import mmread

from metaarbor.consensus.candidates import canonical_nodes
from metaarbor.consensus.harmonize import harmonize
from metaarbor.consensus.quotient import quotient_assemble

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "retina_bipolar")
O2 = os.path.join(HERE, "retina_k2")
O3 = os.path.join(HERE, "retina_k3")
CAP, SEED = 2000, 0

it = json.load(open(os.path.join(O3, "retina_k3_input_trees.json")))
trees = {k: {"parent": it[k]["parent"], "children": it[k]["children"],
             "leaves": it[k]["leaves"]}
         for k in ("mac", "she", "mrca")}
stab = {(k, n): float(v) for k in trees
        for n, v in it[k]["support"].items()}

# ---- raw matrices and reproduced subsets ---------------------------------
print("loading raw macosko/shekhar...")
BC_TYPES = {"RBC (Rod Bipolar cell)": "RBC",
            "BC1A": "BC1A", "BC1B": "BC1B", "BC2": "BC2",
            "BC3A": "BC3A", "BC3B": "BC3B", "BC4": "BC4",
            "BC5A (Cone Bipolar cell 5A)": "BC5A", "BC5B": "BC5B",
            "BC5C": "BC5C", "BC5D": "BC5D", "BC6": "BC6",
            "BC7 (Cone Bipolar cell 7)": "BC7",
            "BC8/9 (mixture of BC8 and BC9)": "BC8_9"}
mac_raw = mmread(os.path.join(D, "macosko_counts.mtx")).tocsc()
she_raw = mmread(os.path.join(D, "shekhar_counts.mtx")).tocsc()
mac_genes = [g for g in open(os.path.join(
    D, "macosko_genes.txt")).read().split("\n") if g]
she_genes = [g for g in open(os.path.join(
    D, "shekhar_genes.txt")).read().split("\n") if g]
mac_ids = [l for l in open(os.path.join(
    D, "macosko_cellids.txt")).read().split("\n") if l]
she_ids = [l for l in open(os.path.join(
    D, "shekhar_cellids.txt")).read().split("\n") if l]
with open(os.path.join(D, "macosko_cells.csv")) as fh:
    mac_cd = {r["cell.id"]: int(float(r["cluster"]))
              for r in csv.DictReader(fh)
              if r["cluster"] not in ("", "NA")}
with open(os.path.join(D, "shekhar_cells.csv")) as fh:
    she_cd = {r["NAME"]: r["CLUSTER"] for r in csv.DictReader(fh)}
mac_keep = [i for i, cid in enumerate(mac_ids)
            if mac_cd.get(cid) in range(26, 34)]
mac_lab = np.asarray([f"mac|c{mac_cd[mac_ids[i]]}" for i in mac_keep])
she_keep = [i for i, cid in enumerate(she_ids)
            if she_cd.get(cid) in BC_TYPES]
she_lab = np.asarray([f"she|{BC_TYPES[she_cd[she_ids[i]]]}"
                      for i in she_keep])

print("loading mrca (reproduced subset)...")
A = ad.read_h5ad(os.path.join(D, "mrca_bipolar.h5ad"), backed="r")
keep = np.asarray(A.obs["accession"] == "GSE243413")
albl = np.asarray(A.obs["author_cell_type"].astype(str))
rng = np.random.default_rng(SEED)
sel = np.zeros(A.n_obs, dtype=bool)
for t in sorted(set(albl)):
    idx = np.where(keep & (albl == t))[0]
    if len(idx) > CAP:
        idx = np.sort(rng.choice(idx, CAP, replace=False))
    sel[idx] = True
sub = A[np.where(sel)[0]].to_memory()
mrca_lab = np.asarray([f"mrca|{t}" for t in
                       sub.obs["author_cell_type"].astype(str)])
saved = np.loadtxt(os.path.join(O3, "mrca_labels.txt"), dtype=str)
assert (mrca_lab == saved).all(), "mrca subset not reproduced"
mrca_genes = [g.upper() for g in
              sub.var["feature_name"].astype(str)]
mrca_raw = sub.raw.X.tocsc()


def first_index(genes):
    out = {}
    for i, g in enumerate(genes):
        out.setdefault(g.upper(), i)
    return out


gi = {"mac": first_index(mac_genes), "she": first_index(she_genes),
      "mrca": first_index(mrca_genes)}
raw = {"mac": (mac_raw, mac_keep, mac_lab, True),
       "she": (she_raw, she_keep, she_lab, True),
       "mrca": (mrca_raw, None, mrca_lab, False)}


def pair_matrices(a, b):
    panel = sorted(set(gi[a]) & set(gi[b]))
    out = {}
    for k in (a, b):
        M, keep_idx, lb, genes_by_row = raw[k]
        cols = [gi[k][g] for g in panel]
        if genes_by_row:                    # genes x cells mtx
            Xk = M[cols, :][:, keep_idx].T.tocsr()
        else:                               # cells x genes h5ad
            Xk = M[:, cols].tocsr()
        out[k] = (np.asarray(Xk.todense()), lb)
    return panel, out


combined_sel, certified = {}, []
summaries = {}
for a, b in (("mac", "she"), ("mac", "mrca"), ("she", "mrca")):
    panel, mats = pair_matrices(a, b)
    print(f"\n=== {a}<->{b}: {len(panel)} pairwise genes ===")
    ds = {k: {"counts": mats[k][0], "labels": mats[k][1],
              "gene_names": panel,
              "lib": mats[k][0].sum(axis=1)} for k in (a, b)}
    harm = harmonize(ds, {k: trees[k] for k in (a, b)}, n_hvg=1000,
                     n_boot=200, stability=stab)
    for (ki, kj), recs in harm["decisions"]["selections"].items():
        combined_sel[f"{ki}>{kj}"] = {
            n: {"selected": r["selected"],
                "matched": bool(r["matched"]),
                "support": (None if r["support"] is None or
                            r["support"] != r["support"]
                            else float(r["support"]))}
            for n, r in recs.items()}
    pc = [{"members": dict(nd["members"]),
           "support": float(nd["mean_boot_support"])}
          for nd in harm["backbone"]["nodes"]
          if nd["status"] == "backbone" and len(nd["members"]) >= 2]
    certified.extend(pc)
    summaries[f"{a}_{b}"] = {"panel": len(panel),
                             "certified": [sorted(m["members"]
                                                  .items())
                                           for m in pc]}
    for m in pc:
        print("   certified:", sorted(m["members"].items()),
              f"{m['support']:.3f}")
    del mats, ds, harm
    gc.collect()

canon = {k: canonical_nodes(trees[k])[1] for k in trees}
gq_r = quotient_assemble(trees, canon, combined_sel)
gq_c = quotient_assemble(trees, canon, combined_sel,
                         certified=certified)
truth = json.load(open(os.path.join(O3, "retina_k3_truth.json")))
group_of = {}
for grp, d in truth.items():
    for ch in d["cherries"]:
        for l in ch:
            group_of[l] = grp
    for l in d["mac"]:
        group_of[l] = grp
for tag, gq in (("raw", gq_r), ("certified", gq_c)):
    sh = sum(1 for v in gq["vertices"].values() if v["shared"])
    tri = [r for r, v in gq["vertices"].items()
           if len(v["members"]) == 3]
    wrong = [r for r, v in gq["vertices"].items() if v["shared"] and
             len({group_of[m] for m in v["members"].values()
                  if m in group_of}) > 1]
    print(f"\npair-specific K=3 quotient ({tag}): "
          f"vertices={len(gq['vertices'])} shared={sh} "
          f"three_atlas={len(tri)} forest={gq['is_forest']} "
          f"certs={len(gq['certificates'])} "
          f"wrong_group={len(wrong)}")
    vid_of = {}
    for r, v in gq["vertices"].items():
        for d_, m_ in v["members"].items():
            vid_of[(d_, m_)] = r
    c27 = vid_of[("mac", "mac|c27")]
    print(f"  c27 vertex: "
          f"{sorted(gq['vertices'][c27]['members'].items())}")
json.dump({"summaries": summaries,
           "decisions": combined_sel, "certified": certified},
          open(os.path.join(O3, "retina_k3_pairwise_panels.json"),
               "w"), indent=1)
print("done")
