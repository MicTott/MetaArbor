"""OTHarmonizer arm of the frozen matched comparison.

Identical inputs to MetaArbor: the same two Allen count matrices with
each atlas's own label column (v2 subclass, v3 cluster). Pipeline
follows the OTHarmonizer tutorial verbatim: normalize_total 1e4 ->
log1p -> HVG (batch_key, subset) -> oth.scVI (epoch_num=80, defaults)
-> do_harmonization(latent, ...) with sample_size=500.

Three insertion-order runs on the same scVI latent:
  default   (OTHarmonizer's own leiden-ratio ordering)
  v2_first  (coarse first)
  v3_first  (fine first)
Each resulting tree is serialized to JSON for the shared scorer.
"""
import csv
import json
import os
import sys

import numpy as np
import scanpy as sc
from scipy.io import mmread

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "OTHarmonizer"))
import OTHarmonizer as oth  # noqa: E402

import scvi  # noqa: E402
scvi.settings.seed = 0

D = "/Users/michael.totty/Documents/metaarbor/data/wmb_plilaorb"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oth_out")
os.makedirs(OUT, exist_ok=True)


def load(tag):
    counts = np.asarray(mmread(os.path.join(D, f"counts_{tag}.mtx"))
                        .todense()).T
    with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, cells


genes = open(os.path.join(D, "genes.txt")).read().split()
cA, cellsA = load("10Xv2")
cB, cellsB = load("10Xv3")
X = np.vstack([cA, cB]).astype(np.float32)
import anndata as ad  # noqa: E402
adata = ad.AnnData(X)
adata.var_names = genes
adata.obs["batch"] = ["v2"] * len(cellsA) + ["v3"] * len(cellsB)
adata.obs["annotation"] = ([c["subclass"] for c in cellsA] +
                           [c["cluster"] for c in cellsB])
print("combined:", adata.shape, "| v2 labels:",
      len(set(c["subclass"] for c in cellsA)), "| v3 labels:",
      len(set(c["cluster"] for c in cellsB)))

# tutorial preprocessing, verbatim
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key="batch", subset=True)
print("HVG kept:", adata.shape[1])

latent = oth.scVI(adata, "batch", "annotation", epoch_num=80)
np.save(os.path.join(OUT, "latent.npy"), latent.X)
latent.obs.to_csv(os.path.join(OUT, "latent_obs.csv"))


def serialize(node):
    # labels are plain strings; equals are "&"-joined (see initialize_tree)
    return {"label": str(node.label),
            "children": [serialize(c) for c in node.children]}


runs = {"default": None, "v2_first": ["v2", "v3"],
        "v3_first": ["v3", "v2"]}
for tag, order in runs.items():
    print(f"\n===== do_harmonization: {tag} =====")
    lat = latent.copy()
    try:
        root = oth.do_harmonization(lat, "annotation", "batch",
                                    sample_size=500, batch_order=order)
        with open(os.path.join(OUT, f"oth_tree_{tag}.json"), "w") as fh:
            json.dump(serialize(root), fh)
        print(f"wrote oth_tree_{tag}.json")
    except Exception as e:  # keep other runs alive; report faithfully
        print(f"RUN FAILED ({tag}): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
print("done")
