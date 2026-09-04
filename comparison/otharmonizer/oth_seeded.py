"""OTHarmonizer SEEDED-sampling control: distinguishes core-method
instability from the avoidable implementation-level randomness in
get_metacells (unseeded np.random.choice of 100 cells/annotation).

Patch: get_metacells draws from a RandomState seeded deterministically
per (annotation, sample_size) — identical metacell draws in every run
and in both insertion orders. Everything downstream (partial OT,
transmission analysis, tree assembly) is untouched.

Runs: v2-first x2 (determinism check -> must be identical),
v3-first x2, on the same trained scVI latent as the main comparison.
Remaining between-order disagreement under seeded sampling is the pure
insertion-order effect of the core method.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "OTHarmonizer"))
import OTHarmonizer as oth  # noqa: E402
import OTHarmonizer.harmonize as H  # noqa: E402
import anndata as ad  # noqa: E402

OUT = os.path.join(HERE, "oth_out")


def get_metacells_seeded(adata, cell_type_col, sample_size=100):
    meta_cell_list, label_list = [], []
    cell_type_counts = adata.obs[cell_type_col].value_counts()
    for cell_type in cell_type_counts.index:
        cell_indices = adata.obs[
            adata.obs[cell_type_col] == cell_type].index
        if cell_type_counts[cell_type] >= sample_size:
            seed = int(hashlib.md5(
                f"{cell_type}|{sample_size}".encode()).hexdigest()[:8],
                16)
            rs = np.random.RandomState(seed)
            sampled_idx = rs.choice(cell_indices, size=sample_size,
                                    replace=False)
        else:
            sampled_idx = cell_indices
        meta_cell_list.append(adata[sampled_idx, :].X)
        label_list.extend([cell_type] * len(sampled_idx))
    return np.vstack(meta_cell_list), np.array(label_list)


H.get_metacells = get_metacells_seeded

X = np.load(os.path.join(OUT, "latent.npy"))
obs = pd.read_csv(os.path.join(OUT, "latent_obs.csv"), index_col=0)
latent = ad.AnnData(X)
latent.obs["annotation"] = obs["annotation"].tolist()
latent.obs["batch"] = obs["batch"].tolist()


def serialize(node):
    return {"label": str(node.label),
            "children": [serialize(c) for c in node.children]}


for tag, order in (("seeded_v2f_a", ["v2", "v3"]),
                   ("seeded_v2f_b", ["v2", "v3"]),
                   ("seeded_v3f_a", ["v3", "v2"]),
                   ("seeded_v3f_b", ["v3", "v2"])):
    np.random.seed(0)   # any residual global-stream use is fixed too
    lat = latent.copy()
    root = oth.do_harmonization(lat, "annotation", "batch",
                                sample_size=500, batch_order=order)
    with open(os.path.join(OUT, f"oth_tree_{tag}.json"), "w") as fh:
        json.dump(serialize(root), fh)
    print("wrote", tag)
print("done")
