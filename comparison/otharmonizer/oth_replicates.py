"""OTHarmonizer sampling-stability replicates: reuse the trained scVI
latent, run do_harmonization 5x per insertion order with seeded
np.random (get_metacells samples 100 cells/annotation with unseeded
np.random.choice — this is the method's internal stochasticity)."""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "OTHarmonizer"))
import OTHarmonizer as oth  # noqa: E402
import anndata as ad  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(HERE, "oth_out")
X = np.load(os.path.join(OUT, "latent.npy"))
obs = pd.read_csv(os.path.join(OUT, "latent_obs.csv"), index_col=0)
latent = ad.AnnData(X)
latent.obs["annotation"] = obs["annotation"].tolist()
latent.obs["batch"] = obs["batch"].tolist()


def serialize(node):
    return {"label": str(node.label),
            "children": [serialize(c) for c in node.children]}


for order_tag, order in (("v2f", ["v2", "v3"]), ("v3f", ["v3", "v2"])):
    for rep in range(5):
        np.random.seed(1000 + rep)
        lat = latent.copy()
        root = oth.do_harmonization(lat, "annotation", "batch",
                                    sample_size=500, batch_order=order)
        path = os.path.join(OUT, f"oth_tree_{order_tag}_r{rep}.json")
        with open(path, "w") as fh:
            json.dump(serialize(root), fh)
        print("wrote", os.path.basename(path))
print("done")
