"""OTHarmonizer arm of the FROZEN retina K=2 comparison (reviewer
step 4). Identical inputs to the frozen MetaArbor record (e887341):
the committed bipolar subsets (same cells, same 20,808 shared genes),
published labels only (mac|c26..c33, she|BC types), batch = dataset.
OTHarmonizer receives NEITHER MetaArbor's inferred trees NOR the
ON/OFF/RBC truth during construction. Tutorial pipeline verbatim:
normalize_total 1e4 -> log1p -> HVG(batch_key, subset) -> oth.scVI
(epoch_num=80, seed 0) -> do_harmonization(sample_size=500).

Runs: default ordering, mac_first, she_first, plus TWO repeat calls
(r0, r1) of the default on the same latent - OTHarmonizer's
get_metacells contains unseeded sampling (Allen finding), so repeats
measure its internal stochasticity.

STACK DEVIATION, declared: the Allen-era comparison used the pinned
old stack (scvi 1.2.1 / anndata 0.10.9 / numpy 1.26 / POT 0.9.3),
whose environment no longer exists; this run uses scvi 1.4.2 /
anndata 0.11.4 / numpy 2.3.5 / POT 0.9.7. Both methods in THIS
comparison run on the same machine and inputs; cross-era OTH numbers
are not comparable and are not compared.

Run: python retina_oth.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "OTHarmonizer"))
import OTHarmonizer as oth  # noqa: E402
import anndata as ad  # noqa: E402
import scanpy as sc  # noqa: E402
import scvi  # noqa: E402

scvi.settings.seed = 0
RK = os.path.join(HERE, "..", "..", "pkgs", "python", "examples",
                  "retina_k2")
OUT = os.path.join(HERE, "retina_oth_out")
os.makedirs(OUT, exist_ok=True)

genes = [g for g in open(os.path.join(
    RK, "shared_genes.txt")).read().split("\n") if g]
mac_X = np.asarray(mmread(os.path.join(RK, "mac_bipolar.mtx"))
                   .todense())
she_X = np.asarray(mmread(os.path.join(RK, "she_bipolar.mtx"))
                   .todense())
mac_lab = np.loadtxt(os.path.join(RK, "mac_labels.txt"), dtype=str)
she_lab = np.loadtxt(os.path.join(RK, "she_labels.txt"), dtype=str)
X = np.vstack([mac_X, she_X]).astype(np.float32)
adata = ad.AnnData(X)
adata.var_names = genes
adata.obs["batch"] = (["mac"] * len(mac_lab) + ["she"] * len(she_lab))
adata.obs["annotation"] = np.concatenate([mac_lab, she_lab])
print("combined:", adata.shape, "| labels:",
      len(set(adata.obs["annotation"])))
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key="batch", subset=True)
print("HVG kept:", adata.shape[1])
latent = oth.scVI(adata, "batch", "annotation", epoch_num=80)
np.save(os.path.join(OUT, "latent.npy"), latent.X)


def serialize(node):
    return {"label": str(node.label),
            "children": [serialize(c) for c in node.children]}


runs = [("default", None), ("mac_first", ["mac", "she"]),
        ("she_first", ["she", "mac"]), ("default_r1", None)]
for tag, order in runs:
    print(f"===== do_harmonization: {tag} =====")
    lat = latent.copy()
    try:
        root = oth.do_harmonization(lat, "annotation", "batch",
                                    sample_size=500,
                                    batch_order=order)
        json.dump(serialize(root),
                  open(os.path.join(OUT, f"oth_retina_{tag}.json"),
                       "w"))
        print(f"wrote oth_retina_{tag}.json")
    except Exception as e:
        print(f"RUN FAILED ({tag}): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
print("done")
