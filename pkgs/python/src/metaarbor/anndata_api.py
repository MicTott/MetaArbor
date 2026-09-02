"""AnnData-first entry points. The core functions (kernel/walk/fugw) take
plain arrays; these wrappers only translate."""
from __future__ import annotations

import numpy as np

from . import kernel


def measure_adata(adata_a, adata_b, label_key_a, label_key_b,
                  layer=None, n_hvg=1000, lib_key=None, chunk=2000):
    """Run the measurement layer on two AnnData objects. Genes are matched
    by var_names intersection. `lib_key`: obs column with full-gene totals
    when X carries a gene subset; defaults to summing X."""
    shared = sorted(set(adata_a.var_names) & set(adata_b.var_names))
    if len(shared) < 100:
        raise ValueError(f"only {len(shared)} shared genes")

    def get(ad):
        X = ad[:, shared].layers[layer] if layer else ad[:, shared].X
        return X

    lib_a = (np.asarray(adata_a.obs[lib_key]) if lib_key else None)
    lib_b = (np.asarray(adata_b.obs[lib_key]) if lib_key else None)
    return kernel.measure(
        get(adata_a), np.asarray(adata_a.obs[label_key_a]),
        get(adata_b), np.asarray(adata_b.obs[label_key_b]),
        gene_names=shared, n_hvg=n_hvg, lib_a=lib_a, lib_b=lib_b, chunk=chunk)


def tree_from_obs(adata, level_keys):
    """Build the taxonomy tree from obs level columns (coarse -> fine)."""
    from .tree import tree_from_levels
    rows = adata.obs[list(level_keys)].drop_duplicates().itertuples(index=False)
    return tree_from_levels([tuple(map(str, r)) for r in rows], list(level_keys))
