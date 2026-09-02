# metaarbor

Integration-free alignment of cell-type taxonomies across atlases of
different resolutions. Primary implementation of the MetaArbor method:

- **Measurement kernel** — MetaNeighbor-derived rank voting with an exact
  vote-cache additivity property: any tree node's score is a column sum of
  cached leaf votes, so hierarchical questions cost nothing extra.
- **MetaArbor-Walk** (frozen) — votes navigate (each query cell votes for its
  argmax training leaf), AUROC contrasts decide (sibling contrast with a
  practical margin, paired bootstrap), with a clear-votes override.
- **Frozen FUGW estimator** (optional, `pip install metaarbor[ot]`) —
  fused unbalanced Gromov-Wasserstein transport with
  **refinement-invariant marginals**: annotation refinement redistributes a
  branch's transport capacity, it never creates capacity.

Both estimators were frozen and validated on the Allen whole-mouse-brain
PL-ILA-ORB benchmark across three batch conditions (random half-split,
donor-held-out, 10Xv2 vs 10Xv3). Walk and kernel are bit-comparable with the companion R package
`MetaArbor` via a shared portable bootstrap stream (`tests/test_parity.py`);
Transport is python-only and regression-gated against the frozen benchmark.

## Install

```bash
pip install -e ".[all,test]"      # anndata + POT + pytest
```

## Quick start (arrays)

```python
import metaarbor as tn

m = ma.measure(counts_a, labels_a, counts_b, labels_b,
               gene_names=genes)                  # cells x genes
tree_b = ma.tree_from_levels(level_rows, ["class", "subclass", "cluster"])
S_dir = {q: dict(zip(m["costs"]["cols"], m["costs"]["S"][i]))
         for i, q in enumerate(m["costs"]["rows"])}
mapping = ma.baseline_map(m["cache_a"], labels_a, tree_b, S_dir)
```

AnnData wrappers: `metaarbor.anndata_api.measure_adata` /
`tree_from_obs`. FUGW: `metaarbor.fugw.fugw_map` (frozen configuration
as defaults).
