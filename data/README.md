# Derived data provenance

All derived from the Allen Brain Cell atlas, release **20230630** (Yao et al.
2023), AWS bucket `allen-brain-cell-atlas`, CC-BY-NC-4.0. Nothing here is raw
source data; everything can be rebuilt from the bucket.

| Path | Contents | Built by |
|---|---|---|
| `wmb_frontal_cell_metadata.csv` | per-cell metadata (taxonomy labels, donor, platform, expression package) for ROIs ACA, PL-ILA-ORB, AI — 309,126 cells | streaming filter of `views/cell_metadata_with_cluster_annotation.csv` (1.4 GB, not stored) |
| `wmb_frontal_summary.json` | cell/level counts per ROI and platform | same |
| `wmb_cache/<pkg>.npz` + `_aux.npz` | sparse raw counts (all 32,285 genes) for PL-ILA-ORB cells only, per expression package; aux = cell labels, gene ids, full-gene UMI totals | download→extract→delete of the six Isocortex `-raw.h5ad` (~50 GB transferred, not stored) |
| `wmb_plilaorb/` | the benchmark input: per-platform counts on a 4,724-gene superset (union of each platform's top-3000 log1p-CPM-variance genes), cells.csv labels, full-gene lib sizes | `export_wmb_subset.py`: cluster floor ≥30 cells/platform (103 clusters pass), cap 300 cells/cluster/platform, seed 20260902 |

Caveats that travel with these files:

- **PL-ILA-ORB only** — dissection-level, not CCF-registered; cells from
  adjacent structures ride along at low frequency, which is exactly why the
  cluster floor exists.
- The cap (300/cluster/platform) trades power for runtime; rare-cluster
  results are noisier than abundant-cluster results by construction.
- Library sizes in `lib_*.txt` are full-gene totals; CPM computed against the
  4,724-gene subset alone would be distorted — always pass them to
  `tn_lognorm(lib=)`.
- 10Xv2 vs 10Xv3 is a real, strong batch axis (chemistry + depth); donor
  splits within platform are the milder axis (donor ids are in cells.csv).
