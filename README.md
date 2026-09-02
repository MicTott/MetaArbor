# TreeNeighbor

**Installable packages live in [pkgs/](pkgs/README.md)** — the primary
python package (`treeneighbor-py`: kernel, frozen walk, refinement-invariant
marginals, frozen FUGW, AnnData API) and the pure-R companion
(`TreeNeighbor-R`). Both are cross-language parity-gated against the frozen
benchmark. This top level is the research repo that produced those results.

An integration-free framework for aligning independently constructed cell-type
taxonomies across atlases of different resolutions, using MetaNeighbor-derived
molecular evidence with either direct hierarchical selection (interpretable
baseline) or MetaNeighbor-informed fused unbalanced Gromov–Wasserstein
transport (the ambitious estimator, kept only if it beats the baseline on
held-out hierarchy recovery). See [DESIGN.md](DESIGN.md) for the full
specification and [NOTES.md](NOTES.md) for prototype findings that revised it.

## Status

Baseline and FUGW estimators implemented; simulation-validated; **first real
benchmark run complete** (Allen WMB PL-ILA-ORB, 10Xv2 vs 10Xv3 cross-platform
split, curated subclass→cluster ground truth):

- stage 2: 100/103 self-RBH across the platform gap, median self-AUROC 0.998
- stage 5 baseline: 21/23 subclasses exact, 84/103 clusters correct
- stage 6: FUGW scored 12/23 under the original 0.90-mass rule + uniform
  marginals; a layered post-mortem (NOTES.md items 12-13) attributed the gap
  to the decision rule (~5-7 queries) and a uniform-marginal capacity
  artifact that smeared singleton families (predicted 4.48×, observed
  4.2-4.8×). With hierarchy-balanced marginals FUGW reaches 23/23 on this
  benchmark — **direct selection remains the primary method (validated,
  interpretable); hierarchy-FUGW is a co-equal challenger** pending the
  same frozen validation battery and the cross-taxonomy test

**Validated** with the frozen estimator across three batch severities
(random / donor-held-out / cross-platform; `analysis/06_batch_conditions.R`):
the robustness curve is flat — self-AUROC 0.998 and sibling margin ~0.013 at
every severity, forward 21-22/23 throughout, and zero wrong-branch errors in
either direction. Reverse failures (16-19, constant across severities) are
the IT-layer continuum and lineage intermediates, not batch effects. Scope
caveat: this is robustness *within the jointly curated Allen taxonomy*;
independent cross-atlas taxonomies are the next test.

| Path | Contents |
|---|---|
| `R/kernel.R` | native voting kernel: joint HVGs, rank-standardized cross-dataset votes, the per-cell vote cache, symmetrized directional AUROC leaf costs |
| `R/tree.R` | tree from taxonomy level columns, leaf sets, path distances (FUGW structure input) |
| `R/baseline.R` | hierarchical selection: sibling-contrast merge test, context compactness, relation calls (leaf / family / unmatched / discordant) |
| `R/simulate.R` | two-atlas hierarchical simulation with dialable batch effects |
| `tests/test_kernel.R` | exact additivity of cached votes, root neutrality, AUROC boundaries, directional asymmetry |
| `analysis/01_simulation_validation.R` | 32/32 node selections correct at batch_sd ∈ {0, 0.5, 1}, both directions |
| `analysis/02_edge_cases.R` | novel population → unmatched; two-family mixture → discordant |
| `data/wmb_frontal_*` | Allen WMB frontal-cortex (ACA, PL-ILA-ORB, AI) per-cell metadata for the benchmark: 309k cells, taxonomy labels, donor, platform, expression package |
| `analysis/07–12`, `figures/` | intuition figure set from the frozen benchmark (walk diagrams, error topology, vote heatmaps, batch conditions, threshold sensitivity, FUGW decomposition); plotting tables in `figures/tables/` |

Run everything:

```bash
Rscript tests/test_kernel.R && Rscript analysis/01_simulation_validation.R && Rscript analysis/02_edge_cases.R
```

(~1 min total. No package installation; base R + Matrix/matrixStats.)

## Next

1. Benchmark stage 1–2 (DESIGN §4.3): download the six Isocortex h5ad packages
   (~50 GB), subset to frontal cells, RBH anchors at cluster level across the
   10Xv2/10Xv3 split.
2. Forests for subset-of-family queries; donor-level bootstrap.
3. FUGW estimator via POT/reticulate, consuming `costs$M` and
   `tn_leaf_path_dist()`.
