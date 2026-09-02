# TreeNeighbor packages

Two installable packages built from the validated research code (top-level
`R/` + `analysis/` stay untouched as the historical record that produced the
frozen benchmark results):

| Path | What | Ships |
|---|---|---|
| `treeneighbor-py/` | **Primary** python package | kernel, trees, frozen walk, refinement-invariant marginals, frozen FUGW (optional `[ot]`), AnnData wrappers |
| `TreeNeighbor-R/` | R companion | kernel, trees, frozen walk, marginals, simulation — pure R, no python dependency (FUGW is python-only) |
| `parity/` | cross-language gate | R script that (1) proves the packaged walk reproduces the saved frozen benchmark exactly and (2) exports fixtures |
| `fixtures/` | parity fixtures | real Allen platform vote caches, trees, similarity, packaged-R walk outputs, simulation inputs/outputs, MINSTD check vector |

## Cross-language guarantee

Both packages draw bootstrap indices from the same portable MINSTD stream
(exact in R doubles, plain ints in python) with a per-query seed and a
documented draw order. The chain of custody:

1. `parity/01_export_and_gate.R` — packaged R walk must reproduce the saved
   frozen-benchmark selections/relations **exactly** (GATE PASSED
   2026-09-02: 23/23 forward, 103/103 reverse identical).
2. `treeneighbor-py/tests/test_parity.py` — python walk on the exported
   real caches must match the packaged-R outputs for every query, plus a
   simulation end-to-end check (HVGs, similarity matrix, selections).

## JHPCE installation

Python (primary):

```bash
module load conda            # or your preferred python >= 3.10 module
conda create -n treeneighbor python=3.12 -y
conda activate treeneighbor
pip install ./treeneighbor-py[all,test]   # anndata + POT
pytest treeneighbor-py/tests/test_core.py # smoke test (no fixtures needed)
```

R companion:

```bash
module load R                # >= 4.2
R CMD INSTALL TreeNeighbor-R
Rscript TreeNeighbor-R/tests/test_all.R
```

The full parity suite additionally needs `fixtures/` (~60 MB), generated on
any machine holding the benchmark results by
`Rscript pkgs/parity/01_export_and_gate.R`.

## Frozen configurations (do not tune on validation data)

- Walk: alpha 0.05, n_boot 200, min_auroc 0.6, margin 0.01,
  vote_override 0.9, min_compact 0.7.
- FUGW: cost 1 − S, rho 0.3, alpha 0.9 (design weight), epsilon 0,
  refinement-invariant marginals both sides; argmax-family +
  mass-confidence readouts.
