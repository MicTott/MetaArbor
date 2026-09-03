# MetaArbor packages

MetaArbor (formerly TreeNeighbor; renamed at tag `v0.1-treeneighbor-final`)
aligns independently constructed cell-type taxonomies across atlases of
different resolutions, without expression-space integration. Two inference
modes share one measurement kernel:

- **MetaArbor-Walk** — frozen hierarchical selection: votes navigate, AUROC
  contrasts decide. Implemented in both packages, cross-language
  parity-gated.
- **MetaArbor-Transport** — frozen FUGW with refinement-invariant
  tree-intrinsic marginals. **Python-only**, regression-gated against the
  frozen benchmark (not cross-language: there is no R transport
  implementation to compare).

| Path | What | Ships |
|---|---|---|
| `python/` | **primary, standalone** — `metaarbor` | kernel, trees, Walk, refinement-invariant marginals, Transport (`[ot]`), interpretation layer (summaries, agreement categories, accessors), packaged plots + result bundle (`[viz]`), AnnData wrappers |
| `r/` | R companion (DEFERRED) — `MetaArbor`, `ma_*` | kernel, trees, Walk, marginals, simulation; a numeric interpretation layer exists in-tree but is unvalidated and deferred until the R package is built out later |
| `parity/` | cross-language gate | R script proving the packaged Walk reproduces the saved frozen benchmark exactly, and exporting fixtures |
| `fixtures/` | parity fixtures (regenerable, gitignored) | real Allen platform vote caches, trees, similarity, packaged-R walk outputs, simulation, MINSTD check vector |

Directory names are lowercase/`r` specifically to avoid case-insensitive
filesystem collisions (macOS merges `metaarbor/` and `MetaArbor/`).

## Parity scope — what is guaranteed where

Both implementations draw bootstrap indices from the same portable MINSTD
stream (exact in R doubles) with per-query seeds and a documented draw
order. The chain of custody:

1. `parity/01_export_and_gate.R` — the packaged R Walk must reproduce the
   saved frozen-benchmark selections/relations **exactly** (23/23 forward,
   103/103 reverse).
2. `python/tests/test_parity.py` — the python Walk must match the packaged-R
   outputs for every query on the real caches, plus a simulation
   end-to-end check (HVGs, similarity within 1e-9, identical selections).
3. Transport (python-only) is regression-gated: it must reproduce the
   frozen battery's platform result (23/23 argmax family, 21 confident, the
   same two underconfident queries, matching P-Q gap).

Serialization note: vote caches are plain R lists in RDS and plain CSV/NumPy
arrays on the python side — nothing stores a language or module class path,
so renames cannot break readability (verified by re-running all gates after
the rename).

## Node-evidence table

`metaarbor.node_evidence(...)` (python) / `ma_node_evidence(...)` (R) emit
one row per (query, visited split, child) combining: vote fraction, child
one-vs-all AUROC, the reverse-fold directional AUROC (when the reverse
cache is supplied), sibling-contrast bootstrap lower bound, parent-contrast
bound, override/margin verdicts, the decision, transport-mass share
(python, when a coupling is supplied), and the query's selected relation.
Evidence rows re-derive the walk with the same per-query seeds, so recorded
decisions are exactly the map's decisions.

## JHPCE installation

Python (primary):

```bash
module load conda            # or any python >= 3.10
conda create -n metaarbor python=3.12 -y
conda activate metaarbor
git clone https://github.com/MicTott/MetaArbor
pip install "./metaarbor/pkgs/python[all,test]"   # anndata + POT + matplotlib
python -m pytest metaarbor/pkgs/python/tests -q   # 10 pass, 4 parity skips
```

Optional ete4 rendering on the cluster: `pip install
"./metaarbor/pkgs/python[ete]"` and set `QT_QPA_PLATFORM=offscreen` for
static `render()`; `explore()` (web) needs no Qt at all.

R companion:

```bash
module load R                # >= 4.2
R CMD INSTALL r
Rscript r/tests/test_all.R
```

The full parity suite additionally needs `fixtures/`, generated on any
machine holding the benchmark results by
`Rscript parity/01_export_and_gate.R`.

## Frozen configurations (do not tune on validation data)

- Walk: alpha 0.05, n_boot 200, min_auroc 0.6, margin 0.01,
  vote_override 0.9, min_compact 0.7.
- Transport: cost 1 − S, rho 0.3, alpha 0.9 (design weight), epsilon 0,
  refinement-invariant marginals both sides; argmax-family +
  mass-confidence readouts.
