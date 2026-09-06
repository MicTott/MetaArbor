# MetaArbor Projection prototype — build & test report

Branch `projection-prototype` (b2e5425), isolated from main. Nothing
frozen was modified; the projector consumes the frozen measurement
layer and its vote computation is parity-tested against
`kernel.vote_cache`.

## What was built

`metaarbor/projection.py` (branch-only, not exported at top level):

- `build_projector(refs, tree, cap_per_leaf=50)` — one or more labeled
  reference atlases, stratified-subsampled per leaf (atlas-balanced;
  coverage-aware splits: an atlas informs a split only when it carries
  cells for every child).
- `project(projector, counts, gene_names, ...)` — routes every query
  cell root-to-leaf. At each split the cell's correlations to the
  node's own reference cells are re-ranked among themselves (the
  per-cell analogue of the Walk's parent-context principle) and each
  child scores the max over its leaves' mean local votes. Stop rule =
  local vote margin (top minus runner-up) >= `min_margin` (0.10).
- Per-cell outputs: `best_leaf` (always — forced descent),
  `path_score`, `resolved_node`/`resolved_depth` (confident descent),
  `stop_margin`, `credible` children at the stop, `top_leaf_mean`
  (out-of-reference score; 0.5 = rank-random), `max_corr`, and the
  global leaf-vote matrix for flat-baseline comparison.

## Two design findings (both material)

1. **Global ranks cannot drive sibling contrasts.** Every leaf under
   the correct family scores high globally, compressing local
   confidence to uselessness (nothing resolved past family). Local
   re-ranking within each node's reference cells restored the
   contrasts — the same principle as the Walk's parent-context
   compactness, rediscovered per cell.
2. **Mean-union navigation is size-biased.** Scoring a child by the
   mean local vote over all its cells cost ~10 points of Allen
   subclass accuracy (85.9% vs the flat baseline's 95.6%): a
   heterogeneous class dilutes its own most distinctive subclass.
   Scoring each child by the max over its leaves' mean local votes
   recovered the flat baseline exactly — echoing why the Walk uses
   vote-guided navigation rather than union scores.

## Accuracy — synthetic gates (tests/test_projection.py, 5/5 pass)

| Gate | Result |
|---|---|
| Vote parity with kernel.vote_cache | exact |
| Held-out batch, true subtypes | leaf >= 90%, family >= 98%, wrong-family among resolved <= 1% |
| Family-only cells (no subtype signal) | >= 60% stop exactly at family; <= 25% resolve deep (measured: 85% / 7%) |
| Novel family | <= 10% resolve deep (measured 2%; 73% held at root); OOR AUROC >= 0.8 |
| Determinism | bit-identical reruns |

`min_margin=0.10` was chosen from the synthetic operating curve
(sweep 0.05-0.20), not tuned on Allen: at 0.10, 76% of true-subtype
cells resolve fully while family-only leakage is 7% and novel leakage
2%.

## Accuracy — Allen held-out platform (real batch axis)

**Test 1 — fine reference, coarse-labeled query.** Reference = 10Xv3
cells with cluster labels under the curated 4-level tree; query = all
22,067 10Xv2 cells; truth = each v2 cell's own subclass (scoring only).

| Metric | Hierarchical | Flat vote baseline |
|---|---|---|
| Subclass accuracy (best_leaf -> subclass) | **95.6%** | 95.6% |
| Class accuracy | **99.85%** | 99.85% |
| Wrong-class rate | 0.15% | 0.15% |
| Resolved to subclass depth or deeper | 63.3% | n/a |
| Subclass accuracy among resolved cells | **99.9%** | n/a |

The headline: the hierarchy costs nothing on forced accuracy and its
abstention is close to perfectly calibrated in effect — cells that
resolve confidently are right 99.9% of the time; essentially all error
lives in the 37% that abstain to a broader node (where the honest
answer *is* the broader node).

**Test 2 — coarse reference, fine-labeled query.** Reference = 10Xv2
subclasses (2-level tree); query = all 13,842 10Xv3 cells: subclass
80.3%, class 99.8%, wrong-class 0.19%. Lower subclass accuracy is
expected — a 2-level, 23-leaf reference gives the router far less
structure — and the class-level near-perfection shows failures stay
within the right family.

## Speed (CPU, single process, Apple silicon)

| ref cap/leaf | ref cells | query cells | seconds | cells/s |
|---|---|---|---|---|
| 25 | 2,575 | 2,000 | 1.6 | 1,283 |
| 25 | 2,575 | 22,067 | 13.5 | 1,630 |
| 50 | 4,864 | 5,000 | 5.9 | 840 |
| 50 | 4,864 | 22,067 | 23.6 | 936 |

Build time is negligible (0.1 s). Cost scales ~linearly in query and
reference size; a 1M-cell query against the capped v3 reference
extrapolates to ~20-30 min single-threaded CPU. The feared
N_query x N_reference blowup never materializes because the reference
is stratified-capped and deep splits touch shrinking subtrees.

## Honest limitations

- `path_score` and `stop_margin` are evidence scores, NOT calibrated
  probabilities; calibration (donor-level cross-fitting) is future
  work, as is the credible-leaf-set formalization.
- OOR detection relies on `top_leaf_mean` (rank votes); `max_corr`
  does not separate novel cells (rank normalization discards absolute
  similarity). A margin-plus-OOR combined flag is future work.
- Single-reference evaluation on Allen; the multi-atlas averaging path
  is implemented and coverage-aware but only exercised by the
  synthetic two-reference gates.
- min_margin fixed at 0.10 from synthetics; per-node thresholds or
  calibration may beat one global constant.
- The mean-vs-max navigation comparison was run on the same Allen data
  used for reporting (the choice is principled and pre-echoed by the
  Walk's design, but an independent confirmation dataset would be
  cleaner — the amygdala atlases are the natural candidate).

## Relation to MetaArbor

The projector is the per-cell counterpart of the frozen Walk: votes
navigate, local contrasts decide, abstention is a first-class outcome.
It can consume any reference tree — curated Allen, Bonsai, or a
MetaArbor-reconciled hierarchy — which keeps its standalone identity
while making the reconciled tree its flagship reference. It is also
the natural home for the asymmetric fine-to-coarse question from the
amygdala audit: project a label's cells into the reconciled tree
rather than weakening the consensus rules.
