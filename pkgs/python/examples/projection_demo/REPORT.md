# MetaArbor Projection prototype — report v2 (after adversarial review)

Branch `projection-prototype`. The round-1 review found five real
problems; all are fixed, each with an adversarial test that now guards
it. The headline numbers CHANGED as the review predicted they would.

## What the review found, and what was done

1. **Max-over-leaves multiplicity bias** (a 10-leaf branch got 10
   draws at the max; the 1-vs-10 null forced 99.7% of cells into the
   leaf-rich branch). Replaced by a refinement-calibrated statistic:
   per-label-block exact-null z of the mean local rank vote,
   Bonferroni-corrected best block per child in LOG space, and a
   null-uniform RATIO margin `1 - p_best/p_second` (uniform for any
   number of children; the log scale never saturates float64 — a
   plain `(1-p)^n` score saturated to 1.0 on Allen and destroyed all
   margins). Null test now: ~balanced assignment, <=15% pass root.
2. **Transductive feature selection** (query batch changed a cell's
   result: 18/30 flipped). The projector is now FITTED: reference-only
   HVGs and preprocessing frozen in `build_projector()`; a
   composition-invariance test asserts single-cell == in-batch
   results exactly.
3. **Ordinal tie-breaking in local ranks.** Tie-average ranks
   everywhere; identical reference profiles now abstain (>=90%) and
   results are invariant to reference row order, label renaming, and
   atlas order (subsampling streams keyed by reference NAME).
4. **Could not consume reconciled trees.** `label_maps` lets reference
   labels sit at INTERNAL nodes (informing splits above, silent at and
   below — partially observed subtrees); `from_harmonize()` adapts a
   harmonize() result directly. Tests: coarse-only references never
   resolve below their labels; mixed-resolution references work;
   harmonize round-trip runs end to end.
5. **Sparse/memory.** scipy.sparse references and queries supported;
   queries stream through in blocks (`block=20000`), and per-cell
   independence makes blocking exact (tested with odd block sizes).

Renames per review: `stop_candidates` (a candidate set, not a credible
set), `stop_margin` = NaN sentinel when no split failed,
`mean_max_corr`, `max_label_vote` (null grows with label count).
`path_score` was dropped (not comparable across depths);
`path_margins` (per-split, NaN-padded) replaces it and supports
offline coverage-risk curves. "Calibration" language removed: the
Allen numbers are SELECTIVE accuracy at a threshold.

`min_margin = 0.99` was re-derived on the synthetic operating curve
(prespecified criterion: smallest threshold with family-only deep
leakage <=10% and novel-family deep leakage <=5%), before any Allen
run. On a ratio scale it reads: descend only when the winner's
corrected p is 100x smaller than the runner-up's.

## Synthetic gates (14/14)

Core five (parity, held-out accuracy, family-only abstention, novelty,
determinism+blocking) plus the review's six adversarial checks
(refinement null, ties/permutation invariance, atlas order, query
composition, internal-node labels + harmonize adapter + mixed
resolution, sparse equivalence, unequal coverage).

## Allen held-out platform (thresholds fixed beforehand)

**Test 1** — v3 cluster reference (4,864 cells after cap), all 22,067
v2 cells, truth = each cell's own subclass:

| metric | value |
|---|---|
| subclass accuracy (micro / MACRO) | 93.6% / 88.3% |
| class accuracy | 99.0% (wrong-class 0.99%) |
| coverage (resolved to subclass depth) | 93.6% |
| selective subclass accuracy (covered cells) | 96.3% |
| flat global-vote baseline | 95.4% |

Coverage-risk curve (threshold 0 -> 0.99): coverage 98.3% -> 93.6%,
risk 6.5% -> 3.8% — smooth and monotone. Sensitivity: seed variation
+-0.2%; cap 25 vs 50 trades coverage (87% vs 94%) against selective
accuracy (98.2% vs 96.1%) monotonically — the pathological cap flip
from the saturating statistic is gone.

**Honest reading vs round 1:** the uncalibrated max flattered the
prototype (95.6% forced, 99.9% selective at only 63% coverage). The
calibrated statistic gives up ~1.9 points of forced accuracy against
the flat baseline (93.6% vs 95.4%) and buys: refinement invariance,
a 30-point coverage gain at 96.3% selective accuracy, valid abstention
semantics, internal-node references, and per-split interpretability.
The flat baseline remains the right yardstick and is reported
alongside.

**Failure structure (per-subclass table committed):** the two
zero-coverage subclasses (Astro-TE, Microglia) have 100% best-leaf
accuracy — pure conservative abstention at their fine splits. The two
low-accuracy subclasses are 003 L5/6 IT TPE-ENT (0.4% — absorbed into
the adjacent IT subclasses: the deep-layer IT continuum that every
MetaArbor analysis has flagged) and rare 046 Sst Chodl (2%, absorbed
by bulk Sst). Errors are structured and known, not random.

**Test 2** — coarse v2 subclass reference -> all 13,842 v3 cells:
subclass 86.1%, class 99.9%, coverage 88.6%.

## Speed

725-1,650 cells/s CPU across caps and query sizes (22k cells in
13-24 s; build <0.5 s). Sparse input and query blocking are in; the
million-cell path now has bounded memory by construction.

## Still open

- Selective accuracy is not calibration; donor-level cross-fit
  calibration remains future work, as does a principled OOR flag
  (max_label_vote's null grows with label count).
- The amygdala atlases remain the independent confirmation the
  statistic choices deserve.
- Module remains one file; a class-based fit/transform split would be
  the packaging step if this graduates.
