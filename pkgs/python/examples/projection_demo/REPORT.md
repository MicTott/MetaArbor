# MetaArbor Projection prototype — consolidated report (round 4)

Branch `projection-prototype`. This file supersedes the earlier
stacked v1/v2 reports, whose superseded claims ("null-uniform",
"calibrated", "refinement-invariant") are retired; the review history
lives in the git log.

## What the method is

A hierarchical MetaNeighbor-style reference classifier with selective
abstention. A FITTED projector (reference-only features and
preprocessing; per-cell results independent of query composition)
routes each cell root-to-leaf through a reference tree. At each split,
the cell's Spearman correlations to that node's own reference cells
are re-ranked locally (tie-average); each reference label under a
child forms a block scored by a z-statistic whose mean/variance are
the EXACT finite-population moments of the cell's realized local rank
vector (tie-robust; the normal tail is the one approximation);
children carry the Bonferroni-corrected best block on the log scale;
the stop rule is a MULTIPLICITY-ADJUSTED RELATIVE-EVIDENCE MARGIN
(1 - p_best/p_second). The margin is an evidence ratio, NOT a
calibrated null probability (dependent p-values sharing one rank
partition; the empirical null passes ~2x the idealized rate).

Outputs per cell: `best_label` (always a real reference label,
possibly coarse), `best_node` (always a valid tree node — a leaf is
never invented beneath a coarse reference), `resolved_node`/depth,
`stop_margin` (NaN when fully resolved), `stop_candidates` (candidate
set, not a credible set), `path_margins` (offline coverage-risk),
`max_label_vote` (OOR evidence; null grows with label count),
`mean_max_corr`, and the global `label_vote` matrix.

Multi-atlas references: stratified per-label caps (name-keyed
subsample streams; atlas-order invariant); the formal COMBINATION RULE
is that a reference contributes to a split only when it covers EVERY
child of that split, so children are always compared on identical
reference subsets. Cross-atlas label collisions raise unless both
atlases map the label to the same tree node. Reference labels may sit
at internal nodes (coarse atlases in a reconciled hierarchy);
`from_harmonize()` adapts a harmonize() result, affiliates included.

The frozen operating point is `min_margin = 0.98`, chosen on synthetic
operating curves under three prespecified constraints (family-only
deep leakage <=10%, novel deep leakage <=5%, pure-null root pass
<=15%). The 15% null criterion is an engineering choice, not a
universal constant; it is FROZEN and will not be retuned after seeing
amygdala results — coverage-risk curves are reported instead.

## Gates (tests/test_projection.py, 19/19)

Parity with kernel.vote_cache; held-out-batch accuracy with ~0
wrong-family among resolved; family-only cells never resolve beyond
family; novel-family cells resolve shallow with separable OOR;
determinism incl. odd block sizes; refinement null (1-vs-10
exchangeable leaves: balanced, bounded); SAME-cells relabeling
(1/2/10 pseudo-leaves: parent choice stable >=95%, null bounded —
margins shift with block size, hence "multiplicity-adjusted", never
"refinement-invariant"); tie abstention + reference-order,
label-renaming, atlas-order invariance; query-composition invariance;
coarse-only references stop at their labels with valid best_node;
mixed-resolution references; harmonize adapter incl. affiliate
mapping; sparse==dense; unequal coverage; CROSSED three-child partial
coverage (split abstains for everyone; adding one full-coverage
reference reproduces its solo decisions exactly); label-collision
guard; gene_panel decouples features from the cap.

## Allen held-out platform (single reference; thresholds fixed first)

v3 cluster reference (4,864 cells) -> all 22,067 v2 cells, truth =
each cell's own subclass:

| metric | value |
|---|---|
| subclass accuracy micro / macro | 93.6% / 88.3% |
| class accuracy | 99.0% (wrong-class 0.99%) |
| coverage (resolved to subclass depth) | 94.4% |
| selective subclass accuracy | 95.8% |
| flat global-vote baseline | 95.4% |

Coverage-risk (threshold 0 -> 0.99): coverage 98.3% -> 93.6%, risk
6.5% -> 3.8%, smooth and monotone. Seeds move results +-0.2%. The cap
is an influential parameter: cap 25 gives 94.8-95.3% forced / 88-89%
coverage / 97.8-98.1% selective; cap 50 gives 93.5-93.8% / 94.4-94.6%
/ 95.6-95.8%. Reverse direction (coarse v2 reference -> v3 cells):
subclass 86.1%, class 99.9%.

Errors concentrate in biologically plausible hard cases: the
deep-layer IT continuum (003 L5/6 IT TPE-ENT at 0.4%, absorbed by
adjacent IT subclasses), rare 046 Sst Chodl absorbed by bulk Sst, and
two non-neuronal subclasses that abstain at their fine splits while
being 100% correct at best_label.

The defensible claim: multiplicity-adjusted hierarchical evidence
yields a useful SELECTIVE classifier — it trades some forced leaf
accuracy for concentrating errors among cells that stop at broader
nodes. It is not a calibrated classifier.

## Speed

725-1,690 cells/s CPU (22k cells in 13-24 s; build <0.5 s). Blocks
densify only the union of fitted panel columns, with library sizes
computed from the full gene set before subsetting.

## Status per review

- GREEN: frozen single-reference evaluation (amygdala next, via
  examples/project_run.py; thresholds frozen).
- AMBER: combined multi-atlas projection — the collision guard and the
  full-coverage combination rule are now in with adversarial tests,
  but independent (amygdala) validation comes before trusting it.
- Not merged to the stable API; branch-only.

## Open items

Donor-level cross-fit calibration; principled OOR flag; sparse HVG
computation or supplied panels for whole-transcriptome-scale fitting
(gene_panel exists); class-based fit/transform refactor if this
graduates; MetaArbor projection vs MapMyCells on the same reconciled
reference as the decisive comparison.
