# MetaArbor-Consensus — design (experimental branch)

**Status: prototype on `consensus-prototype`. Isolation rules:** consumes
the frozen pairwise MetaArbor API only (Walk, Transport, kernel untouched);
no amygdala data or results inform any rule here; consensus rules are
frozen on simulation + Allen pseudo-donor trees BEFORE touching any real
multi-atlas problem. Not merged into the released package until then.

## Problem

Given K ≥ 3 independently constructed atlas/donor trees, produce:

1. **Consensus backbone tree** — meta-clades with stable canonical IDs.
2. **Private branches** — donor-specific populations, retained as such.
3. **Conflict graph** — reproducible but mutually incompatible placements,
   reported, never silently resolved.

## Objects and naming

- **Meta-clade**: a set of nodes, at most one per source tree (members may
  sit at different annotation depths — resolution difference is expected).
- **Canonical identity** is a stable ID (`MA-C0042`). Modal or
  marker-anchored names are DISPLAY names; every published label from
  every source atlas is retained as a synonym in a provenance table
  (original label → meta-clade ID → display name → support). Nomenclature
  changes never change the computational object.

## Pipeline (v1, deliberately greedy and interpretable)

1. **Per-donor trees** — pluggable builder (curated taxonomy, Bonsai,
   hierarchical clustering). Building trees from the MetaArbor kernel
   itself creates a build/align dependence: allowed only if flagged and
   compared against an independent builder.
2. **Rooting is an explicit prerequisite.** Ancestry is meaningless under
   arbitrary roots (Bonsai trees especially). v1 requires rooted inputs
   under a stated convention (curated trees: the given root; built trees:
   a declared rooting rule applied before alignment). Operating on
   unrooted splits/quartets is a documented later extension, not v1.
3. **Pairwise evidence** — frozen MetaArbor alignments for all K(K-1)/2
   pairs. **Raw AUROCs are never averaged across pairs** (AUROC depends on
   the negative set and resolution): support is counted from calibrated
   per-pair DECISIONS (reciprocal matches with bootstrap support), not
   from pooled scores.
4. **Candidate meta-clades**:
   a. seed with reciprocal, bootstrap-supported pairwise node matches;
   b. expand each seed across the remaining datasets via the pairwise
      decisions;
   c. retain a group only if its membership is INVARIANT when re-seeded
      from each constituent node (cycle consistency builds candidates);
   d. hand surviving candidates to greedy selection (ancestry consistency
      decides which can coexist).
5. **Ancestry-poset compatibility**: a meta-clade set is compatible iff
   the induced partial order is tree-like — for every pair (M1, M2) the
   relation (ancestor / descendant / disjoint) agrees across every source
   tree where both have members, and no pair interleaves. Violations are
   emitted to the conflict graph.
6. **Greedy backbone**: rank candidates by support (below), accept in
   order iff ancestry-compatible with everything accepted; where support
   fails, stop resolving — **polytomies, never forced binary splits**.
7. **Support denominator = ELIGIBLE donors**, with an explicit detection
   model. v1: donor d is eligible for M iff d has ≥ floor cells in M's
   parent context. v1.5 (probabilistic): with expected prevalence p_M and
   n = d's parent-context cells,
       P(detect) = 1 - (1 - p_M)^n ;
   then adequately powered + absent → evidence for private/absent;
   inadequately powered → UNKNOWN (never counted against support);
   present + aligned → supports. Sampling must not buy or forfeit
   consensus support.
8. **Private branches**: unmatched-with-power populations retained as
   private, attached at their best-supported ancestor.

## Prespecified predictions (written before any run)

- Conflicts and polytomies will concentrate on continuum regions (Allen:
  the deep-layer IT gradient). A consensus that instead churns in
  well-separated territory (GABAergic, non-neuronal) is buggy.
- The flagship failure mode to beat: a RARE PRIVATE branch under extreme
  donor imbalance — the eligibility model either proves itself there or
  the design fails.

## Validation battery (gates, in order)

1. Simulation: same latent tree + batch distortion → backbone == truth.
2. Simulation: missing + genuinely unique branches → private/absent vs
   unknown calls match the generative truth.
3. Simulation: unequal resolution + severe abundance imbalance →
   support invariant to cell counts.
4. **Flagship**: rare private branch (low p_M) in one small donor, absent
   elsewhere, extreme imbalance → called private (not unknown, not
   forced-matched), and NOT penalized in donors lacking power.
5. Leave-one-donor-out stability (consensus analog of the batch battery).
6. Name-permutation invariance (unit test).
7. Allen pseudo-donors (donors × platforms; curated taxonomy strictly
   held out as truth). Baselines: pooled Bonsai, majority-rule consensus,
   ECLAIR, scTree, treeArches/scHPL, CellHint.

## Module map (this package)

- `simulate.py` — multi-donor generator for scenarios 1-4 (implemented).
- `eligibility.py` — v1 floor + v1.5 probabilistic model (implemented).
- `poset.py` — ancestry-relation extraction + compatibility checker
  (implemented; conflicts emitted, not raised).
- `candidates.py` — pairwise decisions → seed-invariant candidate groups
  (contracts stubbed; consumes frozen `metaarbor` pairwise API).
- `backbone.py` — greedy ancestry-compatible selection + polytomy
  emission + provenance/naming table (contracts stubbed).
