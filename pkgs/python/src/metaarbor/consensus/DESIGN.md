# MetaArbor-Consensus — design

**Status: merged to main; exported as `metaarbor.harmonize()` and
labeled BETA/EXPERIMENTAL.** Isolation rules retained: consumes the
frozen pairwise MetaArbor API only (Walk, Transport, kernel untouched);
consensus rules were frozen on simulation + Allen gates before any real
multi-atlas problem. The structural-review fixes (0.6.0/0.7.0: ancestry
cycles, complete Walk semantics incl. compactness gate, live stability,
multi-landing affiliates, dynamic eligibility, completeness invariant
with rejection routing) are documented in CHANGELOG.md.

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
4. **Candidate meta-clades** (corrected after review):
   a. RECIPROCITY: canonicalize unary-equivalent nodes WITHIN each tree
      (chain collapse), then a reciprocal match is mutual selection —
      A_i -> B_j and B_j -> A_i — between canonical nodes. Cross-atlas
      leaf-set equality is impossible (disjoint cells and labels) and is
      neither required nor referenced.
   b. build the candidate graph from reciprocal, bootstrap-supported
      edges; MISSING datasets are permitted — a meta-clade needs
      agreement only among ELIGIBLE OBSERVED datasets, plus ancestry
      compatibility. Every stable unmatched node enters as a SINGLETON
      candidate (the route to private branches).
   c. seed invariance is a VALIDATION DIAGNOSTIC reported per candidate,
      not an absolute all-pairs requirement (one underpowered dataset
      must not fragment a real meta-clade).
   d. hand candidates to hierarchical greedy selection (step 6).
5. **Ancestry-poset compatibility**: a meta-clade set is compatible iff
   the induced partial order is tree-like — for every pair (M1, M2) the
   relation (ancestor / descendant / disjoint) agrees across every source
   tree where both have members, and no pair interleaves. Violations are
   emitted to the conflict graph.
6. **Greedy backbone — hierarchical, not globally sorted** (corrected):
   global support sorting is pathological (1/1 eligible outranks 5/6) and
   cannot guarantee a parent exists before its children. Process
   ANCESTORS BEFORE DESCENDANTS, ranking candidates WITHIN each parent
   context. Candidate classes are separated up front:
     - BACKBONE candidate: supported by >= 2 datasets with sufficient
       eligible-donor support;
     - PRIVATE candidate: stable within one dataset AND no correspondence
       elsewhere AND high predicted detection power in the adequately
       powered others (all three required);
     - UNKNOWN elsewhere: other datasets lacked detection power — never
       counted against, never called private.
   Accept in order iff ancestry-compatible; where support fails, stop
   resolving — **polytomies, never forced binary splits**.
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
   held out as truth) — scored by clade PRECISION/RECALL, quartet
   agreement and topology distance, NOT exact tree equality (honest
   polytomies may legitimately beat the fully resolved truth). Donor
   leaves come from INDEPENDENT per-donor clustering — reusing identical
   curated cluster labels in every donor makes correspondence
   artificially easy, even after name permutation. Baselines: pooled
   Bonsai, majority-rule consensus, ECLAIR, scTree, treeArches/scHPL,
   CellHint.

## Module map (this package)

- `simulate.py` — multi-donor generator for scenarios 1-4 (implemented).
- `eligibility.py` — v1 floor; v2 Beta-binomial LOO posterior model
  with closed-form P(detect) and lower credible bounds (implemented).
- `poset.py` — ancestry-relation extraction + compatibility checker
  (implemented; conflicts emitted, not raised).
- `candidates.py` — pairwise decisions → seed-invariant candidate groups
  (implemented; consumes frozen `metaarbor` pairwise API and applies
  the COMPLETE frozen Walk decision including the compactness gate).
- `backbone.py` — greedy ancestry-compatible selection + polytomy
  emission + provenance/naming table (implemented; ancestry-cycle
  detection, ambiguous parents placed at the common accepted ancestor,
  eligibility evaluated against accepted claims at adjudication time).

## Gate results and rule-correction log (2026-09-02)

All four simulation gates PASS end-to-end (simulate -> pairwise ->
candidates -> hierarchical greedy backbone; examples/consensus_gates.py;
figures in examples/consensus_demo/):

- batch: 16/16 truth nodes (4 families 3/3 + 12 leaves), 0 conflicts.
- missing_unique: F4 backbone at (2,3) with d1 = powered absence; P1
  private.
- resolution_imbalance: families 4/4 at 3/3; flat d0 = 16
  unresolved_in_dataset rows; 8 weak twins honestly unknown, ZERO
  spurious privates; support fully abundance-invariant.
- rare_private (FLAGSHIP): F1.rare = private, parent = F1's meta-clade;
  d0 powered absence at 1.000; d1 honestly unknown at power 0.854; F1
  unfragmented at 3/3.

The unresolved_in_dataset rule was corrected TWICE by the gates (frozen
thresholds untouched; both corrections structural):
1. Original parent-containment form made private detection impossible in
   principle (any subtype's walk lands in its family) — caught by the
   flagship.
2. Landing-based revision let weak reciprocity-failure twins be
   mis-called private — caught by resolution_imbalance.
Final form: unresolved iff (a) parent member terminal in the dataset's
canonical tree, or (b) FREE (unclaimed) canonical structure exists below
the parent there. Landings remain asymmetric evidence only.

RESIDUAL RESOLVED (affiliate rule, reviewed and implemented): a
singleton whose one-way selection lands on an accepted meta-clade
lacking a member in the singleton's dataset attaches as a visibly
marked asymmetric AFFILIATE alias (never counted as reciprocal
support). Multi-atlas hardening (structural review): landings are
collected across ALL other atlases, pair_relation disagreement
disqualifies a landing, incompatible landings emit an
affiliate_incompatible_landings conflict and no attachment, nested
landings attach at the coarsest target, and consolidated subtrees are
never affiliated (an alias cannot carry topology).
