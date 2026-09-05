# Changelog

## 0.7.0 (2026-09-04) — review round 2: placement honesty + contracts

- Ambiguous parents (two incomparable accepted ancestors) are no longer
  forced under an arbitrary parent: the candidate attaches at the
  deepest COMMON accepted ancestor (polytomy) or root, alongside the
  ambiguous_parent conflict.
- Eligibility is re-evaluated at adjudication time, so claims accepted
  earlier in the same processing cohort are visible (no stale shared
  state within a round).
- Gene-name contract: names validated for uniqueness/length; both
  missing -> neutral positional names under an explicit identical-order
  warning (path previously raised TypeError); MIN_SHARED_GENES=100
  documented as the alignment floor.
- Stability contract: harmonize() now requires either a COMPLETE
  stability map over internal canonical nodes or an explicit
  trust_trees=True; silent 1.0 defaults are gone.
- support semantics: every node carries support_type — cross_atlas
  (backbone/private tuples), input_topology (single-atlas placements
  and expansions; support=None — existence is certain, cross-atlas
  support is NOT implied), unplaced (routed rejections and tripwire
  repairs).
- Audit terminology (examples/audit_unanchored.py): MetaNeighbor-style
  categories (one_way_match / conflicting_matches / no_supported_match
  / insufficient_power / atlas_specific) with an evidence column;
  compactness-gated walks are no_supported_match with
  distributed_evidence — NOT conflicting biology.
- Consensus Walk parity wording: node walks apply the frozen Walk's
  selection and compactness gates; baseline_map's root-stop
  discordant-vs-unmatched diagnostic distinction is not reproduced.

## 0.6.0 (2026-09-04) — structural review fixes (K>=3 synthesis)

- Ancestry cycles detected (SCC) -> ancestry_cycle genuine-conflict +
  rejection of all members; ambiguous_parent conflict for incomparable
  accepted parents (silent min-index linearization removed).
- Consensus consumes the complete frozen Walk decision: the 0.70
  compactness gate is applied to node walks; gated selections are
  discordant and never seed reciprocal edges (landing preserved in
  gated_selected).
- Stability propagated (infer_tree support) into STABILITY_FLOOR
  screening.
- Affiliates: all landings collected, pair_relation honored,
  incompatible landings -> conflict, consolidated subtrees never
  affiliated.
- Gene columns aligned by name (intersect/reorder both matrices).
- Free-structure-below evaluated against ACCEPTED claims only.
- Entry validation: dataset labels == tree leaves. Rejection fallback
  topologically ordered.
- Allen truth case under complete semantics: 71 exact / 27
  consistent-coarse / 4 wrong / 1 root / 0 missing (98/103
  truth-consistent).

## 0.5.0 (2026-09-04) — rejection routes the claim, never its labels

Root cause of missing labels: greedy_backbone's rejected list was
terminal for member labels. route_rejected() surfaces unrepresented
members of rejected claims as unplaced_single_atlas nodes with the
rejection reason as provenance; repair_completeness() remains as a
permanent tripwire that must report zero.

## 0.4.1 (2026-09-04) — completeness as a core invariant

Every input label must appear in the assembly (member / marked
affiliate / private / explicitly unplaced). repair_completeness()
reinstates anything upstream loses, flagged assembly_repair. Offline
audit tooling (examples/audit_unanchored.py) and audit-input
persistence in the K-atlas runner.

## 0.4.0 (2026-09-04) — harmonize() public API (beta)

Consensus harmonization exported at top level: harmonize() +
plot_reconciled_tree(). Reconciled-hierarchy synthesis over frozen
pairwise Walk evidence; K-atlas runner (examples/harmonize_k3.py).

## 0.3.0 (2026-09-03) — patristic structure metric becomes the Transport
default

On cross-taxonomy amygdala pairs, patristic and chord metrics score
near-identically and both edge out hop; Allen (shared taxonomy) cannot
separate them. `fugw_map` now builds patristic structure per atlas when
expression is supplied (`expr_a`/`expr_b`); explicit CA/CB still win;
without expression it falls back to hop with a warning
(`structure="hop"` for silent hop).

## 0.2.1 (2026-09-03) — molecular structure metrics as a first-class
Transport option

- `fugw_map(..., CA=, CB=)` accepts per-atlas structure matrices
  (default remains tree hop distance — the frozen configuration).
- `structure_matrices(counts, labels, tree, leaves, kind="chord"|
  "patristic")` builds them from each atlas's own expression (chord
  pseudobulk distances, or their NNLS patristic projection onto the
  supplied topology); `patristic_matrix` promoted from the experiment
  script into the package.
- Allen evidence to date: all structure metrics score identically under
  a shared taxonomy (saturated benchmark); the discriminating comparison
  on depth-mismatched cross-taxonomy pairs is prespecified in NOTES.

## 0.2.0 (2026-09-03) — FUGW parameterization fix

The design objective `alpha*M + (1-alpha)*GW + rho*R (+ eps*H)` maps to
POT's GW-coefficient-1 form by dividing the WHOLE objective by
`(1-alpha)`; earlier releases co-scaled only `alpha`, silently weakening
mass relaxation by `(1-alpha)` (10x at the frozen alpha=0.9). Diagnosed
via a zero-mass solver-trajectory collapse on the Yu-Allen amygdala pair
(FUGW is nonconvex; no claim about the global optimum).

- `fugw.solve` now implements the mathematically correct co-scaling as
  its ONLY behavior. Analyses produced under the previous
  parameterization are preserved by git history (tag `v0.4-release-ready`
  and earlier), not by an API option.
- `alpha=1` is rejected; the explicit `molecular_only()` mode replaces
  that endpoint.
- Zero-mass/NaN outcomes raise `MassCollapsedError` (interpretable
  diagnostic) instead of propagating NaN couplings.
- Allen three-condition revalidation at the frozen design weights:
  argmax family 23/23 in every condition; confidence 21 -> 20 with the
  delta confined to the deep-layer IT continuum; zero cross-family.

## 0.1.0 — initial release (frozen Walk + Transport, interpretation
layer, tree inference, publication figures).
