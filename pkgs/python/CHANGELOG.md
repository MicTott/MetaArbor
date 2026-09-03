# Changelog

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
