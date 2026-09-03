# Changelog

## 0.2.0 (2026-09-03) — FUGW parameter-contract correction (versioned)

- `fugw.solve` gains an explicit `convention` parameter:
  - `"design-v2"` (new default): the design objective
    `alpha*M + (1-alpha)*GW + rho*R (+ eps*H)` is mapped to POT with the
    WHOLE objective divided by `(1-alpha)`: `alpha/(1-alpha)`,
    `rho/(1-alpha)`, `epsilon/(1-alpha)`.
  - `"pot-v1"`: the released 0.1.0 behavior (only alpha scaled),
    preserved verbatim so every frozen result reproduces.
  Rationale: v1 silently weakened mass relaxation by `(1-alpha)`; at the
  frozen `alpha=0.9` mass destruction was 10x cheaper than the design
  objective intends. Diagnosed via zero-mass collapse on the Yu-Allen
  amygdala pair (solver-trajectory collapse; FUGW is nonconvex, so this
  is not a proof about the global optimum).
- `alpha=1` is no longer accepted by `solve`; the explicit
  `molecular_only()` mode replaces that endpoint.
- Zero-mass/NaN outcomes now raise `MassCollapsedError` (an interpretable
  "mass collapsed" diagnostic with alpha, effective reg_m and last mass)
  instead of propagating NaN couplings.
- The released frozen amygdala result is unchanged: the v1 frozen
  implementation undergoes zero-mass collapse on Yu-Allen, reported as
  "estimator failed to converge" in both directions.

## 0.1.0 — initial release (frozen Walk + Transport, interpretation
layer, tree inference, publication figures).
