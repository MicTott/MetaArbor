"""MetaArbor-Transport: the frozen FUGW estimator with refinement-invariant marginals (NOTES.md
items 14-15; frozen 2026-09-02).

FROZEN configuration: cost = 1 - S (symmetrized MetaNeighbor AUROC),
rho = 0.3, alpha = 0.9 (design weight: cost = a*M + (1-a)*GW, mapped to
POT's linear-term coefficient a/(1-a)), epsilon = 0 (mm solver),
tree-intrinsic recursive marginals on both sides. Readouts: argmax family
and mass-based confidence categories. Requires the optional `pot`
dependency (pip install metaarbor[ot]).
"""
from __future__ import annotations

import numpy as np

from .tree import leaf_path_dist, leaves_under, tree_weights

# The frozen v1 battery/benchmark results were produced under the
# "pot-v1" convention (rho NOT co-scaled): reproduce them by passing
# convention="pot-v1". Under the corrected "design-v2" contract the same
# design weights map to POT reg_m = 0.3 / (1 - 0.9) = 3.0.
FROZEN = {"rho": 0.3, "alpha": 0.9, "epsilon": 0.0}


class MassCollapsedError(RuntimeError):
    """The solver trajectory collapsed toward zero transported mass
    (dominant molecular term + weak mass relaxation). NOTE: FUGW is
    nonconvex — this reports the trajectory's collapse, not a proof that
    the global optimum carries zero mass. Carries `alpha`, `rho_effective`
    and the last finite mass when available."""

    def __init__(self, alpha, rho_effective, mass):
        self.alpha, self.rho_effective, self.mass = alpha, rho_effective, mass
        super().__init__(
            f"FUGW mass collapse (alpha={alpha}, effective POT reg_m="
            f"{rho_effective}): transported mass -> {mass}; the "
            "(alpha, rho) conjunction sits past the feasibility boundary "
            "for this geometry. Increase rho, lower alpha, or use "
            "molecular_only().")


_MASS_FLOOR = 1e-9


def molecular_only(M, wA, wB, rho):
    """Explicit molecular-only unbalanced OT (no GW term) — replaces the
    ill-defined alpha=1 endpoint of solve()."""
    import ot
    pi = ot.unbalanced.mm_unbalanced(np.asarray(wA), np.asarray(wB),
                                     np.asarray(M), reg_m=rho, div="kl")
    pi = np.asarray(pi)
    if not np.all(np.isfinite(pi)) or pi.sum() < _MASS_FLOOR:
        raise MassCollapsedError(1.0, rho, float(np.nansum(pi)))
    return pi, 0.0


def solve(M, CA, CB, wA, wB, alpha=FROZEN["alpha"], rho=FROZEN["rho"],
          epsilon=FROZEN["epsilon"], convention="design-v2"):
    """Fused unbalanced GW via POT. Returns (pi, pq_gap).

    Parameter contract (v2, versioned correction — see CHANGELOG):
    the design objective is the convex-weighted
        alpha * <M, pi> + (1 - alpha) * GW + rho * KL(marginals) [+ eps H].
    POT parameterizes with the GW coefficient fixed at 1, so the WHOLE
    objective is divided by (1 - alpha):
        alpha_pot = alpha / (1 - alpha)
        reg_m_pot = rho / (1 - alpha)          <- v1 failed to co-scale
        eps_pot   = epsilon / (1 - alpha)      <- likewise
    The v1 wrapper scaled only alpha, silently weakening mass relaxation
    by (1 - alpha) — at alpha 0.9, mass destruction was 10x cheaper than
    the design objective intends (diagnosed via the Yu-Allen zero-mass
    collapse; see NOTES). `convention="pot-v1"` reproduces the released
    v1 behavior exactly (rho/epsilon passed through unscaled), preserving
    every frozen result.

    alpha == 1 is no longer reachable here: use `molecular_only()`.
    A zero-mass/NaN outcome raises MassCollapsedError instead of
    propagating NaN couplings.
    """
    import ot
    if alpha >= 1.0:
        raise ValueError("alpha=1 has no GW term: use molecular_only()")
    if alpha < 0:
        raise ValueError("alpha must be in [0, 1)")
    if convention == "design-v2":
        scale = 1.0 / (1.0 - alpha)
        alpha_pot, rho_pot, eps_pot = alpha * scale, rho * scale,             epsilon * scale
    elif convention == "pot-v1":
        alpha_pot = alpha / (1.0 - alpha) if alpha > 0 else 0.0
        rho_pot, eps_pot = rho, epsilon
    else:
        raise ValueError(f"unknown convention {convention!r}")
    solver = "mm" if eps_pot == 0 else "sinkhorn_log"
    try:
        ps, pf, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
            np.asarray(CA) / np.max(CA), np.asarray(CB) / np.max(CB),
            wx=np.asarray(wA), wy=np.asarray(wB),
            reg_marginals=rho_pot, epsilon=eps_pot, divergence="kl",
            unbalanced_solver=solver, alpha=alpha_pot, M=np.asarray(M),
            max_iter=500, tol=1e-8, max_iter_ot=1000, tol_ot=1e-8,
            log=True)
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        # POT surfaces the zero-mass trajectory differently across
        # versions (NaN-in-coupling ValueError, or division inside the
        # renormalisation); all mean the same interpretable thing
        if "nan" in str(e).lower() or "division" in str(e).lower():
            raise MassCollapsedError(alpha, rho_pot, 0.0) from e
        raise
    pi = (np.asarray(ps) + np.asarray(pf)) / 2
    if not np.all(np.isfinite(pi)) or pi.sum() < _MASS_FLOOR:
        raise MassCollapsedError(alpha, rho_pot, float(np.nansum(pi)))
    return pi, float(np.abs(np.asarray(ps) - np.asarray(pf)).sum())


def fugw_map(S, tree_a, tree_b, row_names, col_names, **overrides):
    """The frozen pipeline: intrinsic marginals from each tree alone, path
    distances as structure, solve, and per-query argmax-family + confidence
    read-outs. `S` symmetrized AUROC (rows = source populations = tree_a
    leaves, cols = target leaves = tree_b leaves)."""
    params = dict(FROZEN)
    params.update(overrides)
    CA, la = leaf_path_dist(tree_a)
    CB, lb = leaf_path_dist(tree_b)
    ra = [la.index(r) for r in row_names]
    cb = [lb.index(c) for c in col_names]
    CA = CA[np.ix_(ra, ra)]
    CB = CB[np.ix_(cb, cb)]
    wA = tree_weights(tree_a)
    wB = tree_weights(tree_b)
    pi, gap = solve(1 - np.asarray(S), CA, CB,
                    [wA[r] for r in row_names], [wB[c] for c in col_names],
                    **params)  # convention passes through via overrides
    return {"pi": pi, "pq_gap": gap, "rows": list(row_names),
            "cols": list(col_names), "params": params}


def decompose(pi, row_names, col_names, family_of_leaf, family_leaves):
    """Per-query family-mass decomposition (NOTES.md items 12-13 read-outs).
    `family_of_leaf`: dict target leaf -> its family; `family_leaves`:
    dict query -> list of its true target leaves."""
    fams = sorted(set(family_of_leaf.values()))
    fam_idx = {f: [j for j, c in enumerate(col_names)
                   if family_of_leaf[c] == f] for f in fams}
    out = []
    for i, q in enumerate(row_names):
        row = pi[i]
        tot = row.sum()
        if tot <= 0:
            out.append({"query": q, "argmax_family": None, "true_mass": 0.0,
                        "category": "cross_family_failure"})
            continue
        p = row / tot
        fam_mass = {f: p[fam_idx[f]].sum() for f in fams}
        best = max(fam_mass, key=fam_mass.get)
        inb = np.isin(col_names, family_leaves[q])
        tm = float(p[inb].sum())
        cat = ("cross_family_failure" if best != q else
               "confident_correct" if tm >= 0.9 else
               "underconfident_correct" if tm >= 0.5 else "diffuse_correct")
        out.append({"query": q, "argmax_family": best, "true_mass": tm,
                    "category": cat,
                    "H_family": float(-sum(v * np.log(v)
                                           for v in fam_mass.values() if v > 0)),
                    "eff_leaves": float(np.exp(-np.sum(
                        p[p > 0] * np.log(p[p > 0]))))})
    return out
