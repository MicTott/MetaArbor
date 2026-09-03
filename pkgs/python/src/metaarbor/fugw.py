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

# Design weights; under the correct objective these map to POT
# reg_m = 0.3 / (1 - 0.9) = 3.0. Results published before metaarbor
# 0.2.0 used a mis-scaled parameterization and are reproducible from git
# history (tag v0.4-release-ready and earlier), not from this API.
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
          epsilon=FROZEN["epsilon"]):
    """Fused unbalanced GW via POT. Returns (pi, pq_gap).

    The design objective is the convex-weighted
        alpha * <M, pi> + (1 - alpha) * GW + rho * KL(marginals) [+ eps H].
    POT fixes the GW coefficient at 1, so the whole objective is divided
    by (1 - alpha):
        alpha_pot = alpha / (1 - alpha)
        reg_m_pot = rho / (1 - alpha)
        eps_pot   = epsilon / (1 - alpha)
    This co-scaling is the ONLY behavior. (Releases before metaarbor
    0.2.0 failed to co-scale rho/epsilon, silently weakening mass
    relaxation by (1 - alpha); analyses produced under that
    parameterization are preserved by the git history, tag
    v0.4-release-ready and earlier, not by this API.)

    alpha == 1 has no GW term: use `molecular_only()`. A zero-mass/NaN
    outcome raises MassCollapsedError (solver-trajectory collapse; FUGW
    is nonconvex, so this is not a claim about the global optimum).
    """
    import ot
    if alpha >= 1.0:
        raise ValueError("alpha=1 has no GW term: use molecular_only()")
    if alpha < 0:
        raise ValueError("alpha must be in [0, 1)")
    scale = 1.0 / (1.0 - alpha)
    alpha_pot, rho_pot, eps_pot = alpha * scale, rho * scale,         epsilon * scale
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
        if "nan" in str(e).lower() or "division" in str(e).lower():
            raise MassCollapsedError(alpha, rho_pot, 0.0) from e
        raise
    pi = (np.asarray(ps) + np.asarray(pf)) / 2
    if not np.all(np.isfinite(pi)) or pi.sum() < _MASS_FLOOR:
        raise MassCollapsedError(alpha, rho_pot, float(np.nansum(pi)))
    return pi, float(np.abs(np.asarray(ps) - np.asarray(pf)).sum())


def fugw_map(S, tree_a, tree_b, row_names, col_names, CA=None, CB=None,
             **overrides):
    """The frozen pipeline: intrinsic marginals from each tree alone,
    structure distances, solve, and per-query argmax-family + confidence
    read-outs. `S` symmetrized AUROC (rows = source populations = tree_a
    leaves, cols = target leaves = tree_b leaves).

    Structure defaults to tree HOP distances (the frozen configuration).
    Pass `CA` / `CB` (arrays over row_names / col_names) to substitute a
    molecular metric per atlas — build them with
    `branch_fit.structure_matrices(..., kind="chord"|"patristic")`, each
    atlas from its own expression only. On the Allen benchmark all three
    structure metrics scored identically (the benchmark is saturated
    under a shared taxonomy); on depth-mismatched cross-taxonomy pairs
    they are expected to differ, which is the prespecified comparison."""
    params = dict(FROZEN)
    params.update(overrides)
    if CA is None:
        CAh, la = leaf_path_dist(tree_a)
        ra = [la.index(r) for r in row_names]
        CA = CAh[np.ix_(ra, ra)]
    else:
        CA = np.asarray(CA)
    if CB is None:
        CBh, lb = leaf_path_dist(tree_b)
        cb = [lb.index(c) for c in col_names]
        CB = CBh[np.ix_(cb, cb)]
    else:
        CB = np.asarray(CB)
    wA = tree_weights(tree_a)
    wB = tree_weights(tree_b)
    pi, gap = solve(1 - np.asarray(S), CA, CB,
                    [wA[r] for r in row_names], [wB[c] for c in col_names],
                    **params)
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
