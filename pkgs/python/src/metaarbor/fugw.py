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

FROZEN = {"rho": 0.3, "alpha": 0.9, "epsilon": 0.0}


def solve(M, CA, CB, wA, wB, alpha=FROZEN["alpha"], rho=FROZEN["rho"],
          epsilon=FROZEN["epsilon"]):
    """Fused unbalanced GW via POT. Returns (pi, pq_gap)."""
    import ot
    solver = "mm" if epsilon == 0 else "sinkhorn_log"
    ps, pf, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
        np.asarray(CA) / np.max(CA), np.asarray(CB) / np.max(CB),
        wx=np.asarray(wA), wy=np.asarray(wB),
        reg_marginals=rho, epsilon=epsilon, divergence="kl",
        unbalanced_solver=solver,
        alpha=(alpha / (1 - alpha) if alpha > 0 else 0.0), M=np.asarray(M),
        max_iter=500, tol=1e-8, max_iter_ot=1000, tol_ot=1e-8, log=True)
    pi = (ps + pf) / 2
    return pi, float(np.abs(ps - pf).sum())


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
