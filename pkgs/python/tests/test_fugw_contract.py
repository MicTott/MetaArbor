"""FUGW contract tests (single, mathematically correct parameterization):
the co-scaling algebra pinned against a direct POT call, the explicit
molecular-only mode, finiteness of returned couplings, the zero-mass
guard."""
import numpy as np
import pytest

pytest.importorskip("ot")

from metaarbor.fugw import (FROZEN, MassCollapsedError, molecular_only,
                            solve)

def toy_problem(seed=0, na=4, nb=6):
    rs = np.random.RandomState(seed)
    M = rs.rand(na, nb)
    CA = np.abs(rs.rand(na, na)); CA = (CA + CA.T) / 2; np.fill_diagonal(CA, 0)
    CB = np.abs(rs.rand(nb, nb)); CB = (CB + CB.T) / 2; np.fill_diagonal(CB, 0)
    wA = np.full(na, 1 / na)
    wB = np.full(nb, 1 / nb)
    return M, CA, CB, wA, wB


def test_coscaling_algebra_pinned_against_pot():
    """solve(alpha, rho) must equal a direct POT call with ALL of
    alpha/(1-a), rho/(1-a) — the algebra that v1 got wrong."""
    import ot
    M, CA, CB, wA, wB = toy_problem()
    alpha, rho = 0.7, 0.6
    pi, _ = solve(M, CA, CB, wA, wB, alpha=alpha, rho=rho, epsilon=0.0)
    ps, pf, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
        CA / CA.max(), CB / CB.max(), wx=wA, wy=wB,
        reg_marginals=rho / (1 - alpha), epsilon=0.0, divergence="kl",
        unbalanced_solver="mm", alpha=alpha / (1 - alpha), M=M,
        max_iter=500, tol=1e-8, max_iter_ot=1000, tol_ot=1e-8, log=True)
    assert np.allclose(pi, (ps + pf) / 2, atol=1e-10)


def test_returns_finite_coupling_at_frozen_weights():
    M, CA, CB, wA, wB = toy_problem(1)
    pi, gap = solve(M, CA, CB, wA, wB, alpha=FROZEN["alpha"],
                    rho=FROZEN["rho"])
    assert np.all(np.isfinite(pi)) and pi.sum() > 0 and np.isfinite(gap)


def test_molecular_only_mode():
    import ot
    M, _, _, wA, wB = toy_problem(2)
    pi, gap = molecular_only(M, wA, wB, rho=0.5)
    ref = ot.unbalanced.mm_unbalanced(wA, wB, M, reg_m=0.5, div="kl")
    assert np.allclose(pi, ref) and gap == 0.0
    with pytest.raises(ValueError):
        solve(M, np.zeros((4, 4)), np.zeros((6, 6)), wA, wB, alpha=1.0)


def test_zero_mass_guard_raises_interpretable_error():
    # dominant uniform molecular cost + pathologically weak relaxation:
    # the trajectory drives mass to zero; the guard must convert POT's
    # failure into an interpretable diagnostic
    rs = np.random.RandomState(3)
    na, nb = 5, 7
    M = 3.0 + rs.rand(na, nb)
    CA = np.zeros((na, na)) + 1e-9
    CB = np.zeros((nb, nb)) + 1e-9
    wA = np.full(na, 1 / na)
    wB = np.full(nb, 1 / nb)
    with pytest.raises(MassCollapsedError) as ei:
        solve(M, CA, CB, wA, wB, alpha=0.9, rho=1e-5)
    assert "mass" in str(ei.value).lower()
    assert ei.value.alpha == 0.9


def test_structure_override_and_helpers():
    """CA/CB overrides flow through fugw_map; structure_matrices builds
    chord and patristic metrics from one atlas's own expression; hop
    default is unchanged when overrides are absent."""
    from metaarbor import tree_from_levels
    from metaarbor.branch_fit import structure_matrices
    from metaarbor.fugw import fugw_map

    rs = np.random.RandomState(7)
    rows_a = [("A", "a1"), ("A", "a2"), ("B", "b1")]
    rows_b = [("X", "x1"), ("X", "x2"), ("Y", "y1"), ("Y", "y2")]
    ta = tree_from_levels(rows_a, ["fam", "leaf"])
    tb = tree_from_levels(rows_b, ["fam", "leaf"])
    qn, cols = ["a1", "a2", "b1"], ["x1", "x2", "y1", "y2"]
    S = 0.5 + 0.4 * rs.rand(3, 4)

    counts = rs.poisson(3.0, size=(200, 300)).astype(float)
    labels = np.repeat(cols, 50)
    Cc, st = structure_matrices(counts, labels, tb, cols, kind="chord",
                                n_hvg=200)
    assert Cc.shape == (4, 4) and st is None and np.allclose(Cc, Cc.T)
    Cp, fit = structure_matrices(counts, labels, tb, cols,
                                 kind="patristic", n_hvg=200)
    assert Cp.shape == (4, 4) and 0 <= fit["stress"]

    hop = fugw_map(S, ta, tb, qn, cols, rho=1.0, alpha=0.5)
    chord = fugw_map(S, ta, tb, qn, cols, CB=Cc, rho=1.0, alpha=0.5)
    assert np.all(np.isfinite(chord["pi"]))
    # overriding CB with actual hop distances reproduces the default
    from metaarbor.tree import leaf_path_dist
    CBh, lb = leaf_path_dist(tb)
    CBh = CBh[np.ix_([lb.index(c) for c in cols],
                     [lb.index(c) for c in cols])]
    same = fugw_map(S, ta, tb, qn, cols, CB=CBh, rho=1.0, alpha=0.5)
    assert np.allclose(same["pi"], hop["pi"], atol=1e-10)


def test_patristic_default_with_expression():
    import warnings
    from metaarbor import tree_from_levels
    from metaarbor.branch_fit import structure_matrices
    from metaarbor.fugw import fugw_map

    rs = np.random.RandomState(9)
    ta = tree_from_levels([("A", "a1"), ("A", "a2"), ("B", "b1")],
                          ["fam", "leaf"])
    tb = tree_from_levels([("X", "x1"), ("X", "x2"), ("Y", "y1"),
                           ("Y", "y2")], ["fam", "leaf"])
    qn, cols = ["a1", "a2", "b1"], ["x1", "x2", "y1", "y2"]
    S = 0.5 + 0.4 * rs.rand(3, 4)
    ea = {"counts": rs.poisson(3.0, size=(150, 300)).astype(float),
          "labels": np.repeat(qn, 50)}
    eb = {"counts": rs.poisson(3.0, size=(200, 300)).astype(float),
          "labels": np.repeat(cols, 50)}
    auto = fugw_map(S, ta, tb, qn, cols, expr_a=ea, expr_b=eb,
                    rho=1.0, alpha=0.5)
    CA, _ = structure_matrices(ea["counts"], ea["labels"], ta, qn,
                               kind="patristic")
    CB, _ = structure_matrices(eb["counts"], eb["labels"], tb, cols,
                               kind="patristic")
    manual = fugw_map(S, ta, tb, qn, cols, CA=CA, CB=CB,
                      rho=1.0, alpha=0.5)
    assert np.allclose(auto["pi"], manual["pi"], atol=1e-10)
    # no expression: warns and falls back to hop; structure="hop" silent
    with pytest.warns(UserWarning):
        fugw_map(S, ta, tb, qn, cols, rho=1.0, alpha=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fugw_map(S, ta, tb, qn, cols, structure="hop", rho=1.0, alpha=0.5)
