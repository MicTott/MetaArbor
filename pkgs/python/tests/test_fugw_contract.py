"""FUGW v2 parameter-contract tests: co-scaling equivalence, the explicit
molecular-only mode, the zero-mass guard, v1 reproducibility, and (when
the anonymised Yu-Allen fixture is present) the exact collapse
reproduction plus its resolution under the corrected contract."""
import os

import numpy as np
import pytest

pytest.importorskip("ot")

from metaarbor.fugw import (FROZEN, MassCollapsedError, molecular_only,
                            solve)

FX = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures",
                  "fugw_nan_fixture.npz")


def toy_problem(seed=0, na=4, nb=6):
    rs = np.random.RandomState(seed)
    M = rs.rand(na, nb)
    CA = np.abs(rs.rand(na, na)); CA = (CA + CA.T) / 2; np.fill_diagonal(CA, 0)
    CB = np.abs(rs.rand(nb, nb)); CB = (CB + CB.T) / 2; np.fill_diagonal(CB, 0)
    wA = np.full(na, 1 / na)
    wB = np.full(nb, 1 / nb)
    return M, CA, CB, wA, wB


def test_v2_equals_v1_with_coscaled_rho():
    M, CA, CB, wA, wB = toy_problem()
    alpha, rho = 0.7, 0.6
    pi2, _ = solve(M, CA, CB, wA, wB, alpha=alpha, rho=rho,
                   convention="design-v2")
    pi1, _ = solve(M, CA, CB, wA, wB, alpha=alpha, rho=rho / (1 - alpha),
                   convention="pot-v1")
    assert np.allclose(pi2, pi1, atol=1e-10)


def test_alpha_zero_identical_across_conventions():
    M, CA, CB, wA, wB = toy_problem(1)
    pi2, _ = solve(M, CA, CB, wA, wB, alpha=0.0, rho=0.5,
                   convention="design-v2")
    pi1, _ = solve(M, CA, CB, wA, wB, alpha=0.0, rho=0.5,
                   convention="pot-v1")
    assert np.allclose(pi2, pi1, atol=1e-12)


def test_molecular_only_mode():
    import ot
    M, _, _, wA, wB = toy_problem(2)
    pi, gap = molecular_only(M, wA, wB, rho=0.5)
    ref = ot.unbalanced.mm_unbalanced(wA, wB, M, reg_m=0.5, div="kl")
    assert np.allclose(pi, ref) and gap == 0.0
    with pytest.raises(ValueError):
        solve(M, np.zeros((4, 4)), np.zeros((6, 6)), wA, wB, alpha=1.0)


def test_zero_mass_guard_raises_interpretable_error():
    # dominant uniform molecular cost + near-zero relaxation under the
    # LEGACY convention: mass is driven toward zero
    rs = np.random.RandomState(3)
    na, nb = 5, 7
    M = 3.0 + rs.rand(na, nb)
    CA = np.zeros((na, na)) + 1e-9
    CB = np.zeros((nb, nb)) + 1e-9
    wA = np.full(na, 1 / na)
    wB = np.full(nb, 1 / nb)
    with pytest.raises(MassCollapsedError) as ei:
        solve(M, CA, CB, wA, wB, alpha=0.9, rho=1e-4, convention="pot-v1")
    assert "mass" in str(ei.value).lower()
    assert ei.value.alpha == 0.9


@pytest.mark.skipif(not os.path.exists(FX),
                    reason="anonymised Yu-Allen fixture not present "
                           "(drop fugw_nan_fixture.npz into pkgs/fixtures)")
def test_fixture_collapse_and_v2_resolution():
    d = np.load(FX, allow_pickle=True)
    M, CA, CB, wa, wb = (d["M"], d["CA"], d["CB"], d["wa"], d["wb"])
    # released frozen v1 behavior: collapse on this geometry
    with pytest.raises(MassCollapsedError):
        solve(M, CA, CB, wa, wb, alpha=FROZEN["alpha"], rho=FROZEN["rho"],
              convention="pot-v1")
    # corrected co-scaled objective at the same design weights
    pi, _ = solve(M, CA, CB, wa, wb, alpha=FROZEN["alpha"],
                  rho=FROZEN["rho"], convention="design-v2")
    assert np.all(np.isfinite(pi)) and pi.sum() > 1e-6
