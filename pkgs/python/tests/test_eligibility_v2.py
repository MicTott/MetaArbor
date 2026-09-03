"""v2 eligibility tests (staged; copied into pkgs/python/tests on apply).

Pins the corrected model: LOO Beta-binomial posterior, exact closed-form
detection power, single-donor lower credible bound, and the ascertainment
direction (v2 power <= naive point-estimate power when evidence is thin).
"""
import math

import pytest

from metaarbor.consensus import (call_v2, p_detect, p_detect_posterior,
                                 prevalence_lower, prevalence_posterior)


def test_posterior_pooling_and_closed_form():
    a, b = prevalence_posterior([5, 3], [100, 60])   # k=8, n=160
    assert a == pytest.approx(8.5) and b == pytest.approx(152.5)
    # closed form equals Monte Carlo integration of 1-(1-p)^n
    import numpy as np
    rs = np.random.RandomState(0)
    p = rs.beta(a, b, 200000)
    mc = 1 - np.mean((1 - p) ** 50)
    assert p_detect_posterior(a, b, 50) == pytest.approx(mc, abs=2e-3)


def test_uncertainty_makes_power_conservative():
    # same point prevalence (5%), thin vs rich evidence: the thin posterior
    # must claim LESS detection power than the naive point-estimate model
    a_thin, b_thin = prevalence_posterior([1], [20])
    naive = p_detect(0.05, 40)
    assert p_detect_posterior(a_thin, b_thin, 40) < naive + 1e-9
    # and with rich evidence it approaches the naive value
    a_rich, b_rich = prevalence_posterior([500], [10000])
    assert p_detect_posterior(a_rich, b_rich, 40) == pytest.approx(
        naive, abs=0.02)


def test_single_donor_lower_bound():
    lo = prevalence_lower(2, 100)          # 2% observed once
    assert 0 < lo < 0.02                   # bound sits below the point est.
    hi_evidence = prevalence_lower(200, 10000)
    assert hi_evidence == pytest.approx(0.02, abs=0.005)


def test_call_v2_three_way():
    # rich evidence of a 50% subpopulation elsewhere; 20 parent cells here
    assert call_v2(False, [50], [100], 20) == "private_or_absent"
    # thin evidence of a rare subpopulation; 20 parent cells: unknown
    assert call_v2(False, [1], [100], 20) == "unknown"
    assert call_v2(True, [1], [100], 20) == "supported"
