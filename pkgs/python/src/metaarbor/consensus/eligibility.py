"""Eligibility / detection model (DESIGN.md step 7).

Support denominators count ELIGIBLE donors: sampling must not buy or
forfeit consensus support. v1 is a parent-context cell floor; v1.5 is the
probabilistic model  P(detect) = 1 - (1 - p_M)^n  over a donor's
parent-context cells, yielding the three-way call:

  supported          present and aligned
  private_or_absent  adequately powered, but absent
  unknown            inadequately powered — never counted against support
"""
from __future__ import annotations


def p_detect(prevalence, n_parent_cells):
    """P(at least one cell of a subpopulation at `prevalence` among
    `n_parent_cells` draws)."""
    if n_parent_cells <= 0:
        return 0.0
    return 1.0 - (1.0 - float(prevalence)) ** int(n_parent_cells)


def eligible_v1(n_parent_cells, floor=30):
    """v1: parent-context cell floor."""
    return n_parent_cells >= floor


def eligible_v15(prevalence, n_parent_cells, power=0.95):
    """v1.5: donor is eligible iff detection probability >= `power`."""
    return p_detect(prevalence, n_parent_cells) >= power


def call(present, prevalence, n_parent_cells, power=0.95):
    """Three-way per-(donor, meta-clade) call."""
    if present:
        return "supported"
    if eligible_v15(prevalence, n_parent_cells, power):
        return "private_or_absent"
    return "unknown"


def support(calls):
    """(supporting, eligible) from a list of three-way calls; `unknown`
    is excluded from the denominator."""
    supp = sum(c == "supported" for c in calls)
    elig = sum(c != "unknown" for c in calls)
    return supp, elig


# --- v2: ascertainment-bias-corrected detection model (DESIGN.md step 7) --

from scipy.special import betaln
from scipy.stats import beta as _beta_dist

JEFFREYS = (0.5, 0.5)


def prevalence_posterior(k_list, n_list, prior=JEFFREYS):
    """Beta posterior (a, b) for prevalence in the parent context, pooled
    over the supplied donors (LEAVE OUT the donor whose eligibility is
    being evaluated before calling)."""
    a0, b0 = prior
    k, n = sum(map(int, k_list)), sum(map(int, n_list))
    return a0 + k, b0 + (n - k)


def p_detect_posterior(a, b, n_parent_cells):
    """P(detect | posterior Beta(a, b), n draws), integrating prevalence
    uncertainty exactly: 1 - E_p[(1-p)^n] = 1 - B(a, b+n)/B(a, b)."""
    n = int(n_parent_cells)
    if n <= 0:
        return 0.0
    import math
    return 1.0 - math.exp(betaln(a, b + n) - betaln(a, b))


def prevalence_lower(k, n, q=0.05, prior=JEFFREYS):
    """Lower credible bound of prevalence from ONE donor (single-donor
    clades must use this, never the raw point estimate)."""
    a0, b0 = prior
    return float(_beta_dist.ppf(q, a0 + int(k), b0 + int(n) - int(k)))


def call_v2(present, k_other, n_other, n_parent_cells, power=0.95,
            prior=JEFFREYS):
    """Three-way call for one (donor, meta-clade) with LOO prevalence:
    `k_other`/`n_other` are clade and parent-context cell counts in the
    OTHER (supporting) donors only. When no other donor supports the
    clade, callers must use `prevalence_lower` on the single supporting
    donor to assess power elsewhere; this function then receives that
    donor's counts via k_other/n_other."""
    if present:
        return "supported"
    a, b = prevalence_posterior(k_other, n_other, prior)
    if p_detect_posterior(a, b, n_parent_cells) >= power:
        return "private_or_absent"
    return "unknown"
