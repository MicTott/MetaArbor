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
