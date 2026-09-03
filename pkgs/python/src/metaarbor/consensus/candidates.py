"""Candidate meta-clade formation (DESIGN.md step 4). CONTRACT STUBS —
the v1 algorithm is specified; implementation lands after the simulation
gates for eligibility/poset are green.

Consumes ONLY the frozen pairwise metaarbor API (measure/baseline_map/
fugw_map); never alters Walk or Transport.
"""
from __future__ import annotations


def pairwise_decisions(measures, alpha=0.05, n_boot=200):
    """Calibrated per-pair reciprocal match decisions with bootstrap
    support. Raw AUROCs are NEVER averaged across pairs; only decisions
    cross this boundary. Returns {(i, j): [(node_i, node_j, support)]}."""
    raise NotImplementedError("v1 contract; lands after sim gates")


def candidate_groups(decisions, trees):
    """Seed from reciprocal supported matches; expand across datasets;
    retain a group only if membership is invariant when re-seeded from
    each constituent node. Returns [dict tree_key -> node]."""
    raise NotImplementedError("v1 contract; lands after sim gates")
