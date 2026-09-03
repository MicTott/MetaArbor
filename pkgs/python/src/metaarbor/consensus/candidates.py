"""Candidate meta-clade formation (DESIGN.md step 4). CONTRACT STUBS —
the v1 algorithm is specified; implementation lands after the simulation
gates for eligibility/poset are green.

Consumes ONLY the frozen pairwise metaarbor API (measure/baseline_map/
fugw_map); never alters Walk or Transport.
"""
from __future__ import annotations


def pairwise_decisions(measures, alpha=0.05, n_boot=200):
    """Calibrated per-pair reciprocal decisions: canonicalize
    unary-equivalent nodes WITHIN each tree, then a match is mutual
    selection (A_i -> B_j and B_j -> A_i) between canonical nodes, with
    bootstrap support attached. Raw AUROCs are NEVER averaged across
    pairs; only decisions cross this boundary.
    Returns {(i, j): [(node_i, node_j, support)]}."""
    raise NotImplementedError("corrected contract; lands after sim gates")


def candidate_groups(decisions, trees):
    """Candidate graph from reciprocal supported edges. Missing datasets
    permitted: a group needs agreement only among eligible observed
    datasets (+ ancestry compatibility). Every stable unmatched node
    enters as a SINGLETON candidate. Seed invariance is computed and
    reported as a per-candidate diagnostic, not enforced.
    Returns [dict tree_key -> node] with diagnostics."""
    raise NotImplementedError("corrected contract; lands after sim gates")
