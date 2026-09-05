"""MetaArbor-Consensus (BETA/EXPERIMENTAL).

Consumes the frozen pairwise MetaArbor API; Walk and Transport are never
altered. See DESIGN.md in this package for the specification, isolation
rules, prespecified predictions, and validation gates.
"""
from . import diagnostics  # noqa: F401
from .backbone import FROZEN, classify_edge_conflicts, greedy_backbone, provenance_table
from .candidates import candidate_groups, canonical_nodes, pairwise_decisions
from .eligibility import (
                          call,
                          call_v2,
                          eligible_v1,
                          eligible_v15,
                          p_detect,
                          p_detect_posterior,
                          prevalence_lower,
                          prevalence_posterior,
                          support,
)
from .harmonize import harmonize
from .plot_reconciled import plot_reconciled_tree
from .poset import compatible, pair_relation, relation
from .simulate import latent_tree, scenario, simulate_donors

__all__ = [
    "call", "call_v2", "eligible_v1", "eligible_v15", "p_detect",
    "p_detect_posterior", "prevalence_lower", "prevalence_posterior",
    "support", "FROZEN", "classify_edge_conflicts", "greedy_backbone",
    "provenance_table", "candidate_groups", "canonical_nodes",
    "pairwise_decisions", "compatible", "pair_relation", "relation",
    "latent_tree", "scenario", "simulate_donors", "diagnostics",
    "harmonize", "plot_reconciled_tree",
]
