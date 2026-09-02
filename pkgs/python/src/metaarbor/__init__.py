"""treeneighbor: integration-free alignment of cell-type taxonomies across
atlases of different resolutions.

Primary implementation of the TreeNeighbor method: a MetaNeighbor-derived
voting kernel with an exact vote-cache additivity property, a frozen
hierarchical walk estimator (votes navigate, AUROC contrasts decide), and a
frozen FUGW transport estimator with refinement-invariant tree-intrinsic
marginals. Validated against the reference R implementation on the Allen
WMB PL-ILA-ORB benchmark (see the repo's NOTES.md).
"""
from .kernel import (aggregate_cache, auroc, leaf_costs, lognorm, measure,
                     node_auroc, node_mean_scores, node_scores,
                     rank_normalize, variable_genes, vote_cache)
from .rng import Minstd
from .tree import (ancestors, leaf_path_dist, leaves_under, tree_from_levels,
                   tree_weights)
from .walk import baseline_map, compactness, select_node

__version__ = "0.1.0"

__all__ = [
    "measure", "vote_cache", "aggregate_cache", "leaf_costs", "auroc",
    "node_scores", "node_mean_scores", "node_auroc", "lognorm",
    "rank_normalize", "variable_genes",
    "tree_from_levels", "leaves_under", "ancestors", "leaf_path_dist",
    "tree_weights",
    "select_node", "baseline_map", "compactness",
    "Minstd", "__version__",
]
