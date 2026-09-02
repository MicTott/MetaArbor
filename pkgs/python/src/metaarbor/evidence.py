"""Common node-evidence table: one row per (query, visited split, child),
assembling everything both inference modes know about a candidate node —
directional AUROC, the sibling contrast with its bootstrap lower bound, vote
fraction, the override/margin verdicts, MetaArbor-Transport mass (when a
coupling is supplied), and the query's selected relation.

Evidence rows are re-derived with the same per-query seeds as
`baseline_map`, so the recorded decisions are exactly the map's decisions.
"""
from __future__ import annotations

import numpy as np

from .kernel import auroc
from .tree import leaves_under
from .walk import baseline_map, select_node


def node_evidence(cache, test_labels, tree, S_dir, base_seed=7,
                  cache_rev=None, labels_rev=None, transport=None,
                  **walk_kwargs):
    """Build the evidence table.

    cache/test_labels/tree/S_dir: as for `baseline_map` (test population
      side scored against the training tree).
    cache_rev/labels_rev (optional): the reverse-fold cache whose leaves are
      the query populations, for the second directional AUROC — for child
      node c and query q: AUROC of the reverse-side cells belonging to c
      scored on q's aggregated column.
    transport (optional): dict with "pi" (rows = queries) plus "rows"/"cols"
      as returned by `fugw.fugw_map`; adds each child's transport-mass
      share of the query's coupling row.
    """
    test_labels = np.asarray(test_labels)
    queries = sorted(set(test_labels))
    mapping = {m["query"]: m for m in
               baseline_map(cache, test_labels, tree, S_dir,
                            base_seed=base_seed, **walk_kwargs)}
    rows = []
    for qi, q in enumerate(queries):
        sel = select_node(cache, test_labels, q, tree, seed=base_seed + qi,
                          trace=True, **walk_kwargs)
        rel = mapping[q]["relation"]
        selected = mapping[q]["selected"]
        if transport is not None:
            r = transport["rows"].index(q)
            prow = transport["pi"][r]
            ptot = prow.sum()
        for t in sel["trace"] or []:
            child_leaves = leaves_under(tree, t["child"])
            rec = dict(t)
            rec["n_leaves"] = len(child_leaves)
            rec["selected"] = selected
            rec["relation"] = rel
            # reverse directional AUROC: reverse-side cells of this child
            # vs the query's aggregated column
            if cache_rev is not None and q in cache_rev["leaves"]:
                col = cache_rev["leaves"].index(q)
                pos = np.isin(np.asarray(labels_rev), child_leaves)
                rec["auroc_rev"] = float(auroc(cache_rev["V"][:, col], pos))
            else:
                rec["auroc_rev"] = np.nan
            if transport is not None and ptot > 0:
                idx = [transport["cols"].index(l) for l in child_leaves
                       if l in transport["cols"]]
                rec["transport_mass"] = float(prow[idx].sum() / ptot)
            else:
                rec["transport_mass"] = np.nan
            rows.append(rec)
    return rows
