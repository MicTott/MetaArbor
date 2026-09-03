"""Pairwise decision layer and candidate formation (DESIGN.md step 4,
corrected contracts).

`pairwise_decisions` is implemented: for every unordered dataset pair it
runs ONE frozen measurement, then frozen node-level Walks in both
directions (the vote cache makes internal-node queries cheap — a node's
cells are just a positive mask), canonicalizes unary-equivalent nodes
WITHIN each tree, and records reciprocal matches with each direction's
bootstrap decision support. Raw AUROCs never cross this boundary — only
calibrated decisions do.

`candidate_groups` (step 4b-d) remains a corrected contract stub.
"""
from __future__ import annotations

import numpy as np

from metaarbor import leaves_under, measure
from metaarbor.branch_fit import _collapse_chains
from metaarbor.walk import select_node


def canonical_nodes(tree):
    """(canonical node list, node -> canonical map): unary-chain interiors
    map to their chain end; canonical set = branching internals + leaves,
    root excluded, deterministic order."""
    red_parent, node_map = _collapse_chains(tree)
    return sorted(red_parent.keys()), node_map


def _decision_support(sel):
    """Bootstrap support of the walk's final decision (mirrors the
    interpretation layer's extraction; tolerant of missing trace keys)."""
    path = sel.get("path") or []
    if not path:
        return float("nan")
    last = path[-1]
    if last.get("override"):
        return float(last.get("vote", 1.0))
    par = last.get("par_gt0")
    if last.get("stopped") and par is not None and not np.isnan(
            last.get("par_lo", float("nan"))) and last["par_lo"] > 0:
        return float(par)
    frac = last.get("sib_gt_margin")
    if frac is None:
        return float("nan")
    return float(1.0 - frac) if last.get("stopped") else float(frac)


def _node_walks(cache, labels, own_tree, own_nodes, target_tree,
                target_node_map, base_seed, **walk_kwargs):
    """Frozen Walk of every canonical node of `own_tree` (its cells as the
    positive set) against `target_tree`. Per-node seed = base_seed + rank
    in the canonical order (deterministic)."""
    labels = np.asarray(labels)
    out = {}
    for rank, node in enumerate(own_nodes):
        lv = set(leaves_under(own_tree, node))
        q_labels = np.where(np.isin(labels, list(lv)), "__q__", "__bg__")
        sel = select_node(cache, q_labels, "__q__", target_tree,
                          seed=base_seed + rank, **walk_kwargs)
        matched = bool(sel["matched"]) and not sel["at_root"]
        out[node] = {
            "selected": (target_node_map[sel["selected"]]
                         if matched else None),
            "support": _decision_support(sel) if matched else float("nan"),
            "matched": matched,
        }
    return out


def pairwise_decisions(datasets, trees, n_hvg=1000, base_seed=211,
                       **walk_kwargs):
    """Calibrated reciprocal decisions for all unordered dataset pairs.

    datasets: {key: dict(counts=cells x genes, labels=per-cell leaf labels,
                         gene_names=list, lib=optional full-gene totals)}
    trees:    {key: metaarbor tree over that dataset's leaf labels}

    Reciprocity (corrected): after within-tree canonicalization, node a of
    tree_i and node b of tree_j match iff the frozen Walk selects
    canon(a) -> b AND canon(b) -> a. Cross-atlas leaf-set comparison is
    never used (leaf sets are disjoint by construction).

    Returns dict:
      matches:    {(ki, kj): [dict(node_i, node_j, support_ij, support_ji)]}
      selections: {(ki, kj): one-way walk results ki -> kj (diagnostics)}
      unmatched:  {key: canonical nodes with no reciprocal match in ANY
                   pair — the singleton-candidate pool}
    """
    keys = sorted(datasets)
    canon = {k: canonical_nodes(trees[k]) for k in keys}
    matches, selections = {}, {}
    matched_anywhere = {k: set() for k in keys}
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            ki, kj = keys[a_i], keys[b_i]
            di, dj = datasets[ki], datasets[kj]
            m = measure(di["counts"], di["labels"], dj["counts"],
                        dj["labels"],
                        gene_names=di.get("gene_names",
                                          dj.get("gene_names")),
                        n_hvg=n_hvg, lib_a=di.get("lib"),
                        lib_b=dj.get("lib"))
            nodes_i, map_i = canon[ki]
            nodes_j, map_j = canon[kj]
            fwd = _node_walks(m["cache_a"], di["labels"], trees[ki],
                              nodes_i, trees[kj], map_j,
                              base_seed, **walk_kwargs)
            rev = _node_walks(m["cache_b"], dj["labels"], trees[kj],
                              nodes_j, trees[ki], map_i,
                              base_seed + 5000, **walk_kwargs)
            selections[(ki, kj)] = fwd
            selections[(kj, ki)] = rev
            pair = []
            for a, ra in fwd.items():
                b = ra["selected"]
                if b is None:
                    continue
                rb = rev.get(b)
                if rb and rb["selected"] == a:
                    pair.append({"node_i": a, "node_j": b,
                                 "support_ij": ra["support"],
                                 "support_ji": rb["support"]})
                    matched_anywhere[ki].add(a)
                    matched_anywhere[kj].add(b)
            matches[(ki, kj)] = pair
    unmatched = {k: [n for n in canon[k][0]
                     if n not in matched_anywhere[k]] for k in keys}
    return {"matches": matches, "selections": selections,
            "unmatched": unmatched}


def candidate_groups(decisions, trees):
    """Candidate graph from reciprocal supported edges. Missing datasets
    permitted: a group needs agreement only among eligible observed
    datasets (+ ancestry compatibility). Every stable unmatched node
    enters as a SINGLETON candidate. Seed invariance is computed and
    reported as a per-candidate diagnostic, not enforced.
    Returns [dict tree_key -> node] with diagnostics."""
    raise NotImplementedError("corrected contract; next implementation step")
