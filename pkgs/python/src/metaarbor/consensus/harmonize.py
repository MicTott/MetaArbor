"""harmonize(): the MetaArbor front door for tree synthesis (K >= 2,
Walk evidence).

Produces a BEST-SUPPORTED RECONCILED HIERARCHY — a genuinely new tree in
which labels from every atlas sit at their supported levels, without
privileging any atlas:

- equivalent cross-atlas labels collapse into one meta-clade carrying all
  original labels as aliases;
- a coarse label sits as an internal node ABOVE finer descendants;
- finer structure resolved by only one atlas attaches beneath its parent
  meta-clade as `single_atlas` nodes (existence is certain — it is that
  atlas's own data; only the cross-atlas correspondence is unresolved);
- genuinely private branches persist with full internal topology;
- affiliates (unreciprocated twins) ride as aliases on their meta-clade;
- incompatible evidence is rejected into the conflict graph, leaving
  polytomies rather than forced splits.
"""
from __future__ import annotations

import numpy as np

from .backbone import FROZEN, greedy_backbone
from .candidates import candidate_groups, canonical_nodes, pairwise_decisions


def _display(labels):
    """Modal cleaned name across member labels (dataset prefixes 'ds|'
    stripped); originals are aliases, never destroyed."""
    names = [str(l).split("|", 1)[-1] for l in labels]
    return sorted(names, key=lambda n: (names.count(n), -len(n)))[-1]


def harmonize(datasets, trees, n_hvg=1000, n_boot=200, base_seed=211,
              frozen=FROZEN, **walk_kwargs):
    """Run pairwise Walk evidence -> candidates -> hierarchical greedy
    backbone -> assembled reconciled hierarchy.

    datasets: {key: dict(counts, labels, gene_names, lib=optional)}
    trees:    {key: metaarbor tree over that dataset's labels} (use
              metaarbor.infer_tree for flat label sets — never a star)

    Returns dict with `tree` (id -> node record: parent, children, status,
    members, aliases, display), plus decisions / candidates / backbone /
    conflicts / provenance passthroughs.
    """
    dec = pairwise_decisions(datasets, trees, n_hvg=n_hvg,
                             base_seed=base_seed, n_boot=n_boot,
                             **walk_kwargs)
    cands = candidate_groups(dec, trees)
    bb = greedy_backbone(cands, trees, datasets,
                         selections=dec["selections"], frozen=frozen)

    nodes = {}
    for nd in bb["nodes"]:
        aliases = sorted(nd["members"].values())
        aliases += sorted(f'{a["node"]}' for a in nd.get("affiliates", []))
        nodes[nd["id"]] = {
            "parent": nd["parent"], "status": nd["status"],
            "members": dict(nd["members"]), "aliases": aliases,
            "display": _display(aliases),
            "support": nd["support"],
            "subtree_parent": nd.get("subtree_parent"),
        }

    # single-atlas placements: unknown-class singletons exist with
    # certainty in their own atlas; attach beneath the nearest accepted
    # ancestor with correspondence marked unresolved
    from metaarbor.branch_fit import _collapse_chains
    member_to_id = {}
    for mid, nd in nodes.items():
        for ds, n_ in nd["members"].items():
            member_to_id[(ds, n_)] = mid
    ucount = 0
    for u in bb["unknown"]:
        c = u["candidate"]
        (ds, node), = c["members"].items()
        rp, _ = _collapse_chains(trees[ds])
        p, parent_id = rp.get(node), None
        while p not in (None, "root"):
            if (ds, p) in member_to_id:
                parent_id = member_to_id[(ds, p)]
                break
            p = rp.get(p)
        ucount += 1
        uid = f"MA-U{ucount:04d}"
        nodes[uid] = {"parent": parent_id, "status": "single_atlas",
                      "members": {ds: node}, "aliases": [node],
                      "display": _display([node]),
                      "support": (1, 1), "subtree_parent": None}

    children = {i: [] for i in nodes}
    roots = []
    for i, nd in nodes.items():
        if nd["parent"] is None:
            roots.append(i)
        else:
            children[nd["parent"]].append(i)
    for k in children:
        children[k].sort(key=lambda i: (nodes[i]["status"],
                                        nodes[i]["display"]))
    for i, nd in nodes.items():
        nd["children"] = children[i]

    return {"tree": nodes, "roots": sorted(roots),
            "decisions": dec, "candidates": cands, "backbone": bb,
            "conflicts": bb["conflicts"], "affiliates": bb["affiliates"],
            # internal nodes whose descendants matched but which
            # themselves formed no candidate: the topological signature of
            # incompatible grouping layers (they resolve as polytomies)
            "unplaced_internals": cands["unresolved_internals"],
            "frozen": dict(frozen)}
