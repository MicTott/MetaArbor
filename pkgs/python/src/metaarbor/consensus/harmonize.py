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


def route_rejected(nodes, trees, rejected, affiliates):
    """Rejection routes the CLAIM, never its constituent labels.

    A rejected candidate is a failed cross-atlas (or private) claim.
    Its member labels fall back to their own atlases as explicit
    `unplaced_single_atlas` nodes that carry the rejection reason
    verbatim (`insufficient_support` / `no_support` /
    `ancestry_incompatible` / `ancestry_cycle`) and the candidate id as
    provenance — `ancestry_incompatible` and `ancestry_cycle` are
    additionally conflict evidence, already recorded by the backbone in
    its conflict list. Fallback placement is topologically ordered
    (parents before children by input-tree depth). Neither category is
    ever relabeled private. A member already represented elsewhere
    (another accepted or unknown candidate, an expansion, an affiliate
    alias) is skipped — no duplicate placement. A rejected consolidated
    subtree surfaces with its original within-atlas topology preserved.

    Returns the list of fallback placements performed.
    """
    from metaarbor.branch_fit import _collapse_chains
    member_to_id = {}
    for mid, nd in nodes.items():
        for ds, m in nd["members"].items():
            member_to_id.setdefault((ds, m), mid)
    aff = {(a["dataset"], a["node"]) for a in affiliates}

    def depth(ds, node):
        rp, _ = _collapse_chains(trees[ds])
        d, p = 0, rp.get(node)
        while p not in (None, "root"):
            d += 1
            p = rp.get(p)
        return d

    # topological fallback order: a rejected parent must place before a
    # rejected child, or the child attaches one level too high
    jobs = []
    for rej in rejected:
        c = rej["candidate"]
        for ds, node in sorted(c["members"].items()):
            jobs.append((depth(ds, node), ds, node, rej))
    routed, ucount = [], 0
    for _d, ds, node, rej in sorted(
            jobs, key=lambda t: (t[0], t[1], t[2])):
        c = rej["candidate"]
        reason = rej["reason"]
        if (ds, node) in member_to_id or (ds, node) in aff:
            continue                      # represented elsewhere
        rp, _ = _collapse_chains(trees[ds])
        p, parent_id = rp.get(node), None
        while p not in (None, "root"):
            if (ds, p) in member_to_id:
                parent_id = member_to_id[(ds, p)]
                break
            p = rp.get(p)
        ucount += 1
        uid = f"MA-X{ucount:04d}"
        rec = {"reason": reason, "candidate_id": c["candidate_id"]}
        sub = c.get("provenance", {}).get("subtree_nodes") or []
        sub_un = sorted(x for x in set(sub) - {node}
                        if (ds, x) not in member_to_id and
                        (ds, x) not in aff)
        idmap = {node: uid}
        for k, x in enumerate(sub_un, 1):
            idmap[x] = f"{uid}.{k:02d}"
        for x, xid in idmap.items():
            if x == node:
                px = parent_id
            else:
                p2 = rp.get(x)
                while p2 not in (None, "root") and p2 not in idmap:
                    p2 = rp.get(p2)
                px = idmap.get(p2, uid)
            nodes[xid] = {"parent": px,
                          "status": "unplaced_single_atlas",
                          "members": {ds: x}, "aliases": [x],
                          "display": _display([x]),
                          "support": (0, 0), "subtree_parent": None,
                          "rejection": dict(rec),
                          **({"expanded": True} if x != node else {})}
            member_to_id[(ds, x)] = xid
            routed.append({"dataset": ds, "label": x,
                           "node_id": xid, "reason": reason,
                           "candidate_id": c["candidate_id"]})
    return routed


def repair_completeness(nodes, trees, affiliates):
    """COMPLETENESS INVARIANT (core contract, not a display repair):
    every input-tree leaf must occur in the assembly as a member, a
    marked affiliate alias, or an explicitly unplaced single-atlas node.

    Any label the upstream layers lost is reinstated here as an
    `unplaced_single_atlas` node using its exact input-tree parentage
    (nearest represented ancestor in that atlas's own tree; root when
    none), flagged `assembly_repair` and given support (0, 0) so it is
    excluded from inferred-support counts. The invariant is asserted
    after repair; a violation past this point is a bug.

    Returns the list of repairs performed (empty when upstream was
    already complete).
    """
    from metaarbor.branch_fit import _collapse_chains
    represented = {ds: set() for ds in trees}
    for nd in nodes.values():
        for ds, m in nd["members"].items():
            represented.setdefault(ds, set()).add(m)
    for aff in affiliates:
        represented.setdefault(aff["dataset"], set()).add(aff["node"])
    member_to_id = {}
    for mid, nd in nodes.items():
        for ds, m in nd["members"].items():
            member_to_id.setdefault((ds, m), mid)
    repairs = []
    for ds in sorted(trees):
        rp, _ = _collapse_chains(trees[ds])
        for leaf in trees[ds]["leaves"]:
            if leaf in represented[ds]:
                continue
            p, parent_id = rp.get(leaf), None
            while p not in (None, "root"):
                if (ds, p) in member_to_id:
                    parent_id = member_to_id[(ds, p)]
                    break
                p = rp.get(p)
            rid = f"MA-R{len(repairs) + 1:04d}"
            nodes[rid] = {"parent": parent_id,
                          "status": "unplaced_single_atlas",
                          "members": {ds: leaf}, "aliases": [leaf],
                          "display": _display([leaf]),
                          "support": (0, 0), "subtree_parent": None,
                          "assembly_repair": True}
            member_to_id[(ds, leaf)] = rid
            represented[ds].add(leaf)
            repairs.append({"dataset": ds, "label": leaf,
                            "node_id": rid, "parent": parent_id})
    for ds in trees:
        missing = set(trees[ds]["leaves"]) - represented[ds]
        assert not missing, \
            f"completeness invariant violated for {ds}: {sorted(missing)}"
    return repairs


def _display(labels):
    """Modal cleaned name across member labels (dataset prefixes 'ds|'
    stripped); originals are aliases, never destroyed."""
    names = [str(l).split("|", 1)[-1] for l in labels]
    return sorted(names, key=lambda n: (names.count(n), -len(n)))[-1]


def harmonize(datasets, trees, n_hvg=1000, n_boot=200, base_seed=211,
              stability=None, frozen=FROZEN, **walk_kwargs):
    """Run pairwise Walk evidence -> candidates -> hierarchical greedy
    backbone -> assembled reconciled hierarchy.

    datasets: {key: dict(counts, labels, gene_names, lib=optional)}
    trees:    {key: metaarbor tree over that dataset's labels} (use
              metaarbor.infer_tree for flat label sets — never a star)

    Returns dict with `tree` (id -> node record: parent, children, status,
    members, aliases, display), plus decisions / candidates / backbone /
    conflicts / provenance passthroughs.
    """
    # entry validation: the completeness invariant is only as strong as
    # its reference set — every observed dataset label must be a tree
    # leaf and vice versa, or the tree silently misdescribes the data
    for ds in sorted(datasets):
        obs = set(np.unique(np.asarray(datasets[ds]["labels"])))
        lv = set(trees[ds]["leaves"])
        if obs != lv:
            raise ValueError(
                f"{ds}: dataset labels != tree leaves; "
                f"labels-not-in-tree={sorted(obs - lv)[:5]}, "
                f"tree-leaves-unobserved={sorted(lv - obs)[:5]}")

    if stability is None:
        import warnings
        warnings.warn(
            "no stability map supplied: STABILITY_FLOOR screening of "
            "private clades is INACTIVE (trust-supplied-trees mode). "
            "Pass stability={(dataset, node): support}, e.g. from "
            "infer_tree()['support'], to screen inferred clades.",
            UserWarning)
        stability = {}

    dec = pairwise_decisions(datasets, trees, n_hvg=n_hvg,
                             base_seed=base_seed, n_boot=n_boot,
                             **walk_kwargs)
    cands = candidate_groups(dec, trees)
    bb = greedy_backbone(cands, trees, datasets,
                         selections=dec["selections"],
                         stability=stability, frozen=frozen)

    nodes = {}
    for nd in bb["nodes"]:
        aliases = sorted(nd["members"].values())
        # affiliates ride as VISIBLY MARKED aliases and never count as
        # reciprocal support (support tuples derive from members only)
        aliases += sorted(f'\u2248 {a["node"]}'
                          for a in nd.get("affiliates", []))
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

    # expand consolidated subtrees ("absorbed is never discarded" applies
    # to the ASSEMBLED tree too): every private node and every
    # single-atlas node whose candidate absorbed a subtree gets its full
    # internal topology as expanded child nodes
    def expand(uid, ds, root_node, sub_nodes):
        rp, _ = _collapse_chains(trees[ds])
        sub = set(sub_nodes)
        idmap = {root_node: uid}
        k = 0
        for x in sorted(sub):
            if x == root_node:
                continue
            k += 1
            xid = f"{uid}.{k:02d}"
            idmap[x] = xid
        for x, xid in idmap.items():
            if x == root_node:
                continue
            p = rp.get(x)
            while p not in (None, "root") and p not in sub:
                p = rp.get(p)
            nodes[xid] = {"parent": idmap.get(p, uid),
                          "status": nodes[uid]["status"],
                          "members": {ds: x}, "aliases": [x],
                          "display": _display([x]), "support": (1, 1),
                          "subtree_parent": None, "expanded": True}

    for nd in bb["nodes"]:
        sp = nd.get("subtree_parent")
        if sp and len(sp) > 1:
            (ds, root_node), = nd["members"].items()
            expand(nd["id"], ds, root_node, sp.keys())
    for u in bb["unknown"]:
        c = u["candidate"]
        sub = c.get("provenance", {}).get("subtree_nodes") or []
        if len(sub) > 1:
            (ds, root_node), = c["members"].items()
            uid = [i for i, nd in nodes.items()
                   if nd["members"].get(ds) == root_node]
            if uid:
                expand(uid[0], ds, root_node, sub)

    routed = route_rejected(nodes, trees, bb["rejected"],
                            bb["affiliates"])
    # final tripwire: after rejection routing this must find NOTHING;
    # a nonzero repair count means a new, undiagnosed loss pathway
    repairs = repair_completeness(nodes, trees, bb["affiliates"])

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
            "rejection_fallbacks": routed, "repairs": repairs,
            "decisions": dec, "candidates": cands, "backbone": bb,
            "conflicts": bb["conflicts"], "affiliates": bb["affiliates"],
            # internal nodes whose descendants matched but which
            # themselves formed no candidate: the topological signature of
            # incompatible grouping layers (they resolve as polytomies)
            "unplaced_internals": cands["unresolved_internals"],
            "frozen": dict(frozen)}
