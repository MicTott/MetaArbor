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
    """Meta-clade candidates from the pairwise decision layer (DESIGN.md
    step 4b-d, corrected).

    Grouping: reciprocal edges sorted by min directional support
    (descending, deterministic tie-breaks), merged by union-find under a
    DATASET-UNIQUENESS constraint (at most one canonical node per dataset
    per group); refused merges are returned as `edge_conflicts`. Missing
    datasets are permitted. One-way selections are preserved and attached
    as ASYMMETRIC evidence (resolution mismatch, not failure). Seed
    invariance is reported as the fraction of co-present dataset pairs
    joined by a direct reciprocal edge — a diagnostic, never a filter.

    Singleton consolidation: a PRIVATE-SUBTREE candidate is rooted at each
    unmatched canonical node whose parent is matched-or-root and whose
    subtree contains no matched node; its unmatched descendants are
    absorbed into it. Unmatched internals with matched descendants are
    emitted as `unresolved_internals` (evidence of resolution mismatch),
    not candidates.

    Returns dict(candidates=[...], edge_conflicts=[...],
    unresolved_internals={ds: [...]}); each candidate carries members
    (<= 1 node per dataset), reciprocal and asymmetric edges, missing
    datasets, seed_invariance, nearest matched ancestor per member,
    kind ("multi" | "private_subtree"), and full provenance.
    """
    keys = sorted(trees)
    canon = {k: canonical_nodes(trees[k]) for k in keys}
    red_parent = {k: _collapse_chains(trees[k])[0] for k in keys}

    # ---- support-ordered union-find with dataset uniqueness -------------
    edges = []
    for (ki, kj), pair in decisions["matches"].items():
        for m in pair:
            supp = min(m["support_ij"], m["support_ji"])
            edges.append((-(supp if np.isfinite(supp) else 0.0),
                          ki, m["node_i"], kj, m["node_j"], m))
    edges.sort(key=lambda e: (e[0], e[1], e[2], e[3], e[4]))

    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    members_of = {}

    def datasets_of(root):
        return {ds for ds, _ in members_of.get(root, {root[0]: None}) or []}

    for x in [(k, n) for k in keys for n in canon[k][0]]:
        parent[x] = x
        members_of[x] = {x}

    edge_conflicts = []
    group_edges = {}
    for _, ki, ni, kj, nj, m in edges:
        a, b = find((ki, ni)), find((kj, nj))
        if a == b:
            group_edges.setdefault(a, []).append((ki, ni, kj, nj, m))
            continue
        ds_a = {ds for ds, _ in members_of[a]}
        ds_b = {ds for ds, _ in members_of[b]}
        if ds_a & ds_b:
            edge_conflicts.append({"edge": (ki, ni, kj, nj),
                                   "support": min(m["support_ij"],
                                                  m["support_ji"]),
                                   "reason": "dataset_uniqueness"})
            continue
        parent[b] = a
        members_of[a] |= members_of[b]
        del members_of[b]
        ge = group_edges.pop(b, [])
        group_edges.setdefault(a, []).extend(ge)
        group_edges[a].append((ki, ni, kj, nj, m))

    groups = [ms for ms in members_of.values() if len(ms) > 1]
    matched_nodes = {x for ms in groups for x in ms}

    # ---- nearest matched ancestor helper --------------------------------
    node_group = {}
    groups_sorted = sorted(groups, key=lambda ms: sorted(ms))
    for gi, ms in enumerate(groups_sorted):
        for x in ms:
            node_group[x] = gi

    def nearest_matched_ancestor(k, node):
        p = red_parent[k].get(node)
        while p not in (None, "root"):
            if (k, p) in matched_nodes:
                return p, node_group[(k, p)]
            p = red_parent[k].get(p)
        return None, None

    # ---- asymmetric evidence --------------------------------------------
    asym_by_node = {}
    for (ki, kj), sel in decisions["selections"].items():
        for a, ra in sel.items():
            b = ra["selected"]
            if b is None:
                continue
            rec = {"from": (ki, a), "to": (kj, b),
                   "support": ra["support"]}
            asym_by_node.setdefault((ki, a), []).append(rec)
            asym_by_node.setdefault((kj, b), []).append(rec)

    def asym_for(member_set, recip_set):
        out = []
        for x in member_set:
            for rec in asym_by_node.get(x, []):
                key = (rec["from"][0], rec["from"][1],
                       rec["to"][0], rec["to"][1])
                if key not in recip_set and rec not in out:
                    out.append(rec)
        return out

    candidates = []
    for gi, ms in enumerate(groups_sorted):
        root = find(next(iter(ms)))
        redges = [{"from": (ki, ni), "to": (kj, nj),
                   "support_ij": m["support_ij"],
                   "support_ji": m["support_ji"]}
                  for ki, ni, kj, nj, m in group_edges.get(root, [])]
        recip_keys = set()
        for e in redges:
            recip_keys.add((*e["from"], *e["to"]))
            recip_keys.add((*e["to"], *e["from"]))
        ds_present = sorted({ds for ds, _ in ms})
        n_pairs = len(ds_present) * (len(ds_present) - 1) // 2
        direct = {frozenset((e["from"][0], e["to"][0])) for e in redges}
        candidates.append({
            "members": {ds: n for ds, n in sorted(ms)},
            "reciprocal_edges": redges,
            "asymmetric_edges": asym_for(ms, recip_keys),
            "missing": [k for k in keys if k not in ds_present],
            "seed_invariance": (len(direct) / n_pairs) if n_pairs else 1.0,
            "nearest_matched_ancestor": {
                ds: nearest_matched_ancestor(ds, n)
                for ds, n in sorted(ms)},
            "kind": "multi",
            "provenance": {"n_members": len(ms)},
        })

    # ---- private-subtree consolidation ----------------------------------
    unresolved = {k: [] for k in keys}
    for k in keys:
        nodes, node_map = canon[k]
        matched_k = {n for (kk, n) in matched_nodes if kk == k}

        def subtree_canon(n):
            out, stack = [], [n]
            while stack:
                v = stack.pop()
                out.append(v)
                stack.extend(c for c in nodes
                             if red_parent[k].get(c) == v)
            return out

        def depth_of(n):
            d, p = 0, red_parent[k].get(n)
            while p not in (None, "root"):
                d += 1
                p = red_parent[k].get(p)
            return d

        # top-down: a node that goes unresolved (its subtree contains a
        # matched descendant) remains a valid scan point for its OWN
        # unmatched children — otherwise leaves hanging between an
        # unresolved internal and the matched region below it would be
        # silently dropped from the candidate pool entirely
        unresolved_k = set()
        for n in sorted(nodes, key=depth_of):
            if n in matched_k:
                continue
            p = red_parent[k].get(n)
            parent_ok = (p == "root" or p in matched_k or
                         p in unresolved_k)
            if not parent_ok:
                continue                      # absorbed by an ancestor
            sub = subtree_canon(n)
            if any(x in matched_k for x in sub):
                unresolved_k.add(n)
                unresolved[k].append(n)
                continue
            anc_node, anc_gi = nearest_matched_ancestor(k, n)
            candidates.append({
                "members": {k: n},
                "reciprocal_edges": [],
                "asymmetric_edges": asym_for({(k, n)}, set()),
                "missing": [x for x in keys if x != k],
                "seed_invariance": 1.0,
                "nearest_matched_ancestor": {k: (anc_node, anc_gi)},
                "kind": "private_subtree",
                "provenance": {"subtree_nodes": sorted(sub)},
            })

    candidates.sort(key=lambda c: (c["kind"],
                                   sorted(c["members"].items())))
    for i, c in enumerate(candidates):
        c["candidate_id"] = f"cand:{i+1:04d}"
    return {"candidates": candidates, "edge_conflicts": edge_conflicts,
            "unresolved_internals": unresolved}
