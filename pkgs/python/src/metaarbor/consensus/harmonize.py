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
- unreciprocated one-way calls (the old "affiliates") persist as
  directional ANNOTATIONS, never as structure;
- incompatible evidence is refused and ledgered, leaving the honest
  graph: when merged ancestry admits no unique parent, the node keeps
  ALL its minimal parents and the result is a DAG with certificates.

ASSEMBLY = QUOTIENT (ASSEMBLY2): the certification layer
(greedy_backbone: eligibility, detectability, stability,
MIN_DATASETS, MIN_SUPPORT) outputs prequalified reciprocal merges;
quotient assembly consumes ONLY those, preserves every input node
and ancestry edge (I1 by construction — no routing, no repair, no
interleaving), and exposes unresolved parentage explicitly.
`tree` node records carry `parents` (ALL minimal parents — the
authoritative field), `projected_parent` (the first minimal parent,
a DISPLAY PROJECTION when `conflicting`), and `conflicting`. There
is deliberately no plain `parent` key: tree-only consumers must
check `is_forest` and either refuse a DAG, take an explicit
projection policy, or visibly carry the certificates they drop
(result contract, ASSEMBLY2 Section 11).
"""
from __future__ import annotations

import numpy as np

from .backbone import FROZEN, greedy_backbone
from .candidates import candidate_groups, canonical_nodes, pairwise_decisions
from .quotient import quotient_assemble


def _display(labels):
    """Modal cleaned name across member labels (dataset prefixes 'ds|'
    stripped); originals are aliases, never destroyed."""
    names = [str(l).split("|", 1)[-1] for l in labels]
    return sorted(names, key=lambda n: (names.count(n), -len(n)))[-1]


def harmonize(datasets, trees, n_hvg=1000, n_boot=200, base_seed=211,
              stability=None, trust_trees=False, frozen=FROZEN,
              **walk_kwargs):
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
        if not trust_trees:
            raise ValueError(
                "supply stability={(dataset, node): support} (e.g. from "
                "infer_tree()['support']) so STABILITY_FLOOR can screen "
                "private clades, or pass trust_trees=True to state "
                "explicitly that the supplied trees are trusted as-is "
                "(curated trees). Silent trust is not available.")
        stability = {}
    else:
        # an INCOMPLETE map would silently imply perfect stability for
        # its gaps — require an entry for every internal canonical node
        missing = []
        for ds in sorted(trees):
            nodes_c, _ = canonical_nodes(trees[ds])
            leaves = set(trees[ds]["leaves"])
            for n_ in nodes_c:
                if n_ not in leaves and (ds, n_) not in stability:
                    missing.append((ds, n_))
        if missing:
            raise ValueError(
                "stability map is incomplete for internal nodes "
                f"(first missing: {missing[:5]}); supply every internal "
                "canonical node's support, or trust_trees=True with "
                "stability=None")

    dec = pairwise_decisions(datasets, trees, n_hvg=n_hvg,
                             base_seed=base_seed, n_boot=n_boot,
                             **walk_kwargs)
    cands = candidate_groups(dec, trees)
    bb = greedy_backbone(cands, trees, datasets,
                         selections=dec["selections"],
                         stability=stability, frozen=frozen)

    # ---- certification output: the ONLY judgments assembly consumes ----
    keys = sorted(datasets)
    canon = {k: canonical_nodes(trees[k])[1] for k in keys}
    certified, priv = [], set()
    for nd in bb["nodes"]:
        if nd["status"] == "backbone" and len(nd["members"]) >= 2:
            certified.append({"members": dict(nd["members"]),
                              "support": float(nd["mean_boot_support"]),
                              "id": nd["id"]})
        elif nd["status"] == "private":
            (pds, pnode), = nd["members"].items()
            sub = nd.get("subtree_parent") or {pnode: None}
            priv.update((pds, x) for x in sub)

    sel_str = {f"{i}>{j}": recs
               for (i, j), recs in dec["selections"].items()}
    g = quotient_assemble(
        {k: {"parent": trees[k]["parent"],
             "children": trees[k]["children"],
             "leaves": list(trees[k]["leaves"])} for k in keys},
        canon, sel_str, certified=certified)

    # ---- tree records over quotient vertices --------------------------
    vids = sorted(g["vertices"], key=str)
    vid2id = {r: f"MA-Q{k + 1:04d}" for k, r in enumerate(vids)}
    cert_recs = [(frozenset(c["members"].items()), c)
                 for c in certified]
    nodes = {}
    for r in vids:
        v = g["vertices"][r]
        mem = dict(v["members"])
        mem_set = set(mem.items())
        aliases = sorted(mem.values())
        contributing = [c for fs, c in cert_recs if fs <= mem_set]
        if v["shared"]:
            status, stype = "backbone", "cross_atlas"
            support = max((c["support"] for c in contributing),
                          default=None)
        else:
            (ds1, m1), = mem.items()
            status = "private" if (ds1, m1) in priv else "single_atlas"
            support, stype = None, "input_topology"
        pids = sorted(vid2id[p] for p in g["parents"][r])
        nodes[vid2id[r]] = {
            # NO plain "parent" key: `parents` is the authoritative
            # (possibly multi-valued) minimal-parent set, and
            # `projected_parent` is the first minimal parent — a
            # DISPLAY PROJECTION when `conflicting`, never the result
            "projected_parent": pids[0] if pids else None,
            "parents": pids,
            "conflicting": bool(g["statuses"][r]["conflicting"]),
            "status": status, "members": mem, "aliases": aliases,
            "display": _display(aliases),
            "support": support, "support_type": stype,
            "certified_ids": [c["id"] for c in contributing],
            "subtree_parent": None,
        }

    children = {i: [] for i in nodes}
    roots = []
    for i, nd in nodes.items():
        if nd["projected_parent"] is None:
            roots.append(i)
        else:
            children[nd["projected_parent"]].append(i)
    for k in children:
        children[k].sort(key=lambda i: (nodes[i]["status"],
                                        nodes[i]["display"]))
    for i, nd in nodes.items():
        nd["children"] = children[i]

    # completeness (I1 also asserted inside quotient_assemble)
    placed = {(ds, m) for nd in nodes.values()
              for ds, m in nd["members"].items()}
    for ds in keys:
        missing = [l for l in trees[ds]["leaves"]
                   if (ds, l) not in placed]
        assert not missing, \
            f"completeness invariant violated for {ds}: {missing}"

    certificates = [{"node": vid2id[c["vertex"]],
                     "minimal_parents": [vid2id[p]
                                         for p in c["minimal_parents"]]}
                    for c in g["certificates"]]
    annotations = [dict(a, source_id=vid2id[a["source_vid"]],
                        target_id=vid2id[a["target_vid"]])
                   for a in g["annotations"]]
    bbid2new = {}
    for nid, nd in nodes.items():
        for cid in nd["certified_ids"]:
            bbid2new[cid] = nid
    affiliates = [dict(a, attached_to=bbid2new.get(a["attached_to"],
                                                   a["attached_to"]))
                  for a in bb["affiliates"]]

    return {"tree": nodes, "roots": sorted(roots),
            "is_forest": g["is_forest"], "certificates": certificates,
            "annotations": annotations, "ledger": g["ledger"],
            "quotient": g,
            "decisions": dec, "candidates": cands, "backbone": bb,
            "conflicts": bb["conflicts"], "affiliates": affiliates,
            # internal nodes whose descendants matched but which
            # themselves formed no candidate: the topological signature of
            # incompatible grouping layers (they resolve as polytomies)
            "unplaced_internals": cands["unresolved_internals"],
            "frozen": dict(frozen)}
