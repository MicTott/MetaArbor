"""Consensus cuts — derived VIEWS over a reconciled hierarchy.

The harmonize() output is a PROVENANCE object: by the completeness
invariant it carries every input label of every atlas, so it is
necessarily larger than any single atlas. That is the audit ledger,
not the taxonomy a user annotates with. A consensus cut derives the
usable taxonomy at a chosen evidence level, the way one reads a
multi-resolution atlas at a chosen depth. Nothing here adds evidence
or changes any acceptance rule — cuts are pure filters/collapses of an
existing assembly, and every input label remains accounted for (kept,
alias-merged, or listed in the unresolved ledger; never dropped).

Evidence tiers (from node status; a future directional-containment
status would flow through as its own tier without rework):

  certified       backbone meta-clade with members from >= 2 atlases —
                  the cross-atlas-supported vocabulary
  atlas_specific  private clade (stability + powered absence elsewhere)
  provisional     single-atlas placement (existence certain in its own
                  atlas; cross-atlas correspondence unresolved)
  unresolved      unplaced_single_atlas (rejected-claim fallbacks and
                  tripwire repairs) — moved to a side ledger, with
                  rejection reasons when provenance rows are supplied

Levels:
  "strict"    certified nodes only; everything else summarized in a
              remainder grouped by nearest certified ancestor.
  "default"   certified + NAMED atlas_specific/provisional detail
              nested beneath its nearest kept ancestor; anonymous
              single-atlas scaffolding (inferred n## interiors) is
              spliced out; unplaced go to the unresolved ledger.
  "complete"  the full assembly, tiers annotated (audit view).

`consensus_cut` accepts a harmonize() result dict, its "tree" mapping,
or the serialized tree.json written by examples/harmonize_k3.py.
`cut_to_tree` converts a cut into a standard metaarbor tree dict
(parent/children/leaves) usable with to_newick or as a projection
reference tree.
"""
from __future__ import annotations

import re

_INTERNAL_ID = re.compile(r"^(?:[^|]+\|)?n\d+$")


def _is_internal_name(s):
    """Heuristic for inferred/curated INTERNAL node names: infer_tree's
    neutral ids ('n07', 'ds|n07') and tree_from_levels' level-prefixed
    ids ('family:A|F1'). Supply leaf_labels to consensus_cut for exact
    control when atlas labels could collide with these patterns."""
    s = str(s)
    if _INTERNAL_ID.match(s):
        return True
    head = s.split("|", 1)[0]
    return ":" in head and not s.startswith("≈")


def _norm_nodes(obj):
    nodes = obj.get("tree", obj) if isinstance(obj, dict) else obj
    out = {}
    for i, nd in nodes.items():
        out[i] = {"parent": nd.get("parent"),
                  "status": nd.get("status", ""),
                  "members": dict(nd.get("members", {})),
                  "aliases": list(nd.get("aliases", [])),
                  "display": nd.get("display", str(i)),
                  "assembly_repair": bool(nd.get("assembly_repair"))}
    return out


def _named(nd, leaf_labels):
    """Does this node carry at least one ORIGINAL atlas label?"""
    for ds, m in nd["members"].items():
        if leaf_labels is not None:
            pool = (leaf_labels.get(ds, ()) if isinstance(
                leaf_labels, dict) else leaf_labels)
            if m in pool:
                return True
        elif not _is_internal_name(m):
            return True
    return False


def _tier(nd):
    st = nd["status"]
    if st == "backbone" and len(nd["members"]) >= 2:
        return "certified"
    if st == "backbone":
        return "provisional"          # single-member backbone (rare)
    if st == "private":
        return "atlas_specific"
    if st == "single_atlas":
        return "provisional"
    if st == "unplaced_single_atlas":
        return "unresolved"
    return "provisional"


def consensus_cut(harm_or_nodes, level="default", leaf_labels=None,
                  provenance_rows=None):
    """Derive the consensus taxonomy view at `level`.

    harm_or_nodes: harmonize() result, its tree mapping, or the
                   serialized tree.json content.
    leaf_labels:   optional exact original-label sets — flat set of
                   member strings or {dataset: set}; default uses the
                   internal-name heuristic (see _is_internal_name).
    provenance_rows: optional iterable of dicts (provenance.csv rows)
                   used only to attach rejection reasons to the
                   unresolved ledger.

    Returns dict:
      nodes        {id: {parent, children, display, aliases, members,
                    tier, status, n_datasets}} — the kept view
      roots        top-level ids
      unresolved   [{node_id, display, members, reason}] side ledger
      alias_index  {original member label -> kept node id} for every
                   label that remains in the taxonomy (aliases of
                   merged equivalences map to their meta-clade)
      summary      counts by tier, per-atlas label accounting,
                   top_level_count, n_collapsed_anonymous
      level
    Every input label is accounted for: alias_index + unresolved
    together cover all original labels (asserted).
    """
    if level not in ("strict", "default", "complete"):
        raise ValueError(f"unknown level {level!r}")
    nodes = _norm_nodes(harm_or_nodes)
    tiers = {i: _tier(nd) for i, nd in nodes.items()}

    reasons = {}
    for row in provenance_rows or []:
        if row.get("rejection_reason"):
            reasons[row.get("consensus_node", "")] = \
                row["rejection_reason"]

    def keep(i):
        t = tiers[i]
        if t == "unresolved":
            return False
        if level == "complete":
            return True
        if level == "strict":
            return t == "certified"
        # default: certified always; other tiers only when NAMED —
        # anonymous single-atlas scaffolding is spliced out
        return t == "certified" or _named(nodes[i], leaf_labels)

    kept = {i for i in nodes if keep(i)}

    def kept_ancestor(i):
        p = nodes[i]["parent"]
        while p is not None and p not in kept:
            p = nodes[p]["parent"]
        return p

    out_nodes = {}
    for i in sorted(kept):
        nd = nodes[i]
        out_nodes[i] = {
            "parent": kept_ancestor(i), "children": [],
            "display": nd["display"], "aliases": list(nd["aliases"]),
            "members": dict(nd["members"]), "tier": tiers[i],
            "status": nd["status"],
            "n_datasets": len(nd["members"]),
        }
    for i, nd in out_nodes.items():
        if nd["parent"] is not None:
            out_nodes[nd["parent"]]["children"].append(i)
    for nd in out_nodes.values():
        nd["children"].sort()
    roots = sorted(i for i, nd in out_nodes.items()
                   if nd["parent"] is None)

    unresolved = []
    for i, nd in nodes.items():
        if tiers[i] == "unresolved":
            unresolved.append({
                "node_id": i, "display": nd["display"],
                "members": dict(nd["members"]),
                "reason": reasons.get(
                    i, "assembly_repair" if nd["assembly_repair"]
                    else "rejected_claim"),
            })
    unresolved.sort(key=lambda r: r["node_id"])

    # ---- label accounting: every original label lands somewhere ----------
    def original_members(nd):
        out = []
        for ds, m in nd["members"].items():
            if leaf_labels is not None:
                pool = (leaf_labels.get(ds, ()) if isinstance(
                    leaf_labels, dict) else leaf_labels)
                if m in pool:
                    out.append(m)
            elif not _is_internal_name(m):
                out.append(m)
        return out

    alias_index = {}
    for i in kept:
        for m in original_members(nodes[i]):
            alias_index.setdefault(m, i)
    dropped = {}                       # labels on non-kept, non-unres.
    for i, nd in nodes.items():
        if i in kept or tiers[i] == "unresolved":
            continue
        anc = kept_ancestor(i)
        for m in original_members(nd):
            if m not in alias_index:
                dropped[m] = anc
    if level != "strict":
        missing = set(dropped)
        assert not missing, \
            f"cut lost labels (bug): {sorted(missing)[:5]}"
    else:
        # strict: labels below the certified backbone are summarized
        # into the remainder of their nearest certified ancestor
        for m, anc in dropped.items():
            unresolved.append({"node_id": anc or "", "display": m,
                               "members": {}, "reason":
                               "below_certified_backbone"})

    unres_labels = {r["display"] for r in unresolved} | {
        m for r in unresolved for m in r["members"].values()}
    all_orig = set()
    for nd in nodes.values():
        all_orig.update(original_members(nd))
    unaccounted = all_orig - set(alias_index) - unres_labels
    assert not unaccounted, \
        f"label accounting hole (bug): {sorted(unaccounted)[:5]}"

    tier_counts = {}
    for i in kept:
        tier_counts[tiers[i]] = tier_counts.get(tiers[i], 0) + 1
    per_atlas = {}
    for m, i in alias_index.items():
        for ds, mm in nodes[i]["members"].items():
            if mm == m:
                pa = per_atlas.setdefault(ds, {})
                pa[tiers[i]] = pa.get(tiers[i], 0) + 1
    return {
        "level": level, "nodes": out_nodes, "roots": roots,
        "unresolved": unresolved, "alias_index": alias_index,
        "summary": {
            "n_nodes": len(out_nodes),
            "top_level_count": len(roots) if len(roots) > 1 else
                len(out_nodes[roots[0]]["children"]) if roots else 0,
            "tier_counts": tier_counts,
            "per_atlas_label_tiers": per_atlas,
            "n_unresolved": len(unresolved),
            "n_collapsed_anonymous": len(nodes) - len(out_nodes) -
                sum(1 for i in nodes if tiers[i] == "unresolved"),
        },
    }


def cut_to_tree(cut):
    """Convert a cut into a standard metaarbor tree dict
    (parent/children/leaves, rooted at 'root') — usable with
    metaarbor.to_newick, leaves_under, or as a projection reference
    tree (labels then map to their kept node ids)."""
    parent = {"root": None}
    children = {"root": list(cut["roots"])}
    for i, nd in cut["nodes"].items():
        parent[i] = nd["parent"] if nd["parent"] is not None else "root"
        children[i] = list(nd["children"])
    leaves = [i for i, nd in cut["nodes"].items()
              if not nd["children"]]
    return {"parent": parent, "children": children, "leaves": leaves}
