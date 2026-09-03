"""Ancestry-poset compatibility (DESIGN.md step 5).

A meta-clade holds at most one node per source tree. For two meta-clades
the per-tree relation between their members is ancestor / descendant /
disjoint (or equal). A candidate SET is compatible iff every pair's
relation agrees across all trees where both have members, and the induced
relations form a laminar family (no interleaving). Disagreements are
emitted as conflict records, never silently resolved.
"""
from __future__ import annotations

from metaarbor import ancestors, leaves_under


def relation(tree, a, b):
    """Relation of node a to node b within one tree."""
    if a == b:
        return "equal"
    if a in ancestors(tree, b):
        return "ancestor"
    if b in ancestors(tree, a):
        return "descendant"
    la, lb = set(leaves_under(tree, a)), set(leaves_under(tree, b))
    return "disjoint" if not (la & lb) else "interleaved"


def pair_relation(m1, m2, trees):
    """Consensus relation of meta-clades m1, m2 (dicts: tree_key -> node).
    Returns (relation, conflicts): relation is the agreed value or None;
    conflicts lists (tree_key, seen_relation) when trees disagree."""
    seen = {}
    for k in set(m1) & set(m2):
        seen[k] = relation(trees[k], m1[k], m2[k])
    vals = set(seen.values())
    if not vals:
        return None, []                    # no co-occurring tree: unrelated
    if vals <= {"equal"}:
        return "equal", []
    if len(vals) == 1:
        return vals.pop(), []
    return None, sorted(seen.items())


def compatible(accepted, cand, trees):
    """Is candidate meta-clade `cand` ancestry-compatible with every
    accepted meta-clade? Returns (ok, conflicts)."""
    conflicts = []
    for other in accepted:
        rel, disagreement = pair_relation(cand, other, trees)
        if disagreement:
            conflicts.append({"other": other, "disagreement": disagreement})
        elif rel == "interleaved":
            conflicts.append({"other": other,
                              "disagreement": [("*", "interleaved")]})
    return (not conflicts), conflicts
