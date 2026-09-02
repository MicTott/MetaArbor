"""Trees over atlas terminal populations, and refinement-invariant marginals.

A tree is {"parent": {id: parent_id or None}, "children": {id: [ids]},
"leaves": [leaf ids]}. Built from taxonomy level columns (coarse -> fine),
internal ids "level:label", root id "root" - identical id scheme to the R
companion so selections are directly comparable.
"""
from __future__ import annotations


def tree_from_levels(rows, level_names):
    """`rows`: iterable of tuples ordered coarse -> fine (last item = leaf
    label); `level_names`: names for each column."""
    rows = sorted(set(tuple(r) for r in rows))
    n_lev = len(level_names)
    parent = {"root": None}
    for r in rows:
        for k in range(n_lev):
            is_last = k == n_lev - 1
            node = r[k] if is_last else f"{level_names[k]}:{r[k]}"
            par = "root" if k == 0 else f"{level_names[k-1]}:{r[k-1]}"
            if node in parent and parent[node] != par:
                raise ValueError(f"{node} maps to multiple parents")
            parent[node] = par
    children = {i: [] for i in parent}
    for i, p in parent.items():
        if p is not None:
            children[p].append(i)
    for k in children:
        children[k].sort()
    leaves = [i for i in parent if not children[i]]
    return {"parent": parent, "children": children, "leaves": sorted(leaves)}


def leaves_under(tree, node):
    if not tree["children"][node]:
        return [node]
    out, stack = [], list(tree["children"][node])
    while stack:
        x = stack.pop(0)
        if tree["children"][x]:
            stack.extend(tree["children"][x])
        else:
            out.append(x)
    return sorted(out)


def ancestors(tree, node):
    out = []
    while tree["parent"][node] is not None:
        node = tree["parent"][node]
        out.append(node)
    return out


def leaf_path_dist(tree):
    """Hop-distance matrix between leaves (FUGW structure input)."""
    import numpy as np
    lv = tree["leaves"]
    paths = {l: [l] + ancestors(tree, l) for l in lv}
    d = np.zeros((len(lv), len(lv)))
    for a in range(len(lv)):
        pa = paths[lv[a]]
        for b in range(a):
            pb = paths[lv[b]]
            common = next(x for x in pa if x in set(pb))
            d[a, b] = d[b, a] = pa.index(common) + pb.index(common)
    return d, lv


def tree_weights(tree, all_nodes=False):
    """Refinement-invariant marginals (NOTES.md item 14): mass 1 at the
    root, split equally at every internal node. A pure function of ONE tree:
    independent of any paired atlas, invariant to label names, and refining
    a leaf into k children redistributes its branch mass without changing
    the branch total - annotation resolution never buys transport capacity.
    Siblings are treated as equally important conceptual units, not equally
    abundant populations (a deliberate, stated modeling choice)."""
    w = {"root": 1.0}
    queue = ["root"]
    while queue:
        v = queue.pop(0)
        kids = tree["children"][v]
        for c in kids:
            w[c] = w[v] / len(kids)
        queue.extend(kids)
    if all_nodes:
        return w
    return {l: w[l] for l in tree["leaves"]}
