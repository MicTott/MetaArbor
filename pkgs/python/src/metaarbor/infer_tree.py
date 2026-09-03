"""Within-atlas hierarchy inference for flat label sets.

MetaArbor aligns trees; it must not silently convert flat labels into a
star (root -> every label), which discards transcriptomic relationships,
denies Walk any candidate family node, and makes "discordant" mean "no
family was ever constructed" rather than "biologically incoherent."

Tree policy (explicit, enforced by this module):
  - supplied curated tree            -> use it (tree_from_levels)
  - flat labels, no tree             -> INFER a tree by default (this
                                        module)
  - star representation              -> only by explicit request
                                        (`star_tree`), reported as the
                                        flat/star baseline
  - too little information to infer  -> error, never a silent star

Prespecified builder (frozen before real cross-atlas use): pseudobulk per
label -> atlas-own HVGs -> ranked-expression chord distances -> average
linkage (UPGMA) -> Felsenstein cell-bootstrap clade support -> splits
below the support floor collapse into polytomies -> neutral internal IDs
(n01, n02, ... deterministic by clade size then leaf names; never borrow
names from another atlas). Returns the tree plus per-node support and
provenance.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .branch_fit import pseudobulk_distances


def star_tree(labels):
    """Explicit flat/star baseline representation: root -> every label.
    Use only deliberately; alignment against a star can make leaf calls
    but can never produce family calls."""
    labels = sorted(set(map(str, labels)))
    if len(labels) < 2:
        raise ValueError("need at least two labels")
    parent = {"root": None, **{l: "root" for l in labels}}
    children = {"root": labels, **{l: [] for l in labels}}
    return {"parent": parent, "children": children, "leaves": labels}


def is_star_tree(tree):
    """True when every child of the root is a leaf (no internal nodes)."""
    return all(not tree["children"][c] for c in tree["children"]["root"])


def _clades(Z, leaves):
    """Clades (as frozensets of leaf names) from a scipy linkage matrix."""
    n = len(leaves)
    sets = {i: frozenset([leaves[i]]) for i in range(n)}
    out = []
    for k, (a, b, _, _) in enumerate(Z):
        s = sets[int(a)] | sets[int(b)]
        sets[n + k] = s
        out.append(s)
    return out


def infer_tree(counts, labels, lib=None, n_hvg=2000, n_boot=100,
               support_floor=0.7, seed=0, assume_log=False):
    """Infer a within-atlas hierarchy over `labels` from `counts`
    (cells x genes) using ONLY this atlas's expression.

    Returns dict(tree, support: {node_id: bootstrap proportion},
    clades: {node_id: sorted leaf list}, provenance).
    """
    labels = np.asarray([str(l) for l in labels])
    uniq = sorted(set(labels))
    if len(uniq) < 3:
        raise ValueError(
            f"only {len(uniq)} labels — too little information to infer a "
            "hierarchy; supply a tree or request star_tree() explicitly")
    D, leaves = pseudobulk_distances(counts, labels, lib=lib, n_hvg=n_hvg,
                                     assume_log=assume_log)
    Z = linkage(squareform(D, checks=False), method="average")
    point = [c for c in _clades(Z, leaves) if 1 < len(c) < len(leaves)]

    rs = np.random.RandomState(seed)
    idx_by = {l: np.flatnonzero(labels == l) for l in leaves}
    counts_hits = {c: 0 for c in point}
    for _ in range(n_boot):
        take = np.concatenate([rs.choice(ix, len(ix), replace=True)
                               for ix in idx_by.values()])
        Db, lb = pseudobulk_distances(counts[take], labels[take],
                                      lib=None if lib is None else
                                      np.asarray(lib)[take],
                                      n_hvg=n_hvg, assume_log=assume_log)
        boot_set = set(c for c in _clades(
            linkage(squareform(Db, checks=False), method="average"), lb)
            if 1 < len(c) < len(lb))
        for c in counts_hits:
            if c in boot_set:
                counts_hits[c] += 1
    support = {c: counts_hits[c] / n_boot for c in point}
    kept = [c for c in point if support[c] >= support_floor]

    # laminar family -> tree with polytomies; deterministic neutral IDs
    kept.sort(key=lambda c: (len(c), tuple(sorted(c))))
    ids = {c: f"n{k+1:02d}" for k, c in enumerate(kept)}
    all_set = frozenset(leaves)

    def parent_of(s):
        supersets = [c for c in kept if s < c]
        if not supersets:
            return "root"
        return ids[min(supersets, key=len)]

    parent = {"root": None}
    children = {"root": []}
    for c in kept:
        parent[ids[c]] = parent_of(c)
        children[ids[c]] = []
    for l in leaves:
        parent[l] = parent_of(frozenset([l]))
        children[l] = []
    for node, p in parent.items():
        if p is not None:
            children[p].append(node)
    for k in children:
        children[k].sort()
    tree = {"parent": parent, "children": children, "leaves": list(leaves)}
    return {"tree": tree,
            "support": {ids[c]: support[c] for c in kept},
            "all_candidate_support": {tuple(sorted(c)): s
                                      for c, s in support.items()},
            "clades": {ids[c]: sorted(c) for c in kept},
            "provenance": {"method": "pseudobulk-chord-UPGMA",
                           "n_hvg": n_hvg, "n_boot": n_boot,
                           "support_floor": support_floor, "seed": seed,
                           "n_labels": len(uniq),
                           "n_internal_kept": len(kept)}}


def annotate_star_relations(summary_rows, tree):
    """Interpretation annotation (estimator outputs unchanged): on a star
    target, 'discordant' usually means the distributed signal had no
    candidate family node — report it as distributed_no_target_clade."""
    if not is_star_tree(tree):
        return summary_rows
    out = []
    for r in summary_rows:
        r = dict(r)
        for key in ("walk_relation", "relation"):
            if r.get(key) == "discordant":
                r[key + "_note"] = "distributed_no_target_clade"
        out.append(r)
    return out
