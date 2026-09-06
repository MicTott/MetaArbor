"""metaarbor.projection — hierarchical reference mapping (PROTOTYPE,
`projection-prototype` branch).

Projects INDIVIDUAL CELLS into a reference hierarchy with hierarchical
abstention, reusing the frozen MetaArbor measurement layer (lognorm,
joint HVGs, per-cell rank normalization, MetaNeighbor-style rank
voting). Nothing frozen is modified.

Method
------
- Reference: one or MORE labeled atlases whose labels are leaves of one
  reference tree. References are stratified-subsampled per leaf
  (atlas-balanced evidence; a dominant atlas cannot swamp the votes).
- Evidence per query cell: rank-standardized vote weights toward every
  reference leaf (exactly `kernel.vote_cache`; parity-tested), computed
  per reference atlas and combined as COVERAGE-AWARE averages of
  size-normalized child scores — an atlas contributes to a split only
  when it carries cells for EVERY child of that split.
- Routing: each cell walks the tree root-to-leaf. At every split the
  vote is recomputed LOCALLY: the cell's correlations to the reference
  cells under that node are re-ranked among themselves (the per-cell
  analogue of the frozen Walk's parent-context principle), and each
  child scores the MAXIMUM over its leaves' mean local votes —
  navigation follows the child's most similar leaf, so a heterogeneous
  child cannot dilute its own most distinctive type (mean-union scores
  are size-biased; on the Allen benchmark mean navigation lost ~10
  points of subclass accuracy that max navigation recovers exactly).
  Global ranks would compress sibling contrasts (every leaf under the
  right family scores high globally); local re-ranking restores them.
  Two descents are tracked at once:
    * FORCED descent -> `best_leaf` (every cell gets a terminal
      candidate; `path_score` is the product of local fractions —
      an evidence score, NOT a calibrated probability);
    * CONFIDENT descent -> `resolved_node` (the walk stops the first
      time the LOCAL VOTE MARGIN — top child's mean local vote minus
      the runner-up's — drops below `min_margin`; deeper coordinates
      are not interpreted). The margin is the per-cell analogue of the
      Walk's sibling-contrast stop.
- Per-cell outputs additionally include the credible children at the
  stop, the stop confidence, an out-of-reference score
  (`top_leaf_mean`, the mean vote weight per reference cell of the best
  leaf; 0.5 = rank-random), and the maximum Spearman similarity to any
  reference cell (`max_corr`).

Relation to the frozen Walk: this is the per-cell counterpart of
`select_node` — votes navigate; the stop rule here is a per-cell local
confidence (population AUROC contrasts are undefined for n = 1).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .kernel import lognorm, rank_normalize, variable_genes
from .tree import leaves_under

DEFAULTS = {"cap_per_leaf": 50, "n_hvg": 1000, "min_margin": 0.10}


def _votes_and_maxcorr(test_norm, train_norm, train_labels, chunk=2000):
    """kernel.vote_cache's exact vote computation (parity-tested) plus
    each test cell's maximum Spearman correlation to any train cell."""
    train_labels = np.asarray(train_labels)
    leaves = sorted(set(train_labels))
    n_train = train_norm.shape[0]
    col = np.asarray([leaves.index(l) for l in train_labels])
    ind = np.zeros((n_train, len(leaves)))
    ind[np.arange(n_train), col] = 1.0
    n_test = test_norm.shape[0]
    V = np.zeros((n_test, len(leaves)))
    mc = np.zeros(n_test)
    for s in range(0, n_test, chunk):
        co = test_norm[s:s + chunk] @ train_norm.T
        mc[s:s + chunk] = co.max(axis=1)
        w = np.apply_along_axis(rankdata, 1, co) / n_train
        V[s:s + chunk] = w @ ind
    sizes = np.asarray([(train_labels == l).sum() for l in leaves],
                       dtype=float)
    return {"V": V, "leaves": leaves, "leaf_sizes": sizes,
            "max_corr": mc}


def build_projector(refs, tree, cap_per_leaf=DEFAULTS["cap_per_leaf"],
                    seed=0):
    """Build a projector from labeled reference atlases.

    refs: list of dicts {counts (cells x genes), labels (leaf labels of
          `tree`), gene_names, lib=optional, name=optional}
    tree: metaarbor tree whose leaves are the reference label space.

    Each reference is stratified-subsampled to at most `cap_per_leaf`
    cells per leaf (seeded). Labels not in the tree's leaves raise;
    leaves absent from a reference are allowed (coverage-aware splits).
    """
    leaves = set(tree["leaves"])
    rs = np.random.RandomState(seed)
    stored = []
    for ri, ref in enumerate(refs):
        labels = np.asarray(ref["labels"])
        bad = sorted(set(labels) - leaves)
        if bad:
            raise ValueError(
                f"ref {ri}: labels not in tree leaves: {bad[:5]}")
        keep = []
        for l in sorted(set(labels)):
            idx = np.flatnonzero(labels == l)
            if len(idx) > cap_per_leaf:
                idx = rs.choice(idx, cap_per_leaf, replace=False)
            keep.append(np.sort(idx))
        keep = np.concatenate(keep)
        stored.append({
            "counts": np.asarray(ref["counts"])[keep],
            "labels": labels[keep],
            "gene_names": list(ref["gene_names"]),
            "lib": (np.asarray(ref["lib"])[keep]
                    if ref.get("lib") is not None else None),
            "name": ref.get("name", f"ref{ri}"),
        })
    return {"refs": stored, "tree": tree,
            "cap_per_leaf": cap_per_leaf, "seed": seed}


def _split_index(tree, cache_leaves):
    """Per internal node: list of per-child leaf-index arrays into the
    cache's leaf order, or None when this reference does not cover
    every child of the split (coverage-aware exclusion)."""
    pos = {l: i for i, l in enumerate(cache_leaves)}
    out = {}
    for v in tree["children"]:
        kids = tree["children"][v]
        if not kids or v == "root" and False:
            continue
        if not kids:
            continue
        per_child = []
        ok = True
        for c in kids:
            idx = [pos[l] for l in leaves_under(tree, c) if l in pos]
            if not idx:
                ok = False
                break
            per_child.append(np.asarray(idx))
        out[v] = per_child if ok else None
    return out


def project(projector, counts, gene_names, lib=None,
            n_hvg=DEFAULTS["n_hvg"],
            min_margin=DEFAULTS["min_margin"], chunk=2000,
            assume_log=False):
    """Project query cells (cells x genes) into the reference tree.

    Returns dict of per-cell arrays:
      best_leaf        forced-descent terminal candidate (always set)
      path_score       product of local child fractions along the
                       forced path (evidence score, uncalibrated)
      resolved_node    deepest node reached with every local vote
                       margin >= min_margin (== best_leaf when fully
                       confident)
      resolved_depth   depth of resolved_node (root = 0)
      stop_margin      the local vote margin at the first failing split
                       (1.0 when none failed)
      credible         ';'-joined children within min_margin of the top
                       at the stop; '' if resolved
      top_leaf_mean    out-of-reference score: mean vote weight per
                       reference cell of the best leaf (0.5 = random)
      max_corr         max Spearman correlation to any reference cell
    plus 'leaves' and 'leaf_mean' (n_cells x n_leaves averaged
    size-normalized vote matrix) for flat-baseline comparisons.
    """
    tree = projector["tree"]
    q_ln = lognorm(counts, lib, assume_log)
    caches, splits, mats = [], [], []
    for ref in projector["refs"]:
        if list(ref["gene_names"]) == list(gene_names):
            qi = np.arange(len(gene_names))
            ri = np.arange(len(gene_names))
        else:
            rpos = {g: i for i, g in enumerate(ref["gene_names"])}
            common = [g for g in gene_names if g in rpos]
            if len(common) < 100:
                raise ValueError(
                    f"{ref['name']}: only {len(common)} shared genes")
            qpos = {g: i for i, g in enumerate(gene_names)}
            qi = np.asarray([qpos[g] for g in common])
            ri = np.asarray([rpos[g] for g in common])
        r_ln = lognorm(ref["counts"][:, ri] if len(ri) !=
                       ref["counts"].shape[1] else ref["counts"],
                       ref["lib"], assume_log)
        q_sub = q_ln[:, qi] if len(qi) != q_ln.shape[1] else q_ln
        hvg = variable_genes(q_sub, r_ln, list(np.asarray(
            gene_names)[qi]), n_hvg)
        qn = rank_normalize(q_sub[:, hvg])
        rn = rank_normalize(r_ln[:, hvg])
        cache = _votes_and_maxcorr(qn, rn, ref["labels"], chunk)
        caches.append(cache)
        splits.append(_split_index(tree, cache["leaves"]))
        # per-node reference-cell rows for local re-ranking
        labs = np.asarray(ref["labels"])
        rows = {}
        for v in tree["children"]:
            kids = tree["children"][v]
            if not kids:
                continue
            per_child = []
            ok = True
            for c in kids:
                blocks = []
                for l in leaves_under(tree, c):
                    idx = np.flatnonzero(labs == l)
                    if len(idx):
                        blocks.append(idx)
                if not blocks:
                    ok = False
                    break
                per_child.append(blocks)
            rows[v] = per_child if ok else None
        mats.append({"qn": qn, "rn": rn, "rows": rows})

    n = q_ln.shape[0]
    # averaged size-normalized leaf matrix over the tree's leaf order
    tree_leaves = list(tree["leaves"])
    leaf_mean = np.zeros((n, len(tree_leaves)))
    leaf_cov = np.zeros(len(tree_leaves))
    for cache in caches:
        pos = {l: i for i, l in enumerate(cache["leaves"])}
        for j, l in enumerate(tree_leaves):
            if l in pos:
                leaf_mean[:, j] += (cache["V"][:, pos[l]] /
                                    cache["leaf_sizes"][pos[l]])
                leaf_cov[j] += 1
    cov = np.maximum(leaf_cov, 1.0)
    leaf_mean = leaf_mean / cov[None, :]

    def _ranks(co):
        """dense ranks along axis 1 (argsort-of-argsort; correlation
        ties are measure-zero on float data), scaled to (0, 1]."""
        r = np.empty_like(co)
        idx = np.argsort(co, axis=1)
        r[np.arange(co.shape[0])[:, None], idx] = \
            np.arange(1, co.shape[1] + 1)[None, :]
        return r / co.shape[1]

    def child_scores(v, mask):
        """Coverage-aware, atlas-balanced per-child LOCAL vote scores
        for the cells in `mask` at node v: the cells' correlations to
        the reference cells under v are re-ranked among themselves and
        each child gets its size-normalized mean local vote. None if no
        reference covers the whole split."""
        kids = tree["children"][v]
        sub = np.flatnonzero(mask)
        acc = np.zeros((len(sub), len(kids)))
        n_cov = 0
        for m in mats:
            per_child = m["rows"].get(v)
            if per_child is None:
                continue
            n_cov += 1
            rows_v = np.concatenate([np.concatenate(b)
                                     for b in per_child])
            co = m["qn"][sub] @ m["rn"][rows_v].T
            w = _ranks(co)
            off = 0
            for ci, blocks in enumerate(per_child):
                best = None
                for idx in blocks:
                    mvote = w[:, off:off + len(idx)].mean(axis=1)
                    best = mvote if best is None else \
                        np.maximum(best, mvote)
                    off += len(idx)
                acc[:, ci] += best
        if n_cov == 0:
            return None
        return acc / n_cov

    cur = np.full(n, "root", dtype=object)
    best_leaf = np.full(n, "", dtype=object)
    resolved = np.full(n, "root", dtype=object)
    broken = np.zeros(n, dtype=bool)
    stop_conf = np.ones(n)
    credible = np.full(n, "", dtype=object)
    path_score = np.ones(n)
    depth_of = {"root": 0}

    # topological order (parents first)
    order, stack = [], ["root"]
    while stack:
        v = stack.pop(0)
        order.append(v)
        for c in tree["children"].get(v, []):
            depth_of[c] = depth_of[v] + 1
            stack.append(c)

    for v in order:
        kids = tree["children"].get(v, [])
        mask = np.asarray(cur == v)
        if not kids or not mask.any():
            if not kids and mask.any():
                best_leaf[mask] = v
            continue
        S = child_scores(v, mask)
        sub = np.flatnonzero(mask)
        if S is None:
            # no reference resolves this split: stop here; best_leaf
            # falls back to the strongest leaf below by leaf_mean
            below = [tree_leaves.index(l) for l in leaves_under(tree, v)]
            best_leaf[sub] = np.asarray(tree_leaves, dtype=object)[
                np.asarray(below)[np.argmax(leaf_mean[np.ix_(
                    sub, below)], axis=1)]]
            stop_conf[sub] = np.where(broken[sub], stop_conf[sub], 0.0)
            broken[sub] = True
            continue
        # LOCAL VOTE MARGIN: top child's mean local vote minus the
        # runner-up's. A subtype-true cell separates strongly (~0.5+);
        # a cell carrying no signal at this split sits near 0. Ratios
        # of clipped excesses are avoided (a tiny excess over a zero
        # sibling must not look confident).
        orderk = np.argsort(-S, axis=1)
        top = orderk[:, 0]
        s1 = S[np.arange(len(sub)), top]
        s2 = (S[np.arange(len(sub)), orderk[:, 1]]
              if S.shape[1] > 1 else np.zeros(len(sub)))
        conf = s1 - s2                       # the margin
        E = np.clip(S - 0.5, 0.0, None)
        tot = E.sum(axis=1)
        e1 = E[np.arange(len(sub)), top]
        frac = np.where(tot > 0, e1 / np.maximum(tot, 1e-300), 0.0)
        path_score[sub] *= frac
        newly = (~broken[sub]) & (conf < min_margin)
        if newly.any():
            nb = sub[newly]
            resolved[nb] = v
            stop_conf[nb] = conf[newly]
            kid_arr = np.asarray(kids, dtype=object)
            for x, row in zip(np.flatnonzero(newly), nb):
                cred = kid_arr[S[x] >= s1[x] - min_margin]
                credible[row] = ";".join(map(str, cred))
            broken[nb] = True
        keep_going = ~broken[sub]
        resolved[sub[keep_going]] = np.asarray(
            kids, dtype=object)[top[keep_going]]
        cur[sub] = np.asarray(kids, dtype=object)[top]

    unassigned = best_leaf == ""
    if unassigned.any():                       # safety (shouldn't occur)
        best_leaf[unassigned] = np.asarray(tree_leaves, dtype=object)[
            np.argmax(leaf_mean[unassigned], axis=1)]

    return {
        "best_leaf": best_leaf.astype(str),
        "path_score": path_score,
        "resolved_node": resolved.astype(str),
        "resolved_depth": np.asarray([depth_of.get(r, 0)
                                      for r in resolved]),
        "stop_margin": stop_conf,
        "credible": credible.astype(str),
        "top_leaf_mean": leaf_mean.max(axis=1),
        "max_corr": np.mean([c["max_corr"] for c in caches], axis=0),
        "leaves": tree_leaves,
        "leaf_mean": leaf_mean,
        "params": {"n_hvg": n_hvg, "min_margin": min_margin,
                   "n_refs": len(caches)},
    }
