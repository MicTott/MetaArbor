"""metaarbor.projection — hierarchical reference mapping (PROTOTYPE,
`projection-prototype` branch).

Projects INDIVIDUAL CELLS into a reference hierarchy with hierarchical
abstention. The projector is FITTED at build time (reference-only
feature selection and preprocessing); a cell's result depends only on
the cell and the projector — never on which other cells accompany it.

Evidence model (per split, per reference atlas)
-----------------------------------------------
At node v the cell's Spearman correlations to v's own reference cells
are re-ranked among themselves with tie-AVERAGE ranks (the per-cell
analogue of the frozen Walk's parent-context principle; global ranks
compress sibling contrasts). Each reference LABEL under a child
contributes one block; a block's statistic is the exact-null z-score of
its mean local rank vote (mean 0.5; variance of a without-replacement
mean of m draws from the finite rank population — no distributional
assumption enters the mean/variance; a normal tail approximates the
p-value, evaluated in LOG space so no signal strength saturates
float64). A child's evidence is its Bonferroni-corrected best block on
the log scale, lp_c = min_block log p + log n_blocks, so a child with
ten labels gets no multiplicity advantage over a child with one — the
statistic is refinement-calibrated (an uncorrected max over leaves
forces cells into leaf-rich branches even under a pure null).
Navigation follows the smallest lp (averaged over the reference
atlases covering the split); the stop rule uses the RATIO margin

    margin = 1 - p_best / p_second = -expm1(lp1 - lp2)

which is null-uniform on [0, 1] for any number of children (the ratio
of the two smallest of k iid uniforms is itself uniform) and, unlike
(1-p)^n scores, never saturates: two overwhelming-but-unequal
children keep a large margin.

Reference labels may sit at INTERNAL nodes of the tree (a coarse
atlas's labels in a reconciled hierarchy): such cells inform every
split ABOVE their node and are excluded from splits at/below it —
partially observed subtrees, not wrong fine labels.
`from_harmonize(harm)` adapts a `metaarbor.harmonize()` result into
(tree, per-dataset label->node maps) for direct use as a reference.

Outputs are labeled for what they are: evidence scores and candidate
sets — not calibrated probabilities and not statistical credible sets.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.stats import norm, rankdata

from .kernel import lognorm
from .tree import leaves_under

DEFAULTS = {"cap_per_label": 50, "n_hvg": 1000, "min_margin": 0.99,
            "block": 20000}


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def _dense(x):
    return np.asarray(x.todense() if sp.issparse(x) else x,
                      dtype=np.float64)


def build_projector(refs, tree, label_maps=None,
                    cap_per_label=DEFAULTS["cap_per_label"],
                    n_hvg=DEFAULTS["n_hvg"], seed=0):
    """Fit a projector from labeled reference atlases.

    refs: list of dicts {counts (cells x genes; dense or scipy sparse),
          labels, gene_names, lib=optional, name=optional}
    tree: metaarbor tree. Reference labels must map to tree NODES —
          leaves by default, or via label_maps[i][label] -> node_id
          (internal nodes allowed: coarse labels in a reconciled tree).

    Fitting freezes, per reference: a stratified subsample (at most
    `cap_per_label` cells per label), reference-only HVGs (top-variance
    genes of the reference alone), and the log-normalized HVG matrix.
    Nothing about any future query enters the fit.
    """
    import zlib
    nodes_all = set(tree["parent"]) | {"root"}
    stored = []
    for ri, ref in enumerate(refs):
        # per-reference stream keyed by the reference NAME, so results
        # are invariant to the order references are listed in (unnamed
        # references fall back to their position)
        name = ref.get("name", f"ref{ri}")
        rs = np.random.RandomState(
            (seed + zlib.crc32(str(name).encode())) % (2 ** 31 - 1))
        labels = np.asarray(ref["labels"])
        lmap = (label_maps[ri] if label_maps else
                {l: l for l in set(labels)})
        bad = sorted(l for l in set(labels)
                     if lmap.get(l, l) not in nodes_all)
        if bad:
            raise ValueError(
                f"ref {ri}: labels map to no tree node: {bad[:5]}")
        gn = list(ref["gene_names"])
        counts = ref["counts"]
        if len(gn) != counts.shape[1]:
            raise ValueError(f"ref {ri}: {len(gn)} gene_names for "
                             f"{counts.shape[1]} columns")
        if len(set(gn)) != len(gn):
            raise ValueError(f"ref {ri}: duplicate gene_names")
        keep = []
        for l in sorted(set(labels)):
            idx = np.flatnonzero(labels == l)
            if len(idx) > cap_per_label:
                idx = rs.choice(idx, cap_per_label, replace=False)
            keep.append(np.sort(idx))
        keep = np.concatenate(keep)
        sub = counts.tocsr()[keep] if sp.issparse(counts) else \
            np.asarray(counts)[keep]
        lib = (np.asarray(ref["lib"])[keep]
               if ref.get("lib") is not None else None)
        ln = lognorm(_dense(sub), lib)
        hvg = np.sort(np.argsort(ln.var(axis=0))[::-1][:n_hvg])
        stored.append({
            "ln_hvg": ln[:, hvg],
            "hvg_names": [gn[i] for i in hvg],
            "labels": labels[keep],
            "label_nodes": {l: lmap.get(l, l)
                            for l in sorted(set(labels))},
            "name": ref.get("name", f"ref{ri}"),
        })
    return {"refs": stored, "tree": tree, "cap_per_label": cap_per_label,
            "n_hvg": n_hvg, "seed": seed}


def from_harmonize(harm):
    """Adapt a metaarbor.harmonize() result into (tree, label_maps by
    dataset): a projector-ready tree over the reconciled node ids plus,
    per dataset, each original label's node. Coarse labels land on
    internal nodes; unplaced labels on their unplaced nodes."""
    nodes = harm["tree"]
    parent = {"root": None}
    children = {"root": []}
    for i, nd in nodes.items():
        parent[i] = nd["parent"] if nd["parent"] is not None else "root"
        children.setdefault(i, [])
    for i in nodes:
        children.setdefault(parent[i], []).append(i)
    for k in children:
        children[k].sort()
    leaves = [i for i in sorted(nodes) if not children.get(i)]
    tree = {"parent": parent, "children": children, "leaves": leaves}
    label_maps = {}
    for i, nd in nodes.items():
        for ds, member in nd["members"].items():
            label_maps.setdefault(ds, {})[member] = i
    return tree, label_maps


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _rank_norm(expr):
    """Tie-average Spearman preparation (kernel.rank_normalize
    semantics, vectorized): per-cell average ranks, centered,
    L2-normalized, so qn @ rn.T is cell-cell Spearman correlation."""
    r = rankdata(expr, axis=1, method="average")
    r = r - r.mean(axis=1, keepdims=True)
    n = np.linalg.norm(r, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return r / n


def _null_sigma(m, n):
    """Exact s.d. of the mean of m ranks drawn WITHOUT replacement from
    the scaled rank population {1..n}/n (mean 0.5)."""
    if n <= 1 or m <= 0:
        return np.inf
    pop_var = (n * n - 1.0) / (12.0 * n * n)
    return float(np.sqrt(pop_var / m * max(n - m, 0) / max(n - 1, 1)))


def _prepare_split_blocks(tree, ref):
    """Per internal node: list per child of [(label, ref-row idx)]
    blocks. A reference label maps to a tree node; its cells form a
    block under the child whose subtree contains that node for every
    split strictly ABOVE the node, and carry no signal at or below."""
    node_of = ref["label_nodes"]
    labels = np.asarray(ref["labels"])
    out = {}
    for v in tree["children"]:
        kids = tree["children"][v]
        if not kids:
            continue
        per_child = []
        for c in kids:
            span = {c}
            stack = list(tree["children"].get(c, []))
            while stack:
                x = stack.pop()
                span.add(x)
                stack.extend(tree["children"].get(x, []))
            blocks = []
            for l in sorted(node_of):
                if node_of[l] in span:
                    idx = np.flatnonzero(labels == l)
                    if len(idx):
                        blocks.append((l, idx))
            per_child.append(blocks)
        out[v] = per_child
    return out


def _score_children(qn_sub, rn, per_child):
    """Refinement-calibrated child evidence for one reference at one
    split: local tie-average ranks over the split's reference cells,
    exact-null z per label block, Bonferroni-corrected best block per
    child ON THE LOG SCALE (norm.logsf — no float64 saturation at any
    signal strength). Returns lp (n_cells x n_children; SMALLER = more
    evidence; NaN for a child this reference does not cover), or None
    when fewer than two children carry blocks."""
    covered = [ci for ci, blocks in enumerate(per_child) if blocks]
    if len(covered) < 2:
        return None
    rows = np.concatenate([idx for ci in covered
                           for _l, idx in per_child[ci]])
    n_v = len(rows)
    co = qn_sub @ rn[rows].T
    w = rankdata(co, axis=1, method="average") / n_v
    lp = np.full((qn_sub.shape[0], len(per_child)), np.nan)
    off = 0
    for ci in covered:
        best = None
        for _l, idx in per_child[ci]:
            m = len(idx)
            s = w[:, off:off + m].mean(axis=1)
            z = (s - 0.5) / _null_sigma(m, n_v)
            l_ = norm.logsf(z)
            best = l_ if best is None else np.minimum(best, l_)
            off += m
        lp[:, ci] = np.minimum(best + np.log(len(per_child[ci])), 0.0)
    return lp


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def project(projector, counts, gene_names, lib=None,
            min_margin=DEFAULTS["min_margin"],
            block=DEFAULTS["block"], assume_log=False):
    """Project query cells into the reference tree.

    Per-cell outputs (dict of arrays):
      best_leaf         terminal node of the forced descent (always)
      resolved_node     deepest node with every q-margin >= min_margin
      resolved_depth    its depth (root = 0)
      stop_margin       q-margin at the first failing split (NaN when
                        fully resolved — no split failed)
      stop_candidates   ';'-joined children within min_margin of the
                        top at the stop (a candidate set, NOT a
                        statistical credible set); '' when resolved
      path_margins      (n_cells x max_depth) q-margin at each split of
                        the forced path, NaN elsewhere — re-threshold
                        offline for coverage-risk curves
      max_label_vote    max global mean vote over reference labels
                        (out-of-reference evidence; its null level
                        GROWS with the number of labels)
      mean_max_corr     mean over reference atlases of the cell's max
                        Spearman correlation to any reference cell
      label_vote        (n_cells x n_labels) global vote matrix, with
                        'labels' (flat-baseline comparisons)

    Query cells are processed in blocks of `block` rows (memory
    bounded); every per-cell quantity is independent of the other
    cells in the call by construction.
    """
    tree = projector["tree"]
    refs = projector["refs"]
    qpos = {g: i for i, g in enumerate(gene_names)}
    panels = []
    for ref in refs:
        pr = [(i, qpos[g]) for i, g in enumerate(ref["hvg_names"])
              if g in qpos]
        if len(pr) < 100:
            raise ValueError(f"{ref['name']}: only {len(pr)} of its "
                             "fitted HVGs present in the query genes")
        ridx = np.asarray([i for i, _ in pr])
        qidx = np.asarray([j for _, j in pr])
        rn = _rank_norm(ref["ln_hvg"][:, ridx])
        panels.append({"qidx": qidx, "rn": rn,
                       "blocks": _prepare_split_blocks(tree, ref),
                       "labels_sorted": sorted(ref["label_nodes"])})
    all_labels = sorted({l for ref in refs for l in ref["label_nodes"]})
    lab_pos = {l: i for i, l in enumerate(all_labels)}

    order, depth_of, stack = [], {"root": 0}, ["root"]
    while stack:
        v = stack.pop(0)
        order.append(v)
        for c in tree["children"].get(v, []):
            depth_of[c] = depth_of[v] + 1
            stack.append(c)
    max_depth = max(depth_of.values()) if depth_of else 1

    outs = []
    counts = counts.tocsr() if sp.issparse(counts) else counts
    n_total = counts.shape[0]
    for s0 in range(0, n_total, block):
        sl = slice(s0, min(s0 + block, n_total))
        q_ln = lognorm(_dense(counts[sl]),
                       None if lib is None else np.asarray(lib)[sl],
                       assume_log)
        n = q_ln.shape[0]
        qns = [_rank_norm(q_ln[:, p["qidx"]]) for p in panels]

        label_vote = np.zeros((n, len(all_labels)))
        label_cov = np.zeros(len(all_labels))
        mean_mc = np.zeros(n)
        for ref, p, qn in zip(refs, panels, qns):
            labs = np.asarray(ref["labels"])
            co = qn @ p["rn"].T
            mean_mc += co.max(axis=1) / len(refs)
            w = rankdata(co, axis=1, method="average") / co.shape[1]
            for l in p["labels_sorted"]:
                idx = np.flatnonzero(labs == l)
                label_vote[:, lab_pos[l]] += w[:, idx].mean(axis=1)
                label_cov[lab_pos[l]] += 1
        label_vote /= np.maximum(label_cov, 1)[None, :]

        cur = np.full(n, "root", dtype=object)
        best_leaf = np.full(n, "", dtype=object)
        resolved = np.full(n, "root", dtype=object)
        broken = np.zeros(n, dtype=bool)
        stop_margin = np.full(n, np.nan)
        stop_cand = np.full(n, "", dtype=object)
        path_margins = np.full((n, max_depth), np.nan)

        for v in order:
            kids = tree["children"].get(v, [])
            mask = np.asarray(cur == v)
            if not mask.any():
                continue
            if not kids:
                best_leaf[mask] = v
                continue
            sub = np.flatnonzero(mask)
            q_sum, q_cnt = None, None
            for p, qn in zip(panels, qns):
                qc = _score_children(qn[sub], p["rn"],
                                     p["blocks"].get(v, []))
                if qc is None:
                    continue
                filled = ~np.isnan(qc)
                if q_sum is None:
                    q_sum = np.where(filled, qc, 0.0)
                    q_cnt = filled.astype(float)
                else:
                    q_sum += np.where(filled, qc, 0.0)
                    q_cnt += filled
            if q_sum is None:
                # no reference resolves this split: interpretation
                # stops here; the forced candidate falls back to the
                # strongest label vote among the labels below
                below = [l for l in all_labels
                         if l in lab_pos and _label_below(
                             refs, l, tree, v)]
                nb = sub[~broken[sub]]
                stop_margin[nb] = 0.0
                broken[sub] = True
                if below:
                    bi = np.asarray([lab_pos[l] for l in below])
                    best_leaf[sub] = np.asarray(below, dtype=object)[
                        np.argmax(label_vote[np.ix_(sub, bi)], axis=1)]
                else:
                    best_leaf[sub] = v
                cur[sub] = "__done__"
                continue
            # mean log-p across covering references (geometric-mean
            # evidence); SMALLER = stronger, so sort ascending
            LP = np.where(q_cnt > 0, q_sum / np.maximum(q_cnt, 1),
                          np.inf)
            orderk = np.argsort(LP, axis=1)
            top = orderk[:, 0]
            lp1 = LP[np.arange(len(sub)), top]
            lp2 = (LP[np.arange(len(sub)), orderk[:, 1]]
                   if LP.shape[1] > 1 else np.full(len(sub), np.inf))
            with np.errstate(over="ignore"):
                margin = np.where(np.isfinite(lp2),
                                  -np.expm1(np.minimum(lp1 - lp2, 0.0)),
                                  1.0)
            S = -LP
            path_margins[sub, depth_of[v]] = margin
            newly = (~broken[sub]) & (margin < min_margin)
            if newly.any():
                nb = sub[newly]
                resolved[nb] = v
                stop_margin[nb] = margin[newly]
                kid_arr = np.asarray(kids, dtype=object)
                for x, row in zip(np.flatnonzero(newly), nb):
                    with np.errstate(over="ignore"):
                        rm = -np.expm1(np.minimum(
                            lp1[x] - LP[x], 0.0))
                    cand = kid_arr[np.where(np.isfinite(LP[x]),
                                            rm < min_margin, False)]
                    stop_cand[row] = ";".join(map(str, cand))
                broken[nb] = True
            keep = ~broken[sub]
            resolved[sub[keep]] = np.asarray(
                kids, dtype=object)[top[keep]]
            cur[sub] = np.asarray(kids, dtype=object)[top]

        left = np.asarray([b == "" for b in best_leaf])
        if left.any():                       # safety net
            best_leaf[left] = cur[left]
        outs.append({
            "best_leaf": best_leaf.astype(str),
            "resolved_node": resolved.astype(str),
            "resolved_depth": np.asarray(
                [depth_of.get(r, 0) for r in resolved]),
            "stop_margin": stop_margin,
            "stop_candidates": stop_cand.astype(str),
            "path_margins": path_margins,
            "max_label_vote": label_vote.max(axis=1),
            "mean_max_corr": mean_mc,
            "label_vote": label_vote,
        })

    res = {k: np.concatenate([o[k] for o in outs])
           for k in outs[0] if outs[0][k].ndim == 1}
    res["path_margins"] = np.vstack([o["path_margins"] for o in outs])
    res["label_vote"] = np.vstack([o["label_vote"] for o in outs])
    res["labels"] = all_labels
    res["params"] = {"min_margin": min_margin,
                     "n_refs": len(refs), "block": block}
    return res


def _label_below(refs, label, tree, v):
    """True when `label`'s node lies within v's subtree in any ref."""
    for ref in refs:
        nd = ref["label_nodes"].get(label)
        if nd is None:
            continue
        span = {v}
        stack = list(tree["children"].get(v, []))
        while stack:
            x = stack.pop()
            span.add(x)
            stack.extend(tree["children"].get(x, []))
        if nd in span:
            return True
    return False
