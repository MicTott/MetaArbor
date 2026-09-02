"""MetaArbor measurement kernel (DESIGN.md section 3a).

Convention: matrices are CELLS x GENES (AnnData orientation). The R
companion uses genes x cells; parity tests transpose at the boundary.

Vote scores are mean rank-standardized network weights; they are additive
over disjoint training unions with a fixed network (the vote-cache theorem),
so any node's score is a column sum over its leaves' cached columns. AUROCs
are never aggregated - always recomputed from summed scores.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata


def lognorm(counts, lib=None, assume_log=False):
    """log1p-CPM. `counts` cells x genes (dense or sparse). Pass full-gene
    per-cell totals as `lib` when `counts` carries only a gene subset."""
    if assume_log:
        return np.asarray(counts.todense() if sparse.issparse(counts) else counts,
                          dtype=np.float64)
    X = np.asarray(counts.todense() if sparse.issparse(counts) else counts,
                   dtype=np.float64)
    if lib is None:
        lib = X.sum(axis=1)
    lib = np.asarray(lib, dtype=np.float64).copy()
    lib[lib == 0] = 1.0
    return np.log1p(X / lib[:, None] * 1e6)


def variable_genes(expr_a, expr_b, gene_names, n_top=1000):
    """Joint HVGs: intersection of each dataset's top-variance genes on
    log-normalized expression. Returns gene indices into `gene_names`."""
    va = expr_a.var(axis=0)
    vb = expr_b.var(axis=0)
    top_a = set(np.argsort(va)[::-1][:n_top])
    top_b = set(np.argsort(vb)[::-1][:n_top])
    hvg = sorted(top_a & top_b)
    if len(hvg) < 50:
        import warnings
        warnings.warn(f"only {len(hvg)} joint HVGs; costs may be unstable")
    return np.asarray(hvg, dtype=int)


def rank_normalize(expr):
    """Rank each cell's profile within itself (Spearman preparation), center,
    L2-normalize, so that X_a @ X_b.T is the cell-cell Spearman correlation.
    `expr` cells x genes; returns same orientation."""
    r = np.apply_along_axis(rankdata, 1, expr)
    r -= r.mean(axis=1, keepdims=True)
    n = np.linalg.norm(r, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return r / n


def vote_cache(test_norm, train_norm, train_labels, chunk=2000):
    """Per-test-cell summed rank-standardized weights to each training leaf.

    Returns dict with V (n_test x n_leaf), leaves (sorted labels), and
    leaf_sizes. Column sums over any leaf set give that union's vote score;
    only the ordering across test cells matters for AUROC.
    """
    train_labels = np.asarray(train_labels)
    leaves = sorted(set(train_labels))
    n_train = train_norm.shape[0]
    ind = sparse.csr_matrix(
        (np.ones(n_train), (np.arange(n_train),
                            [leaves.index(l) for l in train_labels])),
        shape=(n_train, len(leaves)))
    n_test = test_norm.shape[0]
    V = np.zeros((n_test, len(leaves)))
    for s in range(0, n_test, chunk):
        co = test_norm[s:s + chunk] @ train_norm.T
        w = np.apply_along_axis(rankdata, 1, co) / n_train
        V[s:s + chunk] = w @ ind
    sizes = np.asarray([(train_labels == l).sum() for l in leaves], dtype=float)
    return {"V": V, "leaves": leaves, "leaf_sizes": sizes}


def aggregate_cache(cache, mapping):
    """Exact coarse cache from a fine one: column sums per coarse label
    (the additivity theorem used as an algorithm)."""
    coarse = sorted(set(mapping[l] for l in cache["leaves"]))
    cols = {c: [i for i, l in enumerate(cache["leaves"]) if mapping[l] == c]
            for c in coarse}
    V = np.column_stack([cache["V"][:, cols[c]].sum(axis=1) for c in coarse])
    sizes = np.asarray([cache["leaf_sizes"][cols[c]].sum() for c in coarse])
    return {"V": V, "leaves": coarse, "leaf_sizes": sizes}


def auroc(scores, positive):
    """Mann-Whitney AUROC with average tie ranks."""
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(scores)
    return (r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def node_scores(cache, node_leaves):
    idx = [cache["leaves"].index(l) for l in node_leaves]
    return cache["V"][:, idx].sum(axis=1)


def node_mean_scores(cache, node_leaves):
    idx = [cache["leaves"].index(l) for l in node_leaves]
    return cache["V"][:, idx].sum(axis=1) / cache["leaf_sizes"][idx].sum()


def node_auroc(cache, test_labels, query, node_leaves):
    return auroc(node_scores(cache, node_leaves),
                 np.asarray(test_labels) == query)


def leaf_costs(cache_a, labels_a, cache_b, labels_b):
    """Symmetrized costs M = 1 - (AUC_ab + AUC_ba)/2 and both folds."""
    la = sorted(set(np.asarray(labels_a)))
    lb = cache_a["leaves"]
    auc_b_to_a = np.full((len(la), len(lb)), np.nan)
    for i, qi in enumerate(la):
        pos = np.asarray(labels_a) == qi
        for j in range(len(lb)):
            auc_b_to_a[i, j] = auroc(cache_a["V"][:, j], pos)
    auc_a_to_b = np.full((len(la), len(lb)), np.nan)
    lb_labels = np.asarray(labels_b)
    for j, qj in enumerate(lb):
        pos = lb_labels == qj
        for i in range(len(la)):
            auc_a_to_b[i, j] = auroc(cache_b["V"][:, cache_b["leaves"].index(la[i])], pos)
    S = (auc_a_to_b + auc_b_to_a) / 2
    return {"M": 1 - S, "S": S, "auc_a_to_b": auc_a_to_b,
            "auc_b_to_a": auc_b_to_a, "rows": la, "cols": lb}


def measure(counts_a, labels_a, counts_b, labels_b, gene_names,
            n_hvg=1000, lib_a=None, lib_b=None, assume_log=False, chunk=2000):
    """Whole measurement layer for two labeled datasets (cells x genes)."""
    ea = lognorm(counts_a, lib_a, assume_log)
    eb = lognorm(counts_b, lib_b, assume_log)
    hvg = variable_genes(ea, eb, gene_names, n_hvg)
    na = rank_normalize(ea[:, hvg])
    nb = rank_normalize(eb[:, hvg])
    cache_a = vote_cache(na, nb, labels_b, chunk)
    cache_b = vote_cache(nb, na, labels_a, chunk)
    costs = leaf_costs(cache_a, labels_a, cache_b, labels_b)
    return {"hvg": [gene_names[i] for i in hvg], "cache_a": cache_a,
            "cache_b": cache_b, "labels_a": np.asarray(labels_a),
            "labels_b": np.asarray(labels_b), "costs": costs}
