"""Molecular branch lengths for visualization (phylogram mode).

Fits nonnegative branch lengths to a FIXED taxonomy topology from a
within-atlas molecular distance between leaves, so that patristic (tree)
distances approximate observed leaf-to-leaf distances. Visualization only:
nothing here touches MetaArbor-Walk or MetaArbor-Transport, and the fit
uses the target atlas alone — cross-atlas assignments never enter (that
would make the visualization circular).

Method: unary chains are collapsed before fitting (their split is not
identifiable — only the chain total is; intermediate chain nodes are placed
at the chain end, so topologically equivalent nodes geometrically
coincide); edge lengths solve min ||A b - d||^2 with b >= 0 over all leaf
pairs (path-incidence NNLS); lengths are normalized so the median observed
leaf-leaf distance is 1. Fit quality is reported as Pearson r and Kruskal
normalized stress between observed and fitted patristic distances — a poor
fit is itself a finding: the supplied topology does not represent the
atlas's transcriptomic geometry additively.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.stats import rankdata

from .kernel import lognorm
from .tree import ancestors


def pseudobulk_distances(counts, labels, lib=None, n_hvg=2000,
                         assume_log=False):
    """Chord distance between rank-based pseudobulk profiles:
    d = sqrt(2 * (1 - rho_Spearman)), which on centered, L2-normalized
    ranked vectors is exactly their Euclidean distance — a proper metric,
    unlike 1 - rho. `counts` cells x genes from the TARGET atlas alone;
    HVGs are chosen from this atlas alone (top variance on log1p-CPM)."""
    X = lognorm(counts, lib=lib, assume_log=assume_log)
    labels = np.asarray(labels)
    leaves = sorted(set(labels))
    hvg = np.argsort(X.var(axis=0))[::-1][:n_hvg]
    P = np.vstack([X[labels == l][:, hvg].mean(axis=0) for l in leaves])
    R = np.apply_along_axis(rankdata, 1, P)
    R -= R.mean(axis=1, keepdims=True)
    n = np.linalg.norm(R, axis=1, keepdims=True)
    n[n == 0] = 1
    C = (R / n) @ (R / n).T
    D = np.sqrt(np.maximum(2 * (1 - C), 0))
    np.fill_diagonal(D, 0)
    return D, leaves


def _collapse_chains(tree):
    """Chain-collapsed topology. `kept` = root, leaves, and branching
    internals; a unary chain's interior nodes map to the kept node at the
    chain's lower end (so topologically equivalent nodes share a position).
    Returns (reduced_parent over kept non-root nodes, node -> kept map)."""
    kept = {v for v in tree["parent"]
            if v == "root" or len(tree["children"][v]) != 1}

    def down(v):
        while len(tree["children"][v]) == 1:
            v = tree["children"][v][0]
        return v

    node_map = {v: ("root" if v == "root" else down(v))
                for v in tree["parent"]}
    red_parent = {}
    for v in kept:
        if v == "root":
            continue
        p = tree["parent"][v]
        while p != "root" and p not in kept:
            p = tree["parent"][p]
        red_parent[v] = p
    return red_parent, node_map


def fit_branch_lengths(tree, D, leaves):
    """NNLS branch lengths on the chain-collapsed topology. Returns
    cumulative root distance for EVERY original node, per-edge lengths,
    and fit statistics (Pearson r, normalized stress, scale)."""
    red_parent, node_map = _collapse_chains(tree)
    edges = sorted(red_parent)          # one edge per kept non-root node
    e_ix = {e: k for k, e in enumerate(edges)}

    def path_edges(leaf):
        out, r = [], node_map[leaf]
        while r != "root":
            out.append(e_ix[r])
            r = red_parent[r]
        return out

    rows, d = [], []
    for a in range(len(leaves)):
        pa = set(path_edges(leaves[a]))
        for b_ in range(a):
            pb = set(path_edges(leaves[b_]))
            row = np.zeros(len(edges))
            for e in pa.symmetric_difference(pb):
                row[e] = 1.0
            rows.append(row)
            d.append(D[a, b_])
    A = np.vstack(rows)
    d = np.asarray(d)
    scale = float(np.median(d[d > 0]))
    b, _ = nnls(A, d / scale)
    # degree-2 reduced root: only the SUM of the two root-adjacent edges is
    # identified by leaf distances (classic rooted-tree identifiability);
    # split it equally by convention — all patristic distances preserved
    root_kids = [e for e in edges if red_parent[e] == "root"]
    if len(root_kids) == 2:
        tot = b[e_ix[root_kids[0]]] + b[e_ix[root_kids[1]]]
        b[e_ix[root_kids[0]]] = b[e_ix[root_kids[1]]] = tot / 2
    fitted = A @ b
    obs = d / scale
    r = float(np.corrcoef(obs, fitted)[0, 1])
    stress = float(np.sqrt(np.sum((obs - fitted) ** 2) / np.sum(obs ** 2)))
    pos = {"root": 0.0}

    def depth_of(kv):
        if kv not in pos:
            pos[kv] = depth_of(red_parent[kv]) + b[e_ix[kv]]
        return pos[kv]

    positions = {v: depth_of(node_map[v]) if node_map[v] != "root" else 0.0
                 for v in tree["parent"]}
    lengths = {e: float(b[e_ix[e]]) for e in edges}
    return {"positions": positions, "edge_lengths": lengths,
            "pearson_r": r, "stress": stress, "scale": scale}


def patristic_matrix(tree, positions, leaves):
    """Pairwise patristic distances between `leaves` from fitted cumulative
    root positions (path through the LCA)."""
    P = np.zeros((len(leaves), len(leaves)))
    anc = {l: [l] + ancestors(tree, l) for l in leaves}
    for a in range(len(leaves)):
        sa = anc[leaves[a]]
        seta = set(sa)
        for b_ in range(a):
            lca = next(x for x in anc[leaves[b_]] if x in seta)
            P[a, b_] = P[b_, a] = (positions[leaves[a]] +
                                   positions[leaves[b_]] -
                                   2 * positions[lca])
    return P


def structure_matrices(counts, labels, tree, leaves, kind="chord",
                       lib=None, n_hvg=2000, assume_log=False):
    """One atlas's GW structure matrix over `leaves`, from ITS OWN
    expression only (never cross-atlas — that would be circular).

    kind:
      "chord"     raw within-atlas chord distances between rank-based
                  pseudobulks (no tree metric imposed)
      "patristic" chord distances NNLS-projected onto the supplied
                  topology (fitted branch lengths -> patristic matrix)
    Returns (C, fit_stats_or_None).
    """
    D, dl = pseudobulk_distances(counts, labels, lib=lib, n_hvg=n_hvg,
                                 assume_log=assume_log)
    ix = [dl.index(l) for l in leaves]
    D = D[np.ix_(ix, ix)]
    if kind == "chord":
        return D, None
    if kind == "patristic":
        fit = fit_branch_lengths(tree, D, list(leaves))
        P = patristic_matrix(tree, fit["positions"], list(leaves))
        return P * fit["scale"], {"pearson_r": fit["pearson_r"],
                                  "stress": fit["stress"]}
    raise ValueError(f"unknown kind {kind!r} (chord|patristic)")
