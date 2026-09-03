"""Stable accessors for MetaArbor's underlying matrices, plus tidy CSV
exporters. Read-only views over quantities the estimators already compute.

A note on `node_auroc_matrix`: raw one-vs-all node AUROC is an
INTERPRETABILITY measure, not a replacement for the sibling-relative Walk
decision. It carries two documented biases: large heterogeneous unions
dilute toward 0.5 (root neutrality at scale), and easy backgrounds saturate
parent and child alike near 1 — precisely the effects the Walk's
vote-guided navigation and sibling contrast exist to sidestep.
"""
from __future__ import annotations

import csv
import gzip
import os

import numpy as np

from .kernel import auroc, node_scores
from .tree import leaves_under


def vote_fraction_matrix(cache, test_labels):
    """Query x target-leaf argmax-vote fractions: each query cell votes for
    its argmax mean-rank training leaf. Rows sum to 1."""
    test_labels = np.asarray(test_labels)
    queries = sorted(set(test_labels))
    ms = cache["V"] / cache["leaf_sizes"]
    top = ms.argmax(axis=1)
    M = np.zeros((len(queries), len(cache["leaves"])))
    for i, q in enumerate(queries):
        idx, cnt = np.unique(top[test_labels == q], return_counts=True)
        M[i, idx] = cnt / cnt.sum()
    return M, queries, list(cache["leaves"])


def node_auroc_matrix(cache, test_labels, tree, nodes=None):
    """Query x target-node one-vs-all AUROCs from the vote cache. `nodes`
    defaults to the cache leaves; pass internal node ids for aggregated
    views. See the module docstring for size/saturation caveats."""
    test_labels = np.asarray(test_labels)
    queries = sorted(set(test_labels))
    if nodes is None:
        nodes = list(cache["leaves"])
    M = np.full((len(queries), len(nodes)), np.nan)
    for j, n in enumerate(nodes):
        lv = [l for l in leaves_under(tree, n) if l in cache["leaves"]] \
             if n not in cache["leaves"] else [n]
        if not lv:
            continue
        s = node_scores(cache, lv)
        for i, q in enumerate(queries):
            M[i, j] = auroc(s, test_labels == q)
    return M, queries, list(nodes)


def family_mass(pi, col_names, family_of_leaf, normalize=True):
    """Family/node-aggregated FUGW mass. Rows normalized to 1 by default
    (queries comparable); `normalize=False` returns raw coupling mass."""
    fams = sorted(set(family_of_leaf.values()))
    idx = {f: [j for j, c in enumerate(col_names) if family_of_leaf[c] == f]
           for f in fams}
    P = np.asarray(pi, dtype=float)
    if normalize:
        tot = P.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        P = P / tot
    return np.column_stack([P[:, idx[f]].sum(axis=1) for f in fams]), fams


def walk_traces(cache, test_labels, tree, base_seed=7, **walk_kwargs):
    """Complete Walk decision traces for every query (same per-query seeds
    as `baseline_map`; decisions identical by construction)."""
    from .walk import select_node
    test_labels = np.asarray(test_labels)
    rows = []
    for qi, q in enumerate(sorted(set(test_labels))):
        sel = select_node(cache, test_labels, q, tree, seed=base_seed + qi,
                          trace=True, **walk_kwargs)
        rows.extend(sel["trace"] or [])
    return rows


def write_csv(rows_or_matrix, path, row_names=None, col_names=None):
    """Tidy CSV writer for list-of-dicts or (matrix, names) exports.
    Gzip when the path ends in .gz."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        if isinstance(rows_or_matrix, list):
            keys = list(rows_or_matrix[0].keys())
            w.writerow(keys)
            for r in rows_or_matrix:
                w.writerow([r.get(k) for k in keys])
        else:
            w.writerow([""] + list(col_names))
            for i, rn in enumerate(row_names):
                w.writerow([rn] + [f"{x:.10g}" for x in rows_or_matrix[i]])
    return os.path.abspath(path)
