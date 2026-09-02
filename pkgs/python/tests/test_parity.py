"""Cross-language parity against the reference R implementation.

Fixtures come from pkgs/parity/01_export_and_gate.R, which itself gates on
reproducing the saved frozen benchmark. Two tiers:

- simulation end-to-end: identical counts in, HVG set / similarity matrix /
  walk selections out
- real vote caches: the frozen walk on the exported Allen platform caches
  must reproduce the packaged-R selections and relations for all 23 forward
  and 103 reverse queries (bit-comparable decisions via the shared MINSTD
  stream)
"""
import csv
import gzip
import os

import numpy as np
import pytest

from treeneighbor import baseline_map, measure, tree_from_levels

FX = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")


def rcsv(name):
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def load_matrix(name):
    hdr, rows = rcsv(name)
    return hdr, np.asarray([[float(x) for x in r] for r in rows])


def load_cache(v_name, leaves_name):
    _, V = load_matrix(v_name)
    hdr, rows = rcsv(leaves_name)
    leaves = [r[hdr.index("leaf")] for r in rows]
    sizes = np.asarray([float(r[hdr.index("size")]) for r in rows])
    return {"V": V, "leaves": leaves, "leaf_sizes": sizes}


def load_labels(name):
    hdr, rows = rcsv(name)
    return np.asarray([r[0] for r in rows])


def load_walk(name):
    hdr, rows = rcsv(name)
    out = {}
    for r in rows:
        d = dict(zip(hdr, r))
        out[d["query"]] = (d["selected"] if d["selected"] != "NA" else None,
                           d["relation"])
    return out


def s_dir_from_csv(name):
    hdr, rows = rcsv(name)
    cols = hdr[1:]
    S = {}
    for r in rows:
        S[r[0]] = {c: float(v) for c, v in zip(cols, r[1:])}
    return S


def compare_walk(py_map, r_map):
    mismatches = []
    for row in py_map:
        r_sel, r_rel = r_map[row["query"]]
        if row["selected"] != r_sel or row["relation"] != r_rel:
            mismatches.append((row["query"], row["selected"], r_sel,
                               row["relation"], r_rel))
    assert not mismatches, f"walk parity failed: {mismatches}"


def test_simulation_end_to_end():
    # first column is the gene name; matrices are genes x cells in R
    _, rows_a = rcsv("sim_countsA.csv.gz")
    _, rows_b = rcsv("sim_countsB.csv.gz")
    genes = [r[0] for r in rows_a]
    A = np.asarray([[float(x) for x in r[1:]] for r in rows_a]).T  # cells x genes
    B = np.asarray([[float(x) for x in r[1:]] for r in rows_b]).T
    la = load_labels("sim_labelsA.csv.gz")
    lb = load_labels("sim_labelsB.csv.gz")
    m = measure(A, la, B, lb, gene_names=genes, n_hvg=1000)
    r_hvg = set(load_labels("sim_hvg.csv.gz"))
    assert set(m["hvg"]) == r_hvg
    hdr_s, s_rows = rcsv("sim_S.csv.gz")
    r_S = {(r[0], c): float(v) for r in s_rows for c, v in zip(hdr_s[1:], r[1:])}
    for i, qi in enumerate(m["costs"]["rows"]):
        for j, cj in enumerate(m["costs"]["cols"]):
            assert abs(m["costs"]["S"][i, j] - r_S[(qi, cj)]) < 1e-9
    leaves = sorted(set(lb))
    tree_b = tree_from_levels([(l.split(".")[0], l) for l in leaves],
                              ["family", "leaf"])
    S_dir = {q: {c: r_S[(q, c)] for c in leaves} for q in sorted(set(la))}
    py_map = baseline_map(m["cache_a"], la, tree_b, S_dir)
    compare_walk(py_map, load_walk("sim_walkR.csv.gz"))


def test_real_cache_walk_parity_forward():
    cache = load_cache("cacheA_V.csv.gz", "cacheA_leaves.csv.gz")
    labels = load_labels("labelsA_subclass.csv.gz")
    hdr, rows = rcsv("tree_levels_b.csv.gz")
    tree_b = tree_from_levels([tuple(r) for r in rows], hdr)
    S_dir = s_dir_from_csv("S_sub.csv.gz")
    py_map = baseline_map(cache, labels, tree_b, S_dir)
    compare_walk(py_map, load_walk("walkR_forward.csv.gz"))


def test_real_cache_walk_parity_reverse():
    cache = load_cache("cacheBsub_V.csv.gz", "cacheBsub_leaves.csv.gz")
    labels = load_labels("labelsB_cluster.csv.gz")
    subclasses = sorted(load_labels("subclasses.csv.gz"))
    tree_a = tree_from_levels([(s,) for s in subclasses], ["leaf"])
    S = s_dir_from_csv("S_sub.csv.gz")
    # reverse direction: rows = cluster queries, cols = subclasses
    clusters = sorted(set(labels))
    S_dir = {c: {q: S[q][c] for q in S} for c in clusters}
    py_map = baseline_map(cache, labels, tree_a, S_dir)
    compare_walk(py_map, load_walk("walkR_reverse.csv.gz"))
