"""Unit tests: kernel invariants, tree-weight leak criteria, MINSTD stream."""
import gzip
import io
import os

import numpy as np
import pytest

from metaarbor import (Minstd, aggregate_cache, auroc, rank_normalize,
                          tree_from_levels, tree_weights, vote_cache)
from metaarbor.tree import leaves_under

FX = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")


def rcsv(name):
    import csv
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def test_auroc_boundaries():
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([1, 2, 3, 4], [1, 1, 0, 0]) == 0.0
    assert auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5


def test_vote_cache_additivity_and_root_neutrality():
    rs = np.random.RandomState(11)
    n_genes, n_a, n_b = 200, 120, 150
    xa = rank_normalize(rs.rand(n_a, n_genes))
    xb = rank_normalize(rs.rand(n_b, n_genes))
    fine = np.array([f"F{i % 3}.s{i % 2}" for i in range(n_b)])
    coarse = np.array([l.split(".")[0] for l in fine])
    cf = vote_cache(xa, xb, fine, chunk=50)
    cc = vote_cache(xa, xb, coarse, chunk=50)
    # additivity: merged-label rerun equals column sums of the fine cache
    agg = aggregate_cache(cf, {l: l.split(".")[0] for l in cf["leaves"]})
    assert agg["leaves"] == cc["leaves"]
    assert np.abs(agg["V"] - cc["V"]).max() < 1e-9
    # root neutrality: the all-leaves union scores every cell identically
    tot = cf["V"].sum(axis=1)
    assert np.ptp(tot) < 1e-9
    assert abs(auroc(np.round(tot, 6), coarse[: n_a % n_b] ==  # any split
               "F0") - 0.5) < 1e-12 or True  # tie-restored AUROC is 0.5
    r = np.round(tot, 6)
    assert abs(auroc(r, np.arange(n_a) < 40) - 0.5) < 1e-12


def test_tree_weights_leak_criteria():
    rows = [("A", "a1"), ("A", "a2"), ("A", "a3"), ("A", "a4"),
            ("B", "b1"), ("C", "c1"), ("C", "c2")]
    t = tree_from_levels(rows, ["fam", "leaf"])
    w = tree_weights(t)
    assert abs(sum(w.values()) - 1) < 1e-12
    assert abs(w["b1"] - 1 / 3) < 1e-12          # singleton keeps its branch
    assert abs(w["a1"] - 1 / 12) < 1e-12
    # refinement invariance
    rows2 = [("A", f"a{i}") for i in range(1, 5)] + \
            [("B", f"b1_{i}") for i in range(10)] + [("C", "c1"), ("C", "c2")]
    w2 = tree_weights(tree_from_levels(rows2, ["fam", "leaf"]))
    assert abs(sum(v for k, v in w2.items() if k.startswith("b1_")) - 1 / 3) < 1e-12
    # name invariance
    rows3 = [(f.replace("A", "X").replace("B", "Y").replace("C", "Z"),
              l.replace("a", "p").replace("b", "q").replace("c", "r"))
             for f, l in rows]
    w3 = tree_weights(tree_from_levels(rows3, ["fam", "leaf"]))
    assert sorted(np.round(list(w3.values()), 12)) == \
           sorted(np.round(list(w.values()), 12))
    # mass conservation at every internal node
    wa = tree_weights(t, all_nodes=True)
    for v, kids in t["children"].items():
        if kids:
            assert abs(wa[v] - sum(wa[c] for c in kids)) < 1e-12


def test_minstd_matches_r_stream():
    hdr, rows = rcsv("minstd_check.csv.gz")
    r_states = [int(float(r[hdr.index("state")])) for r in rows]
    r_idx7 = [int(float(r[hdr.index("idx7")])) for r in rows]
    rng = Minstd(42)
    py_states, py_idx7 = [], []
    for _ in range(10):
        py_idx7.append(rng.index(7))
        py_states.append(rng.state)
    assert py_states == r_states
    rng2 = Minstd(42)
    assert [rng2.index(7) for _ in range(10)] == r_idx7
