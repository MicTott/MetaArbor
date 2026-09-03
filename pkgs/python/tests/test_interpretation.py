"""Interpretation-layer tests: agreement semantics, transport summary
invariants, outcome classification, and plot rendering (Agg smoke) on a
small synthetic problem. The heavy identity/parity checks against the real
Allen fixtures live in test_parity.py and pkgs/parity/02_interp_py.py."""
import os
import tempfile

import numpy as np
import pytest

from metaarbor import (agreement, alignment_summary, classify_outcome,
                       family_mass, rank_normalize, transport_summary,
                       tree_from_levels, vote_cache, walk_summary)

ROWS = [("A", "A1", "a1"), ("A", "A1", "a2"), ("A", "A2", "a3"),
        ("B", "B1", "b1"), ("B", "B1", "b2")]
TREE = tree_from_levels(ROWS, ["cls", "fam", "leaf"])
LEAVES = ["a1", "a2", "a3", "b1", "b2"]
FAM = {"a1": "A1", "a2": "A1", "a3": "A2", "b1": "B1", "b2": "B1"}


def synth_measurement(seed=3, n_per=40, n_genes=120):
    rs = np.random.RandomState(seed)
    centers = {l: rs.rand(n_genes) * 4 for l in LEAVES}
    def cells(lbls):
        X = np.vstack([centers[l] + rs.rand(n_genes) for l in lbls])
        return rank_normalize(X)
    train_labels = np.repeat(LEAVES, n_per)
    test_labels = np.repeat(["A1", "A2", "B1"], n_per)
    test_leaf = {"A1": ["a1", "a2"], "A2": ["a3"], "B1": ["b1", "b2"]}
    tl = [rs.choice(test_leaf[l]) for l in test_labels]
    cache = vote_cache(cells(tl), cells(train_labels), train_labels, chunk=50)
    return cache, test_labels


def test_agreement_semantics():
    assert agreement("fam:A1", "fam:A1", TREE) == "agree"
    # unary chain: fam:A2 -> a3, identical leaf sets => equivalent, not depth
    assert agreement("a3", "fam:A2", TREE) == "topologically_equivalent"
    assert agreement("fam:A2", "a3", TREE) == "topologically_equivalent"
    assert agreement("cls:A", "fam:A1", TREE) == "same_branch_different_depth"
    assert agreement("fam:A1", "cls:A", TREE) == "same_branch_different_depth"
    assert agreement("fam:A1", "fam:B1", TREE) == "conflicting_branch"
    assert agreement("fam:A1", None, TREE) == "walk_only"
    assert agreement(None, "fam:B1", TREE) == "transport_only"
    assert agreement(None, None, TREE) == "both_unmatched"


def test_transport_summary_invariants():
    pi = np.array([[0.5, 0.4, 0.05, 0.03, 0.02],
                   [0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.01, 0.01, 0.01, 0.6, 0.37]])
    rows = transport_summary(pi, ["A1", "Z", "B1"], LEAVES, FAM, tree=TREE)
    by = {r["query"]: r for r in rows}
    assert by["A1"]["transport_family"] == "A1"
    assert by["A1"]["transport_node"] == "fam:A1"
    assert by["A1"]["transport_bin"] == "confident"
    assert by["Z"]["transport_bin"] == "unmatched"
    assert by["B1"]["transport_bin"] == "confident"
    M, fams = family_mass(pi, LEAVES, FAM)
    sums = M.sum(axis=1)
    assert np.allclose(sums[[0, 2]], 1, atol=1e-12)


def test_classify_outcome():
    assert classify_outcome("fam:A1", "fam:A1", TREE) == "correct"
    assert classify_outcome("cls:A", "fam:A1", TREE) == "premature_stop"
    assert classify_outcome("a1", "fam:A1", TREE) == "too_deep"
    assert classify_outcome("fam:A2", "fam:A1", TREE) == "adjacent_same_class"
    assert classify_outcome("fam:B1", "fam:A1", TREE) == "wrong_branch"
    assert classify_outcome(None, "fam:A1", TREE) == "unmatched"


def test_end_to_end_summary_and_plots():
    pytest.importorskip("matplotlib")
    cache, test_labels = synth_measurement()
    S_dir = {q: {l: 0.9 for l in LEAVES} for q in ("A1", "A2", "B1")}
    wk = walk_summary(cache, test_labels, TREE, S_dir)
    assert {r["query"] for r in wk} == {"A1", "A2", "B1"}
    # deterministic: rerun equals first run
    assert wk == walk_summary(cache, test_labels, TREE, S_dir)
    pi = np.eye(3, 5, k=0) * 0.9 + 0.02
    ts = transport_summary(pi, ["A1", "A2", "B1"], LEAVES, FAM, tree=TREE)
    al = alignment_summary(wk, ts, TREE)
    assert all("agreement" in r for r in al)
    from metaarbor import (plot_alignment_tree, plot_error_tree,
                           plot_evidence_heatmap, plot_query_path,
                           plot_transport_heatmap, walk_traces)
    with tempfile.TemporaryDirectory() as td:
        figs = []
        figs.append(plot_alignment_tree(al, TREE)[0])
        figs.append(plot_evidence_heatmap(cache, test_labels, TREE,
                                          family_of_leaf=FAM)[0])
        figs.append(plot_transport_heatmap(pi, ["A1", "A2", "B1"], LEAVES,
                                           TREE, family_of_leaf=FAM)[0])
        tr = walk_traces(cache, test_labels, TREE)
        figs.append(plot_query_path(tr, "A1")[0])
        truth = {"A1": "fam:A1", "A2": "fam:A2", "B1": "fam:B1"}
        figs.append(plot_error_tree(al, truth, TREE)[0])
        for i, f in enumerate(figs):
            p = os.path.join(td, f"f{i}.png")
            f.savefig(p)
            assert os.path.getsize(p) > 0
