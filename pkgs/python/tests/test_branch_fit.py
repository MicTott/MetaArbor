"""Branch-length fitting: exact patristic recovery on a toy tree with known
additive lengths, chain-node coincidence, degree-2-root convention, and
chord-distance metric properties."""
import numpy as np

from metaarbor import fit_branch_lengths, tree_from_levels
from metaarbor.branch_fit import pseudobulk_distances

ROWS = [("A", "A1", "a1"), ("A", "A1", "a2"), ("A", "A2", "a3"),
        ("B", "B1", "b1"), ("B", "B1", "b2")]
TREE = tree_from_levels(ROWS, ["cls", "fam", "leaf"])
L = {"cls:A": 0.30, "cls:B": 0.50, "fam:A1": 0.20, "a1": 0.05, "a2": 0.07,
     "a3": 0.40, "b1": 0.10, "b2": 0.12, "fam:B1": 0.25}
LEAVES = ["a1", "a2", "a3", "b1", "b2"]
PATHS = {"a1": ["a1", "fam:A1", "cls:A"], "a2": ["a2", "fam:A1", "cls:A"],
         "a3": ["a3", "cls:A"], "b1": ["b1", "fam:B1", "cls:B"],
         "b2": ["b2", "fam:B1", "cls:B"]}


def toy_D():
    D = np.zeros((5, 5))
    for i, a in enumerate(LEAVES):
        for j, b in enumerate(LEAVES):
            if i < j:
                D[i, j] = D[j, i] = sum(
                    L[e] for e in set(PATHS[a]) ^ set(PATHS[b]))
    return D


def test_exact_additive_recovery():
    fit = fit_branch_lengths(TREE, toy_D(), LEAVES)
    sc = fit["scale"]
    assert abs(fit["pearson_r"] - 1) < 1e-9 and fit["stress"] < 1e-7
    for e in ("a1", "a2", "a3", "b1", "b2", "fam:A1"):
        assert abs(fit["edge_lengths"][e] * sc - L[e]) < 1e-8
    # unary cls:B merged into fam:B1's edge; degree-2 root sum preserved and
    # split equally by convention
    tot = (fit["edge_lengths"]["cls:A"] + fit["edge_lengths"]["fam:B1"]) * sc
    assert abs(tot - (L["cls:A"] + L["cls:B"] + L["fam:B1"])) < 1e-8
    assert abs(fit["edge_lengths"]["cls:A"] -
               fit["edge_lengths"]["fam:B1"]) < 1e-9
    # chain interiors coincide with their chain ends
    assert abs(fit["positions"]["fam:A2"] - fit["positions"]["a3"]) < 1e-12
    assert abs(fit["positions"]["cls:B"] - fit["positions"]["fam:B1"]) < 1e-12


def test_chord_distance_is_metric_on_ranks():
    rs = np.random.RandomState(5)
    counts = rs.poisson(3, size=(300, 200)).astype(float)
    labels = np.repeat(["x", "y", "z"], 100)
    D, leaves = pseudobulk_distances(counts, labels, n_hvg=150)
    assert leaves == ["x", "y", "z"]
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0) and np.all(D >= 0)
    # triangle inequality (chord distance is Euclidean on standardized ranks)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert D[i, j] <= D[i, k] + D[k, j] + 1e-12


def test_phylogram_and_newick():
    import pytest
    pytest.importorskip("matplotlib")
    import tempfile, os
    from metaarbor import plot_alignment_phylogram, to_newick
    fit = fit_branch_lengths(TREE, toy_D(), LEAVES)
    rows = [{"query": "Q1", "walk_selected": "fam:A1",
             "transport_node": "fam:A1", "agreement": "agree",
             "walk_decision_support": 0.9, "transport_mass": 0.95},
            {"query": "Q2", "walk_selected": "a3",
             "transport_node": "fam:A2",
             "agreement": "topologically_equivalent",
             "walk_decision_support": 0.8, "transport_mass": 0.9}]
    fig, ax = plot_alignment_phylogram(rows, TREE,
                                       node_positions=fit["positions"])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "p.png")
        fig.savefig(p)
        assert os.path.getsize(p) > 0
    nwk = to_newick(TREE, fit["edge_lengths"])
    assert nwk.endswith(";") and nwk.count("(") == nwk.count(")")
    for leaf in LEAVES:
        assert leaf in nwk
    assert ":" in nwk  # branch lengths attached
