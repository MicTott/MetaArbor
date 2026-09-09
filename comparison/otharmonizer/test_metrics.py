"""Golden tests for the OTHarmonizer-metric port (metrics.py).

Goldens were produced by running github.com/Duck-Boss/OTHarmonizer's
own TEDS/PCBS/AH_F1 side by side with this port (exact parity on every
case, including two committed real trees). The upstream clone is NOT
needed to run these tests. One documented deviation: upstream count_f1
raises ZeroDivisionError when a tree has no '&'-merged equal
relations; this port returns component F1 = 0.0 instead.

Run standalone:  python -m pytest comparison/otharmonizer/test_metrics.py
(requires: zss)
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import from_nested, score_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

T1 = {"label": "root", "children": [
    {"label": "A", "children": [{"label": "A1&A1b", "children": []},
                                {"label": "A2", "children": []}]},
    {"label": "B", "children": [{"label": "B1", "children": []}]}]}
R1 = {"label": "root", "children": [
    {"label": "A", "children": [{"label": "A1&A1b", "children": []},
                                {"label": "A2&B1", "children": []}]},
    {"label": "B", "children": []}]}
T2 = {"label": "root", "children": [
    {"label": "X&Y", "children": [
        {"label": "u", "children": []}, {"label": "v", "children": []},
        {"label": "w&q", "children": []}]},
    {"label": "Z", "children": []}]}
R2 = {"label": "root", "children": [
    {"label": "X&Y&Z", "children": [
        {"label": "u&v", "children": []},
        {"label": "w", "children": [{"label": "q", "children": []}]}]}]}


def test_golden_simple():
    got = score_all(T1, R1)
    assert got["TED"] == 2.0
    assert got["TEDS"] == pytest.approx(0.8, abs=1e-12)
    assert got["PCB"] == 3
    assert got["RFS"] == pytest.approx(0.4, abs=1e-12)
    assert got["PCBS"] == pytest.approx(0.7666666666666666, abs=1e-12)
    assert got["AH_F1"] == pytest.approx(0.7962962962962963, abs=1e-12)


def test_golden_merged_multifurcation():
    got = score_all(T2, R2)
    assert got["TED"] == 6.0
    assert got["TEDS"] == pytest.approx(0.4, abs=1e-12)
    assert got["PCB"] == 5
    assert got["RFS"] == pytest.approx(0.0, abs=1e-12)
    assert got["PCBS"] == pytest.approx(0.16666666666666666, abs=1e-12)
    assert got["AH_F1"] == pytest.approx(0.5918803418803419, abs=1e-12)


def test_golden_real_committed_trees():
    o1 = json.load(open(os.path.join(HERE, "oth_tree_default.json")))
    o2 = json.load(open(os.path.join(HERE,
                                     "oth_tree_seeded_v2f_a.json")))
    got = score_all(o1, o2)
    assert got["TEDS"] == pytest.approx(0.863248, abs=1e-6)
    assert got["PCBS"] == pytest.approx(0.899137, abs=1e-6)
    assert got["AH_F1"] == pytest.approx(0.901696, abs=1e-6)


def test_identity_and_bounds():
    for t in (T1, T2):
        got = score_all(t, t)
        assert got["TED"] == 0.0 and got["TEDS"] == 1.0
        assert got["PCBS"] == pytest.approx(1.0)
        assert got["AH_F1"] == pytest.approx(1.0)
    got = score_all(T1, R2)
    for k in ("TEDS", "PCBS", "AH_F1"):
        assert 0.0 <= got[k] <= 1.0


def test_no_equals_does_not_crash():
    """Upstream raises ZeroDivisionError here; the port returns the
    component F1 as 0 (documented deviation)."""
    plain = {"label": "root", "children": [
        {"label": "A", "children": [{"label": "A1", "children": []}]}]}
    got = score_all(plain, plain)
    assert got["TEDS"] == 1.0
    assert 0.0 <= got["AH_F1"] <= 1.0


def test_from_nested_roundtrip():
    n = from_nested(T2)
    assert n.label == "root"
    assert [c.label for c in n.children] == ["X&Y", "Z"]
    assert n.children[0].children[2].label == "w&q"


def test_internal_label_equals_pendant_leaf():
    """A label on an internal node is scored exactly as if it were a
    pendant leaf attached at that node: every pairwise LCA (and hence
    both metrics) is identical under the two encodings."""
    from metrics import cophenetic_spearman, triplet_scores
    internal = {"label": "root", "children": [
        {"label": "P", "children": [{"label": "x", "children": []},
                                    {"label": "y", "children": []}]},
        {"label": "Q", "children": [{"label": "z", "children": []}]}]}
    pendant = {"label": "root", "children": [
        {"label": "__i1__", "children": [
            {"label": "P", "children": []},
            {"label": "x", "children": []},
            {"label": "y", "children": []}]},
        {"label": "__i2__", "children": [
            {"label": "Q", "children": []},
            {"label": "z", "children": []}]}]}
    ref = {"label": "root", "children": [
        {"label": "P", "children": [{"label": "x", "children": []},
                                    {"label": "z", "children": []}]},
        {"label": "Q", "children": [{"label": "y", "children": []}]}]}
    assert cophenetic_spearman(internal, ref) == pytest.approx(
        cophenetic_spearman(pendant, ref))
    assert triplet_scores(internal, ref) == pytest.approx(
        triplet_scores(pendant, ref))
    # and the parent-of-its-children triplet is UNRESOLVED, not scored:
    # {P, x, y} has all three pairwise LCAs at P's node in `internal`
    rec, agr = triplet_scores(internal, internal)
    assert rec == 1.0 and agr == 1.0   # self-score sanity under both


def test_reference_unresolved_triplets_not_counted():
    """Triplets the reference leaves unresolved (polytomies) are
    excluded from recovery's denominator: a tree that RESOLVES them
    (rightly or wrongly) is neither rewarded nor punished there."""
    from metrics import triplet_scores
    ref = {"label": "root", "children": [
        {"label": "A", "children": [{"label": "a", "children": []},
                                    {"label": "b", "children": []}]},
        {"label": "c", "children": []},
        {"label": "d", "children": []}]}
    # groups (a,b) correctly AND invents (c,d) — ref is silent on {c,d}
    over = {"label": "root", "children": [
        {"label": "X", "children": [{"label": "a", "children": []},
                                    {"label": "b", "children": []}]},
        {"label": "Y", "children": [{"label": "c", "children": []},
                                    {"label": "d", "children": []}]}]}
    rec, agr = triplet_scores(over, ref)
    assert rec == 1.0            # every ref-resolved triplet recovered
    assert agr < 1.0             # extra resolution shows up ONLY here


def test_unary_chain_invariance():
    """Inserting unary (single-child) anonymous chains never changes
    triplet resolution; cophenetic Spearman may move (depth-sensitive)
    and is documented as convention/depth-profile-bound."""
    from metrics import triplet_scores
    base = {"label": "root", "children": [
        {"label": "A", "children": [{"label": "a1", "children": []},
                                    {"label": "a2", "children": []}]},
        {"label": "B", "children": [{"label": "b1", "children": []},
                                    {"label": "b2", "children": []}]}]}
    chained = {"label": "root", "children": [
        {"label": "__u1__", "children": [{"label": "__u2__", "children": [
            {"label": "A", "children": [
                {"label": "a1", "children": []},
                {"label": "__u3__", "children": [
                    {"label": "a2", "children": []}]}]}]}]},
        {"label": "B", "children": [{"label": "b1", "children": []},
                                    {"label": "b2", "children": []}]}]}
    ref = {"label": "root", "children": [
        {"label": "F", "children": [{"label": "a1", "children": []},
                                    {"label": "b1", "children": []}]},
        {"label": "G", "children": [{"label": "a2", "children": []},
                                    {"label": "b2", "children": []},
                                    {"label": "A", "children": []},
                                    {"label": "B", "children": []}]}]}
    assert triplet_scores(base, ref) == pytest.approx(
        triplet_scores(chained, ref))
    assert triplet_scores(base, base) == pytest.approx(
        triplet_scores(chained, base))


def test_scores_restricted_to_shared_labels():
    """Labels present in only one tree are excluded (pairwise shared
    universe); adding tree-only extra labels changes nothing."""
    from metrics import cophenetic_spearman, triplet_scores
    ref = {"label": "root", "children": [
        {"label": "A", "children": [{"label": "a1", "children": []},
                                    {"label": "a2", "children": []}]},
        {"label": "B", "children": [{"label": "b1", "children": []},
                                    {"label": "b2", "children": []}]}]}
    tree = {"label": "root", "children": [
        {"label": "X", "children": [{"label": "a1", "children": []},
                                    {"label": "a2", "children": []}]},
        {"label": "Y", "children": [{"label": "b1", "children": []},
                                    {"label": "b2", "children": []}]}]}
    extra = {"label": "root", "children": [
        {"label": "X", "children": [{"label": "a1", "children": []},
                                    {"label": "a2", "children": []},
                                    {"label": "novel1", "children": []}]},
        {"label": "Y", "children": [{"label": "b1", "children": []},
                                    {"label": "b2", "children": []}]},
        {"label": "novel2", "children": []}]}
    assert cophenetic_spearman(tree, ref) == pytest.approx(
        cophenetic_spearman(extra, ref))
    assert triplet_scores(tree, ref) == pytest.approx(
        triplet_scores(extra, ref))


def test_cophenetic_and_triplets_basics():
    from metrics import cophenetic_spearman, triplet_scores
    # identical trees: perfect scores
    assert cophenetic_spearman(T2, T2) == pytest.approx(1.0)
    rec, agr = triplet_scores(T1, T1)
    assert rec == 1.0 and agr == 1.0
    # a STAR cannot game either metric
    star = {"label": "root", "children": [
        {"label": l, "children": []}
        for l in ("A1", "A2", "B1", "B2", "C1")]}
    ref = {"label": "root", "children": [
        {"label": "A", "children": [{"label": "A1", "children": []},
                                    {"label": "A2", "children": []}]},
        {"label": "B", "children": [{"label": "B1", "children": []},
                                    {"label": "B2", "children": []}]},
        {"label": "C1", "children": []}]}
    assert cophenetic_spearman(star, ref) == 0.0
    rec, _agr = triplet_scores(star, ref)
    assert rec == 0.0                      # recovers nothing resolved
    # a correct two-family tree recovers everything
    good = {"label": "root", "children": [
        {"label": "X", "children": [{"label": "A1", "children": []},
                                    {"label": "A2", "children": []}]},
        {"label": "Y", "children": [{"label": "B1", "children": []},
                                    {"label": "B2", "children": []}]},
        {"label": "C1", "children": []}]}
    rec, agr = triplet_scores(good, ref)
    assert rec == 1.0 and agr == 1.0
    assert cophenetic_spearman(good, ref) == pytest.approx(1.0)
    # a WRONG grouping scores clearly lower
    bad = {"label": "root", "children": [
        {"label": "X", "children": [{"label": "A1", "children": []},
                                    {"label": "B1", "children": []}]},
        {"label": "Y", "children": [{"label": "A2", "children": []},
                                    {"label": "B2", "children": []}]},
        {"label": "C1", "children": []}]}
    rec_b, _ = triplet_scores(bad, ref)
    assert rec_b == 0.0
    assert cophenetic_spearman(bad, ref) < 0.1
