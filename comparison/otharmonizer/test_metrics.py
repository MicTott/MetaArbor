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
