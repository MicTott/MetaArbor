"""Consensus scaffold tests: simulation generator truth-consistency, the
eligibility model's three-way calls (including the flagship rare-private
regime), and ancestry-poset compatibility logic."""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.consensus import (call, compatible, eligible_v15, p_detect,
                                 pair_relation, relation, scenario, support)


def test_simulator_scenarios_and_truth():
    for name in ("batch", "missing_unique", "resolution_imbalance",
                 "rare_private"):
        sim = scenario(name, seed=1, n_genes=400, cells_per_leaf=40)
        assert len(sim["donors"]) == 3
        for d in sim["donors"]:
            assert d["counts"].shape[0] == len(d["labels"])
    sim = scenario("missing_unique", seed=1, n_genes=400, cells_per_leaf=40)
    assert sim["truth_presence"][(1, "F4.s1")] == "absent"
    assert sim["truth_presence"][(2, "P1")] == "private"
    assert "P1" in sim["donors"][2]["labels"]
    sim = scenario("rare_private", seed=1, n_genes=400, cells_per_leaf=40)
    d2 = sim["donors"][2]
    n_rare = (d2["labels"] == "F1.rare").sum()
    assert n_rare >= 1
    assert sim["prevalence"]["F1.rare"] == pytest.approx(0.02)
    # extreme imbalance: donor 0 much larger than donor 2
    assert sim["donors"][0]["counts"].shape[0] > \
        4 * sim["donors"][2]["counts"].shape[0]


def test_eligibility_three_way_calls():
    assert p_detect(0.5, 20) > 0.999
    assert p_detect(0.01, 20) < 0.2
    # 50% subpopulation, 20 parent cells: powered -> absence is evidence
    assert call(False, 0.5, 20) == "private_or_absent"
    # 1% subpopulation, 20 parent cells: unpowered -> unknown
    assert call(False, 0.01, 20) == "unknown"
    assert call(True, 0.01, 20) == "supported"
    # rare-private flagship arithmetic: p=0.02 needs n >= ~150 for 95%
    assert not eligible_v15(0.02, 100)
    assert eligible_v15(0.02, 200)
    supp, elig = support(["supported", "unknown", "private_or_absent"])
    assert (supp, elig) == (1, 2)   # unknown excluded from the denominator


def test_poset_relations_and_compatibility():
    rows = [("A", "A1", "a1"), ("A", "A1", "a2"), ("A", "A2", "a3"),
            ("B", "B1", "b1"), ("B", "B1", "b2")]
    t1 = tree_from_levels(rows, ["cls", "fam", "leaf"])
    t2 = tree_from_levels(rows, ["cls", "fam", "leaf"])
    trees = {"d1": t1, "d2": t2}
    assert relation(t1, "cls:A", "fam:A1") == "ancestor"
    assert relation(t1, "a1", "fam:A1") == "descendant"
    assert relation(t1, "fam:A1", "fam:B1") == "disjoint"
    m_top = {"d1": "cls:A", "d2": "cls:A"}
    m_mid = {"d1": "fam:A1", "d2": "fam:A1"}
    m_other = {"d1": "fam:B1", "d2": "fam:B1"}
    rel, conf = pair_relation(m_top, m_mid, trees)
    assert rel == "ancestor" and not conf
    ok, conflicts = compatible([m_top, m_other], m_mid, trees)
    assert ok and not conflicts
    # disagreement across trees -> conflict, not acceptance
    m_bad = {"d1": "fam:A1", "d2": "fam:B1"}
    ok, conflicts = compatible([m_mid], m_bad, trees)
    assert not ok and conflicts
