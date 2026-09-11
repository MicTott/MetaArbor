"""Consensus-cut invariants:

1. STRICT keeps only certified (>=2-atlas backbone) nodes; everything
   below is summarized in the remainder — never silently lost.
2. DEFAULT keeps named atlas detail nested under kept ancestors,
   splices anonymous scaffolding, routes unplaced to the ledger, and
   accounts for EVERY original label (alias_index + unresolved).
3. Views are ordered: strict <= default <= complete node counts.
4. Serialized tree.json input gives the identical cut as the in-memory
   harmonize result.
5. cut_to_tree produces a valid metaarbor tree (newick-convertible).
6. A routed/unplaced node keeps its rejection reason when provenance
   rows are supplied.
"""
import json

import numpy as np
import pytest

from metaarbor import to_newick, tree_from_levels
from metaarbor.consensus.cut import consensus_cut, cut_to_tree
from metaarbor.consensus.harmonize import harmonize

from test_harmonize import simulate

FAM = {f"F{i}.s{j}": f"F{i}" for i in (1, 2, 3) for j in (1, 2)}


@pytest.fixture(scope="module")
def world():
    leaves = sorted(FAM)
    leaves_a = [l for l in leaves if not l.startswith("F3")]
    trees = {
        "A": tree_from_levels([(f"A|{FAM[l]}", f"A|{l}")
                               for l in leaves_a], ["family", "leaf"]),
        "B": tree_from_levels([(f"B|{FAM[l]}", f"B|{l}")
                               for l in leaves], ["family", "leaf"]),
    }
    dA = simulate(leaves_a, FAM, batch_seed=5)
    dA["labels"] = np.asarray([f"A|{l}" for l in dA["labels"]])
    dB = simulate(leaves, FAM, batch_seed=6)
    dB["labels"] = np.asarray([f"B|{l}" for l in dB["labels"]])
    harm = harmonize({"A": dA, "B": dB}, trees, n_hvg=700, n_boot=100,
                     trust_trees=True)
    labels = {"A": set(trees["A"]["leaves"]),
              "B": set(trees["B"]["leaves"])}
    return harm, trees, labels


def test_strict_certified_only_and_remainder(world):
    harm, trees, labels = world
    cut = consensus_cut(harm, level="strict", leaf_labels=labels)
    assert cut["nodes"], "no certified backbone found"
    for nd in cut["nodes"].values():
        assert nd["tier"] == "certified" and nd["n_datasets"] >= 2
    # every original label is accounted for: merged into a meta-clade
    # or listed in the remainder
    accounted = set(cut["alias_index"]) | \
        {r["display"] for r in cut["unresolved"]} | \
        {m for r in cut["unresolved"] for m in r["members"].values()}
    for ds in labels:
        assert labels[ds] <= accounted


def test_default_completeness_and_tiers(world):
    harm, trees, labels = world
    cut = consensus_cut(harm, level="default", leaf_labels=labels)
    # every original label lands in alias_index or unresolved
    accounted = set(cut["alias_index"]) | \
        {r["display"] for r in cut["unresolved"]} | \
        {m for r in cut["unresolved"] for m in r["members"].values()}
    for ds in labels:
        assert labels[ds] <= accounted
    # the private F3 family is present and tiered atlas_specific
    tiers = {cut["nodes"][i]["tier"] for i in cut["nodes"]
             if any("F3" in a for a in cut["nodes"][i]["aliases"])}
    assert "atlas_specific" in tiers
    # parents are valid and acyclic
    for i, nd in cut["nodes"].items():
        seen = {i}
        p = nd["parent"]
        while p is not None:
            assert p in cut["nodes"] and p not in seen
            seen.add(p)
            p = cut["nodes"][p]["parent"]


def test_view_ordering(world):
    harm, trees, labels = world
    ns = [len(consensus_cut(harm, level=lv, leaf_labels=labels)["nodes"])
          for lv in ("strict", "default", "complete")]
    assert ns[0] <= ns[1] <= ns[2]


def test_serialized_input_identical(world):
    harm, trees, labels = world
    serial = json.loads(json.dumps(
        {i: {"parent": nd.get("projected_parent",
                              nd.get("parent")),
             "status": nd["status"],
             "members": nd["members"], "aliases": nd["aliases"],
             "display": nd["display"],
             "assembly_repair": bool(nd.get("assembly_repair"))}
         for i, nd in harm["tree"].items()}))
    a = consensus_cut(harm, level="default", leaf_labels=labels)
    b = consensus_cut(serial, level="default", leaf_labels=labels)
    assert a["alias_index"] == b["alias_index"]
    assert sorted(a["nodes"]) == sorted(b["nodes"])
    assert a["summary"] == b["summary"]


def test_cut_to_tree_and_newick(world):
    harm, trees, labels = world
    cut = consensus_cut(harm, level="default", leaf_labels=labels)
    t = cut_to_tree(cut)
    assert set(t["leaves"]) <= set(t["parent"])
    for c, kids in t["children"].items():
        for k in kids:
            assert t["parent"][k] == c or (c == "root" and
                                           t["parent"][k] == "root")
    nwk = to_newick(t)
    assert nwk.endswith(";") and nwk.count("(") == nwk.count(")")


def test_unresolved_ledger_keeps_reason(world):
    harm, trees, labels = world
    # inject a synthetic unplaced node the way route_rejected would
    nodes = {i: dict(nd) for i, nd in harm["tree"].items()}
    nodes["MA-X9999"] = {"parent": None,
                         "status": "unplaced_single_atlas",
                         "members": {"B": "B|ghost"},
                         "aliases": ["B|ghost"], "display": "ghost",
                         "assembly_repair": False}
    prov = [{"consensus_node": "MA-X9999",
             "rejection_reason": "ancestry_incompatible"}]
    cut = consensus_cut(nodes, level="default",
                        leaf_labels={"A": labels["A"],
                                     "B": labels["B"] | {"B|ghost"}},
                        provenance_rows=prov)
    led = {r["node_id"]: r for r in cut["unresolved"]}
    assert led["MA-X9999"]["reason"] == "ancestry_incompatible"
    assert "MA-X9999" not in cut["nodes"]


def _dag_harm():
    """Minimal quotient-era harm dict: one vertex with two minimal
    parents (a real certificate)."""
    nodes = {
        "MA-Q0001": {"projected_parent": None, "parents": [],
                     "conflicting": True, "status": "backbone",
                     "members": {"A": "A|p1"}, "aliases": ["A|p1"],
                     "display": "p1"},
        "MA-Q0002": {"projected_parent": None, "parents": [],
                     "conflicting": True, "status": "backbone",
                     "members": {"B": "B|p2"}, "aliases": ["B|p2"],
                     "display": "p2"},
        "MA-Q0003": {"projected_parent": "MA-Q0001",
                     "parents": ["MA-Q0001", "MA-Q0002"],
                     "conflicting": True, "status": "backbone",
                     "members": {"A": "A|x", "B": "B|x"},
                     "aliases": ["A|x", "B|x"], "display": "x"},
    }
    return {"tree": nodes, "is_forest": False,
            "certificates": [{"node": "MA-Q0003",
                              "minimal_parents": ["MA-Q0001",
                                                  "MA-Q0002"]}]}


def test_cut_refuses_dag_without_projection_policy():
    import pytest
    from metaarbor.consensus.cut import consensus_cut
    with pytest.raises(ValueError, match="DAG"):
        consensus_cut(_dag_harm())


def test_cut_projected_carries_certificates():
    from metaarbor.consensus.cut import consensus_cut
    cut = consensus_cut(_dag_harm(), projection="projected_parent")
    assert cut["projection"] == "projected_parent"
    assert cut["certificates"] == _dag_harm()["certificates"]
    assert cut["summary"]["n_certificates"] == 1


def test_from_harmonize_refuses_dag_without_policy():
    import pytest
    from metaarbor.projection import from_harmonize
    h = _dag_harm()
    with pytest.raises(ValueError, match="DAG"):
        from_harmonize(h)
    tree, _ = from_harmonize(h, projection="projected_parent")
    assert "MA-Q0003" in tree["parent"]


def test_viz_marks_projected_parent():
    from metaarbor.viz import nested_from_harmonize
    h = _dag_harm()
    nested = nested_from_harmonize(h["tree"],
                                   {"A|p1", "B|p2", "A|x", "B|x"})

    def find(n):
        hits = []
        if n.get("edge") == "projected":
            hits.append(n["label"])
        for c in n["children"]:
            hits.extend(find(c))
        return hits
    assert find(nested) == ["A|x&B|x"]
