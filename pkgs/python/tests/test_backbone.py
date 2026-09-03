"""Hierarchical greedy backbone: acceptance order (ancestors first),
parent linkage, class separation (backbone/private/unknown — 1/1 can
never outrank multi-dataset support), unresolved_in_dataset semantics,
private subtree topology preservation, and edge-conflict classification."""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.consensus.backbone import (FROZEN, classify_edge_conflicts,
                                          greedy_backbone,
                                          provenance_table)


def toy_tree(key, fams):
    rows = []
    for f, leaves in fams.items():
        for l in leaves:
            rows.append((f"{key}|{f}", f"{key}|{l}"))
    return tree_from_levels(rows, ["family", "leaf"])


def labels_for(key, fams, per_leaf):
    out = []
    for f, leaves in fams.items():
        for l in leaves:
            out += [f"{key}|{l}"] * per_leaf.get(l, 50)
    return np.asarray(out)


def cand(members, kind="multi", inv=1.0, edges=None, prov=None):
    return {"members": members, "kind": kind, "seed_invariance": inv,
            "reciprocal_edges": edges or [], "asymmetric_edges": [],
            "missing": [], "nearest_matched_ancestor": {},
            "provenance": prov or {}, "candidate_id": "x"}


@pytest.fixture()
def world():
    fams = {"F1": ["F1.a", "F1.b"], "F2": ["F2.a", "F2.b"]}
    fams_d2 = dict(fams, P=["p1", "p2"], R=["r1"])
    trees = {k: toy_tree(k, fams) for k in ("d0", "d1")}
    trees["d2"] = toy_tree("d2", fams_d2)
    per = {"p1": 20, "p2": 20, "r1": 1}
    datasets = {
        "d0": {"labels": labels_for("d0", fams, {})},
        "d1": {"labels": labels_for("d1", fams, {})},
        "d2": {"labels": labels_for("d2", fams_d2, per)},
    }
    e = [{"from": ("d0", "x"), "to": ("d1", "y"),
          "support_ij": 0.9, "support_ji": 0.9}]
    cands = [
        cand({f"d{i}": f"family:d{i}|F1" for i in range(3)}, edges=e * 3),
        cand({f"d{i}": f"family:d{i}|F2" for i in range(3)}, edges=e * 3),
        cand({f"d{i}": f"d{i}|F1.a" for i in range(3)}, edges=e * 3),
        cand({f"d{i}": f"d{i}|F1.b" for i in range(3)}, edges=e * 3),
        # private singleton with clear power elsewhere (40/440 in d2)
        cand({"d2": "family:d2|P"}, kind="private_subtree",
             prov={"subtree_nodes": ["family:d2|P", "d2|p1", "d2|p2"]}),
        # unpowered singleton (1 cell): must be unknown, not private
        cand({"d2": "d2|r1"}, kind="private_subtree",
             prov={"subtree_nodes": ["d2|r1"]}),
    ]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"cand:{i+1:04d}"
    return trees, datasets, {"candidates": cands, "edge_conflicts": []}


def test_order_parents_classes_and_private_topology(world):
    trees, datasets, cand_out = world
    out = greedy_backbone(cand_out, trees, datasets)
    nodes = {nd["candidate_id"]: nd for nd in out["nodes"]}
    # families accepted before leaves; leaf parents = their family node
    f1 = nodes["cand:0001"]
    leaf_a = nodes["cand:0003"]
    assert f1["status"] == "backbone" and f1["parent"] is None
    assert leaf_a["parent"] == f1["id"]
    assert int(f1["id"].split("MA-C")[1]) < int(leaf_a["id"].split("MA-C")[1])
    # full 3/3 support
    assert f1["support"] == (3, 3)
    # private singleton accepted as private with FULL subtree topology
    p = nodes["cand:0005"]
    assert p["status"] == "private"
    assert p["subtree_parent"] == {"d2|p1": "family:d2|P",
                                   "d2|p2": "family:d2|P",
                                   "family:d2|P": None}
    # unpowered singleton -> unknown, never private, never backbone
    assert "cand:0006" not in nodes
    assert any(u["candidate"]["candidate_id"] == "cand:0006"
               for u in out["unknown"])
    # raw eligibility rows exist for every (dataset, candidate) pair
    assert len(out["eligibility_table"]) == 3 * len(cand_out["candidates"])
    # frozen thresholds recorded with the output
    assert out["frozen"] == FROZEN
    # provenance: every member label mapped, names preserved as synonyms
    prov = provenance_table(out)
    assert any(r["source_node"] == "d0|F1.a" and r["status"] == "backbone"
               for r in prov)


def test_unresolved_in_dataset_excluded_from_denominator(world):
    trees, datasets, cand_out = world
    # drop d0 from the F1.a leaf candidate; its members' one-way walks into
    # d0 land AT d0's F1 family (containment without resolution)
    cand_out["candidates"][2]["members"] = {"d1": "d1|F1.a",
                                            "d2": "d2|F1.a"}
    selections = {("d1", "d0"): {"d1|F1.a": {"selected": "family:d0|F1",
                                             "support": 0.8,
                                             "matched": True}}}
    out = greedy_backbone(cand_out, trees, datasets, selections=selections)
    rows = [r for r in out["eligibility_table"]
            if r["candidate"] == "cand:0003"]
    d0 = next(r for r in rows if r["dataset"] == "d0")
    assert d0["call"] == "unresolved_in_dataset"
    nd = next(nd for nd in out["nodes"]
              if nd["candidate_id"] == "cand:0003")
    assert nd["support"] == (2, 2)     # d0 out of the denominator entirely


def test_edge_conflict_classification(world):
    trees, _, cand_out = world
    cands = cand_out["candidates"]
    conflicts = [
        # refused endpoint is a CHILD of the group's d0 member -> resolution
        {"edge": ("d0", "d0|F1.a", "d1", "family:d1|F1"),
         "support": 0.5, "reason": "dataset_uniqueness"},
        # refused endpoint is a DISJOINT family -> genuine conflict
        {"edge": ("d0", "family:d0|F2", "d1", "family:d1|F1"),
         "support": 0.4, "reason": "dataset_uniqueness"},
    ]
    out = classify_edge_conflicts(conflicts, cands, trees)
    assert out[0]["class"] == "resolution_mismatch"
    assert out[1]["class"] == "genuine_conflict"
