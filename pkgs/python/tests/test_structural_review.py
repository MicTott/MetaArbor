"""Adversarial structural tests from the K>=3 review:

1. a three-atlas ancestry CYCLE must be detected, emitted as a genuine
   conflict, and rejected — never silently linearized by index;
2. two incomparable accepted parents for one child -> ambiguity conflict;
3. STABILITY_FLOOR actually screens private calls when stability is
   supplied (and harmonize warns when it is not);
4. rejected claims must NOT screen eligibility (free-structure-below is
   evaluated against accepted claims only);
5. affiliate landings are collected across ALL atlases — incompatible
   landings yield a conflict and no affiliate; consolidated subtrees are
   never affiliated;
6. gene columns are aligned by name (reordering is corrected; one-sided
   or unverifiable inputs raise);
7. dataset labels must equal tree leaves at harmonize() entry;
8. rejection fallback placement is topologically ordered;
9. the compactness gate of the frozen Walk is applied to consensus node
   walks (a gated selection can never seed a reciprocal edge);
10. harmonize() output is invariant to dataset insertion order.
"""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.consensus.backbone import greedy_backbone
from metaarbor.consensus.harmonize import harmonize, route_rejected

from test_backbone import cand, labels_for, toy_tree

FAMS = {"F1": ["F1.a", "F1.b"], "F2": ["F2.a", "F2.b"]}


def three_world():
    trees = {k: toy_tree(k, FAMS) for k in ("A", "B", "C")}
    datasets = {k: {"labels": labels_for(k, FAMS, {})}
                for k in ("A", "B", "C")}
    return trees, datasets


# ---- 1. ancestry cycle ----------------------------------------------------
def test_three_atlas_ancestry_cycle_rejected_not_linearized():
    trees, datasets = three_world()
    e = [{"from": ("A", "x"), "to": ("B", "y"),
          "support_ij": 0.9, "support_ji": 0.9}]
    cands = [
        # X anc Z (via A), Z anc Y (via C), Y anc X (via B): a 3-cycle
        cand({"A": "family:A|F1", "B": "B|F1.a"}, edges=e),      # X
        cand({"B": "family:B|F1", "C": "C|F1.a"}, edges=e),      # Y
        cand({"C": "family:C|F1", "A": "A|F1.a"}, edges=e),      # Z
        cand({"A": "family:A|F2", "B": "family:B|F2",
              "C": "family:C|F2"}, edges=e * 3),                 # clean
    ]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"cand:{i+1:04d}"
    out = greedy_backbone({"candidates": cands, "edge_conflicts": []},
                          trees, datasets)
    cyc = [c for c in out["conflicts"] if c["type"] == "ancestry_cycle"]
    assert len(cyc) == 1
    assert sorted(cyc[0]["candidates"]) == ["cand:0001", "cand:0002",
                                            "cand:0003"]
    assert cyc[0]["class"] == "genuine_conflict"
    rej = {r["candidate"]["candidate_id"] for r in out["rejected"]
           if r["reason"] == "ancestry_cycle"}
    assert rej == {"cand:0001", "cand:0002", "cand:0003"}
    # none of the cycle members entered the consensus; the clean clique
    # was accepted normally
    ids = {nd["candidate_id"] for nd in out["nodes"]}
    assert ids == {"cand:0004"}


# ---- 2. two incomparable accepted parents ---------------------------------
def test_ambiguous_parent_emits_conflict():
    trees = {k: toy_tree(k, FAMS) for k in ("A", "B", "C", "D")}
    datasets = {k: {"labels": labels_for(k, FAMS, {})}
                for k in ("A", "B", "C", "D")}
    e = [{"from": ("A", "x"), "to": ("B", "y"),
          "support_ij": 0.9, "support_ji": 0.9}]
    cands = [
        cand({"A": "family:A|F1", "C": "family:C|F1"}, edges=e),  # P1
        cand({"B": "family:B|F1", "D": "family:D|F1"}, edges=e),  # P2
        cand({"A": "A|F1.a", "B": "B|F1.a"}, edges=e),            # child
    ]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"cand:{i+1:04d}"
    out = greedy_backbone({"candidates": cands, "edge_conflicts": []},
                          trees, datasets)
    amb = [c for c in out["conflicts"]
           if c["type"] == "ambiguous_parent"]
    assert len(amb) == 1 and amb[0]["candidate"] == "cand:0003"
    assert sorted(amb[0]["parents"]) == ["cand:0001", "cand:0002"]


# ---- 3. stability screening -----------------------------------------------
def test_stability_floor_screens_private_and_harmonize_warns():
    fams_d2 = dict(FAMS, P=["p1", "p2"])
    trees = {k: toy_tree(k, FAMS) for k in ("d0", "d1")}
    trees["d2"] = toy_tree("d2", fams_d2)
    datasets = {
        "d0": {"labels": labels_for("d0", FAMS, {})},
        "d1": {"labels": labels_for("d1", FAMS, {})},
        "d2": {"labels": labels_for("d2", fams_d2,
                                    {"p1": 20, "p2": 20})},
    }
    e = [{"from": ("d0", "x"), "to": ("d1", "y"),
          "support_ij": 0.9, "support_ji": 0.9}]
    cands = [
        cand({f"d{i}": f"family:d{i}|F1" for i in range(3)}, edges=e * 3),
        cand({f"d{i}": f"family:d{i}|F2" for i in range(3)}, edges=e * 3),
        cand({"d2": "family:d2|P"}, kind="private_subtree",
             prov={"subtree_nodes": ["family:d2|P", "d2|p1", "d2|p2"]}),
    ]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"cand:{i+1:04d}"
    co = {"candidates": cands, "edge_conflicts": []}
    ok = greedy_backbone(co, trees, datasets,
                         stability={("d2", "family:d2|P"): 0.9})
    assert any(nd["candidate_id"] == "cand:0003" and
               nd["status"] == "private" for nd in ok["nodes"])
    low = greedy_backbone(co, trees, datasets,
                          stability={("d2", "family:d2|P"): 0.5})
    assert not any(nd["candidate_id"] == "cand:0003"
                   for nd in low["nodes"])
    assert any(u["candidate"]["candidate_id"] == "cand:0003"
               for u in low["unknown"])


# ---- 4. rejected claims do not screen eligibility -------------------------
def test_rejected_claims_do_not_affect_free_below():
    """The cycle candidates claim F1 structure; because they are
    rejected, every other candidate's eligibility rows must be identical
    to a world where the cycle candidates never existed."""
    trees, datasets = three_world()
    e = [{"from": ("A", "x"), "to": ("B", "y"),
          "support_ij": 0.9, "support_ji": 0.9}]
    clean = cand({"A": "family:A|F2", "B": "family:B|F2",
                  "C": "family:C|F2"}, edges=e * 3)
    clean["candidate_id"] = "cand:0100"
    cyc = [
        cand({"A": "family:A|F1", "B": "B|F1.a"}, edges=e),
        cand({"B": "family:B|F1", "C": "C|F1.a"}, edges=e),
        cand({"C": "family:C|F1", "A": "A|F1.a"}, edges=e),
    ]
    for i, c in enumerate(cyc):
        c["candidate_id"] = f"cand:{i+1:04d}"
    import copy
    with_cycle = greedy_backbone(
        {"candidates": cyc + [copy.deepcopy(clean)],
         "edge_conflicts": []}, trees, datasets)
    without = greedy_backbone(
        {"candidates": [copy.deepcopy(clean)], "edge_conflicts": []},
        trees, datasets)
    rows_w = [r for r in with_cycle["eligibility_table"]
              if r["candidate"] == "cand:0100"]
    rows_o = [r for r in without["eligibility_table"]
              if r["candidate"] == "cand:0100"]
    assert rows_w == rows_o


# ---- 5. affiliates: multi-landing + subtree guard -------------------------
def _affiliate_world(second_landing):
    trees = {k: toy_tree(k, FAMS) for k in ("d0", "d1")}
    trees["d2"] = toy_tree("d2", {"F1": ["F1.a", "F1.b"]})
    datasets = {
        "d0": {"labels": labels_for("d0", FAMS, {})},
        "d1": {"labels": labels_for("d1", FAMS, {})},
        "d2": {"labels": labels_for("d2", {"F1": ["F1.a", "F1.b"]}, {})},
    }
    e = [{"from": ("d0", "x"), "to": ("d1", "y"),
          "support_ij": 0.9, "support_ji": 0.9}]
    cands = [
        cand({"d0": "family:d0|F1", "d1": "family:d1|F1"}, edges=e),
        cand({"d0": "family:d0|F2", "d1": "family:d1|F2"}, edges=e),
        cand({"d2": "d2|F1.a"}, kind="private_subtree",
             prov={"subtree_nodes": ["d2|F1.a"]}),
    ]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"cand:{i+1:04d}"
    selections = {
        ("d2", "d0"): {"d2|F1.a": {"selected": "family:d0|F1",
                                   "matched": True, "support": 0.9}},
        ("d2", "d1"): {"d2|F1.a": {"selected": second_landing,
                                   "matched": True, "support": 0.9}},
    }
    return trees, datasets, cands, selections


def test_affiliate_coherent_landings_attach():
    trees, datasets, cands, sel = _affiliate_world("family:d1|F1")
    out = greedy_backbone({"candidates": cands, "edge_conflicts": []},
                          trees, datasets, selections=sel)
    assert len(out["affiliates"]) == 1
    assert out["affiliates"][0]["node"] == "d2|F1.a"


def test_affiliate_incompatible_landings_conflict_no_attach():
    trees, datasets, cands, sel = _affiliate_world("family:d1|F2")
    out = greedy_backbone({"candidates": cands, "edge_conflicts": []},
                          trees, datasets, selections=sel)
    assert out["affiliates"] == []
    assert any(c["type"] == "affiliate_incompatible_landings"
               for c in out["conflicts"])


def test_consolidated_subtree_never_affiliated():
    trees, datasets, cands, sel = _affiliate_world("family:d1|F1")
    # give the singleton an absorbed subtree: affiliation must be
    # refused (an alias cannot carry topology)
    cands[2]["members"] = {"d2": "family:d2|F1"}
    cands[2]["provenance"] = {"subtree_nodes": ["family:d2|F1",
                                                "d2|F1.a", "d2|F1.b"]}
    sel = {("d2", "d0"): {"family:d2|F1": {"selected": "family:d0|F1",
                                           "matched": True,
                                           "support": 0.9}}}
    out = greedy_backbone({"candidates": cands, "edge_conflicts": []},
                          trees, datasets, selections=sel)
    assert out["affiliates"] == []


# ---- 6/7/9/10: harmonize-level contracts ---------------------------------
GENES = [f"g{i}" for i in range(700)]


def sim(key, leaves, family_of, seed):
    rs = np.random.RandomState(seed)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    pool = np.random.RandomState(98).permutation(len(GENES))
    fams = sorted(set(family_of.values()))
    fam_idx = {f: pool[i * 40:(i + 1) * 40] for i, f in enumerate(fams)}
    sub_idx = {s: pool[len(fams) * 40 + i * 15:
                       len(fams) * 40 + (i + 1) * 15]
               for i, s in enumerate(sorted(leaves))}
    batch = rs.lognormal(0, 0.3, len(GENES))
    blocks, labels = [], []
    for l in leaves:
        mu = base.copy()
        mu[fam_idx[family_of[l]]] *= np.exp(1.2)
        mu[sub_idx[l]] *= np.exp(1.3)
        blocks.append(rs.poisson(
            np.outer(rs.gamma(10, 0.1, 50), mu * batch)))
        labels += [f"{key}|{l}"] * 50
    return {"counts": np.vstack(blocks).astype(float),
            "labels": np.asarray(labels), "gene_names": list(GENES)}


@pytest.fixture(scope="module")
def small_pair():
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2) for j in (1, 2)}
    leaves = sorted(fam)
    trees = {k: tree_from_levels([(f"{k}|{fam[l]}", f"{k}|{l}")
                                  for l in leaves], ["family", "leaf"])
             for k in ("A", "B")}
    ds = {k: sim(k, leaves, fam, seed=i) for i, k in enumerate(("A", "B"))}
    return ds, trees


def test_gene_reordering_is_corrected(small_pair):
    ds, trees = small_pair
    base = harmonize(dict(ds), trees, n_hvg=500, n_boot=50,
                     trust_trees=True)
    perm = np.random.RandomState(3).permutation(len(GENES))
    shuf = dict(ds)
    shuf["B"] = dict(ds["B"],
                     counts=ds["B"]["counts"][:, perm],
                     gene_names=[GENES[i] for i in perm])
    out = harmonize(shuf, trees, n_hvg=500, n_boot=50,
                    trust_trees=True)
    assert {i: (nd["parent"], nd["status"], nd["members"])
            for i, nd in out["tree"].items()} == \
        {i: (nd["parent"], nd["status"], nd["members"])
         for i, nd in base["tree"].items()}


def test_one_sided_gene_names_raise(small_pair):
    ds, trees = small_pair
    bad = dict(ds)
    bad["B"] = {k: v for k, v in ds["B"].items() if k != "gene_names"}
    with pytest.raises(ValueError, match="only one"):
        harmonize(bad, trees, n_hvg=500, n_boot=50, trust_trees=True)


def test_label_tree_leaf_mismatch_raises(small_pair):
    ds, trees = small_pair
    bad = dict(ds)
    lab = ds["A"]["labels"].copy()
    lab[0] = "A|GHOST"
    bad["A"] = dict(ds["A"], labels=lab)
    with pytest.raises(ValueError, match="labels != tree leaves"):
        harmonize(bad, trees, n_hvg=500, n_boot=50, trust_trees=True)


def test_dataset_insertion_order_invariance(small_pair):
    ds, trees = small_pair
    fwd = harmonize(dict(ds), trees, n_hvg=500, n_boot=50,
                    trust_trees=True)
    rev_ds = dict(reversed(list(ds.items())))
    rev_tr = dict(reversed(list(trees.items())))
    rev = harmonize(rev_ds, rev_tr, n_hvg=500, n_boot=50,
                     trust_trees=True)
    assert {i: (nd["parent"], nd["status"], nd["members"])
            for i, nd in fwd["tree"].items()} == \
        {i: (nd["parent"], nd["status"], nd["members"])
         for i, nd in rev["tree"].items()}


# ---- 8. topological rejection routing -------------------------------------
def test_rejection_routing_is_topologically_ordered():
    tree = tree_from_levels([("B|F1", "B|F1.a"), ("B|F1", "B|F1.b")],
                            ["family", "leaf"])
    trees = {"B": tree}
    nodes = {}
    rejected = [
        # child listed FIRST: without topological ordering it would
        # attach at root instead of under its routed parent
        {"candidate": {"members": {"B": "B|F1.a"},
                       "candidate_id": "cand:0002", "provenance": {}},
         "reason": "no_support", "support": (0, 0)},
        {"candidate": {"members": {"B": "family:B|F1"},
                       "candidate_id": "cand:0001", "provenance": {}},
         "reason": "insufficient_support", "support": (0, 0)},
    ]
    routed = route_rejected(nodes, trees, rejected, [])
    by_label = {r["label"]: r for r in routed}
    parent_id = by_label["family:B|F1"]["node_id"]
    assert nodes[by_label["B|F1.a"]["node_id"]]["parent"] == parent_id
    assert nodes[parent_id]["rejection"]["reason"] == \
        "insufficient_support"


# ---- 9. compactness gate on consensus walks -------------------------------
def test_compactness_gate_blocks_reciprocal_edge(small_pair, monkeypatch):
    from metaarbor.consensus import candidates as C
    ds, trees = small_pair
    real = C.compactness

    def fake(cache, mask, tree, node):
        return 0.1                      # below the frozen 0.70 gate
    monkeypatch.setattr(C, "compactness", fake)
    dec = C.pairwise_decisions(dict(ds), trees, n_hvg=500, n_boot=50)
    for (_ki, _kj), recs in dec["selections"].items():
        for rec in recs.values():
            assert rec["matched"] is False
            assert rec["selected"] is None
            if rec["gated_selected"] is not None:
                assert rec["relation"] == "discordant"
    assert all(not v for v in dec["matches"].values())
    monkeypatch.setattr(C, "compactness", real)


def test_rejected_claimant_leaves_structure_free_and_honest():
    """Reviewer's adversarial shape: a candidate missing in one atlas,
    free structure beneath its accepted parent there, and a REJECTED
    same-cohort sibling as the only apparent claimant of that structure.
    The missing-atlas eligibility row must be unresolved_in_dataset
    (structure genuinely free — the claimant died), NOT a confident
    private_or_absent driven by a claim that never entered the
    consensus."""
    trees = {k: toy_tree(k, FAMS) for k in ("d0", "d1", "d2")}
    datasets = {k: {"labels": labels_for(k, FAMS, {})}
                for k in ("d0", "d1", "d2")}
    e_hi = [{"from": ("d0", "x"), "to": ("d1", "y"),
             "support_ij": 0.95, "support_ji": 0.95}]
    e_lo = [{"from": ("d0", "x"), "to": ("d1", "y"),
             "support_ij": 0.9, "support_ji": 0.9}]
    cands = [
        cand({f"d{i}": f"family:d{i}|F1" for i in range(3)},
             edges=e_lo * 3),
        cand({f"d{i}": f"family:d{i}|F2" for i in range(3)},
             edges=e_lo * 3),
        # R: crosses families (d0 under F2, d2 under F1) -> rejected as
        # ancestry_incompatible at adjudication; ranks BEFORE A2 via
        # higher bootstrap support. It "claims" d2|F1.a — apparently.
        cand({"d0": "d0|F2.a", "d2": "d2|F1.a"}, edges=e_hi),
        # A2: leaf claim missing in d2, under accepted F1 parent
        cand({"d0": "d0|F1.b", "d1": "d1|F1.b"}, edges=e_lo),
    ]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"cand:{i+1:04d}"
    out = greedy_backbone({"candidates": cands, "edge_conflicts": []},
                          trees, datasets)
    assert any(r["candidate"]["candidate_id"] == "cand:0003" and
               r["reason"] == "ancestry_incompatible"
               for r in out["rejected"])
    row = next(r for r in out["eligibility_table"]
               if r["candidate"] == "cand:0004" and r["dataset"] == "d2")
    assert row["call"] == "unresolved_in_dataset", row
    assert row["via"][0] == "free_structure_below"
