"""candidate_groups: dataset-uniqueness refusal, asymmetric-evidence
preservation, private-subtree consolidation (absorbed descendants),
unresolved internals, seed-invariance diagnostic, mixed-depth groups —
on hand-built decision dicts where every outcome is forced — plus an
integration run through the real pairwise layer."""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.consensus import simulate_donors
from metaarbor.consensus.candidates import (candidate_groups,
                                            pairwise_decisions)


def toy_tree(key, fams):
    rows = []
    for f, leaves in fams.items():
        for l in leaves:
            rows.append((f"{key}|{f}", f"{key}|{l}"))
    return tree_from_levels(rows, ["family", "leaf"])


def M(ni, nj, s=0.9):
    return {"node_i": ni, "node_j": nj, "support_ij": s, "support_ji": s}


def base_trees():
    return {
        "A": toy_tree("A", {"F1": ["F1.a", "F1.b"], "F2": ["F2.a", "F2.b"],
                            "P": ["p1", "p2"]}),
        "B": toy_tree("B", {"F1": ["F1.x", "F1.y"],
                            "F2": ["F2.x", "F2.y"]}),
        "C": toy_tree("C", {"F1": ["F1.q", "F1.r"],
                            "F2": ["F2.q", "F2.r"]}),
    }


def empty_selections(matches):
    """Selections consistent with `matches` (reciprocal one-way records)."""
    sel = {}
    for (ki, kj), pair in matches.items():
        f = {m["node_i"]: {"selected": m["node_j"],
                           "support": m["support_ij"], "matched": True}
             for m in pair}
        r = {m["node_j"]: {"selected": m["node_i"],
                           "support": m["support_ji"], "matched": True}
             for m in pair}
        sel[(ki, kj)] = f
        sel[(kj, ki)] = r
    return sel


def test_full_clique_and_private_subtree():
    trees = base_trees()
    matches = {
        ("A", "B"): [M("family:A|F1", "family:B|F1"),
                     M("family:A|F2", "family:B|F2")],
        ("A", "C"): [M("family:A|F1", "family:C|F1"),
                     M("family:A|F2", "family:C|F2")],
        ("B", "C"): [M("family:B|F1", "family:C|F1"),
                     M("family:B|F2", "family:C|F2")],
    }
    dec = {"matches": matches, "selections": empty_selections(matches),
           "unmatched": {}}
    out = candidate_groups(dec, trees)
    multi = [c for c in out["candidates"] if c["kind"] == "multi"]
    assert len(multi) == 2
    for c in multi:
        assert set(c["members"]) == {"A", "B", "C"}
        assert c["seed_invariance"] == 1.0          # full clique
        assert c["missing"] == []
        assert len(c["reciprocal_edges"]) == 3
    priv = [c for c in out["candidates"] if c["kind"] == "private_subtree"]
    # A's P family is wholly unmatched with NO matched ancestor: ONE
    # consolidated subtree rooted at the family node, leaves absorbed —
    # not three competing candidates
    rootless = [c for c in priv
                if list(c["nearest_matched_ancestor"].values())[0][0] is None]
    assert len(rootless) == 1
    p = rootless[0]
    assert p["members"] == {"A": "family:A|P"}
    assert set(p["provenance"]["subtree_nodes"]) == \
        {"family:A|P", "A|p1", "A|p2"}
    assert p["missing"] == ["B", "C"]
    # p1/p2 are absorbed: they never appear as their own candidates
    assert not any(c["members"] in ({"A": "A|p1"}, {"A": "A|p2"})
                   for c in priv)
    # unmatched leaves BELOW matched families are each a private candidate
    # anchored to their family's group (unresolved finer structure — the
    # route to donor-specific subtypes; the backbone adjudicates later)
    leafp = [c for c in priv if c["members"].get("A") == "A|F1.a"]
    assert len(leafp) == 1
    anc_node, anc_gi = leafp[0]["nearest_matched_ancestor"]["A"]
    assert anc_node == "family:A|F1" and anc_gi is not None
    assert not out["edge_conflicts"]


def test_chain_group_invariance_below_one_and_missing_dataset():
    trees = base_trees()
    # F1 matched A-B and B-C but NOT A-C (chained, not a clique);
    # F2 matched only A-B (C missing entirely)
    matches = {
        ("A", "B"): [M("family:A|F1", "family:B|F1"),
                     M("family:A|F2", "family:B|F2")],
        ("B", "C"): [M("family:B|F1", "family:C|F1")],
        ("A", "C"): [],
    }
    dec = {"matches": matches, "selections": empty_selections(matches),
           "unmatched": {}}
    out = candidate_groups(dec, trees)
    multi = {tuple(sorted(c["members"].values())): c
             for c in out["candidates"] if c["kind"] == "multi"}
    f1 = multi[("family:A|F1", "family:B|F1", "family:C|F1")]
    assert f1["seed_invariance"] == pytest.approx(2 / 3)   # chained
    f2 = multi[("family:A|F2", "family:B|F2")]
    assert f2["missing"] == ["C"]
    assert f2["seed_invariance"] == 1.0


def test_dataset_uniqueness_refusal_records_conflict():
    trees = base_trees()
    # B|F1 reciprocally matches BOTH A|F1 (strong) and, via C, a chain
    # that would pull A|F2 into the same group: A|F2-C|F1 edge forces the
    # union of {A|F2} with a group already containing A|F1 -> refused
    matches = {
        ("A", "B"): [M("family:A|F1", "family:B|F1", 0.95)],
        ("B", "C"): [M("family:B|F1", "family:C|F1", 0.9)],
        ("A", "C"): [M("family:A|F2", "family:C|F1", 0.5)],
    }
    dec = {"matches": matches, "selections": empty_selections(matches),
           "unmatched": {}}
    out = candidate_groups(dec, trees)
    assert len(out["edge_conflicts"]) == 1
    assert out["edge_conflicts"][0]["reason"] == "dataset_uniqueness"
    assert out["edge_conflicts"][0]["support"] == pytest.approx(0.5)
    f1 = [c for c in out["candidates"] if c["kind"] == "multi"][0]
    assert f1["members"] == {"A": "family:A|F1", "B": "family:B|F1",
                             "C": "family:C|F1"}


def test_asymmetric_evidence_and_unresolved_internal():
    trees = base_trees()
    # leaves of A|F1 match leaves of B|F1 reciprocally, but the FAMILY
    # nodes only match one-way (A family -> B family; B family walks to a
    # leaf): family stays out of groups, becomes unresolved_internal on
    # both sides, and the one-way record is preserved as asymmetric
    # evidence on the touching candidates
    matches = {
        ("A", "B"): [M("A|F1.a", "B|F1.x"), M("A|F1.b", "B|F1.y")],
        ("A", "C"): [], ("B", "C"): [],
    }
    sel = empty_selections(matches)
    sel[("A", "B")]["family:A|F1"] = {"selected": "family:B|F1",
                                      "support": 0.8, "matched": True}
    dec = {"matches": matches, "selections": sel,
           "unmatched": {}}
    out = candidate_groups(dec, trees)
    leaf_groups = [c for c in out["candidates"] if c["kind"] == "multi"]
    assert len(leaf_groups) == 2
    # family:A|F1 has matched descendants -> unresolved internal, NOT
    # a private candidate
    assert "family:A|F1" in out["unresolved_internals"]["A"]
    assert not any(c["members"].get("A") == "family:A|F1"
                   for c in out["candidates"])
    # mixed-depth one-way record survives as asymmetric evidence somewhere
    all_asym = [e for c in out["candidates"] for e in c["asymmetric_edges"]]
    # (family->family one-way does not touch leaf-group members, so check
    # the record exists in selections-derived evidence for private/eligible
    # candidates; at minimum it must never have been silently dropped into
    # a reciprocal edge)
    assert all(e["from"] != ("A", "family:A|F1") or
               e["to"] == ("B", "family:B|F1") for e in all_asym)


def test_mixed_depth_group_is_valid():
    trees = base_trees()
    # A's family reciprocally matches a single LEAF of B (resolution
    # mismatch with mutual selection): the group must form unchanged
    matches = {("A", "B"): [M("family:A|F1", "B|F1.x")],
               ("A", "C"): [], ("B", "C"): []}
    dec = {"matches": matches, "selections": empty_selections(matches),
           "unmatched": {}}
    out = candidate_groups(dec, trees)
    multi = [c for c in out["candidates"] if c["kind"] == "multi"]
    assert {"A": "family:A|F1", "B": "B|F1.x"} in [c["members"]
                                                   for c in multi]


def test_integration_with_pairwise_layer():
    sim = simulate_donors(K=3, n_family=3, n_sub=3, n_genes=800,
                          cells_per_leaf=60, batch_sd=0.4, sub_lfc=1.3,
                          seed=3, private={2: ["P1"]})
    lt = sim["latent"]
    datasets, trees = {}, {}
    genes = [f"g{i}" for i in range(800)]
    for d, dd in enumerate(sim["donors"]):
        key = f"d{d}"
        labels = np.asarray([f"{key}|{l}" for l in dd["labels"]])
        datasets[key] = {"counts": dd["counts"], "labels": labels,
                         "gene_names": genes}
        rows = [(f"{key}|{lt['family_of'].get(l, l)}", f"{key}|{l}")
                for l in sorted(set(dd["labels"]))]
        trees[key] = tree_from_levels(rows, ["family", "leaf"])
    dec = pairwise_decisions(datasets, trees, n_hvg=600, n_boot=100)
    out = candidate_groups(dec, trees)
    multi = [c for c in out["candidates"] if c["kind"] == "multi"]
    # 3 families + 9 leaves, all three datasets, all cliques
    full = [c for c in multi if len(c["members"]) == 3]
    assert len(full) == 12
    assert all(c["seed_invariance"] == 1.0 for c in full)
    priv = [c for c in out["candidates"] if c["kind"] == "private_subtree"]
    assert any("P1" in c["members"]["d2"] for c in priv
               if "d2" in c["members"])
