"""Pairwise decision layer: reciprocal recovery across three donors with
donor-specific label names, a unary-chain tree (canonicalization must
unify chain nodes), and a private clade (must land in the singleton pool,
never force-matched)."""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.consensus import simulate_donors
from metaarbor.consensus.candidates import (canonical_nodes,
                                            pairwise_decisions)


@pytest.fixture(scope="module")
def three_donors():
    # subtype signal strengthened so leaf-level truth is unambiguous —
    # this test checks the decision layer's mechanics, not the frozen
    # Walk's (already-validated) conservatism on borderline subtypes
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
        rows = []
        for l in sorted(set(dd["labels"])):
            fam = lt["family_of"].get(l, l)      # private P1 -> own family
            if d == 1:
                # unary chain: family -> mid (one per leaf) -> leaf
                rows.append((f"{key}|{fam}", f"{key}|mid:{l}", f"{key}|{l}"))
            else:
                rows.append((f"{key}|{fam}", f"{key}|{l}"))
        cols = ["family", "mid", "leaf"] if d == 1 else ["family", "leaf"]
        trees[key] = tree_from_levels(rows, cols)
    return sim, datasets, trees


def test_canonicalization_unifies_chains(three_donors):
    _, _, trees = three_donors
    nodes, node_map = canonical_nodes(trees["d1"])
    # canonical set contains no chain-interior (mid-level) ids
    assert not any(n.startswith("mid:") for n in nodes)
    # every unary mid node maps to the leaf at its chain end
    chain_ids = [k for k in node_map if k.startswith("mid:")]
    assert chain_ids, "expected unary mid-level nodes in d1's tree"
    for c in chain_ids:
        assert node_map[c] in trees["d1"]["leaves"]


def test_reciprocal_recovery_and_private_pool(three_donors):
    sim, datasets, trees = three_donors
    dec = pairwise_decisions(datasets, trees, n_hvg=600, n_boot=100)
    fams = sim["latent"]["families"]

    for (ki, kj), pair in dec["matches"].items():
        got = {(m["node_i"], m["node_j"]) for m in pair}
        # every latent family node reciprocally matched
        for f in fams:
            assert (f"family:{ki}|{f}", f"family:{kj}|{f}") in got, \
                (ki, kj, f, sorted(got))
        # every latent leaf reciprocally matched to its counterpart
        for l in sim["latent"]["leaves"]:
            assert (f"{ki}|{l}", f"{kj}|{l}") in got, (ki, kj, l)
        # no cross-family confusion
        for (a, b) in got:
            base_a = a.split("|", 1)[1]
            base_b = b.split("|", 1)[1]
            fa = base_a.replace("family:", "").split(".")[0].split("|")[-1]
            fb = base_b.replace("family:", "").split(".")[0].split("|")[-1]
            if not base_a.startswith("P") and not base_b.startswith("P"):
                assert fa[:2] == fb[:2], (a, b)
        # supports are finite for matches
        assert all(np.isfinite(m["support_ij"]) and
                   np.isfinite(m["support_ji"]) for m in pair)

    # the private clade never force-matches: it sits in the singleton pool
    assert any("P1" in n for n in dec["unmatched"]["d2"]), \
        dec["unmatched"]["d2"]
    # and no match anywhere involves P1
    for pair in dec["matches"].values():
        assert not any("P1" in m["node_i"] or "P1" in m["node_j"]
                       for m in pair)
