"""harmonize() assembly invariants (quotient assembly):

1. COMPLETENESS — every original leaf of every input tree is a MEMBER
   of exactly one node (I1 by construction; no affiliate aliasing, no
   routing, no repair).
2. ONE-WAY CALLS ARE ANNOTATIONS — an unreciprocated twin (the old
   "affiliate") exists as its own single-atlas member node, and its
   evidence appears in `annotations`, never as structure.
3. PRIVATE SUBTREES keep their full within-atlas topology through the
   inherited ancestry edges (no expansion machinery needed).
4. MULTI-PARENT EXPOSURE — `parents` lists every minimal parent;
   `parent` is the first (a view when `conflicting`); `certificates`
   enumerate unresolved parentage; `is_forest` states the shape.
"""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.consensus.harmonize import harmonize

GENES = [f"g{i}" for i in range(900)]


def simulate(leaves, family_of, n_per=60, batch_seed=0, latent_map=None):
    rs = np.random.RandomState(batch_seed)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    pool = np.random.RandomState(98).permutation(len(GENES))
    fams = sorted(set(family_of.values()))
    fam_idx = {f: pool[i * 40:(i + 1) * 40] for i, f in enumerate(fams)}
    latents = sorted(set((latent_map or {}).get(l, l) for l in leaves))
    sub_idx = {s: pool[len(fams) * 40 + i * 15:
                       len(fams) * 40 + (i + 1) * 15]
               for i, s in enumerate(latents)}
    batch = rs.lognormal(0, 0.4, len(GENES))
    blocks, labels = [], []
    for l in leaves:
        lat = (latent_map or {}).get(l, l)
        mu = base.copy()
        mu[fam_idx[family_of[l]]] *= np.exp(1.2)
        mu[sub_idx[lat]] *= np.exp(1.3)
        lam = np.outer(rs.gamma(10, 0.1, n_per), mu * batch)
        blocks.append(rs.poisson(lam))
        labels += [l] * n_per
    return {"counts": np.vstack(blocks).astype(float),
            "labels": np.asarray(labels), "gene_names": GENES}


@pytest.fixture(scope="module")
def private_world():
    """Atlas B has a whole private family F3 (a consolidated subtree in
    the assembly); atlas A lacks it."""
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2, 3) for j in (1, 2)}
    leaves = sorted(fam)
    leaves_a = [l for l in leaves if not l.startswith("F3")]
    trees = {
        "A": tree_from_levels([(f"A|{fam[l]}", f"A|{l}")
                               for l in leaves_a], ["family", "leaf"]),
        "B": tree_from_levels([(f"B|{fam[l]}", f"B|{l}")
                               for l in leaves], ["family", "leaf"]),
    }
    dA = simulate(leaves_a, fam, batch_seed=5)
    dA["labels"] = np.asarray([f"A|{l}" for l in dA["labels"]])
    dB = simulate(leaves, fam, batch_seed=6)
    dB["labels"] = np.asarray([f"B|{l}" for l in dB["labels"]])
    return harmonize({"A": dA, "B": dB}, trees, n_hvg=700,
                 n_boot=100, trust_trees=True), \
        trees


def test_every_original_leaf_is_a_member(private_world):
    harm, trees = private_world
    placed = {}
    for i, nd in harm["tree"].items():
        for ds, m in nd["members"].items():
            assert (ds, m) not in placed, f"{(ds, m)} placed twice"
            placed[(ds, m)] = i
    for ds, tr in trees.items():
        for leaf in tr["leaves"]:
            assert (ds, leaf) in placed, f"leaf {leaf} missing"


def test_affiliate_evidence_is_annotation_not_structure(private_world):
    harm, _ = private_world
    member_of = {(ds, m) for nd in harm["tree"].values()
                 for ds, m in nd["members"].items()}
    for aff in harm["affiliates"]:
        # the twin exists as a real member somewhere...
        assert (aff["dataset"], aff["node"]) in member_of
        # ...and its one-way evidence is recorded as an annotation
        assert any(a["source"] == aff["node"]
                   for a in harm["annotations"])
        # never as an alias on another node's record
        assert not any(f'\u2248 {aff["node"]}' in nd["aliases"]
                       for nd in harm["tree"].values())


def test_private_subtree_topology_inherited(private_world):
    """B's private F3 family keeps its within-atlas parenthood via
    inherited ancestry: each F3 leaf's parent node holds B's F3
    family node."""
    harm, trees = private_world
    nodes = harm["tree"]
    by_member = {(ds, m): i for i, nd in nodes.items()
                 for ds, m in nd["members"].items()}
    fam = by_member[("B", "family:B|F3")]
    for leaf in ("B|F3.s1", "B|F3.s2"):
        nd = nodes[by_member[("B", leaf)]]
        assert nd["parents"] == [fam]
        assert nd["status"] == nodes[fam]["status"]


def test_multi_parent_contract_exposed(private_world):
    harm, _ = private_world
    assert isinstance(harm["is_forest"], bool)
    assert isinstance(harm["certificates"], list)
    for i, nd in harm["tree"].items():
        assert nd["parents"] == sorted(nd["parents"])
        assert "parent" not in nd          # no plain key: no bypass
        if nd["parents"]:
            assert nd["projected_parent"] == nd["parents"][0]
        else:
            assert nd["projected_parent"] is None
    conflicted = {c["node"] for c in harm["certificates"]}
    conflicted |= {p for c in harm["certificates"]
                   for p in c["minimal_parents"]}
    for i, nd in harm["tree"].items():
        assert nd["conflicting"] == (i in conflicted)
    if harm["is_forest"]:
        assert not harm["certificates"]
        assert all(len(nd["parents"]) <= 1
                   for nd in harm["tree"].values())


def test_backbone_nodes_cite_certification(private_world):
    """Every shared (backbone) node carries the certified merge ids
    that license it — I2 evidence grounding at the harmonize level."""
    harm, _ = private_world
    bb_ids = {nd["id"] for nd in harm["backbone"]["nodes"]
              if nd["status"] == "backbone" and len(nd["members"]) >= 2}
    for nd in harm["tree"].values():
        if nd["status"] == "backbone":
            assert nd["certified_ids"]
            assert set(nd["certified_ids"]) <= bb_ids
        else:
            assert nd["certified_ids"] == []


def test_no_routing_or_repair_exists(private_world):
    """The routing/repair path is retired: the keys are gone, the
    status is never emitted, and the functions no longer exist."""
    harm, _ = private_world
    assert "repairs" not in harm and "rejection_fallbacks" not in harm
    assert all(nd["status"] in ("backbone", "private", "single_atlas")
               for nd in harm["tree"].values())
    import metaarbor.consensus.harmonize as H
    assert not hasattr(H, "route_rejected")
    assert not hasattr(H, "repair_completeness")


def test_tree_consumers_accept_forest_without_policy(private_world):
    """On a forest result the consumers work unchanged — the DAG
    policy must produce no false alarms."""
    from metaarbor.consensus.cut import consensus_cut
    from metaarbor.projection import from_harmonize
    harm, _ = private_world
    assert harm["is_forest"]
    cut = consensus_cut(harm)
    assert cut["projection"] is None and cut["certificates"] == []
    tree, lmaps = from_harmonize(harm)
    assert tree["leaves"]
