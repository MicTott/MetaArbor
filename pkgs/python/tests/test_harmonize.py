"""harmonize() assembly invariants:

1. COMPLETENESS — every original leaf of every input tree appears in the
   assembled tree (as a member of some node or as a visibly marked
   affiliate alias); consolidated subtrees are expanded with their
   internal topology, never hidden inside a collapsed node.
2. AFFILIATE MARKING — affiliates ride as aliases prefixed with the
   approx sign and NEVER appear in any node's members (so they cannot
   contribute to reciprocal support, whose tuples derive from members).
3. Expanded nodes preserve within-atlas parenthood: an expanded node's
   parent chain stays inside its own consolidated subtree until the
   subtree root.
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
    return harmonize({"A": dA, "B": dB}, trees, n_hvg=700, n_boot=100), \
        trees


def test_every_original_leaf_appears(private_world):
    harm, trees = private_world
    placed = set()
    for nd in harm["tree"].values():
        placed.update(nd["members"].values())
        placed.update(a[2:] for a in nd["aliases"]
                      if isinstance(a, str) and a.startswith("≈ "))
    for ds, tr in trees.items():
        for leaf in tr["leaves"]:
            assert leaf in placed, f"leaf {leaf} missing from assembly"


def test_affiliates_marked_and_never_members(private_world):
    harm, _ = private_world
    member_vals = {m for nd in harm["tree"].values()
                   for m in nd["members"].values()}
    for aff in harm["affiliates"]:
        assert aff["node"] not in member_vals
        host = next(nd for nd in harm["tree"].values()
                    if f'≈ {aff["node"]}' in nd["aliases"])
        assert aff["node"] not in host["members"].values()


def test_expanded_subtree_topology_preserved(private_world):
    harm, trees = private_world
    nodes = harm["tree"]
    exp = {i: nd for i, nd in nodes.items() if nd.get("expanded")}
    assert exp, "private F3 subtree should have expanded nodes"
    for i, nd in exp.items():
        root = i.split(".")[0]
        p = nd["parent"]
        while p is not None and nodes[p].get("expanded"):
            p = nodes[p]["parent"]
        assert p == root, "expanded node must chain to its subtree root"
        assert nd["status"] == nodes[root]["status"]


def _represented_leaves(harm, trees, ds):
    leaves = set(trees[ds]["leaves"])
    rep = set()
    for nd in harm["tree"].values():
        m = nd["members"].get(ds)
        if m in leaves:
            rep.add(m)
    for aff in harm["affiliates"]:
        if aff["dataset"] == ds and aff["node"] in leaves:
            rep.add(aff["node"])
    return rep


def test_completeness_invariant_exact_equality(private_world):
    """input labels == represented output labels, per dataset: every
    input-tree leaf appears (member/affiliate/unplaced) and nothing
    else masquerades as a leaf label."""
    harm, trees = private_world
    for ds in trees:
        assert _represented_leaves(harm, trees, ds) == \
            set(trees[ds]["leaves"])
    assert harm["repairs"] == []      # healthy upstream: no repairs


def test_repair_reinstates_lost_label_with_input_parentage(private_world):
    """If upstream ever loses a label, repair_completeness reinstates it
    as an unplaced_single_atlas node under its nearest represented
    input-tree ancestor, flagged assembly_repair, support (0, 0)."""
    import copy
    from metaarbor.consensus.harmonize import repair_completeness
    harm, trees = private_world
    nodes = copy.deepcopy(
        {i: {k: v for k, v in nd.items() if k != "children"}
         for i, nd in harm["tree"].items()})
    victim_ds, victim = "B", "B|F1.s1"
    doomed = [i for i, nd in nodes.items()
              if nd["members"].get(victim_ds) == victim]
    for i in doomed:
        for c in [j for j, nd in nodes.items() if nd["parent"] == i]:
            nodes[c]["parent"] = nodes[i]["parent"]
        del nodes[i]
    repairs = repair_completeness(nodes, trees, harm["affiliates"])
    # deleting the meta-clade loses BOTH atlases' labels; repair must
    # reinstate each in its own atlas's parentage
    assert sorted(r["label"] for r in repairs) == ["A|F1.s1", victim]
    rid = next(r["node_id"] for r in repairs if r["label"] == victim)
    nd = nodes[rid]
    assert nd["status"] == "unplaced_single_atlas"
    assert nd["assembly_repair"] is True
    assert nd["support"] == (0, 0)          # excluded from support counts
    # exact input-tree parentage: nearest represented ancestor of the
    # leaf in B's own tree
    from metaarbor.branch_fit import _collapse_chains
    rp, _ = _collapse_chains(trees[victim_ds])
    p = rp.get(victim)
    while p not in (None, "root"):
        hit = [i for i, x in nodes.items()
               if x["members"].get(victim_ds) == p and i != rid]
        if hit:
            assert nd["parent"] == hit[0]
            break
        p = rp.get(p)
    else:
        assert nd["parent"] is None
