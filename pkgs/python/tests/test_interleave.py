"""Synthetic gates for consensus.interleave (reviewer-specified:
ancestry preservation, conflict, cycle, duplication, order
invariance, and field immutability)."""
import pytest

from metaarbor.consensus.interleave import interleave_core_ancestry


def tree(parent):
    children = {}
    for n, p in parent.items():
        children.setdefault(p, []).append(n)
    for n in parent:
        children.setdefault(n, [])
    return {"parent": parent, "children": children,
            "leaves": [n for n in parent if not children[n]]}


def world():
    """Retina-shaped case: atlas A has OFF clade n4 = {c7, n1{c8,c9}};
    n1 is certified with B's n10; n4 is unrepresented; c7 hoisted."""
    ta = tree({"c7": "n4", "n1": "n4", "c8": "n1", "c9": "n1",
               "n4": "n6", "n5": "n6", "n6": "root", "c0": "n5"})
    tb = tree({"b1": "n10", "b2": "n10", "n10": "n12",
               "n11": "n12", "n12": "root", "b3": "n11"})
    canonical = {"A": {n: n for n in ta["parent"]},
                 "B": {n: n for n in tb["parent"]}}
    nodes = {
        "C1": {"parent": None, "status": "backbone",
               "members": {"A": "n6", "B": "n12"}, "aliases": []},
        "C3": {"parent": "C1", "status": "backbone",
               "members": {"A": "n1", "B": "n10"}, "aliases": []},
        "U1": {"parent": "C1", "status": "single_atlas",
               "members": {"A": "c7"}, "aliases": ["c7"]},
        "U2": {"parent": "C3", "status": "single_atlas",
               "members": {"A": "c8"}, "aliases": ["c8"]},
    }
    return nodes, {"A": ta, "B": tb}, canonical


def test_ancestry_preservation_restores_grouping():
    nodes, it, canon = world()
    out, ledger = interleave_core_ancestry(nodes, it, canon)
    assert ledger == []
    inter = [i for i, nd in out.items() if nd.get("interleaved")]
    assert len(inter) == 1
    i4 = inter[0]
    assert out[i4]["members"] == {"A": "n4"}
    # both the hoisted label and the certified child nest under n4
    assert out["U1"]["parent"] == i4
    assert out["C3"]["parent"] == i4
    assert out[i4]["parent"] == "C1"
    # nodes inside the certified child are untouched
    assert out["U2"]["parent"] == "C3"


def test_conflict_two_atlas_chains_ledgered():
    nodes, it, canon = world()
    # give B an unrepresented intermediate above n10 as well
    itb = tree({"b1": "n10", "b2": "n10", "n10": "n9",
                "n9": "n12", "n11": "n12", "n12": "root",
                "b3": "n11"})
    canon2 = {"A": canon["A"], "B": {n: n for n in itb["parent"]}}
    out, ledger = interleave_core_ancestry(
        nodes, {"A": it["A"], "B": itb}, canon2)
    assert ledger == ["C3"]
    assert out["C3"]["parent"] == "C1"       # unchanged, not forced


def test_no_cycles_and_no_duplication():
    nodes, it, canon = world()
    out, _ = interleave_core_ancestry(nodes, it, canon)
    for i in out:                            # acyclic walk
        seen, x = set(), i
        while x is not None:
            assert x not in seen
            seen.add(x)
            x = out[x].get("parent")
    labels = sorted((d, m) for i, nd in out.items()
                    if not nd.get("interleaved")
                    for d, m in nd["members"].items())
    assert labels == sorted((d, m) for nd in nodes.values()
                            for d, m in nd["members"].items())


def test_order_invariance():
    nodes, it, canon = world()
    base, l1 = interleave_core_ancestry(nodes, it, canon)

    def rev(d):
        if isinstance(d, dict):
            return {k: rev(d[k]) for k in reversed(list(d))}
        return d
    perm, l2 = interleave_core_ancestry(rev(nodes), rev(it),
                                        rev(canon))
    assert {i: nd["parent"] for i, nd in base.items()} == \
        {i: nd["parent"] for i, nd in perm.items()}
    assert l1 == l2


def test_memberships_status_support_unchanged():
    nodes, it, canon = world()
    for nd in nodes.values():
        nd["support"] = (0.9, 0.8)
        nd["display"] = "x"
    out, _ = interleave_core_ancestry(nodes, it, canon)
    for i, nd in nodes.items():
        for k, v in nd.items():
            if k != "parent":
                assert out[i][k] == v


def test_unary_intermediates_collapse():
    """An unrepresented chain with only one downstream child must not
    materialize."""
    ta = tree({"c1": "u1", "u1": "u2", "u2": "n6", "n6": "root",
               "c2": "n6"})
    tb = tree({"b1": "n12", "n12": "root"})
    canon = {"A": {n: n for n in ta["parent"]},
             "B": {n: n for n in tb["parent"]}}
    nodes = {"C1": {"parent": None, "status": "backbone",
                    "members": {"A": "n6", "B": "n12"},
                    "aliases": []},
             "U1": {"parent": "C1", "status": "single_atlas",
                    "members": {"A": "c1"}, "aliases": ["c1"]}}
    out, _ = interleave_core_ancestry(nodes, ta and
                                      {"A": ta, "B": tb}, canon)
    assert not any(nd.get("interleaved") for nd in out.values())
    assert out["U1"]["parent"] == "C1"
