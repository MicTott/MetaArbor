"""Invariant gates for consensus.quotient (ASSEMBLY2 rev2, v1)."""
from metaarbor.consensus.quotient import quotient_assemble


def tree(parent):
    children = {}
    for n, p in parent.items():
        children.setdefault(p, []).append(n)
    for n in parent:
        children.setdefault(n, [])
    leaves = [n for n in parent if not children[n]]
    return {"parent": parent, "children": children, "leaves": leaves}


def ident(t):
    return {n: n for n in t["parent"]}


def mk(sel, s=1.0):
    return {"selected": sel, "matched": True, "support": s}


def retina_shaped():
    """A: n6{n4{c7, n1{c8,c9}}, n5}; B: n12{n10{b1,b2}, n11}.
    Reciprocal: n6<->n12, n1<->n10. c27-shaped: n4 unmerged."""
    ta = tree({"c7": "n4", "n1": "n4", "c8": "n1", "c9": "n1",
               "n4": "n6", "n5": "n6", "c0": "n5", "n6": "root"})
    tb = tree({"b1": "n10", "b2": "n10", "n10": "n12",
               "b3": "n11", "n11": "n12", "n12": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    dec = {"A>B": {"n6": mk("n12"), "n1": mk("n10")},
           "B>A": {"n12": mk("n6"), "n10": mk("n1"),
                   "b3": mk("c0", 0.9)}}
    return trees, canon, dec


def test_conservation_and_interleave_automatic():
    trees, canon, dec = retina_shaped()
    g = quotient_assemble(trees, canon, dec)
    # I1: every leaf present exactly once (asserted internally too)
    labels = {(ds, l) for ds in trees for l in trees[ds]["leaves"]}
    placed = {(ds, m) for v in g["vertices"].values()
              for ds, m in v["members"].items()
              if m in trees[ds]["leaves"]}
    assert placed == labels
    # the merged [n1,n10] vertex sits under UNMERGED n4 (interleaving
    # automatic: n4 was never deleted) and n4 under [n6,n12]
    vid_of = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            vid_of[(ds, m)] = r
    v_n1 = vid_of[("A", "n1")]
    assert vid_of[("B", "n10")] == v_n1          # merged
    v_n4 = vid_of[("A", "n4")]
    assert g["parents"][v_n1] == [v_n4]          # minimal parent = n4
    assert g["parents"][vid_of[("A", "c7")]] == [v_n4]
    v_n6 = vid_of[("A", "n6")]
    assert vid_of[("B", "n12")] == v_n6
    assert g["parents"][v_n4] == [v_n6]
    assert g["is_forest"]


def test_annotations_never_order():
    trees, canon, dec = retina_shaped()
    g = quotient_assemble(trees, canon, dec)
    # b3 -> c0 one-way call becomes an annotation, not structure
    ann = [a for a in g["annotations"] if a["source"] == "b3"]
    assert len(ann) == 1 and not ann[0]["within_merge"]
    vid_of = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            vid_of[(ds, m)] = r
    # b3's parent is still its input parent's class (n11 -> n12)
    assert g["parents"][vid_of[("B", "b3")]] == \
        [vid_of[("B", "n11")]]


def test_order_reversal_refused():
    """A merge that would invert ancestry is refused and ledgered."""
    ta = tree({"a1": "nA", "a2": "nA", "nA": "root"})
    tb = tree({"b1": "nB", "b2": "nB", "nB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    # nA <-> b1 (internal to leaf) AND a1 <-> nB: accepting both
    # yields [nA,b1] <= [nB,a1] <= [nA...] cycle
    dec = {"A>B": {"nA": mk("b1"), "a1": mk("nB", 0.9)},
           "B>A": {"b1": mk("nA"), "nB": mk("a1", 0.9)}}
    g = quotient_assemble(trees, canon, dec)
    # higher-support pair accepted, second refused as incompatible
    assert any(r["reason"] == "incompatible"
               for r in g["ledger"]["refused"])
    assert g["is_forest"]


def test_tie_rule_independent_accepted_competing_unresolved():
    ta = tree({"a1": "nA", "a2": "nA", "a3": "nA", "nA": "root"})
    tb = tree({"b1": "nB", "b2": "nB", "b3": "nB", "nB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    # tie at support 1.0: a1<->b1 and a2<->b2 independent (accept
    # both); a3 competes: a3<->b3 and ALSO nA... make competing via
    # shared endpoint: a3<->b3 and a3... can't have two selections
    # from a3; compete on b3: a3->b3 & nA->b3 reciprocated? b3 can
    # only select one. Compete via b3's single reverse: candidates
    # (a3,b3) needs b3->a3. Build second candidate sharing a3? Not
    # expressible. So competing case: two candidates sharing target
    # slot across pairs is impossible in K=2 by function-ness; use
    # K=3: a3<->c1 and b3<->c1 share endpoint c1.
    tc = tree({"c1": "nC", "c2": "nC", "nC": "root"})
    trees["C"] = tc
    canon["C"] = ident(tc)
    dec = {"A>B": {"a1": mk("b1"), "a2": mk("b2")},
           "B>A": {"b1": mk("a1"), "b2": mk("a2")},
           "A>C": {"a3": mk("c1")}, "C>A": {"c1": mk("a3")},
           "B>C": {"b3": mk("c1")}, "C>B": {"c1": mk("b3")}}
    # C>A and C>B both from c1 — a real endpoint competition at equal
    # support. NOTE: c1 has ONE call per direction; both pairs are
    # reciprocal and tie.
    g = quotient_assemble(trees, canon, dec)
    vid_of = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            vid_of[(ds, m)] = r
    assert vid_of[("A", "a1")] == vid_of[("B", "b1")]   # accepted
    assert vid_of[("A", "a2")] == vid_of[("B", "b2")]   # accepted
    # competing pair(s) involving c1 unresolved, c1 unmerged
    assert any(u for u in g["ledger"]["unresolved_ties"])
    assert vid_of[("C", "c1")] not in (vid_of[("A", "a3")],
                                       vid_of[("B", "b3")])


def test_determinism_insertion_and_renaming():
    trees, canon, dec = retina_shaped()
    base = quotient_assemble(trees, canon, dec)

    def rev(d):
        if isinstance(d, dict):
            return {k: rev(d[k]) for k in reversed(list(d))}
        return d
    perm = quotient_assemble(rev(trees), rev(canon), rev(dec))
    assert base["parents"] == perm["parents"]
    assert base["is_forest"] == perm["is_forest"]
    # label renaming: rename A-side nodes; structure of output
    # (member multisets and parent shape) must be isomorphic
    ren = {n: f"X{n}" for n in trees["A"]["parent"]}
    ta2 = {"parent": {ren[n]: (ren.get(p, p))
                      for n, p in trees["A"]["parent"].items()},
           "children": {}, "leaves": [ren[l]
                                      for l in trees["A"]["leaves"]]}
    ch = {}
    for n, p in ta2["parent"].items():
        ch.setdefault(p, []).append(n)
    for n in ta2["parent"]:
        ch.setdefault(n, [])
    ta2["children"] = ch
    trees2 = {"A": ta2, "B": trees["B"]}
    canon2 = {"A": {ren[n]: ren[n] for n in trees["A"]["parent"]},
              "B": canon["B"]}
    dec2 = {"A>B": {ren[n]: r for n, r in dec["A>B"].items()},
            "B>A": {n: mk(ren[r["selected"]], r["support"])
                    for n, r in dec["B>A"].items()}}
    g2 = quotient_assemble(trees2, canon2, dec2)
    assert len(g2["vertices"]) == len(base["vertices"])
    assert sorted(len(p) for p in g2["parents"].values()) == \
        sorted(len(p) for p in base["parents"].values())
    assert g2["is_forest"] == base["is_forest"]


def test_multi_parent_certificate():
    """Truly incomparable minimal parents -> DAG + certificate,
    never a forced tree."""
    ta = tree({"a1": "nA1", "a2": "nA2", "nA1": "root",
               "nA2": "root"})
    tb = tree({"b1": "nB", "b2": "nB", "nB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    # merge b1<->a1 (so [a1,b1] has parents nA1 and nB — incomparable)
    dec = {"A>B": {"a1": mk("b1")}, "B>A": {"b1": mk("a1")}}
    g = quotient_assemble(trees, canon, dec)
    assert not g["is_forest"]
    assert len(g["certificates"]) == 1
    cert = g["certificates"][0]
    assert len(cert["minimal_parents"]) == 2


def test_component_statuses():
    trees, canon, dec = retina_shaped()
    g = quotient_assemble(trees, canon, dec)
    # everything connects through the merged roots here -> anchored
    assert set(g["component_status"].values()) == {"anchored"}
    # remove all decisions -> no merges -> two unanchored components
    g2 = quotient_assemble(trees, canon, {})
    assert set(g2["component_status"].values()) == {"unanchored"}
    assert all(s == "atlas_specific" for s in g2["statuses"].values())
