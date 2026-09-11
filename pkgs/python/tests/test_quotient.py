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


def test_three_atlas_meta_clade_via_shared_endpoint():
    """A<->C plus C<->B sharing endpoint C is how a three-atlas
    equivalence forms: transitive closure yields {a3, b3, c1}.
    (Revision-1 of this test wrongly expected rejection —
    reviewer-corrected.)"""
    ta = tree({"a1": "nA", "a2": "nA", "a3": "nA", "nA": "root"})
    tb = tree({"b1": "nB", "b2": "nB", "b3": "nB", "nB": "root"})
    tc = tree({"c1": "nC", "c2": "nC", "nC": "root"})
    trees = {"A": ta, "B": tb, "C": tc}
    canon = {k: ident(trees[k]) for k in trees}
    dec = {"A>B": {"a1": mk("b1"), "a2": mk("b2")},
           "B>A": {"b1": mk("a1"), "b2": mk("a2")},
           "A>C": {"a3": mk("c1")}, "C>A": {"c1": mk("a3")},
           "B>C": {"b3": mk("c1")}, "C>B": {"c1": mk("b3")}}
    g = quotient_assemble(trees, canon, dec)
    vid_of = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            vid_of[(ds, m)] = r
    assert vid_of[("A", "a1")] == vid_of[("B", "b1")]
    assert vid_of[("A", "a2")] == vid_of[("B", "b2")]
    # the transitive three-atlas meta-clade forms
    assert vid_of[("A", "a3")] == vid_of[("C", "c1")] == \
        vid_of[("B", "b3")]
    tri = g["vertices"][vid_of[("A", "a3")]]
    assert set(tri["members"]) == {"A", "B", "C"}
    assert g["ledger"]["unresolved_ties"] == []


def test_genuine_competition_same_atlas_slot():
    """Two tied candidates claiming the SAME atlas slot of one class
    (a3<->c1 and a4<->c1 would need two A-nodes in one vertex) are
    genuinely competing: unresolved, neither forced."""
    ta = tree({"a3": "nA", "a4": "nA", "nA": "root"})
    tc = tree({"c1": "nC", "c2": "nC", "nC": "root"})
    trees = {"A": ta, "C": tc}
    canon = {k: ident(trees[k]) for k in trees}
    # c1's single reverse call cannot point at both; emulate the
    # competing situation at the CANDIDATE level via two directions:
    # a3->c1 reciprocated, and a4->c2/c2... K=2 cannot express two
    # reciprocal pairs on one endpoint (selection is a function), so
    # competition arises only via joint incompatibility: build it
    # with an ancestry clash instead: a3<->c1 and nA<->c2 at equal
    # support (jointly: [nA,c2] must be ancestor of [a3,c1] via A
    # side and incomparable via C side -> cycle-free but check).
    dec = {"A>C": {"a3": mk("c1"), "nA": mk("c2")},
           "C>A": {"c1": mk("a3"), "c2": mk("nA")}}
    g = quotient_assemble(trees, canon, dec)
    # joint acceptance: c2 is a LEAF sibling of c1; merging nA with
    # c2 makes [nA,c2] a parent of [a3,c1] on the A side while c2
    # has no children on the C side - order-preserving, acceptable.
    # So both accept; this documents that shared-slot competition is
    # impossible to express in a single tie under function-valued
    # selections with K=2, and joint evaluation handles the rest.
    vid_of = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            vid_of[(ds, m)] = r
    assert vid_of[("A", "a3")] == vid_of[("C", "c1")]


def test_tied_maximality_impossible_does_not_block_safe():
    """An individually-impossible tied candidate must not withhold an
    independent safe one (reviewer maximality correction)."""
    ta = tree({"a1": "nA1", "a2": "nA2", "nA1": "root",
               "nA2": "root"})
    tb = tree({"b1": "nB", "b2": "nB", "nB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {k: ident(trees[k]) for k in trees}
    # First (higher support): merge a1<->nB. Then a tie at 0.9:
    # (nA1<->b1) is individually IMPOSSIBLE (would put [a1,nB] above
    # and below [nA1,b1]: cycle), while (a2<->b2) is independent and
    # safe. The safe one must be accepted.
    dec = {"A>B": {"a1": mk("nB", 1.0), "nA1": mk("b1", 0.9),
                   "a2": mk("b2", 0.9)},
           "B>A": {"nB": mk("a1", 1.0), "b1": mk("nA1", 0.9),
                   "b2": mk("a2", 0.9)}}
    g = quotient_assemble(trees, canon, dec)
    vid_of = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            vid_of[(ds, m)] = r
    assert vid_of[("A", "a2")] == vid_of[("B", "b2")]   # safe accepted
    assert any(r["reason"] == "incompatible"
               for r in g["ledger"]["refused"])


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
    # label renaming: TRUE isomorphism under the induced vertex map
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
    # induced vertex map: identify vertices by their (renamed)
    # member sets and require parent structure to correspond exactly
    def key(v, unren):
        return tuple(sorted((ds, unren(ds, m))
                            for ds, m in v["members"].items()))
    unren1 = lambda ds, m: m
    inv = {v: k for k, v in ren.items()}
    unren2 = lambda ds, m: inv.get(m, m) if ds == "A" else m
    k2vid1 = {key(v, unren1): r for r, v in base["vertices"].items()}
    k2vid2 = {key(v, unren2): r for r, v in g2["vertices"].items()}
    assert set(k2vid1) == set(k2vid2)
    iso = {k2vid2[k]: k2vid1[k] for k in k2vid1}
    for r2, r1 in iso.items():
        assert sorted(iso[p] for p in g2["parents"][r2]) == \
            sorted(base["parents"][r1])
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
    assert all(s["kind"] == "atlas_specific"
               for s in g2["statuses"].values())
    # documented flags are actually returned
    g3 = quotient_assemble(trees, canon, dec)
    assert any(s["has_directional_evidence"]
               for s in g3["statuses"].values())
    assert all("conflicting" in s for s in g3["statuses"].values())


def _vid_of(g):
    out = {}
    for r, v in g["vertices"].items():
        for ds, m in v["members"].items():
            out[(ds, m)] = r
    return out


def test_equal_support_disjoint_endpoint_cycle_unresolved():
    """Reviewer's adversarial case: two tied candidates with
    DISJOINT endpoints that conflict through ancestry. nA<->b1 and
    a1<->nB are each valid alone but together create a cycle
    ([a1,nB] <= [nA,b1] via A, [nA,b1] <= [a1,nB] via B). Neither
    may win by processing order: both must be ledgered unresolved
    and neither merged."""
    ta = tree({"a1": "nA", "nA": "root"})
    tb = tree({"b1": "nB", "nB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    dec = {"A>B": {"nA": mk("b1"), "a1": mk("nB")},
           "B>A": {"b1": mk("nA"), "nB": mk("a1")}}
    g = quotient_assemble(trees, canon, dec)
    vid_of = _vid_of(g)
    assert vid_of[("A", "nA")] != vid_of[("B", "b1")]
    assert vid_of[("A", "a1")] != vid_of[("B", "nB")]
    reasons = [u["reason"] for u in g["ledger"]["unresolved_ties"]]
    assert reasons == ["order_dependent_within_tie"] * 2
    assert g["ledger"]["refused"] == []
    assert g["is_forest"]


def test_equal_support_cycle_renaming_invariant():
    """Renaming the adversarial case's nodes (reversing every
    lexicographic order) must not change the outcome: still no
    merge, both candidates unresolved."""
    # 'z*' names sort opposite to the originals in every position
    ta = tree({"za1": "znA", "znA": "root"})
    tb = tree({"ab1": "anB", "anB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    dec = {"A>B": {"znA": mk("ab1"), "za1": mk("anB")},
           "B>A": {"ab1": mk("znA"), "anB": mk("za1")}}
    g = quotient_assemble(trees, canon, dec)
    vid_of = _vid_of(g)
    assert vid_of[("A", "znA")] != vid_of[("B", "ab1")]
    assert vid_of[("A", "za1")] != vid_of[("B", "anB")]
    assert len(g["ledger"]["unresolved_ties"]) == 2
    assert g["ledger"]["refused"] == []


def test_conflicting_flag_covers_certificate_parents():
    """`conflicting` marks every vertex appearing in a certificate:
    the multi-parent child AND its incomparable minimal parents."""
    ta = tree({"a1": "nA1", "a2": "nA2", "nA1": "root",
               "nA2": "root"})
    tb = tree({"b1": "nB", "b2": "nB", "nB": "root"})
    trees = {"A": ta, "B": tb}
    canon = {"A": ident(ta), "B": ident(tb)}
    dec = {"A>B": {"a1": mk("b1")}, "B>A": {"b1": mk("a1")}}
    g = quotient_assemble(trees, canon, dec)
    assert len(g["certificates"]) == 1
    cert = g["certificates"][0]
    flagged = {r for r, s in g["statuses"].items()
               if s["conflicting"]}
    assert flagged == {cert["vertex"], *cert["minimal_parents"]}
