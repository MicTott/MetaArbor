"""Functional-recovery gate for recursive_assemble — the behavior the
Allen safety test could not exercise (reviewer-specified, one targeted
scenario, no parameter sweep).

Synthetic world, K=3, fragmented by construction:
  latent truth: two families L1 {a,b,c,d} and L2 {e,f,g,h}, each with
  two nested subclades (x: first two, y: last two).
  atlas A: sees everything, fine (leaves A|a..A|h, internal nL1/nL1x/
           nL1y/nL2/nL2x/nL2y).
  atlas B: sees only L1, coarse (leaves B|L1x, B|L1y under nB).
  atlas C: sees only L2, fine (nC2/nC2x/nC2y over C|e..C|h) PLUS an
           adversarial clade nCz (C|z1, C|z2).
  certified core: ONE reciprocal node C1 = {A: nL1, B: nB} — sparse on
  purpose; A's L2 subtree and all of C are root-stranded components.
  one-way calls:
    C's nC2  -> A's nL2   support 1.00  (the reconnecting containment)
    A's nL2  -> C's nC2   support 0.90  (mutual: must be cycle-skipped)
    C's nCz  -> A's nL1x  support 1.00  \\ incomparable targets:
    C's nCz  -> B's B|L1y support 1.00  /  must stay UNRESOLVED
    C|e      -> A's nL2   support 1.00  (audit: consistent after join)
    C|g      -> A's nL1   support 1.00  (audit: cross-family conflict)
    B|L1x    -> A's nL1x  support 1.00  (audit: sibling, conflict-
                                         flagged, never moved)

Required behavior (each is one assertion below):
  1. reconnection      — C's L2 component nests under A's nL2;
  2. nesting recovered — C keeps its own internal structure beneath;
  3. ancestry preserved— every non-root vertex keeps its input parent;
  4. no cycles         — the mutual call is skipped and ledgered;
  5. no new merges     — containment never becomes equivalence: the
                         only multi-atlas vertex is the certified core;
  6. incompatible case — nCz stays at ROOT, ledgered multi_parent;
  7. order invariance  — identical output under reversed insertion
                         order of every input dict.

Run: python -m pytest comparison/otharmonizer/test_recursive.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def world():
    def tree(parent):
        children = {n: [] for n in parent}
        for n, p in parent.items():
            if p is not None and p != "root":
                children.setdefault(p, []).append(n)
        children.setdefault("root", [c for c, p in parent.items()
                                     if p == "root"])
        leaves = [n for n in parent if not children.get(n)]
        return {"parent": parent, "children": children,
                "leaves": leaves}

    tA = tree({"nL1": "root", "nL1x": "nL1", "nL1y": "nL1",
               "A|a": "nL1x", "A|b": "nL1x", "A|c": "nL1y",
               "A|d": "nL1y",
               "nL2": "root", "nL2x": "nL2", "nL2y": "nL2",
               "A|e": "nL2x", "A|f": "nL2x", "A|g": "nL2y",
               "A|h": "nL2y"})
    tB = tree({"nB": "root", "B|L1x": "nB", "B|L1y": "nB"})
    tC = tree({"nC2": "root", "nC2x": "nC2", "nC2y": "nC2",
               "C|e": "nC2x", "C|f": "nC2x", "C|g": "nC2y",
               "C|h": "nC2y",
               "nCz": "root", "C|z1": "nCz", "C|z2": "nCz"})
    itrees = {"A": tA, "B": tB, "C": tC}
    canon = {k: {n: n for n in itrees[k]["parent"]} for k in itrees}
    strict_nodes = {"C1": {"parent": None, "children": [],
                           "members": {"A": "nL1", "B": "nB"}}}
    cert_member = {("A", "nL1"): "C1", ("B", "nB"): "C1"}
    mk = lambda sel, s: {"selected": sel, "matched": True, "support": s}
    decisions = {
        "C>A": {"nC2": mk("nL2", 1.0), "nCz": mk("nL1x", 1.0),
                "C|e": mk("nL2", 1.0), "C|g": mk("nL1", 1.0)},
        "A>C": {"nL2": mk("nC2", 0.9)},
        "C>B": {"nCz": mk("B|L1y", 1.0)},
        "B>A": {"B|L1x": mk("nL1x", 1.0)},
    }
    return strict_nodes, cert_member, itrees, canon, decisions


def test_functional_recovery():
    from recursive_frontier import CORE, recursive_assemble
    strict_nodes, cert_member, itrees, canon, decisions = world()
    r = recursive_assemble(strict_nodes, cert_member, itrees, canon,
                           decisions, t=0.8)
    p = r["parent_of"]

    # 1. reconnection: C's fragment nests under A's L2 clade
    assert p[("C", "nC2")] == ("A", "nL2")
    # 2. known nesting recovered, C's own structure intact beneath
    assert p[("C", "nC2x")] == ("C", "nC2")
    assert p[("C", "C|e")] == ("C", "nC2x")
    assert p[("A", "nL2x")] == ("A", "nL2")
    # 3. ancestry preserved for every non-pending vertex
    for ds in itrees:
        for n in itrees[ds]["parent"]:
            par = itrees[ds]["parent"][n]
            v = (ds, n)
            if cert_member.get((ds, n)) or par in (None, "root"):
                continue
            expect = ((CORE, cert_member[(ds, par)])
                      if (ds, par) in cert_member else (ds, par))
            assert p[v] == expect, (v, p[v], expect)
    # 4. the mutual call was cycle-skipped and ledgered; no cycles exist
    assert [(u, w) for u, w, _s in r["cycle_skipped"]] == \
        [(("A", "nL2"), ("C", "nC2"))]
    for v in p:
        seen, x = set(), v
        while x != "ROOT":
            assert x not in seen, f"cycle through {v}"
            seen.add(x)
            x = p.get(x, "ROOT")
    assert p[("A", "nL2")] == "ROOT"
    # 5. containment never became equivalence: the certified core is
    #    the only vertex holding members from more than one atlas
    assert set(strict_nodes) == {"C1"}
    assert all(v[0] != CORE or v[1] == "C1" for v in p)
    # 6. the incomparable-parent clade stays unresolved, ledgered
    assert p[("C", "nCz")] == "ROOT"
    assert [u for u, _imgs in r["multi_parent"]] == [("C", "nCz")]
    # audit: C|g cross-family and B|L1x sibling calls flagged, not moved
    flagged = {v for (v, _w) in r["conflict_vertices"]}
    assert ("C", "C|g") in flagged and ("B", "B|L1x") in flagged
    assert p[("C", "C|g")] == ("C", "nC2y")       # never moved
    assert ("C", "C|e") not in flagged            # consistent after join


def test_order_invariance():
    from recursive_frontier import recursive_assemble
    strict_nodes, cert_member, itrees, canon, decisions = world()
    base = recursive_assemble(strict_nodes, cert_member, itrees, canon,
                              decisions, t=0.8)

    def rev(d):
        if isinstance(d, dict):
            return {k: rev(d[k]) for k in reversed(list(d))}
        return d

    perm = recursive_assemble(rev(strict_nodes), rev(cert_member),
                              rev(itrees), rev(canon), rev(decisions),
                              t=0.8)
    assert base["parent_of"] == perm["parent_of"]
    assert base["attach_edges"] == perm["attach_edges"]
    assert base["multi_parent"] == perm["multi_parent"]
    assert base["cycle_skipped"] == perm["cycle_skipped"]
