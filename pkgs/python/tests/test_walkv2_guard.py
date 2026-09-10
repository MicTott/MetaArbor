"""Adversarial regression tests for the Walk-v2 reverse-child
coverage guard (examples/walkv2_prototype.py) — reviewer-required
after the winner-return hole: the first implementation could PERMIT
descent into child A on reverse evidence about child B."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "examples"))
from walkv2_prototype import classify_reverse, guard  # noqa: E402


def leaves_under(tree, node):
    if not tree["children"].get(node):
        return [node]
    out, stack = [], list(tree["children"][node])
    while stack:
        x = stack.pop()
        if tree["children"].get(x):
            stack.extend(tree["children"][x])
        else:
            out.append(x)
    return sorted(out)


def mk_tree(parent):
    children = {}
    for n, p in parent.items():
        children.setdefault(p, []).append(n)
    for n in parent:
        children.setdefault(n, [])
    return {"parent": parent, "children": children,
            "leaves": [n for n in parent if not children[n]]}


# target split P with children A, B; source tree with query Q (leaf),
# an internal node QI with child leaf QL, and unrelated leaf X
TGT = mk_tree({"P": "root", "A": "P", "B": "P"})
SRC = mk_tree({"Q": "root", "QI": "root", "QL": "QI", "X": "root"})
CANON = {"A": "A", "B": "B"}
mk = lambda sel: {"selected": sel, "matched": True, "support": 1.0}


def test_winner_return_enforced():
    """Reverse evidence maps B to Q; vote winner is A -> must NOT
    permit descent into A (the hole in revision 1)."""
    rev = {"A": mk("X"), "B": mk("Q")}
    g, _ = guard("P", "Q", "A", TGT, rev, CANON, SRC, leaves_under)
    assert g == "UNRESOLVED"
    # and with the winner being the mapping child, it permits
    g, prov = guard("P", "Q", "B", TGT, rev, CANON, SRC, leaves_under)
    assert g == "PERMIT" and prov == ["B"]


def test_veto_two_maps():
    rev = {"A": mk("Q"), "B": mk("Q")}
    for best in ("A", "B"):
        g, prov = guard("P", "Q", best, TGT, rev, CANON, SRC,
                        leaves_under)
        assert g == "VETO" and sorted(prov) == ["A", "B"]


def test_absent_sibling_is_unresolved():
    """Winner maps to Q but the sibling's call is absent -> rule-6
    precedence: UNRESOLVED, never PERMIT."""
    rev = {"A": mk("Q")}
    g, _ = guard("P", "Q", "A", TGT, rev, CANON, SRC, leaves_under)
    assert g == "UNRESOLVED"


def test_internal_query_descendant_landing():
    """A reverse landing INSIDE the query's subtree counts as
    maps_to_Q (internal-node queries); an ancestor landing is
    uninformative, not elsewhere."""
    assert classify_reverse("QL", "QI", SRC) == "maps_to_Q"
    assert classify_reverse("root", "QL", SRC) != "maps_elsewhere"
    # ancestor-of-query landing: QI is QL's parent
    assert classify_reverse("QI", "QL", SRC) == "uninformative"
    rev = {"A": mk("QL"), "B": mk("X")}
    g, prov = guard("P", "QI", "A", TGT, rev, CANON, SRC,
                    leaves_under)
    assert g == "PERMIT" and prov == ["A"]


def test_uninformative_sibling_blocks_permit():
    """Sibling lands on the query's ancestor (uninformative): absent
    affirmative elsewhere-evidence -> UNRESOLVED."""
    rev = {"A": mk("QL"), "B": mk("QI")}   # B's call covers QL's parent
    g, _ = guard("P", "QL", "A", TGT, rev, CANON, SRC, leaves_under)
    assert g == "UNRESOLVED"
