"""Retina K=2 C3-failure audit and the core-ancestry interleaving
repair (discovered here, no-harm-checked on Allen).

FINDINGS (from the committed retina dumps, 1779de1):
1. The C3 damage is ONE label: mac|c27, hoisted because its input
   parent n04 (= the TRUE mac OFF clade {c27,c28,c29}) has no
   assembly representation. Certified nodes exist above (n06=C0001)
   and below (n01=C0003), so assembly spliced n04 out and c27
   re-parented to the grandparent.
2. WHY n04 is unanchored — a truth-certified SAME-BRANCH NEAR-MISS,
   the first with ground truth: mac n04 -> she n10 (support 1.0) is
   the TRUE OFF<->OFF equivalence, but she n10's reverse call picked
   n01 (a CHILD of n04), so reciprocity certified the finer
   grain (n01<->n10, a containment promoted to anchor) and the
   correct coarse equivalence died. Four she-side calls (n08, n03,
   BC3B, BC4) converge on n04 — the reverse-coverage signal that
   n04 is real.
3. Recursive-v1 does NOT repair the score (0.9318 unchanged):
   it preserves ancestry for non-core vertices only; certified
   nodes keep collapsed cut parentage, so C0003 stays a SIBLING of
   the anonymous n04 vertex instead of nesting beneath it.

REPAIR (prespecified rule, stated before scoring): CORE-ANCESTRY
INTERLEAVING — a certified node is re-parented under the deepest
unrepresented input ancestor of its members when (a) exactly ONE
member atlas supplies intermediate ancestors and (b) every member
atlas's implied core parent equals the cut parent (no
contradiction); >= 2 atlases with distinct chains -> ledgered
unresolved, never forced. Pure representation of existing evidence;
no thresholds.

STATUS: retina is the DISCOVERY set for this rule (0.9318 -> 1.0000,
COPH 0.9415 -> 0.9700, 2 nodes interleaved, 0 conflicts); Allen is
the independent NO-HARM check (0.9117 -> 0.9129, 4 nodes
interleaved, 13 multi-chain cases ledgered, 0 cycles). Amygdala
structural evaluation and promotion into the assembly proper are
future, review-gated steps.

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/retina_k2_audit.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "comparison",
                                "otharmonizer"))
from metrics import MyNode, cophenetic_spearman, triplet_scores  # noqa
from recursive_frontier import CORE, recursive_assemble  # noqa: E402
from metaarbor.consensus.cut import consensus_cut  # noqa: E402


def cparent_fn(itrees, canon):
    def cparent(ds, n):
        q = itrees[ds]["parent"].get(n)
        while q not in (None, "root"):
            qc = canon[ds].get(q, q)
            if qc != n:
                return qc
            q = itrees[ds]["parent"].get(q)
        return None
    return cparent


def interleave(strict_nodes, cert_member, itrees, canon, parent_of):
    """The prespecified core-ancestry interleaving rule. Returns
    (updated parent map, n_moved, ledger)."""
    cparent = cparent_fn(itrees, canon)
    p = dict(parent_of)
    moved, ledger = 0, []
    for i, nd in strict_nodes.items():
        chains, implied = {}, {}
        for ds, m in nd["members"].items():
            ch, x = [], cparent(ds, m)
            while x is not None and (ds, x) not in cert_member:
                ch.append((ds, x))
                x = cparent(ds, x)
            chains[ds] = ch
            implied[ds] = ("ROOT" if x is None else
                           (CORE, cert_member[(ds, x)]))
        nonempty = {ds: ch for ds, ch in chains.items() if ch}
        cut_par = p[(CORE, i)]
        if len(nonempty) == 1 and all(
                v == cut_par or (v == "ROOT" and cut_par == "ROOT")
                for v in implied.values()):
            _ds, ch = next(iter(nonempty.items()))
            p[(CORE, i)] = ch[0]
            moved += 1
        elif len(nonempty) >= 2:
            ledger.append(i)
    for v in p:                                  # cycle guard
        seen, x = set(), v
        while x != "ROOT":
            assert x not in seen, f"cycle at {v}"
            seen.add(x)
            x = p.get(x, "ROOT")
    return p, moved, ledger


def render(p, strict_nodes, label_sets, disp):
    kids = {}
    for v, par in p.items():
        kids.setdefault(par, []).append(v)

    def name_of(v):
        if v[0] == CORE:
            nd = strict_nodes[v[1]]
            pp = [disp(ds, m) for ds, m in sorted(nd["members"].items())
                  if m in label_sets[ds]]
            return "&".join(pp) if pp else f"__c{v[1]}__"
        ds, n = v
        return disp(ds, n) if n in label_sets[ds] else f"__{ds}_{n}__"

    root = MyNode("root")

    def emit(v, pn):
        node = MyNode(name_of(v))
        pn.addkid(node)
        for c in sorted(kids.get(v, []), key=str):
            emit(c, node)
    for v in sorted(kids.get("ROOT", []), key=str):
        emit(v, root)
    return root


if __name__ == "__main__":
    OUT = os.path.join(HERE, "retina_k2")
    it = json.load(open(os.path.join(OUT,
                                     "retina_k2_input_trees.json")))
    truth = json.load(open(os.path.join(OUT, "retina_k2_truth.json")))
    dump = json.load(open(os.path.join(OUT, "retina_k2_tree.json")))
    dec = json.load(open(os.path.join(OUT,
                                      "retina_k2_decisions.json")))
    canon = json.load(open(os.path.join(OUT,
                                        "retina_k2_canonical.json")))
    group_of = {l: g for g, ls in truth.items() for l in ls}
    labels_all = {"mac": {l for l in group_of if l.startswith("mac|")},
                  "she": {l for l in group_of
                          if l.startswith("she|")}}
    itrees = {k: {"parent": it[k]["parent"],
                  "children": it[k]["children"],
                  "leaves": it[k]["leaves"]} for k in ("mac", "she")}
    strict = consensus_cut(dump, level="strict",
                           leaf_labels=labels_all)
    cert_member = {(ds, m): i for i, nd in strict["nodes"].items()
                   for ds, m in nd["members"].items()}

    def truth_tree():
        root = MyNode("root")
        rbc = MyNode("__rbc__")
        root.addkid(rbc)
        for l in truth["RBC_group"]:
            rbc.addkid(MyNode(l.replace("|", "-")))
        cbc = MyNode("__cbc__")
        root.addkid(cbc)
        for grp in ("OFF", "ON"):
            g = MyNode(f"__{grp}__")
            cbc.addkid(g)
            for l in truth[grp]:
                g.addkid(MyNode(l.replace("|", "-")))
        return root

    REF = truth_tree()
    disp = lambda ds, n: n.replace("|", "-")
    r = recursive_assemble(strict["nodes"], cert_member, itrees,
                           canon, dec, 1.0)
    base = render(r["parent_of"], strict["nodes"], labels_all, disp)
    rec0, _ = triplet_scores(base, REF)
    p2, moved, ledger = interleave(strict["nodes"], cert_member,
                                   itrees, canon, r["parent_of"])
    fixed = render(p2, strict["nodes"], labels_all, disp)
    rec1, _ = triplet_scores(fixed, REF)
    print(f"retina: recursive-v1 TRIP={rec0:.4f} -> interleaved "
          f"TRIP={rec1:.4f} (COPH {cophenetic_spearman(base, REF):.4f}"
          f" -> {cophenetic_spearman(fixed, REF):.4f}); "
          f"moved={moved} ledger={ledger}")
