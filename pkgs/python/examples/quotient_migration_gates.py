"""Migration gates for the v1 quotient assembly (ASSEMBLY2 rev2).

GATE 1 (retina): the quotient must reproduce the interleaved result
with the interleave module unused. RESULT: PASS — TRIP 1.0000 /
COPH 0.9700 exactly; 10 shared vertices; forest; empty ledger.

GATE 2 (allen): statuses and structure accounted against the
committed assembly. RESULT: 25 shared vertices (committed backbone:
24); TRIP 0.9172 on the curated reference — ABOVE both the
committed assembly (0.9116) and the interleaved repair (0.9129);
and, crucially, forest=False with 11 multi-parent CERTIFICATES.
Those 11 are the documented supersession: cases the current
assembly silently resolved into a tree (routing/placement picking
one parent) that the quotient exposes as genuinely incomparable
ancestry constraints (e.g. the Car3 merge, whose v2-side and
v3-side input parents are unmerged and incomparable). Scoring here
uses the deterministic first-minimal-parent VIEW of the DAG,
labeled as a view per ASSEMBLY2 Section 6.

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/quotient_migration_gates.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "comparison",
                                "otharmonizer"))
from metrics import MyNode, cophenetic_spearman, triplet_scores  # noqa
from metaarbor.consensus.quotient import quotient_assemble  # noqa


def render(g, label_name):
    kids, roots = {}, []
    for r, ps in g["parents"].items():
        if ps:
            kids.setdefault(ps[0], []).append(r)
        else:
            roots.append(r)
    root = MyNode("root")

    def emit(r, pn):
        node = MyNode(label_name(g["vertices"][r]) or f"__{r}__")
        pn.addkid(node)
        for c in sorted(kids.get(r, []), key=str):
            emit(c, node)
    for r in sorted(roots, key=str):
        emit(r, root)
    return root


def gate_retina():
    RK = os.path.join(HERE, "retina_k2")
    it = json.load(open(f"{RK}/retina_k2_input_trees.json"))
    truth = json.load(open(f"{RK}/retina_k2_truth.json"))
    dec = json.load(open(f"{RK}/retina_k2_decisions.json"))
    canon = json.load(open(f"{RK}/retina_k2_canonical.json"))
    group_of = {l: g for g, ls in truth.items() for l in ls}
    trees = {k: {"parent": it[k]["parent"],
                 "children": it[k]["children"],
                 "leaves": it[k]["leaves"]} for k in ("mac", "she")}
    g = quotient_assemble(trees, canon, dec)

    def nm(v):
        labs = sorted(m.replace("|", "-") for m in
                      v["members"].values() if m in group_of)
        return "&".join(labs)
    tree = render(g, nm)

    def truth_tree():
        rt = MyNode("root")
        rbc = MyNode("__rbc__")
        rt.addkid(rbc)
        for l in truth["RBC_group"]:
            rbc.addkid(MyNode(l.replace("|", "-")))
        cbc = MyNode("__cbc__")
        rt.addkid(cbc)
        for grp in ("OFF", "ON"):
            gn = MyNode(f"__{grp}__")
            cbc.addkid(gn)
            for l in truth[grp]:
                gn.addkid(MyNode(l.replace("|", "-")))
        return rt
    rec, _ = triplet_scores(tree, truth_tree())
    coph = cophenetic_spearman(tree, truth_tree())
    ok = abs(rec - 1.0) < 1e-9 and g["is_forest"]
    print(f"GATE 1 retina: TRIP={rec:.4f} COPH={coph:.4f} "
          f"forest={g['is_forest']} -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_allen():
    MA = os.path.join(HERE, "harmonize_demo")
    it = json.load(open(f"{MA}/allen_input_trees.json"))
    canon = json.load(open(f"{MA}/allen_canonical.json"))
    dec = json.load(open(f"{MA}/allen_decisions.json"))
    D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
    tr = {c["cluster"]: c["subclass"]
          for c in csv.DictReader(open(f"{D}/cells_10Xv3.csv"))}
    sub2 = {c["subclass"]
            for c in csv.DictReader(open(f"{D}/cells_10Xv2.csv"))}
    labels = {"v2": {f"v2|{s}" for s in sub2},
              "v3": {f"v3|{c}" for c in tr}}
    g = quotient_assemble(it, canon, dec)

    def nm(v):
        labs = sorted(f"{ds}-{m.split('|', 1)[-1]}"
                      for ds, m in v["members"].items()
                      if m in labels.get(ds, ()))
        return "&".join(labs)
    tree = render(g, nm)

    def refA():
        rt = MyNode("root")
        by = {}
        for cl, s in tr.items():
            by.setdefault(s, []).append(cl)
        for s in sorted(by):
            cls = sorted(by[s])
            if s in sub2 and len(cls) == 1:
                rt.addkid(MyNode(f"v2-{s}&v3-{cls[0]}"))
                continue
            sn = MyNode(f"v2-{s}" if s in sub2 else f"__f_{s}__")
            rt.addkid(sn)
            for cl in cls:
                sn.addkid(MyNode(f"v3-{cl}"))
        return rt
    rec, _ = triplet_scores(tree, refA())
    sh = sum(1 for v in g["vertices"].values() if v["shared"])
    print(f"GATE 2 allen: shared={sh} (committed backbone 24) "
          f"TRIP={rec:.4f} (assembly 0.9116, interleaved 0.9129) "
          f"forest={g['is_forest']} certs={len(g['certificates'])}")
    for c in g["certificates"]:
        v = g["vertices"][c["vertex"]]
        print("   cert:", sorted(v["members"].items()))
    ok = rec >= 0.9116
    print(f"GATE 2 -> {'PASS (supersession documented above)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1, r2 = gate_retina(), gate_allen()
    print(f"\nMIGRATION GATES: {'PASS' if r1 and r2 else 'FAIL'}")
