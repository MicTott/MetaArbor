"""Migration gates for the v1 quotient assembly (ASSEMBLY2 rev2).

GATE 1 (retina): the quotient must reproduce the interleaved result
with the interleave module unused. RESULT: PASS — TRIP 1.0000 /
COPH 0.9700 exactly; 10 shared vertices; forest; empty ledger.

GATE 2 (allen): ACCOUNTING gate, not a superiority gate. The
quotient output on Allen is a DAG (forest=False) with 11
UNRECONCILED MULTIPLE-PARENT CONSTRAINTS — vertices whose input
ancestry places them below two incomparable minimal ancestors
(e.g. the Car3 merge, whose v2-side and v3-side input parents are
unmerged and incomparable). These are not conflicts between
evidence records; they are relationships the old assembly silently
resolved into a tree by routing order and the quotient declines to
resolve. Because the result is a DAG, NO single TRIP number is a
valid score of it: any tree score depends on an arbitrary
projection choice. An earlier revision of this file reported
TRIP=0.9172 from the first-minimal-parent projection and claimed
improvement over the committed assembly (0.9116); that claim is
WITHDRAWN — the projection was arbitrary. This gate instead
reports (a) the shared-vertex accounting, (b) each unreconciled
constraint with its incomparable minimal parents (verified
incomparable structurally), and (c) OBSERVED SCORES AMONG SAMPLED
PROJECTIONS (baseline first-parent projection plus every
single-constraint alternative flip; combinations not evaluated) —
NOT a projection range. Set QUOTIENT_GATE_EXHAUSTIVE=1 to
enumerate ALL 2^11 = 2048 parent-choice combinations (about 5-10
minutes on several cores), which yields the true min/max over
compatible forest projections. RESULT OF THE EXHAUSTIVE RUN
(2026-09-10): min=0.9129 max=0.9263 over all 2048 projections —
numerically identical to the sampled endpoints, so the one-flip
sample happened to bracket the full range here (that was not
guaranteed in advance). The numbers are descriptive either way;
PASS = the accounting is complete and every listed constraint is
genuinely incomparable.

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


def render(g, label_name, choice=None):
    """Forest projection of the quotient graph. `choice` maps a
    multi-parent vertex to the minimal parent used in this
    projection (default: first). Any such tree is a VIEW, never the
    result (ASSEMBLY2 Section 6)."""
    choice = choice or {}
    kids, roots = {}, []
    for r, ps in g["parents"].items():
        if ps:
            kids.setdefault(choice.get(r, ps[0]), []).append(r)
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


def ancestor_closure(parents):
    memo = {}

    def anc(r):
        if r not in memo:
            memo[r] = set()
            for p in parents[r]:
                memo[r] |= {p} | anc(p)
        return memo[r]
    for r in parents:
        anc(r)
    return memo


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


_SC = {}


def _score_init(g, labels, ref):
    _SC["g"], _SC["labels"], _SC["ref"] = g, labels, ref


def _score_choice(choice):
    g, labels, ref = _SC["g"], _SC["labels"], _SC["ref"]

    def nm(v):
        labs = sorted(f"{ds}-{m.split('|', 1)[-1]}"
                      for ds, m in v["members"].items()
                      if m in labels.get(ds, ()))
        return "&".join(labs)
    s, _ = triplet_scores(render(g, nm, choice), ref)
    return s


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
    ref = refA()

    sh = sum(1 for v in g["vertices"].values() if v["shared"])
    print(f"GATE 2 allen: shared={sh} (committed backbone 24) "
          f"forest={g['is_forest']} "
          f"unreconciled_constraints={len(g['certificates'])}")

    # (b) each constraint listed and VERIFIED incomparable
    anc = ancestor_closure(g["parents"])
    all_incomparable = True
    for c in g["certificates"]:
        v = g["vertices"][c["vertex"]]
        ps = c["minimal_parents"]
        inc = all(p not in anc[q] and q not in anc[p]
                  for i, p in enumerate(ps) for q in ps[i + 1:])
        all_incomparable &= inc
        print("   constraint:", sorted(v["members"].items()),
              f"minimal_parents={len(ps)} incomparable={inc}")

    # (c) descriptive scores of forest projections. Default:
    # baseline (first minimal parent everywhere) + one flip per
    # constraint alternative — OBSERVED SCORES AMONG SAMPLED
    # PROJECTIONS, not a range. Exhaustive mode enumerates every
    # parent-choice combination and IS the projection range. Either
    # way: views, not results; no single number scores a DAG and no
    # superiority claim is made.
    if os.environ.get("QUOTIENT_GATE_EXHAUSTIVE"):
        import itertools
        from concurrent.futures import ProcessPoolExecutor
        certs = g["certificates"]
        combos = list(itertools.product(
            *[c["minimal_parents"] for c in certs]))
        vxs = [c["vertex"] for c in certs]
        jobs = [dict(zip(vxs, combo)) for combo in combos]
        with ProcessPoolExecutor(
                initializer=_score_init,
                initargs=(g, labels, ref)) as ex:
            scores = list(ex.map(_score_choice, jobs,
                                 chunksize=32))
        print(f"   TRIP over ALL {len(scores)} compatible forest "
              f"projections (views, not results): "
              f"min={min(scores):.4f} max={max(scores):.4f} "
              f"(committed assembly 0.9116 for reference; "
              f"no improvement claim)")
    else:
        scores = []
        base, _ = triplet_scores(render(g, nm), ref)
        scores.append(base)
        for c in g["certificates"]:
            for alt in c["minimal_parents"][1:]:
                s, _ = triplet_scores(
                    render(g, nm, {c["vertex"]: alt}), ref)
                scores.append(s)
        print(f"   TRIP observed among {len(scores)} SAMPLED "
              f"projections (baseline + one-at-a-time flips; "
              f"combinations not evaluated; views, not results): "
              f"min={min(scores):.4f} max={max(scores):.4f} "
              f"(committed assembly 0.9116 for reference; "
              f"no improvement claim)")
    ok = all_incomparable and len(g["certificates"]) > 0
    print(f"GATE 2 -> {'PASS (accounting complete)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1, r2 = gate_retina(), gate_allen()
    print(f"\nMIGRATION GATES: {'PASS' if r1 and r2 else 'FAIL'}")
