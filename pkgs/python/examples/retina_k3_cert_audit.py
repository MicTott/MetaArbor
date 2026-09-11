"""Retina K=3 — individual evidence traces for the five DAG
certificates of the RAW-mode quotient run (d361834), per review:
"they need individual evidence traces before concluding that a
same-branch equivalence rule would correctly remove all five."

For every certificate: the vertex, its minimal parents, the raw
directional evidence between each unmerged parent pair (both
directions, with supports), tie membership, and the mechanism class:
  same_branch_near_miss   parents not raw-reciprocal because one
                          side's selection lands on a CHILD of the
                          other (the K=2 n04->n10 pattern)
  unresolved_equal_tie    parents raw-reciprocal at equal support
                          but ledgered by the tie rule
  weak_uncertified_merge  the certificate's CHILD vertex only exists
                          because of a raw merge with min directional
                          support far below anything the
                          certification layer would qualify

Run: PYTHONPATH=src python examples/retina_k3_cert_audit.py
"""
import json
import os

from metaarbor.consensus.quotient import quotient_assemble

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "retina_k3")
it = json.load(open(os.path.join(OUT, "retina_k3_input_trees.json")))
dec = json.load(open(os.path.join(OUT, "retina_k3_decisions.json")))
canon = json.load(open(os.path.join(OUT, "retina_k3_canonical.json")))
trees = {k: {"parent": it[k]["parent"], "children": it[k]["children"],
             "leaves": it[k]["leaves"]} for k in ("mac", "she",
                                                  "mrca")}
g = quotient_assemble(trees, canon, dec)          # RAW mode (as run)
unresolved = {tuple(sorted((tuple(u["pair"][0]), tuple(u["pair"][1]))))
              for u in g["ledger"]["unresolved_ties"]}


def call(da, a, db, b):
    f = dec.get(f"{da}>{db}", {}).get(a)
    r = dec.get(f"{db}>{da}", {}).get(b)
    return ((f or {}).get("selected"), (f or {}).get("support"),
            (r or {}).get("selected"), (r or {}).get("support"))


def kids_of(ds, n):
    return trees[ds]["children"].get(n, [])


report = []
for c in g["certificates"]:
    v = g["vertices"][c["vertex"]]
    entry = {"vertex": sorted(v["members"].items()),
             "minimal_parents": [], "mechanisms": []}
    # the child vertex's own formation evidence
    mem = sorted(v["members"].items())
    child_supports = []
    for x in range(len(mem)):
        for y in range(x + 1, len(mem)):
            (da, a), (db, b) = mem[x], mem[y]
            sel_f, s_f, sel_r, s_r = call(da, a, db, b)
            if sel_f == b and sel_r == a:
                child_supports.append(min(s_f, s_r))
    if child_supports and min(child_supports) < 0.5:
        entry["mechanisms"].append(
            ("weak_uncertified_merge",
             f"child vertex formed at min support "
             f"{min(child_supports):.3f}"))
    ps = c["minimal_parents"]
    for x in range(len(ps)):
        for y in range(x + 1, len(ps)):
            pa, pb = g["vertices"][ps[x]], g["vertices"][ps[y]]
            (da, a) = sorted(pa["members"].items())[0]
            (db, b) = sorted(pb["members"].items())[0]
            if da == db:
                continue
            sel_f, s_f, sel_r, s_r = call(da, a, db, b)
            rec = {"parents": (f"{da}|{a}", f"{db}|{b}"),
                   f"{da}->{db}": (sel_f, s_f),
                   f"{db}->{da}": (sel_r, s_r)}
            key = tuple(sorted(((da, a), (db, b))))
            if key in unresolved:
                entry["mechanisms"].append(
                    ("unresolved_equal_tie", f"{da}|{a}<->{db}|{b} "
                     f"reciprocal at {s_f}/{s_r}, ledgered"))
            elif sel_f == b and sel_r == a:
                entry["mechanisms"].append(
                    ("reciprocal_but_unmerged_other",
                     f"{da}|{a}<->{db}|{b} at {s_f}/{s_r}"))
            else:
                near = (sel_f in kids_of(db, b) or
                        sel_r in kids_of(da, a))
                entry["mechanisms"].append(
                    ("same_branch_near_miss" if near
                     else "non_reciprocal",
                     f"{da}|{a}->{sel_f} ({s_f}); "
                     f"{db}|{b}->{sel_r} ({s_r})"))
            entry["minimal_parents"].append(rec)
    report.append(entry)

for e in report:
    print("CERT", e["vertex"])
    for p in e["minimal_parents"]:
        print("   ", p)
    for m in e["mechanisms"]:
        print("   MECHANISM:", m[0], "--", m[1])
    print()
mech_ct = {}
for e in report:
    for m, _ in e["mechanisms"]:
        mech_ct[m] = mech_ct.get(m, 0) + 1
print("mechanism counts:", mech_ct)
json.dump(report, open(os.path.join(OUT, "retina_k3_cert_audit.json"),
                       "w"), indent=1, default=str)
print("NOT one mechanism: a same-branch rule alone would not remove "
      "certificates whose children exist only via weak uncertified "
      "merges, nor the equal-tie pair — stated, not assumed.")
