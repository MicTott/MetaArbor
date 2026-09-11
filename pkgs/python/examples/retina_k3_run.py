"""Retina bipolar K=3 — stage 2: frozen harmonize (3 atlases),
decision dumps, QUOTIENT assembly as the primary result (scored per
the DAG result contract), legacy assembly alongside, prespecified
checks C1-C6, and the order-swap stability replicate. Everything per
retina_k3_PRESPEC.md (committed a4a9975). No thresholds tuned;
failures reported, never repaired.

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/retina_k3_run.py
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "comparison",
                                "otharmonizer"))
from metrics import (MyNode, cophenetic_spearman,  # noqa: E402
                     triplet_scores)
from metaarbor.consensus.candidates import canonical_nodes  # noqa
from metaarbor.consensus.harmonize import harmonize  # noqa: E402
from metaarbor.consensus.quotient import quotient_assemble  # noqa

OUT = os.path.join(HERE, "retina_k3")
DS = ("mac", "she", "mrca")
it = json.load(open(os.path.join(OUT, "retina_k3_input_trees.json")))
truth = json.load(open(os.path.join(OUT, "retina_k3_truth.json")))
genes = [g for g in open(os.path.join(OUT, "common_genes.txt"))
         .read().split("\n") if g]
X = {k: np.asarray(mmread(os.path.join(OUT, f"{k}_bipolar.mtx"))
                   .todense()) for k in DS}
lab = {k: np.loadtxt(os.path.join(OUT, f"{k}_labels.txt"), dtype=str)
       for k in DS}

group_of = {}
for grp, d in truth.items():
    for ch in d["cherries"]:
        for l in ch:
            group_of[l] = grp
    for l in d["mac"]:
        group_of[l] = grp
ALL_LABELS = sorted(group_of)
labels_of = {k: {l for l in ALL_LABELS if l.startswith(f"{k}|")}
             for k in DS}


def mk_tree(d):
    return {"parent": d["parent"], "children": d["children"],
            "leaves": d["leaves"]}


trees = {k: mk_tree(it[k]) for k in DS}
stability = {(k, n): v for k in DS
             for n, v in it[k]["support"].items()}


def run_harmonize(order):
    ds = {}
    for k in order:
        ds[k] = {"counts": X[k], "labels": lab[k],
                 "gene_names": genes, "lib": X[k].sum(axis=1)}
    return harmonize(ds, {k: trees[k] for k in order}, n_hvg=1000,
                     n_boot=200, stability=stability)


print("harmonize (primary, mac+she+mrca order)...")
harm = run_harmonize(list(DS))
nodes = harm["tree"]

# ---- dumps ----------------------------------------------------------------
dump = {i: {"parent": nd["parent"], "status": nd["status"],
            "members": nd["members"], "aliases": nd["aliases"],
            "display": nd["display"],
            "assembly_repair": bool(nd.get("assembly_repair"))}
        for i, nd in nodes.items()}
json.dump(dump, open(os.path.join(OUT, "retina_k3_tree.json"), "w"),
          indent=1)
sel_out = {}
for (ki, kj), recs in harm["decisions"]["selections"].items():
    sel_out[f"{ki}>{kj}"] = {
        n: {"selected": r["selected"], "matched": bool(r["matched"]),
            "support": (None if r["support"] is None or
                        r["support"] != r["support"]
                        else float(r["support"])),
            "relation": r.get("relation")}
        for n, r in recs.items()}
json.dump(sel_out, open(os.path.join(OUT, "retina_k3_decisions.json"),
                        "w"), indent=1)
canon = {k: canonical_nodes(trees[k])[1] for k in trees}
json.dump(canon, open(os.path.join(OUT, "retina_k3_canonical.json"),
                      "w"))

# ---- PRIMARY: quotient assembly (C6 gate) ---------------------------------
print("\nquotient_assemble (PRIMARY)...")
g = quotient_assemble(trees, canon, sel_out)
sh = sum(1 for v in g["vertices"].values() if v["shared"])
tri = [r for r, v in g["vertices"].items() if len(v["members"]) == 3]
print(f"quotient: {len(g['vertices'])} vertices, {sh} shared, "
      f"{len(tri)} three-atlas meta-clades, "
      f"forest={g['is_forest']}, "
      f"certificates={len(g['certificates'])}")
for c in g["certificates"]:
    print("  constraint:",
          sorted(g["vertices"][c["vertex"]]["members"].items()),
          "parents:", c["minimal_parents"])
print("ledger refused:", g["ledger"]["refused"])
print("ledger unresolved:", g["ledger"]["unresolved_ties"])
for r in tri:
    mem = g["vertices"][r]["members"]
    gs = {group_of[m] for m in mem.values() if m in group_of}
    print(f"  3-atlas vertex {r}: {sorted(mem.items())} groups={gs}")
c6_groups = all(
    len({group_of[m] for m in g["vertices"][r]["members"].values()
         if m in group_of}) <= 1 for r in tri)
c6 = c6_groups  # invariants assert internally; DAG state reported
print(f"C6 quotient gate (3-atlas vertices group-consistent, "
      f"invariants asserted): {'PASS' if c6 else 'FAIL'}")

# ---- C1 completeness ------------------------------------------------------
present = {m for nd in nodes.values()
           for _d, m in nd["members"].items()}
missing = [l for k in DS for l in labels_of[k] if l not in present]
c1 = not missing
status_ct = {}
for nd in nodes.values():
    status_ct[nd["status"]] = status_ct.get(nd["status"], 0) + 1
print(f"\nlegacy node statuses: {status_ct}")
print(f"C1 completeness (legacy; quotient I1 asserted): "
      f"{'PASS' if c1 else 'FAIL ' + str(missing)}")

# ---- C2 wrong-group reciprocal merges (quotient vertices) -----------------
merged = [(r, sorted(m for m in v["members"].values()
                     if m in group_of))
          for r, v in g["vertices"].items()
          if v["shared"]]
wrong = [(r, labs) for r, labs in merged
         if len({group_of[m] for m in labs}) > 1]
print(f"\nshared vertices with labels: "
      f"{sum(1 for _r, l in merged if len(l) >= 2)}")
for r, labs in merged:
    if len(labs) >= 2:
        print(f"  {r}: {labs}")
c2 = not wrong
print(f"C2 zero wrong-group merges: "
      f"{'PASS' if c2 else 'FAIL ' + str(wrong)}")

# ---- C5 fine-fine correspondence ------------------------------------------
NAME_PAIRS = ["RBC", "BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4",
              "BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7"]
vid_of = {}
for r, v in g["vertices"].items():
    for d_, m in v["members"].items():
        vid_of[(d_, m)] = r
name_hits = [t for t in NAME_PAIRS
             if vid_of.get(("she", f"she|{t}")) is not None
             and vid_of.get(("she", f"she|{t}")) ==
             vid_of.get(("mrca", f"mrca|{t}"))]
cross_group = [(r, labs) for r, labs in merged
               if len({group_of[m] for m in labs}) > 1
               and any(m.startswith("she|") for m in labs)
               and any(m.startswith("mrca|") for m in labs)]
mismatch = []
for r, labs in merged:
    she_l = [m[4:] for m in labs if m.startswith("she|")]
    mrca_l = [m[5:] for m in labs if m.startswith("mrca|")]
    for a in she_l:
        for b in mrca_l:
            if a != b and not (a == "BC8_9" and b in ("BC8", "BC9")):
                mismatch.append((r, a, b))
v89 = vid_of.get(("she", "she|BC8_9"))
fam89 = [t for t in ("BC8", "BC9")
         if vid_of.get(("mrca", f"mrca|{t}")) == v89]
ann89 = [(a["direction"], a["source"], a["target"], a["support"])
         for a in g["annotations"]
         if not a["within_merge"]
         and ("BC8" in str(a["source"]) + str(a["target"])
              or "BC9" in str(a["source"]) + str(a["target"]))]
print(f"\nC5 name-identical reciprocal merges: {len(name_hits)}/13 "
      f"{sorted(name_hits)}")
print(f"C5 missed pairs: "
      f"{sorted(set(NAME_PAIRS) - set(name_hits))}")
print(f"C5 within-group name mismatches (reported): {mismatch}")
print(f"C5 BC8_9 family: she|BC8_9 merged with {fam89 or 'none'}; "
      f"related annotations: {ann89}")
c5 = len(name_hits) >= 8 and not cross_group
print(f"C5 (>=8/13 name pairs, zero cross-group she<->mrca): "
      f"{'PASS' if c5 else 'FAIL'}")

# ---- C4 macosko cross-atlas relations -------------------------------------
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
    return out


mac_rel = {m: [] for m in labels_of["mac"]}
for m in labels_of["mac"]:
    r = vid_of[("mac", m)]
    for d_, o in g["vertices"][r]["members"].items():
        if d_ != "mac" and o in group_of \
                and group_of[o] == group_of[m]:
            mac_rel[m].append(("merge", o))
for a in g["annotations"]:
    if a["within_merge"]:
        continue
    src, tgt = a["source"], a["target"]
    d_src, d_tgt = a["direction"].split(">")
    if d_src == "mac" and src in group_of:
        tl = [l for l in leaves_under(trees[d_tgt], tgt)
              if l in group_of]
        if tl and all(group_of[l] == group_of[src] for l in tl):
            mac_rel[src].append((a["direction"], tgt, a["support"]))
    if d_tgt == "mac" and tgt in group_of and src in group_of \
            and group_of[src] == group_of[tgt]:
        mac_rel[tgt].append((a["direction"] + "(incoming)", src,
                             a["support"]))
for m in sorted(mac_rel):
    print(f"  {m} ({group_of[m]}): {mac_rel[m] or 'NONE'}")
c4 = all(mac_rel[m] for m in mac_rel)
print(f"C4 every mac cluster has a group-matched relation: "
      f"{'PASS' if c4 else 'FAIL'}")

# ---- truth scoring (4e) and C3 --------------------------------------------
def truth_tree(keep_atlas=None):
    keep = (lambda l: keep_atlas is None or
            l.startswith(f"{keep_atlas}|"))
    root = MyNode("root")

    def grp_node(gname, parent):
        gn = MyNode(f"__{gname}__")
        parent.addkid(gn)
        for ch in truth[gname]["cherries"]:
            ls = [l for l in ch if keep(l)]
            if len(ls) >= 2:
                cn = MyNode("__ch__")
                gn.addkid(cn)
                for l in ls:
                    cn.addkid(MyNode(l.replace("|", "-")))
            else:
                for l in ls:
                    gn.addkid(MyNode(l.replace("|", "-")))
        for l in truth[gname]["mac"]:
            if keep(l):
                gn.addkid(MyNode(l.replace("|", "-")))
        return gn
    grp_node("RBC_group", root)
    cbc = MyNode("__cbc__")
    root.addkid(cbc)
    grp_node("OFF", cbc)
    grp_node("ON", cbc)
    return root


def render_quotient(choice=None, keep_atlas=None):
    choice = choice or {}
    keep = (lambda l: keep_atlas is None or
            l.startswith(f"{keep_atlas}|"))
    kids, roots = {}, []
    for r, ps in g["parents"].items():
        if ps:
            kids.setdefault(choice.get(r, ps[0]), []).append(r)
        else:
            roots.append(r)
    root = MyNode("root")

    def emit(r, pn):
        labs = sorted(m.replace("|", "-") for m in
                      g["vertices"][r]["members"].values()
                      if m in group_of and keep(m))
        node = MyNode("&".join(labs) if labs else f"__{r}__")
        pn.addkid(node)
        for c in sorted(kids.get(r, []), key=str):
            emit(c, node)
    for r in sorted(roots, key=str):
        emit(r, root)
    return root


def input_label_tree(k):
    t = trees[k]
    root = MyNode("root")

    def build(n, pn):
        node = MyNode(n.replace("|", "-") if n in group_of
                      else f"__{k}_{n}__")
        pn.addkid(node)
        for c in t["children"].get(n, []):
            build(c, node)
    for c in t["children"].get("root", []):
        build(c, root)
    return root


choices = [dict()]
if not g["is_forest"]:
    certs = g["certificates"]
    if len(certs) <= 8:
        vxs = [c["vertex"] for c in certs]
        choices = [dict(zip(vxs, combo)) for combo in
                   itertools.product(*[c["minimal_parents"]
                                       for c in certs])]
        print(f"\nDAG result: scoring over ALL {len(choices)} "
              f"compatible projections")
    else:
        choices = [dict()] + [{c["vertex"]: alt} for c in certs
                              for alt in c["minimal_parents"][1:]]
        print(f"\nDAG result: {len(certs)} certificates — sampled "
              f"projections only (reported as observed scores)")

rows = []
REF_full = truth_tree()
for k in ("she", "mrca"):
    rec, _ = triplet_scores(input_label_tree(k), truth_tree(k))
    rows.append({"tree": f"{k}_input", "TRIP_REC_K": round(rec, 4)})
    print(f"truth scoring {k}_input   TRIP_REC={rec:.4f}")
c3 = True
for k in ("she", "mrca"):
    REF_k = truth_tree(k)
    scores = [triplet_scores(render_quotient(ch, k), REF_k)[0]
              for ch in choices]
    inp = next(r["TRIP_REC_K"] for r in rows
               if r["tree"] == f"{k}_input")
    ok = min(scores) >= inp - 0.02
    c3 &= ok
    rows.append({"tree": f"quotient_{k}_side",
                 "TRIP_REC_K": round(min(scores), 4)})
    print(f"C3 {k}-side: quotient min-over-projections "
          f"{min(scores):.4f} (max {max(scores):.4f}) vs input "
          f"{inp:.4f} -> {'PASS' if ok else 'FAIL'}")
full_scores = [triplet_scores(render_quotient(ch), REF_full)[0]
               for ch in choices]
coph = cophenetic_spearman(render_quotient(choices[0]), REF_full)
rows.append({"tree": "quotient_full",
             "TRIP_REC_K": round(min(full_scores), 4)})
print(f"quotient vs FULL 4e truth: TRIP min={min(full_scores):.4f} "
      f"max={max(full_scores):.4f} COPH(first proj)={coph:.4f}")


def project_legacy(nds):
    kids, roots = {}, []
    for i, nd in nds.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)
    root = MyNode("root")

    def build(i, pn):
        nd = nds[i]
        pp = sorted(m.replace("|", "-") for _d, m in
                    nd.get("members", {}).items() if m in group_of)
        node = MyNode("&".join(pp) if pp else f"__a{i}__")
        pn.addkid(node)
        for c in sorted(kids.get(i, [])):
            build(c, node)
    for r in sorted(roots):
        build(r, root)
    return root


rec_l, _ = triplet_scores(project_legacy(dump), REF_full)
rows.append({"tree": "legacy_assembly", "TRIP_REC_K": round(rec_l, 4)})
print(f"legacy assembly vs FULL 4e truth: TRIP={rec_l:.4f}")
print(f"C3 structural retention (both fine atlases, min over "
      f"projections): {'PASS' if c3 else 'FAIL'}")
with open(os.path.join(OUT, "retina_k3_scores.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["tree", "TRIP_REC_K"])
    w.writeheader()
    w.writerows(rows)

# ---- stability: order swap ------------------------------------------------
def decisions_of(h):
    out = {}
    for (ki, kj), recs in h["decisions"]["selections"].items():
        for n, r in recs.items():
            out[(ki, kj, n)] = r["selected"] if r["matched"] else None
    return out


print("\nreplicate: atlas-order swap (mrca+she+mac)...")
harm_o = run_harmonize(["mrca", "she", "mac"])
base, d_o = decisions_of(harm), decisions_of(harm_o)
common = set(base) & set(d_o)
agree = sum(base[k] == d_o[k] for k in common) / len(common)
print(f"order-swap decision agreement: {agree:.4f} "
      f"({len(common)} decisions)")

json.dump({"order_swap_agreement": agree,
           "quotient": {"vertices": len(g["vertices"]),
                        "shared": sh, "three_atlas": len(tri),
                        "is_forest": g["is_forest"],
                        "certificates": len(g["certificates"])},
           "checks": {"C1": bool(c1), "C2": bool(c2),
                      "C3": bool(c3), "C4": bool(c4),
                      "C5": bool(c5), "C6": bool(c6)}},
          open(os.path.join(OUT, "retina_k3_checks.json"), "w"),
          indent=1)
print("\nCHECKS:", {"C1": c1, "C2": c2, "C3": c3, "C4": c4,
                    "C5": c5, "C6": c6})
