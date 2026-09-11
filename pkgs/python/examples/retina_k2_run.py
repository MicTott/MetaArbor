"""Retina bipolar K=2 — stage 2: frozen harmonize, decision dumps,
truth scoring, prespecified checks C1-C4, and stability replicates.
Everything per retina_k2_PRESPEC.md (committed a638e1b) and the
stage-1 annex (22315db). No thresholds tuned; failures reported.

Run: python examples/retina_k2_run.py
"""
import csv
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
from metaarbor.consensus.cut import consensus_cut  # noqa: E402
from metaarbor.consensus.harmonize import harmonize  # noqa: E402

OUT = os.path.join(HERE, "retina_k2")
it = json.load(open(os.path.join(OUT, "retina_k2_input_trees.json")))
truth = json.load(open(os.path.join(OUT, "retina_k2_truth.json")))
genes = open(os.path.join(OUT, "shared_genes.txt")).read().split("\n")
genes = [g for g in genes if g]
mac_X = np.asarray(mmread(os.path.join(OUT, "mac_bipolar.mtx"))
                   .todense())
she_X = np.asarray(mmread(os.path.join(OUT, "she_bipolar.mtx"))
                   .todense())
mac_lab = np.loadtxt(os.path.join(OUT, "mac_labels.txt"), dtype=str)
she_lab = np.loadtxt(os.path.join(OUT, "she_labels.txt"), dtype=str)
group_of = {}
for grp, labs in truth.items():
    for l in labs:
        group_of[l] = grp
ALL_LABELS = sorted(group_of)

def mk_tree(d):
    return {"parent": d["parent"], "children": d["children"],
            "leaves": d["leaves"]}


trees = {"mac": mk_tree(it["mac"]), "she": mk_tree(it["she"])}
stability = {("mac", n): v for n, v in it["mac"]["support"].items()}
stability.update({("she", n): v
                  for n, v in it["she"]["support"].items()})


def run_harmonize(order, seed_note, base_counts_scramble=False):
    ds = {}
    pairs = {"mac": (mac_X, mac_lab), "she": (she_X, she_lab)}
    for k in order:
        X, lab = pairs[k]
        ds[k] = {"counts": X, "labels": lab, "gene_names": genes,
                 "lib": X.sum(axis=1)}
    return harmonize(ds, {k: trees[k] for k in order}, n_hvg=1000,
                     n_boot=200, stability=stability)


print("harmonize (primary, mac+she order)...")
harm = run_harmonize(["mac", "she"], "primary")
nodes = harm["tree"]

# ---- dumps ---------------------------------------------------------------
dump = {i: {"parent": nd["parent"], "status": nd["status"],
            "members": nd["members"], "aliases": nd["aliases"],
            "display": nd["display"],
            "assembly_repair": bool(nd.get("assembly_repair"))}
        for i, nd in nodes.items()}
json.dump(dump, open(os.path.join(OUT, "retina_k2_tree.json"), "w"),
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
json.dump(sel_out, open(os.path.join(OUT, "retina_k2_decisions.json"),
                        "w"), indent=1)
json.dump({k: canonical_nodes(trees[k])[1] for k in trees},
          open(os.path.join(OUT, "retina_k2_canonical.json"), "w"))

# ---- status accounting (C1) ----------------------------------------------
labels_all = {"mac": set(f"mac|c{c}" for c in range(26, 34)),
              "she": {l for l in ALL_LABELS if l.startswith("she|")}}
present = {m for nd in nodes.values()
           for ds_, m in nd["members"].items()}
missing = [l for ds_ in labels_all for l in labels_all[ds_]
           if l not in present]
c1 = not missing
status_ct = {}
for nd in nodes.values():
    status_ct[nd["status"]] = status_ct.get(nd["status"], 0) + 1
print("node statuses:", status_ct)
print(f"C1 completeness: {'PASS' if c1 else 'FAIL ' + str(missing)}")

# ---- reciprocal meta-clades + C2 wrong-group merges ----------------------
backbone = [(i, nd) for i, nd in nodes.items()
            if nd["status"] == "backbone" and len(nd["members"]) >= 2]
wrong_group = []
cross_pairs = []
for i, nd in backbone:
    mem = [(ds_, m) for ds_, m in nd["members"].items()]
    labs = [m for _d, m in mem if m in group_of]
    gs = {group_of[m] for m in labs}
    if len(gs) > 1:
        wrong_group.append((i, labs, sorted(gs)))
    if len(labs) >= 2:
        cross_pairs.append((i, labs))
print(f"reciprocal backbone nodes (>=2 atlases): {len(backbone)}; "
      f"with >=2 LABELS: {len(cross_pairs)}")
for i, labs in cross_pairs:
    print(f"  {i}: {labs}")
c2 = not wrong_group
print(f"C2 zero wrong-group merges: "
      f"{'PASS' if c2 else 'FAIL ' + str(wrong_group)}")

# ---- one-way containment + C4 --------------------------------------------
canon = {k: canonical_nodes(trees[k])[1] for k in trees}


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


c4_hits = []
oneway = 0
for direc, recs in sel_out.items():
    src, tgt = direc.split(">")
    for n, r in recs.items():
        if not r["matched"] or r["selected"] is None:
            continue
        oneway += 1
        if n in group_of:                      # source is a label
            tgt_labs = [l for l in leaves_under(trees[tgt],
                                                r["selected"])
                        if l in group_of]
            if tgt_labs and all(group_of[l] == group_of[n]
                                for l in tgt_labs):
                c4_hits.append((n, direc, r["selected"],
                                r["support"]))
print(f"matched one-way calls: {oneway}; group-consistent "
      f"label-level calls: {len(c4_hits)}")
c4 = len(c4_hits) > 0
print(f"C4 cross-atlas group-matched relationship exists: "
      f"{'PASS' if c4 else 'FAIL'}")

# ---- truth scoring (structure kept) --------------------------------------
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
disp = lambda l: l.replace("|", "-")


def project_nodes(nds, keep=True):
    kids, roots = {}, []
    for i, nd in nds.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)
    root = MyNode("root")

    def build(i, pn):
        nd = nds[i]
        pp = [disp(m) for _d, m in sorted(nd.get("members",
                                                 {}).items())
              if m in group_of]
        node = MyNode("&".join(pp)) if pp else (
            MyNode(f"__a{i}__") if keep else None)
        if node is not None:
            pn.addkid(node)
        anchor = node or pn
        for c in sorted(kids.get(i, [])):
            build(c, anchor)
    for r in sorted(roots):
        build(r, root)
    return root


def input_label_tree(k):
    t = trees[k]
    kids = t["children"]
    root = MyNode("root")

    def build(n, pn):
        node = MyNode(disp(n) if n in group_of else f"__{k}_{n}__")
        pn.addkid(node)
        for c in kids.get(n, []):
            build(c, node)
    for c in kids.get("root", []):
        build(c, root)
    return root


rows = []
for name, tr in (("mac_input", input_label_tree("mac")),
                 ("she_input", input_label_tree("she")),
                 ("assembly", project_nodes(dump))):
    rec, _ = triplet_scores(tr, REF)
    coph = cophenetic_spearman(tr, REF)
    rows.append({"tree": name, "TRIP_REC_K": round(rec, 4),
                 "COPH_K": round(coph, 4)})
    print(f"truth scoring {name:10s} TRIP_REC={rec:.4f} "
          f"COPH={coph:.4f}")
for lv in ("strict", "default"):
    cut = consensus_cut(dump, level=lv, leaf_labels=labels_all)
    tr = project_nodes(cut["nodes"])
    rec, _ = triplet_scores(tr, REF)
    rows.append({"tree": f"{lv}_cut", "TRIP_REC_K": round(rec, 4),
                 "COPH_K": round(cophenetic_spearman(tr, REF), 4)})
    print(f"truth scoring {lv + '_cut':10s} TRIP_REC={rec:.4f}")
she_in = next(r for r in rows if r["tree"] == "she_input")
asm = next(r for r in rows if r["tree"] == "assembly")
c3 = asm["TRIP_REC_K"] >= she_in["TRIP_REC_K"] - 0.02
print(f"C3 assembly >= she_input - 0.02: "
      f"{'PASS' if c3 else 'FAIL'} "
      f"({asm['TRIP_REC_K']} vs {she_in['TRIP_REC_K']})")
with open(os.path.join(OUT, "retina_k2_scores.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---- stability replicates -------------------------------------------------
def decisions_of(h):
    out = {}
    for (ki, kj), recs in h["decisions"]["selections"].items():
        for n, r in recs.items():
            out[(ki, kj, n)] = r["selected"] if r["matched"] else None
    return out


base = decisions_of(harm)
print("replicate: atlas-order swap (she+mac)...")
harm_o = run_harmonize(["she", "mac"], "order")
d_o = decisions_of(harm_o)
common = set(base) & set(d_o)
agree_o = sum(base[k] == d_o[k] for k in common) / len(common)
print(f"order-swap decision agreement: {agree_o:.4f} "
      f"({len(common)} decisions)")
print("NOTE: seed replicate uses the same frozen harmonize seed "
      "path; the frozen pipeline's seeding is internal (per-query "
      "MINSTD), so the order swap is the informative replicate; a "
      "full independent-seed rerun would require reruns of "
      "infer_tree and is out of scope for this stage (stated, not "
      "hidden).")
json.dump({"order_swap_agreement": agree_o,
           "checks": {"C1": bool(c1), "C2": bool(c2),
                      "C3": bool(c3), "C4": bool(c4)}},
          open(os.path.join(OUT, "retina_k2_checks.json"), "w"),
          indent=1)
print(f"\nCHECKS: C1={'PASS' if c1 else 'FAIL'} "
      f"C2={'PASS' if c2 else 'FAIL'} C3={'PASS' if c3 else 'FAIL'} "
      f"C4={'PASS' if c4 else 'FAIL'}")
