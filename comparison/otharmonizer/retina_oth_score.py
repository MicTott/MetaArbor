"""Score the OTHarmonizer retina trees against the FROZEN MetaArbor
K=2 record (e887341) on identical inputs and the frozen truth tree.
Conventions: structure-kept metrics (both methods; OTH trees have no
anonymous internals so the convention is a no-op for them); '|' in
labels normalized to '-'; truth universe = the 22 published labels.

Run: PYTHONPATH=../../pkgs/python/src python retina_oth_score.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pkgs", "python",
                                "src"))
from metrics import (MyNode, cophenetic_spearman,  # noqa: E402
                     from_nested, triplet_scores)
from metaarbor.consensus.interleave import (  # noqa: E402
    interleave_core_ancestry)

RK = os.path.join(HERE, "..", "..", "pkgs", "python", "examples",
                  "retina_k2")
OTH = os.path.join(HERE, "retina_oth_out")
truth = json.load(open(os.path.join(RK, "retina_k2_truth.json")))
group_of = {l.replace("|", "-"): g for g, ls in truth.items()
            for l in ls}
ALL = sorted(group_of)


def truth_tree(allowed=None):
    allowed = allowed or set(ALL)
    root = MyNode("root")
    rbc = MyNode("__rbc__")
    root.addkid(rbc)
    for l in truth["RBC_group"]:
        if l.replace("|", "-") in allowed:
            rbc.addkid(MyNode(l.replace("|", "-")))
    cbc = MyNode("__cbc__")
    root.addkid(cbc)
    for grp in ("OFF", "ON"):
        g = MyNode(f"__{grp}__")
        cbc.addkid(g)
        for l in truth[grp]:
            if l.replace("|", "-") in allowed:
                g.addkid(MyNode(l.replace("|", "-")))
    return root


def norm_label(l):
    """OTH prefixes labels with '<batch>-'; strip it per '&' part,
    then normalize '|' to '-' ('mac-mac|c26' -> 'mac-c26')."""
    parts = []
    for p in l.split("&"):
        for b in ("mac-", "she-"):
            if p.startswith(b) and "|" in p:
                p = p[len(b):]
                break
        parts.append(p.replace("|", "-"))
    return "&".join(parts)


def norm_nested(d):
    return {"label": norm_label(d["label"]),
            "children": [norm_nested(c) for c in d["children"]]}


def tree_labels(t):
    out = set()
    stack = [t]
    while stack:
        n = stack.pop()
        for part in n.label.split("&"):
            if part in group_of:
                out.add(part)
        stack.extend(n.children)
    return out


def wrong_group_merges(t):
    bad, good = [], []
    stack = [t]
    while stack:
        n = stack.pop()
        parts = [p for p in n.label.split("&") if p in group_of]
        if len(parts) >= 2:
            gs = {group_of[p] for p in parts}
            (bad if len(gs) > 1 else good).append(n.label)
        stack.extend(n.children)
    return good, bad


def root_singletons(t):
    return sum(1 for c in t.children if not c.children
               and any(p in group_of for p in c.label.split("&")))


def score(name, tree):
    labs = tree_labels(tree)
    REF = truth_tree()
    rec, _ = triplet_scores(tree, REF)
    coph = cophenetic_spearman(tree, REF)
    she = {l for l in labs if l.startswith("she-")}
    mac = {l for l in labs if l.startswith("mac-")}
    rec_she, _ = triplet_scores(tree, truth_tree({l for l in ALL
                                                  if l.startswith(
                                                      "she-")}))
    rec_mac, _ = triplet_scores(tree, truth_tree({l for l in ALL
                                                  if l.startswith(
                                                      "mac-")}))
    good, bad = wrong_group_merges(tree)
    row = {"tree": name, "n_labels": len(labs),
           "complete": len(labs) == len(ALL),
           "TRIP_REC": round(rec, 4), "COPH": round(coph, 4),
           "retention_she": round(rec_she, 4),
           "retention_mac": round(rec_mac, 4),
           "good_merges": len(good), "wrong_group_merges": len(bad),
           "root_singletons": root_singletons(tree)}
    print(f"{name:22s} labels={len(labs):2d}/{len(ALL)} "
          f"TRIP={rec:.4f} COPH={coph:.4f} she-ret={rec_she:.4f} "
          f"mac-ret={rec_mac:.4f} merges +{len(good)}/-{len(bad)} "
          f"rootsing={row['root_singletons']}")
    if bad:
        print(f"    WRONG-GROUP: {bad}")
    if good:
        print(f"    merges: {good}")
    return row


# ---- MetaArbor frozen arms ------------------------------------------------
dump = json.load(open(os.path.join(RK, "retina_k2_tree.json")))
it = json.load(open(os.path.join(RK, "retina_k2_input_trees.json")))
canon = json.load(open(os.path.join(RK, "retina_k2_canonical.json")))
itrees = {k: {"parent": it[k]["parent"],
              "children": it[k]["children"],
              "leaves": it[k]["leaves"]} for k in ("mac", "she")}
inter, _ = interleave_core_ancestry(dump, itrees, canon)


def project(nds):
    kids, roots = {}, []
    for i, nd in nds.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)
    root = MyNode("root")

    def build(i, pn):
        nd = nds[i]
        pp = [m.replace("|", "-") for _d, m in
              sorted(nd.get("members", {}).items())
              if m.replace("|", "-") in group_of]
        node = MyNode("&".join(pp)) if pp else MyNode(f"__a{i}__")
        pn.addkid(node)
        for c in sorted(kids.get(i, [])):
            build(c, node)
    for r in sorted(roots):
        build(r, root)
    return root


rows = [score("MA_assembly", project(dump)),
        score("MA_interleaved", project(inter))]
for tag in ("default", "default_r1", "mac_first", "she_first"):
    t = from_nested(norm_nested(json.load(open(os.path.join(
        OTH, f"oth_retina_{tag}.json")))))
    rows.append(score(f"OTH_{tag}", t))
import csv  # noqa: E402
with open(os.path.join(HERE, "retina_oth_scores.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote retina_oth_scores.csv")
