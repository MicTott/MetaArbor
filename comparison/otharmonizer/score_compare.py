"""Shared scorer for the frozen MetaArbor vs OTHarmonizer comparison.

One verdict function scores every tree identically: for each v3
cluster, find its node, climb to the first ancestor carrying any v2
label, and compare that ancestor's v2 subclass set to the curated
subclass (which never entered either method's construction):
  exact              the set is exactly {true subclass}
  consistent_coarse  the set contains the true subclass (breadth kept)
  wrong_lineage      the set excludes it
  root               no v2-bearing ancestor below root
  missing            the cluster is absent from the tree

Topology agreement uses OTHarmonizer's own metrics (TEDS, PCBS, AH-F1)
against one shared curated reference tree (root -> v2 subclass ->
member v3 clusters). MetaArbor's tree is projected to the same label
space first (anonymous inferred internals spliced out; affiliate
aliases excluded).

Stability = per-cluster agreement of predicted v2 sets between runs
(MetaArbor: seed 211 vs 977; OTHarmonizer: three insertion orders).
"""
import csv
import numpy as np
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "OTHarmonizer"))
D = "/Users/michael.totty/Documents/metaarbor/data/wmb_plilaorb"
MA = "/Users/michael.totty/Documents/metaarbor/pkgs/python/examples/harmonize_demo"
OTH = os.path.join(HERE, "oth_out")

with open(os.path.join(D, "cells_10Xv3.csv")) as fh:
    cellsB = list(csv.DictReader(fh))
truth = {c["cluster"]: c["subclass"] for c in cellsB}
clusters = sorted(truth)
subclasses_v2 = set()
with open(os.path.join(D, "cells_10Xv2.csv")) as fh:
    for c in csv.DictReader(fh):
        subclasses_v2.add(c["subclass"])

# ---------------- unified tree form: {id: {parent, v2set, v3set}} ----------


def from_oth(path):
    """OTHarmonizer JSON: nested {label, children}; labels
    'v2-x'/'v3-y' possibly '&'-merged."""
    js = json.load(open(path))
    nodes, ctr = {}, [0]

    def walk(nd, parent):
        i = f"n{ctr[0]}"; ctr[0] += 1
        parts = [p for p in str(nd["label"]).split("&") if p]
        v2 = {p[3:] for p in parts if p.startswith("v2-")}
        v3 = {p[3:] for p in parts if p.startswith("v3-")}
        nodes[i] = {"parent": parent, "v2": v2, "v3": v3}
        for c in nd["children"]:
            walk(c, i)
    for c in js["children"]:
        walk(c, None)
    return nodes


def from_metaarbor(path, leafsets):
    """MetaArbor JSON: {id: {parent, members, aliases, status}}.
    v2 member may be an inferred internal node -> expand via leafsets.
    Affiliate aliases (approx-prefixed) are NOT placements here — they
    are marked provisional associations and are excluded, matching
    their exclusion from reciprocal support."""
    js = json.load(open(path))
    nodes = {}
    for i, nd in js.items():
        v2, v2_is_leaf = set(), False
        m2 = nd["members"].get("v2")
        if m2:
            v2 = {x.split("|", 1)[-1] for x in leafsets["v2"].get(m2, [m2])}
            v2_is_leaf = len(leafsets["v2"].get(m2, [m2])) == 1
        v3 = set()
        m3 = nd["members"].get("v3")
        if m3:
            if m3 in leafsets["v3"] and len(leafsets["v3"][m3]) == 1:
                v3 = {m3.split("|", 1)[-1]}
            elif m3.startswith("v3|"):
                v3 = {m3.split("|", 1)[-1]}
        # v2: full subclass set (verdict climbing); v2lab: label-space
        # projection for the topology metrics — only an ORIGINAL v2 leaf
        # label names a node there; an internal v2 clade member is
        # anonymous structure (its subclasses appear as their own nodes),
        # not an equality group of subclasses
        nodes[i] = {"parent": nd["parent"], "v2": v2, "v3": v3,
                    "v2lab": v2 if v2_is_leaf else set()}
    return nodes


def verdicts(nodes):
    """{cluster: (verdict, frozenset predicted v2 subclasses)}"""
    where = {}
    for i, nd in nodes.items():
        for cl in nd["v3"]:
            if cl in truth and cl not in where:
                where[cl] = i
    out = {}
    for cl in clusters:
        i = where.get(cl)
        if i is None:
            out[cl] = ("missing", frozenset())
            continue
        p, pred = i, None
        while p is not None:
            if nodes[p]["v2"]:
                pred = frozenset(nodes[p]["v2"])
                break
            p = nodes[p]["parent"]
        t = truth[cl]
        if pred is None:
            out[cl] = ("root", frozenset())
        elif pred == {t}:
            out[cl] = ("exact", pred)
        elif t in pred:
            out[cl] = ("consistent_coarse", pred)
        else:
            out[cl] = ("wrong_lineage", pred)
    return out


def summarize(v):
    c = Counter(x[0] for x in v.values())
    breadth = [len(x[1]) for x in v.values()
               if x[0] == "consistent_coarse"]
    return {"exact": c.get("exact", 0),
            "consistent_coarse": c.get("consistent_coarse", 0),
            "wrong_lineage": c.get("wrong_lineage", 0),
            "root": c.get("root", 0), "missing": c.get("missing", 0),
            "median_coarse_breadth": (sorted(breadth)[len(breadth) // 2]
                                      if breadth else 0)}


def agreement(v1, v2):
    same = sum(1 for cl in clusters if v1[cl] == v2[cl])
    same_pred = sum(1 for cl in clusters if v1[cl][1] == v2[cl][1])
    return same / len(clusters), same_pred / len(clusters)


# ---------------- OTHarmonizer's own metrics on shared ref ----------------
from OTHarmonizer.harmonize import myNode  # noqa: E402
from OTHarmonizer.benchmark import benchmark  # noqa: E402


def ref_tree():
    root = myNode("root")
    by_sub = {}
    for cl, s in truth.items():
        by_sub.setdefault(s, []).append(cl)
    for s in sorted(by_sub):
        cls = sorted(by_sub[s])
        if s in subclasses_v2 and len(cls) == 1:
            # single-cluster subclass: the curated relation is EQUAL
            root.addkid(myNode(f"v2-{s}&v3-{cls[0]}"))
            continue
        sn = myNode(f"v2-{s}") if s in subclasses_v2 else myNode(
            f"v3only-{s}")
        root.addkid(sn)
        for cl in cls:
            sn.addkid(myNode(f"v3-{cl}"))
    return root


def to_mynode(nodes):
    """Project a unified tree to the shared label space; splice out
    anonymous nodes (no original v2/v3 labels)."""
    kids = {i: [] for i in nodes}
    roots = []
    for i, nd in nodes.items():
        (roots if nd["parent"] is None else kids[nd["parent"]]).append(i)
    root = myNode("root")

    def build(i, parent_node):
        nd = nodes[i]
        parts = [f"v2-{x}" for x in sorted(nd.get("v2lab", nd["v2"]))] + \
                [f"v3-{x}" for x in sorted(nd["v3"])]
        if parts:
            node = myNode("&".join(parts))
            parent_node.addkid(node)
            attach = node
        else:
            attach = parent_node          # splice anonymous node
        for c in kids[i]:
            build(c, attach)
    for r in roots:
        build(r, root)
    return root


REF = ref_tree()
rows, verd = [], {}
runs = [("MetaArbor", "primary",
         os.path.join(MA, "metaarbor_tree_primary.json")),
        ("MetaArbor", "seed977",
         os.path.join(MA, "metaarbor_tree_seed977.json")),
        ("OTHarmonizer", "default",
         os.path.join(OTH, "oth_tree_default.json")),
        ("OTHarmonizer", "v2_first",
         os.path.join(OTH, "oth_tree_v2_first.json")),
        ("OTHarmonizer", "v3_first",
         os.path.join(OTH, "oth_tree_v3_first.json"))]
runs += [("OTHarmonizer", f"{o}_r{r}",
          os.path.join(OTH, f"oth_tree_{o}_r{r}.json"))
         for o in ("v2f", "v3f") for r in range(5)]
leafsets = json.load(open(os.path.join(MA, "input_tree_leafsets.json")))
for method, tag, path in runs:
    if not os.path.exists(path):
        print(f"SKIP {method}/{tag}: {path} not found")
        continue
    nodes = (from_metaarbor(path, leafsets) if method == "MetaArbor"
             else from_oth(path))
    v = verdicts(nodes)
    verd[(method, tag)] = v
    s = summarize(v)
    print(f"\n===== {method} / {tag} =====")
    print(s)
    bm = benchmark(to_mynode(nodes), REF)
    print("topology metrics vs shared curated ref:", bm)
    rows.append({"method": method, "run": tag, **s,
                 **{k: round(float(x), 4) for k, x in bm.items()}})

print("\n===== stability (per-cluster predicted-set agreement) =====")
stab_rows = []
pairs = [(("MetaArbor", "primary"), ("MetaArbor", "seed977"))]
reps = {o: [("OTHarmonizer", f"{o}_r{r}") for r in range(5)
            if ("OTHarmonizer", f"{o}_r{r}") in verd]
        for o in ("v2f", "v3f")}
for o, rr in reps.items():                     # within-order (sampling)
    pairs += [(rr[i], rr[j]) for i in range(len(rr))
              for j in range(i + 1, len(rr))]
pairs += [(a, b) for a in reps["v2f"] for b in reps["v3f"]]  # between
within, between = [], []
for a, b in pairs:
    if a in verd and b in verd:
        full, _ = agreement(verd[a], verd[b])
        stab_rows.append({"a": "/".join(a), "b": "/".join(b),
                          "verdict_and_set": round(full, 4)})
        if a[0] == b[0] == "OTHarmonizer":
            (within if a[1][:3] == b[1][:3] else between).append(full)
        else:
            print(f"{a} vs {b}: agreement {full:.3f}")
if within:
    print(f"OTHarmonizer within-order sampling agreement: "
          f"mean {np.mean(within):.3f} (n={len(within)}, "
          f"range {min(within):.3f}-{max(within):.3f})")
if between:
    print(f"OTHarmonizer between-order agreement: "
          f"mean {np.mean(between):.3f} (n={len(between)}, "
          f"range {min(between):.3f}-{max(between):.3f})")

with open(os.path.join(OTH, "comparison_summary.csv"), "w",
          newline="") as fh:
    if rows:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
with open(os.path.join(OTH, "comparison_stability.csv"), "w",
          newline="") as fh:
    if stab_rows:
        w = csv.DictWriter(fh, fieldnames=list(stab_rows[0]))
        w.writeheader(); w.writerows(stab_rows)

# per-cluster verdict dump for auditing
with open(os.path.join(OTH, "comparison_per_cluster.csv"), "w",
          newline="") as fh:
    w = csv.writer(fh)
    keys = [k for k in verd]
    w.writerow(["cluster", "curated_subclass"] +
               ["/".join(k) for k in keys])
    for cl in clusters:
        w.writerow([cl, truth[cl]] +
                   [f"{verd[k][cl][0]}:{'|'.join(sorted(verd[k][cl][1]))}"
                    for k in keys])
print("\nwrote comparison_summary.csv, comparison_stability.csv, "
      "comparison_per_cluster.csv ->", OTH)
