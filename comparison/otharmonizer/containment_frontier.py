"""The containment frontier: a continuous family of consensus cuts
between the CERTIFIED core and the full tree, built by attaching
one-way-supported labels at their EVIDENCE targets.

For every label outside the certified core, its own atlas's one-way
Walk call (which passed every molecular gate — signal, margin,
compactness — and failed only reciprocity) names a target node in the
other atlas plus a bootstrap support value. cut(t) = certified core +
every such label with support >= t, attached beneath the certified
node its target maps into (direct member, else the target's nearest
input-tree ancestor that is a certified member; labels whose evidence
points outside the certified core stay excluded and are counted).

Sweeping t traces coverage vs topology recovery against the curated
reference — the tree-level analogue of a selective classifier's
coverage-risk curve. Scores are computed on SHARED labels, so each
point is scored over its own coverage (inherent to selective
evaluation; stated on the figure).

Run: PYTHONPATH=../../pkgs/python/src python containment_frontier.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pkgs", "python",
                                "src"))
from metrics import (MyNode, cophenetic_spearman,  # noqa: E402
                     triplet_scores)
from metaarbor.consensus.cut import consensus_cut  # noqa: E402

D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
MA = os.path.join(HERE, "..", "..", "pkgs", "python", "examples",
                  "harmonize_demo")

with open(os.path.join(D, "cells_10Xv3.csv")) as fh:
    cellsB = list(csv.DictReader(fh))
truth = {c["cluster"]: c["subclass"] for c in cellsB}
subclasses_v2 = set()
with open(os.path.join(D, "cells_10Xv2.csv")) as fh:
    for c in csv.DictReader(fh):
        subclasses_v2.add(c["subclass"])
labels = {"v2": {f"v2|{s}" for s in subclasses_v2},
          "v3": {f"v3|{c}" for c in truth}}
tree_nodes = json.load(open(os.path.join(MA,
                                         "metaarbor_tree_current.json")))
decisions = json.load(open(os.path.join(MA, "allen_decisions.json")))
canon = json.load(open(os.path.join(MA, "allen_canonical.json")))
itrees = json.load(open(os.path.join(MA, "allen_input_trees.json")))


def ref_tree():
    root = MyNode("root")
    by_sub = {}
    for cl, s in truth.items():
        by_sub.setdefault(s, []).append(cl)
    for s in sorted(by_sub):
        cls = sorted(by_sub[s])
        if s in subclasses_v2 and len(cls) == 1:
            root.addkid(MyNode(f"v2-{s}&v3-{cls[0]}"))
            continue
        sn = MyNode(f"v2-{s}" if s in subclasses_v2 else f"v3only-{s}")
        root.addkid(sn)
        for cl in cls:
            sn.addkid(MyNode(f"v3-{cl}"))
    return root


REF = ref_tree()
strict = consensus_cut(tree_nodes, level="strict", leaf_labels=labels)

# certified membership index: (ds, input-tree node) -> certified node id
cert_member = {}
for i, nd in strict["nodes"].items():
    for ds, m in nd["members"].items():
        cert_member[(ds, m)] = i
core_labels = {m for (ds, m) in cert_member if m in labels[ds]}


def evidence_target(ds, label):
    """(certified node id, support) for a label's own one-way call, or
    (None, reason)."""
    other = "v3" if ds == "v2" else "v2"
    c = canon[ds].get(label, label)
    rec = decisions.get(f"{ds}>{other}", {}).get(c)
    if not rec or not rec.get("matched") or rec.get("selected") is None:
        return None, "no_surviving_call"
    tgt = rec["selected"]
    node = cert_member.get((other, tgt))
    p = itrees[other]["parent"].get(tgt)
    while node is None and p not in (None, "root"):
        pc = canon[other].get(p, p)
        node = cert_member.get((other, pc))
        p = itrees[other]["parent"].get(p)
    if node is None:
        return None, "target_outside_core"
    return (node, float(rec["support"])), None


# gather attachable labels
attach = []
excluded = {"no_surviving_call": 0, "target_outside_core": 0}
for ds in ("v2", "v3"):
    for lab in sorted(labels[ds]):
        if lab in core_labels:
            continue
        hit, reason = evidence_target(ds, lab)
        if hit is None:
            excluded[reason] += 1
        else:
            attach.append((ds, lab, hit[0], hit[1]))
print(f"certified core labels: {len(core_labels)} | attachable one-way "
      f"labels: {len(attach)} | excluded: {excluded}")


def frontier_tree(t, keep_structure=False):
    """MyNode label-space tree: certified topology + one-way labels
    with support >= t as children of their evidence targets. With
    keep_structure, unlabeled certified internals become anonymous
    nodes instead of being spliced (COPH_K/TRIP_*_K columns; the
    phylogenetic convention — see rescore_current.py docstring)."""
    kids = {i: list(nd["children"]) for i, nd in strict["nodes"].items()}
    extra = {}
    for ds, lab, node, supp in attach:
        if supp >= t:
            extra.setdefault(node, []).append((ds, lab))

    def parts(nd):
        return [f"{ds}-{m.split('|', 1)[-1]}"
                for ds, m in sorted(nd["members"].items())
                if m in labels[ds]]

    root = MyNode("root")

    def build(i, parent_node):
        nd = strict["nodes"][i]
        pp = parts(nd)
        if pp:
            node = MyNode("&".join(pp))
        elif keep_structure:
            node = MyNode(f"__a{i}__")
        else:
            node = None
        if node is not None:
            parent_node.addkid(node)
        anchor = node or parent_node
        for ds, lab in extra.get(i, []):
            anchor.addkid(MyNode(f"{ds}-{lab.split('|', 1)[-1]}"))
        for c in sorted(kids.get(i, [])):
            build(c, anchor)
    for r in sorted(strict["roots"]):
        build(r, root)
    return root


rows = []
for t in (1.01, 1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.0):
    tr = frontier_tree(t)
    tr_k = frontier_tree(t, keep_structure=True)
    n_att = sum(1 for *_x, s in attach if s >= t)
    cov = (len(core_labels) + n_att) / (len(labels["v2"]) +
                                        len(labels["v3"]))
    coph = cophenetic_spearman(tr, REF)
    rec, agr = triplet_scores(tr, REF)
    coph_k = cophenetic_spearman(tr_k, REF)
    rec_k, agr_k = triplet_scores(tr_k, REF)
    rows.append({"threshold": t, "n_labels": len(core_labels) + n_att,
                 "coverage": round(cov, 4), "COPH": round(coph, 4),
                 "TRIP_REC": round(rec, 4), "TRIP_AGR": round(agr, 4),
                 "COPH_K": round(coph_k, 4),
                 "TRIP_REC_K": round(rec_k, 4),
                 "TRIP_AGR_K": round(agr_k, 4)})
    print(f"t={t:<5} labels={len(core_labels)+n_att:3d} "
          f"cov={cov:.3f} COPH={coph:.4f} TRIP_REC={rec:.4f} | "
          f"kept: COPH={coph_k:.4f} TRIP_REC={rec_k:.4f}")
with open(os.path.join(HERE, "containment_frontier.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote containment_frontier.csv")
