"""Refreshed Allen scoring with the ported OTHarmonizer metrics
(metrics.py; parity-tested): scores the CURRENT (v0.7-semantics)
MetaArbor assembly, its strict/default consensus cuts, the frozen-era
assembly (continuity), and every committed OTHarmonizer tree, all
against the same curated reference used in the frozen comparison.

The reference and the label-space projection rules are unchanged from
score_compare.py: only original labels name nodes ('v2-<subclass>',
'v3-<cluster>'); merged equivalences join with '&'; anonymous
structure (inferred internals) is spliced; affiliates excluded.

TWO SCORING CONVENTIONS for the star-proof metrics. TEDS/PCBS/AH-F1
are label-relation metrics and require every node to be a label, so
anonymous inferred internals MUST be spliced for them. But cophenetic
Spearman and triplet metrics only need the topology relating shared
labels — the standard phylogenetic convention (labeled leaves,
anonymous internals) applies, and splicing anonymous structure
DISCARDS genuinely inferred organization before scoring. Columns
COPH/TRIP_REC/TRIP_AGR use the spliced (label-only) convention;
COPH_K/TRIP_REC_K/TRIP_AGR_K keep anonymous structure. OTHarmonizer
trees have no anonymous nodes, so their _K columns equal the plain
ones by construction — the convention only changes what MetaArbor's
own structure is allowed to count.

Run: PYTHONPATH=../../pkgs/python/src python rescore_current.py
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
                     score_all, triplet_scores)
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
leafsets = json.load(open(os.path.join(MA, "input_tree_leafsets.json")))
labels = {"v2": {f"v2|{s}" for s in subclasses_v2},
          "v3": {f"v3|{c}" for c in truth}}
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


def project_nodes(nodes, keep_structure=False):
    """Assembly/cut node mapping -> label-space MyNode tree. With
    keep_structure, unlabeled internals become anonymous nodes
    ('__aN__') instead of being spliced."""
    kids, roots = {}, []
    for i, nd in nodes.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)

    def parts(nd):
        out = []
        for ds, m in sorted(nd.get("members", {}).items()):
            if m in labels.get(ds, ()):
                out.append(f"{ds.replace('|', '')}-"
                           f"{m.split('|', 1)[-1]}")
        return out

    root = MyNode("root")

    def build(i, parent_node):
        nd = nodes[i]
        pp = parts(nd)
        if pp:
            node = MyNode("&".join(pp))
        elif keep_structure:
            node = MyNode(f"__a{i}__")
        else:
            node = None
        if node is not None:
            parent_node.addkid(node)
        attach = node or parent_node
        for c in sorted(kids.get(i, [])):
            build(c, attach)
    for r in sorted(roots):
        build(r, root)
    return root


def input_tree(ds, keep_structure=True):
    """A single atlas's inferred input tree in label space."""
    par = itrees[ds]["parent"]
    kids, roots = {}, []
    for n, p in par.items():
        (roots if p in (None, "root") else
         kids.setdefault(p, [])).append(n)
    root = MyNode("root")

    def build(n, parent_node):
        if n in labels[ds]:
            node = MyNode(f"{ds}-{n.split('|', 1)[-1]}")
        elif keep_structure:
            node = MyNode(f"__{ds}_{n}__")
        else:
            node = None
        if node is not None:
            parent_node.addkid(node)
        anchor = node or parent_node
        for c in sorted(kids.get(n, [])):
            build(c, anchor)
    for r in sorted(roots):
        build(r, root)
    return root


def load_nested(path):
    return json.load(open(path))


REF = ref_tree()


def star(lbls):
    r = MyNode("root")
    for l in sorted(lbls):
        r.addkid(MyNode(l))
    return r


# STRUCTURE-FREE BASELINES (essential context: TEDS and PCBS are both
# beaten by a flat star of labels, so neither can headline a method
# comparison; AH-F1 is the only one of the three that requires genuine
# relational content — see the finding in the commit log)
v2_l = {f"v2-{s_}" for s_ in subclasses_v2}
v3_l = {f"v3-{c}" for c in truth}
# each run: (method, tag, label-only tree, structure-kept tree or None
# when the two coincide)
runs = [("baseline", "flat_union_star", star(v2_l | v3_l), None),
        ("baseline", "v2_input_star", star(v2_l), None),
        ("baseline", "v3_input_star", star(v3_l), None),
        ("baseline", "v2_input_tree", input_tree("v2", False),
         input_tree("v2", True)),
        ("baseline", "v3_input_tree", input_tree("v3", False),
         input_tree("v3", True))]
cur_path = os.path.join(MA, "metaarbor_tree_current.json")
if os.path.exists(cur_path):
    cur = load_nested(cur_path)
    runs.append(("MetaArbor", "current_assembly", project_nodes(cur),
                 project_nodes(cur, keep_structure=True)))
    for lv in ("default", "strict"):
        cut = consensus_cut(cur, level=lv, leaf_labels=labels)
        runs.append(("MetaArbor", f"{lv}_cut",
                     project_nodes(cut["nodes"]),
                     project_nodes(cut["nodes"], keep_structure=True)))
else:
    print("NOTE: metaarbor_tree_current.json missing — current-era "
          "rows skipped")
old = load_nested(os.path.join(MA, "metaarbor_tree_primary.json"))
runs.append(("MetaArbor", "frozen_v0.8_assembly", project_nodes(old),
             project_nodes(old, keep_structure=True)))
for lv in ("default", "strict"):
    cut = consensus_cut(old, level=lv, leaf_labels=labels)
    runs.append(("MetaArbor", f"frozen_v0.8_{lv}_cut",
                 project_nodes(cut["nodes"]),
                 project_nodes(cut["nodes"], keep_structure=True)))
for fn in sorted(os.listdir(HERE)):
    if fn.startswith("oth_tree_") and fn.endswith(".json"):
        tag = fn[len("oth_tree_"):-len(".json")]
        runs.append(("OTHarmonizer", tag,
                     load_nested(os.path.join(HERE, fn)), None))

rows = []
for method, tag, tree, tree_k in runs:
    got = score_all(tree, REF)
    got["COPH"] = cophenetic_spearman(tree, REF)
    got["TRIP_REC"], got["TRIP_AGR"] = triplet_scores(tree, REF)
    tk = tree if tree_k is None else tree_k
    got["COPH_K"] = cophenetic_spearman(tk, REF)
    got["TRIP_REC_K"], got["TRIP_AGR_K"] = triplet_scores(tk, REF)
    rows.append({"method": method, "run": tag,
                 **{k: round(v, 4) for k, v in got.items()}})
    print(f"{method:12s} {tag:24s} TEDS={got['TEDS']:.4f} "
          f"PCBS={got['PCBS']:.4f} AH_F1={got['AH_F1']:.4f} "
          f"COPH={got['COPH']:.4f} TRIP_REC={got['TRIP_REC']:.4f} | "
          f"kept: COPH={got['COPH_K']:.4f} "
          f"TRIP_REC={got['TRIP_REC_K']:.4f}")
with open(os.path.join(HERE, "rescore_current.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote rescore_current.csv")
