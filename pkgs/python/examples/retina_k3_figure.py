"""K=3 audit figure (package default renderer): the three input
trees beside the CERTIFIED quotient assembly, shared canonical
order, group bands = prespecified truth groups, raw denominators in
subtitles. The quotient panel draws the first-minimal-parent
projection; a dashed edge INTO a vertex marks the three
multi-parent vertices (unreconciled parentage — a view, not the
result; ASSEMBLY2 Section 11).

Run: PYTHONPATH=src python examples/retina_k3_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metaarbor.consensus.candidates import canonical_nodes  # noqa
from metaarbor.consensus.quotient import quotient_assemble  # noqa
from metaarbor.viz import audit_legend, plot_audit_tree  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
O3 = os.path.join(HERE, "retina_k3")
it = json.load(open(os.path.join(O3, "retina_k3_input_trees.json")))
truth = json.load(open(os.path.join(O3, "retina_k3_truth.json")))
dec = json.load(open(os.path.join(O3, "retina_k3_decisions.json")))
cert = json.load(open(os.path.join(
    O3, "retina_k3_certified_merges.json")))
trees = {k: {"parent": it[k]["parent"],
             "children": it[k]["children"],
             "leaves": it[k]["leaves"]}
         for k in ("mac", "she", "mrca")}
canon = {k: canonical_nodes(trees[k])[1] for k in trees}
g = quotient_assemble(trees, canon, dec, certified=cert)

group_of = {}
for grp, d in truth.items():
    for ch in d["cherries"]:
        for l in ch:
            group_of[l] = grp
    for l in d["mac"]:
        group_of[l] = grp
disp = lambda l: l.replace("|", "-")
real = set(group_of)

ORDER_T = ["RBC", "BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4",
           "BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7", "BC8_9",
           "BC8", "BC9"]
order = []
for grp in ("RBC_group", "OFF", "ON"):
    for t in ORDER_T:
        for ds in ("she", "mrca"):
            l = f"{ds}|{t}"
            if group_of.get(l) == grp:
                order.append(disp(l))
    for l in sorted(truth[grp]["mac"]):
        order.append(disp(l))
groups_disp = {disp(l): gr for l, gr in group_of.items()}


def nested_input(k):
    t = trees[k]

    def build(n):
        return {"label": disp(n) if n in real else "",
                "edge": "ancestry",
                "children": [build(c)
                             for c in t["children"].get(n, [])]}
    return {"label": "", "edge": "ancestry",
            "children": [build(c)
                         for c in t["children"].get("root", [])]}


def nested_quotient():
    multi = {c["vertex"] for c in g["certificates"]}
    kids, roots = {}, []
    for r, ps in g["parents"].items():
        if ps:
            kids.setdefault(ps[0], []).append(r)
        else:
            roots.append(r)

    def build(r):
        labs = sorted(disp(m) for m in
                      g["vertices"][r]["members"].values()
                      if m in real)
        edge = ("containment" if r in multi else
                ("certified" if g["vertices"][r]["shared"]
                 else "ancestry"))
        return {"label": "&".join(labs), "edge": edge,
                "children": [build(c) for c in
                             sorted(kids.get(r, []), key=str)]}
    def prune(n):
        n["children"] = [prune(c) for c in n["children"]]
        n["children"] = [c for c in n["children"]
                         if c["label"] or c["children"]]
        return n
    return prune({"label": "", "edge": "ancestry",
                  "children": [build(r)
                               for r in sorted(roots, key=str)]})


fig, axes = plt.subplots(1, 4, figsize=(22, 12))
kw = dict(order=order, groups=groups_disp,
          dataset_of=lambda l: l.split("-")[0])
plot_audit_tree(axes[0], nested_input("mac"), title="Macosko input",
                subtitle="8 clusters, 6285 cells", **kw)
plot_audit_tree(axes[1], nested_input("she"), title="Shekhar input",
                subtitle="14 labels, 23494 cells", **kw)
plot_audit_tree(axes[2], nested_input("mrca"), title="MRCA input",
                subtitle="15 types, 22435 cells (GSE243413 only)",
                **kw)
plot_audit_tree(axes[3], nested_quotient(),
                title="Certified quotient assembly (K=3)",
                subtitle=("18 certified merges; 8 three-atlas meta-clades;\n"
                          "11/13 she&mrca name pairs; DAG: 3 multi-"
                          "parent vertices\n(dashed edge = 1 of 2 "
                          "parents drawn; a view, not the result)"),
                **kw)
audit_legend(fig, groups=groups_disp,
             dataset_names=["mac", "mrca", "she"], ncol=4,
             bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Retina bipolar K=3 — inputs and certified quotient "
             "(frozen MetaArbor; truth groups as bands)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0.05, 1, 0.98))
out = os.path.join(O3, "retina_k3_trees.png")
fig.savefig(out, dpi=160)
print("wrote", out)
