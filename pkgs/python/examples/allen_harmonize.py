"""Allen coarse-vs-fine harmonize with known truth: 10Xv2 at SUBCLASS
(2-level class->subclass tree) vs 10Xv3 at CLUSTER (curated 4-level tree).
Truth = the curated subclass->cluster nesting. The merged tree must place
each 10Xv3 cluster beneath the meta-clade of its true subclass."""
import csv, os, sys
import numpy as np
from scipy.io import mmread
from metaarbor import tree_from_levels
from metaarbor.consensus.harmonize import harmonize
from metaarbor.consensus.plot_reconciled import plot_reconciled_tree
from metaarbor.style import save_pub

D = "../../data/wmb_plilaorb"
def load(tag):
    counts = np.asarray(mmread(os.path.join(D, f"counts_{tag}.mtx")).todense()).T
    lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
    with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells

genes = open(os.path.join(D, "genes.txt")).read().split()
cA, lA, cellsA = load("10Xv2")
cB, lB, cellsB = load("10Xv3")
datasets = {
    "v2": {"counts": cA, "labels": np.asarray([f"v2|{c['subclass']}" for c in cellsA]),
           "gene_names": genes, "lib": lA},
    "v3": {"counts": cB, "labels": np.asarray([f"v3|{c['cluster']}" for c in cellsB]),
           "gene_names": genes, "lib": lB},
}
trees = {
    "v2": tree_from_levels(sorted({(f"v2|{c['class']}", f"v2|{c['subclass']}")
                                   for c in cellsA}), ["class", "subclass"]),
    "v3": tree_from_levels(sorted({(f"v3|{c['class']}", f"v3|{c['subclass']}",
                                    f"v3|{c['supertype']}", f"v3|{c['cluster']}")
                                   for c in cellsB}),
                           ["class", "subclass", "supertype", "cluster"]),
}
harm = harmonize(datasets, trees, n_hvg=1000, n_boot=200)
nodes = harm["tree"]
truth_sub = {f"v3|{c['cluster']}": c["subclass"] for c in cellsB}

# score: each v3 cluster node's ancestor chain must reach the meta-clade
# whose v2 member is its true subclass
member_sub = {}
for mid, nd in nodes.items():
    m = nd["members"].get("v2", "")
    if m.startswith("subclass:") or (m and "subclass" in m):
        member_sub[mid] = m.split("|", 1)[-1]
    elif m:
        member_sub[mid] = m.split("|", 1)[-1]
correct = wrong = missing = 0
for mid, nd in nodes.items():
    mem = nd["members"].get("v3")
    if not mem or mem not in truth_sub:
        continue
    t = truth_sub[mem]
    p, ok = nd["parent"], False
    while p is not None:
        if member_sub.get(p, "").endswith(t):
            ok = True
            break
        p = nodes[p]["parent"]
    if ok:
        correct += 1
    elif nd["parent"] is None:
        missing += 1
    else:
        wrong += 1
by_status = {}
for nd in nodes.values():
    by_status[nd["status"]] = by_status.get(nd["status"], 0) + 1
print("nodes:", by_status)
print(f"v3 clusters placed under their TRUE subclass meta-clade: "
      f"{correct} | wrong lineage: {wrong} | at root: {missing}")
print("conflicts:", len([c for c in harm["conflicts"]
                         if c.get("class") == "genuine_conflict"]),
      "| affiliates:", len(harm["affiliates"]),
      "| unplaced internals:",
      sum(len(v) for v in harm["unplaced_internals"].values()))
fig, _ = plot_reconciled_tree(harm, trees,
                              dataset_names={"v2": "10Xv2 subclasses (coarse)",
                                             "v3": "10Xv3 clusters (fine)"})
out = "examples/harmonize_demo"
save_pub(fig, os.path.join(out, "allen_coarse_vs_fine"), formats=("png",),
         dpi=150)
print("figure written")
