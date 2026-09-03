"""Allen coarse-vs-fine harmonize with STRICT truth holdout (reviewer
gate, step 5 complete).

Construction consumes expression + each atlas's OWN label column only:
10Xv2 cells carry their subclass annotation (that IS the coarse atlas's
labeling); 10Xv3 cells carry their cluster annotation. BOTH input trees
are inferred from expression alone (metaarbor.infer_tree). The curated
Allen class labels and the curated cluster->subclass nesting never enter
construction — they are used exclusively in the scoring block below.

Outputs (examples/harmonize_demo/):
  allen_coarse_vs_fine.png   complete reconciled tree (all consolidated
                             subtrees expanded) beside both input trees
  allen_confusion.csv        predicted-parent x curated-subclass matrix
  allen_misplacements.csv    full-chain audit of every non-exact cluster
"""
import csv
import os

import numpy as np
from scipy.io import mmread

from metaarbor.infer_tree import infer_tree
from metaarbor.tree import leaves_under
from metaarbor.consensus.harmonize import harmonize
from metaarbor.consensus.plot_reconciled import plot_reconciled_tree
from metaarbor.style import save_pub

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
if not os.path.isdir(D):
    D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "harmonize_demo")
os.makedirs(OUT, exist_ok=True)


def load(tag):
    counts = np.asarray(mmread(os.path.join(D, f"counts_{tag}.mtx"))
                        .todense()).T
    lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
    with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells


genes = open(os.path.join(D, "genes.txt")).read().split()
cA, lA, cellsA = load("10Xv2")
cB, lB, cellsB = load("10Xv3")
labA = np.asarray([f"v2|{c['subclass']}" for c in cellsA])
labB = np.asarray([f"v3|{c['cluster']}" for c in cellsB])

# ---- HOLDOUT: input trees inferred from expression + own labels only ------
print("inferring input trees from expression (curated class labels and "
      "cluster->subclass nesting held out of construction)")
inf_a = infer_tree(cA, labA, lib=lA, n_hvg=2000, n_boot=50, seed=0)
inf_b = infer_tree(cB, labB, lib=lB, n_hvg=2000, n_boot=50, seed=0)
trees = {"v2": inf_a["tree"], "v3": inf_b["tree"]}
print(f"  v2 tree: {inf_a['provenance']['n_internal_kept']} supported "
      f"internals over {inf_a['provenance']['n_labels']} subclass leaves")
print(f"  v3 tree: {inf_b['provenance']['n_internal_kept']} supported "
      f"internals over {inf_b['provenance']['n_labels']} cluster leaves")

datasets = {
    "v2": {"counts": cA, "labels": labA, "gene_names": genes, "lib": lA},
    "v3": {"counts": cB, "labels": labB, "gene_names": genes, "lib": lB},
}
harm = harmonize(datasets, trees, n_hvg=1000, n_boot=200)
nodes = harm["tree"]
by_status = {}
for nd in nodes.values():
    by_status[nd["status"]] = by_status.get(nd["status"], 0) + 1
n_expanded = sum(1 for nd in nodes.values() if nd.get("expanded"))
print("nodes:", by_status, f"(of which expanded-subtree: {n_expanded})",
      "| affiliates:", len(harm["affiliates"]),
      "| genuine conflicts:",
      len([c for c in harm["conflicts"]
           if c.get("class") == "genuine_conflict"]),
      "| unplaced internals:",
      sum(len(v) for v in harm["unplaced_internals"].values()))

# ---- scoring over ALL clusters (curated truth enters HERE only) -----------
truth_sub = {f"v3|{c['cluster']}": c["subclass"] for c in cellsB}
all_clusters = sorted(truth_sub)

# locate every cluster in the assembled tree: as a node member (direct or
# expanded) or as a marked affiliate alias
AFF = "≈ "
locate = {}
for mid, nd in nodes.items():
    m = nd["members"].get("v3")
    if m in truth_sub and m not in locate:
        locate[m] = ("node", mid)
for mid, nd in nodes.items():
    for a in nd["aliases"]:
        if isinstance(a, str) and a.startswith(AFF):
            base = a[len(AFF):]
            if base in truth_sub and base not in locate:
                locate[base] = ("affiliate", mid)


def first_v2_ancestor(mid):
    """Climb to the first ancestor (inclusive) carrying a v2 member;
    return (node_id, set of v2 subclasses under that member, display)."""
    p = mid
    while p is not None:
        nd = nodes[p]
        if "v2" in nd["members"]:
            m = nd["members"]["v2"]
            if m in trees["v2"]["leaves"]:
                subs = {m.split("|", 1)[-1]}
            else:
                subs = {l.split("|", 1)[-1]
                        for l in leaves_under(trees["v2"], m)}
            return p, subs, nd["display"]
        p = nd["parent"]
    return None, set(), ""


rows_audit, confusion = [], {}
n_exact = n_consistent = n_wrong = n_root = n_missing = 0
sel_v3 = harm["decisions"]["selections"].get(("v3", "v2"), {})
cands_all = harm["candidates"]["candidates"]
for cl in all_clusters:
    t = truth_sub[cl]
    how, mid = locate.get(cl, (None, None))
    if mid is None:
        n_missing += 1
        verdict, pred_disp = "missing", "NONE"
    else:
        anc, subs, disp = first_v2_ancestor(mid)
        if anc is None:
            n_root += 1
            verdict, pred_disp = "root", "ROOT"
        elif subs == {t}:
            n_exact += 1
            verdict, pred_disp = "exact", disp
        elif t in subs:
            n_consistent += 1
            verdict = "consistent_coarse"
            pred_disp = disp + f" [{len(subs)} subclasses]"
        else:
            n_wrong += 1
            verdict, pred_disp = "wrong_lineage", disp
    confusion[(pred_disp, t)] = confusion.get((pred_disp, t), 0) + 1
    if verdict in ("exact",):
        continue
    # ---- full-chain audit: inferred input tree -> pairwise Walk
    # decision -> candidate group -> assembly ----
    clade_id, purity, clade_n = "", "", ""
    best = None
    for cid, lv in inf_b["clades"].items():
        if cl in lv and (best is None or len(lv) < len(inf_b["clades"][best])):
            best = cid
    if best is not None:
        lv = inf_b["clades"][best]
        subs_in = [truth_sub[x] for x in lv if x in truth_sub]
        clade_id = best
        clade_n = len(lv)
        purity = round(max(subs_in.count(x) for x in set(subs_in)) /
                       len(subs_in), 2)
    sr = sel_v3.get(cl, {})
    cand = next((c for c in cands_all if c["members"].get("v3") == cl or
                 cl in (c.get("provenance", {}).get("subtree_nodes")
                        or [])), None)
    rows_audit.append({
        "cluster": cl.split("|", 1)[-1],
        "curated_subclass": t,
        "verdict": verdict,
        "located_as": how or "absent",
        "predicted_parent": pred_disp,
        "inferred_v3_clade": clade_id,
        "clade_size": clade_n,
        "clade_subclass_purity": purity,
        "walk_selected_in_v2": (sr.get("selected") or ""),
        "walk_support": (round(sr["support"], 3)
                         if sr.get("matched") and
                         np.isfinite(sr.get("support", float("nan")))
                         else ""),
        "candidate_kind": cand["kind"] if cand else "",
        "assembly_parent": ((nodes[mid]["parent"] or "ROOT")
                           if mid else ""),
    })

n_all = len(all_clusters)
print(f"ALL-cluster placement (n = {n_all}): exact {n_exact} | "
      f"consistent-coarse {n_consistent} | wrong-lineage {n_wrong} | "
      f"root {n_root} | missing {n_missing}")
print("clusters located via affiliate alias:",
      sum(1 for v in locate.values() if v[0] == "affiliate"),
      "(marked with '≈' in aliases; excluded from reciprocal "
      "support by construction)")

with open(os.path.join(OUT, "allen_misplacements.csv"), "w",
          newline="") as fh:
    if rows_audit:
        w = csv.DictWriter(fh, fieldnames=list(rows_audit[0]))
        w.writeheader()
        w.writerows(rows_audit)
subs_all = sorted({t for _, t in confusion})
preds_all = sorted({p for p, _ in confusion})
with open(os.path.join(OUT, "allen_confusion.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["predicted_parent\\curated_subclass"] + subs_all)
    for p in preds_all:
        w.writerow([p] + [confusion.get((p, t), 0) for t in subs_all])

fig, _ = plot_reconciled_tree(
    harm, trees,
    dataset_names={"v2": "10Xv2 subclasses (coarse, tree inferred)",
                   "v3": "10Xv3 clusters (fine, tree inferred)"})
save_pub(fig, os.path.join(OUT, "allen_coarse_vs_fine"), formats=("png",),
         dpi=150)
print("wrote figure, allen_confusion.csv, allen_misplacements.csv ->", OUT)
print("HOLDOUT CONFIRMED: construction consumed expression and each "
      "atlas's own label column only; curated class labels and the "
      "curated cluster->subclass nesting appear exclusively in the "
      "scoring block of this script.")
