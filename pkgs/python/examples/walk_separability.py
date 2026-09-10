"""Separability diagnostic for the Walk's evidence scale — two
questions, one script, NO method changes and NO thresholds derived:

Q1 (ceiling compression, cross-atlas): the Walk compares siblings via
1-vs-all node AUROCs computed over the whole test set. Deep in a
family both children sit near ceiling, so the sibling CONTRAST is a
difference of two numbers near 1. Diagnostic: for every split on each
v2 subclass's TRUE path down the curated v3 tree, record both
children's global node AUROCs, their gap, and the LOCAL preference
(the fraction of the query's own cells whose per-cell mean vote
prefers the correct child — the projection prototype's local-contrast
scale). If global gaps shrink toward bootstrap noise with depth while
local preference stays decisive, the compression is binding and a
local-contrast Walk (one-vs-best in the MetaNeighbor sense) is
warranted as a v2 candidate.

Q2 (resolution floor, within-atlas): are sibling leaves distinct cell
types at all? Head-to-head separability: split the v3 cells in half
(seeded), score the test half against the train half with the frozen
rank-vote kernel, and for sibling cluster pairs compute (a) each
cluster's 1-vs-all AUROC (the current scale) and (b) the head-to-head
AUROC using only the two clusters' cells and the score difference
V[:,a] - V[:,b] (the one-vs-best analogue). Pairs are tiered by their
curated relationship: same supertype (deepest), same subclass, same
class, cross-class. If 1-vs-all saturates in every tier while
head-to-head decays toward 0.5 in the deepest tier, then (i) the
current scale cannot make the distinctness call and (ii) head-to-head
AUROC is the natural statistic for a future collapse rule (threshold
to be frozen on synthetics, NOT here — this script only measures).

Run: python examples/walk_separability.py
"""
import csv
import gzip
import os

import numpy as np
from scipy.io import mmread

from metaarbor.kernel import (auroc, lognorm, rank_normalize,
                              variable_genes, vote_cache)
from metaarbor import measure

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "wmb_plilaorb")
FX = os.path.join(REPO, "pkgs", "fixtures")
OUT = os.path.join(HERE, "walk_separability")
os.makedirs(OUT, exist_ok=True)
SEED = 0
N_PAIR_CAP = 200            # per tier, seeded subsample


def load(tag):
    counts = np.asarray(mmread(os.path.join(
        DATA, f"counts_{tag}.mtx")).todense()).T
    lib = np.loadtxt(os.path.join(DATA, f"lib_{tag}.txt"))
    with open(os.path.join(DATA, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells


genes = open(os.path.join(DATA, "genes.txt")).read().split()
lv = list(csv.reader(gzip.open(os.path.join(FX, "tree_levels_b.csv.gz"),
                               "rt")))
rows_lv = lv[1:]
cls_of = {r[3]: r[0] for r in rows_lv}
sub_of = {r[3]: r[1] for r in rows_lv}
sup_of = {r[3]: r[2] for r in rows_lv}
sub_cls = {r[1]: r[0] for r in rows_lv}

# ===================== Q1: cross-atlas compression =========================
print("Q1: measuring v2 -> v3 (frozen kernel)...")
cA, lA, cellsA = load("10Xv2")
cB, lB, cellsB = load("10Xv3")
labA = np.asarray([c["subclass"] for c in cellsA])
labB = np.asarray([c["cluster"] for c in cellsB])
m = measure(cA, labA, cB, labB, genes, lib_a=lA, lib_b=lB)
cache = m["cache_a"]                     # v2 cells scored vs v3 clusters
V, leaves = cache["V"], cache["leaves"]
col = {l: i for i, l in enumerate(leaves)}

# curated v3 tree paths: class -> subclass -> supertype -> cluster
kids = {}
for r in rows_lv:
    kids.setdefault(("root",), set()).add(("class", r[0]))
    kids.setdefault(("class", r[0]), set()).add(("subclass", r[1]))
    kids.setdefault(("subclass", r[1]), set()).add(("supertype", r[2]))
    kids.setdefault(("supertype", r[2]), set()).add(("cluster", r[3]))
under = {}


def leaves_under(node):
    if node in under:
        return under[node]
    if node[0] == "cluster":
        out = [node[1]]
    else:
        out = [l for c in kids.get(node, ()) for l in leaves_under(c)]
    under[node] = out
    return out


def node_cols(node):
    return [col[l] for l in leaves_under(node) if l in col]


DEPTH = {"class": 1, "subclass": 2, "supertype": 3}
q1 = []
for q in sorted(set(labA)):
    if q not in sub_cls:
        continue
    qcells = labA == q
    path = [("root",), ("class", sub_cls[q]), ("subclass", q)]
    for pi in range(len(path) - 1):
        parent, correct = path[pi], path[pi + 1]
        sibs = sorted(kids.get(parent, ()) - {correct})
        cc = node_cols(correct)
        if not cc:
            continue
        s_correct = V[:, cc].mean(axis=1)
        auc_c = auroc(s_correct, qcells)
        for sib in sibs:
            sc = node_cols(sib)
            if not sc:
                continue
            s_sib = V[:, sc].mean(axis=1)
            auc_s = auroc(s_sib, qcells)
            local = float((s_correct[qcells] >
                           s_sib[qcells]).mean())
            q1.append({"query": q, "parent": parent[1] if len(parent) > 1
                       else "root",
                       "depth": DEPTH.get(correct[0], 0),
                       "sibling": sib[1],
                       "auc_correct": round(float(auc_c), 4),
                       "auc_sibling": round(float(auc_s), 4),
                       "global_gap": round(float(auc_c - auc_s), 4),
                       "local_pref": round(local, 4)})
    # deepest level: within own subclass, supertype splits (query's
    # cells vs the sibling supertypes of its own true supertypes) —
    # for a v2 subclass query every supertype under it is "correct";
    # the compression question is between them, answered in Q2.
with open(os.path.join(OUT, "q1_path_contrasts.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(q1[0]))
    w.writeheader()
    w.writerows(q1)
for d in (1, 2):
    sel = [r for r in q1 if r["depth"] == d]
    ac = [r["auc_correct"] for r in sel]
    gp = [abs(r["global_gap"]) for r in sel]
    lp = [r["local_pref"] for r in sel]
    print(f"  depth {d}: n={len(sel)} auc_correct med="
          f"{np.median(ac):.3f} |gap| med={np.median(gp):.3f} "
          f"local_pref med={np.median(lp):.3f}")

# ===================== Q2: within-atlas resolution floor ===================
print("Q2: within-v3 half-split head-to-head separability...")
rs = np.random.RandomState(SEED)
n = cB.shape[0]
perm = rs.permutation(n)
te, tr = perm[: n // 2], perm[n // 2:]
eb = lognorm(cB, lB)
hvg = variable_genes(eb[te], eb[tr], genes, 1000)
nt = rank_normalize(eb[np.ix_(te, hvg)])
nr = rank_normalize(eb[np.ix_(tr, hvg)])
cache2 = vote_cache(nt, nr, labB[tr])
V2, leaves2 = cache2["V"], cache2["leaves"]
col2 = {l: i for i, l in enumerate(leaves2)}
labT = labB[te]

clusters = sorted(set(labB) & set(leaves2))


def tier(a, b):
    if sup_of[a] == sup_of[b]:
        return "same_supertype"
    if sub_of[a] == sub_of[b]:
        return "same_subclass"
    if cls_of[a] == cls_of[b]:
        return "same_class"
    return "cross_class"


pairs = {}
for i, a in enumerate(clusters):
    for b in clusters[i + 1:]:
        pairs.setdefault(tier(a, b), []).append((a, b))
q2 = []
for tname in ("same_supertype", "same_subclass", "same_class",
              "cross_class"):
    ps = pairs.get(tname, [])
    if len(ps) > N_PAIR_CAP:
        idx = rs.choice(len(ps), N_PAIR_CAP, replace=False)
        ps = [ps[i] for i in sorted(idx)]
    for a, b in ps:
        ca_, cb_ = labT == a, labT == b
        if ca_.sum() < 5 or cb_.sum() < 5:
            continue
        one_vs_all_a = auroc(V2[:, col2[a]], ca_)
        one_vs_all_b = auroc(V2[:, col2[b]], cb_)
        both = ca_ | cb_
        h2h = auroc((V2[:, col2[a]] - V2[:, col2[b]])[both], ca_[both])
        q2.append({"tier": tname, "a": a, "b": b,
                   "n_a": int(ca_.sum()), "n_b": int(cb_.sum()),
                   "one_vs_all_a": round(float(one_vs_all_a), 4),
                   "one_vs_all_b": round(float(one_vs_all_b), 4),
                   "head_to_head": round(float(max(h2h, 1 - h2h)), 4)})
with open(os.path.join(OUT, "q2_sibling_separability.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(q2[0]))
    w.writeheader()
    w.writerows(q2)
for tname in ("same_supertype", "same_subclass", "same_class",
              "cross_class"):
    sel = [r for r in q2 if r["tier"] == tname]
    if not sel:
        continue
    ova = [max(r["one_vs_all_a"], r["one_vs_all_b"]) for r in sel]
    h2h = [r["head_to_head"] for r in sel]
    near_chance = sum(1 for x in h2h if x < 0.6)
    print(f"  {tname:15s} n={len(sel):3d} one-vs-all med="
          f"{np.median(ova):.3f} head-to-head med={np.median(h2h):.3f} "
          f"(<0.6: {near_chance})")
print("wrote", OUT)
