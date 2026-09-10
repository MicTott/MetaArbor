"""Walk separability diagnostic, v2 (CORRECTED — supersedes the v1
committed at 52bfc8e, whose two flaws a reviewer identified and this
header records: (1) v1's "local preference" compared UNNORMALIZED
column sums from the globally-ranked cache — not parent-local
re-ranking and not the frozen Walk's leaf-size-normalized votes; its
bimodal distribution (19/64 exactly 0, e.g. one-cluster Car3 "losing"
to many-cluster siblings) was largely a size artifact of the
diagnostic itself; (2) v1 followed curated paths and never replayed
actual frozen Walk decisions, so it could not say whether saturation
is binding; its Q2 used unnormalized score differences, cell-level
(not donor-level) splits, pooled HVGs, and unsigned-only AUROC.)

PART 1 — WALK CEILING AUDIT. Replays the actual frozen Walk (v2
subclass queries down the INFERRED v3 tree, harmonize namespaces,
frozen defaults, per-query seeds) with trace=True, then at every
visited split that ran the bootstrap test (overrides excluded)
re-evaluates the SAME best-vs-second contrast under the 2x2 design:

    ranking:   global cache ranks   | parent-local re-ranks
               (frozen)             | (rankdata over ONLY training
                                    |  cells under the split node)
    negatives: all background       | source siblings only
               (frozen)             | (cells of other v2 subclasses
                                    |  in the query's curated class)

Each variant applies the FROZEN decision rule (paired bootstrap of
summed node scores, 5th percentile > 0.01 margin, n_boot=200, fresh
per-variant Minstd stream seeded base+qi — same procedure, not the
same draws; the (global, all) variant's agreement with the frozen
in-walk decision is reported as the RNG-sensitivity calibration).
Report: of the actual frozen STOPS (not-concentrated), how many
become supported descents under each variant — the direct answer to
whether one-vs-all saturation is binding. Actual descends are also
re-checked (variants could un-support them).

PART 2 — SIBLING DISTINCTNESS AUDIT (corrected). Within-v3,
DONOR-held-out split (train donors -> reference, test donors ->
query cells), HVGs selected on training donors only, per-leaf-size
NORMALIZED scores (V/leaf_sizes, the frozen vote scale), SIGNED
head-to-head AUROC of (score_a - score_b) with a as positive
(orientation preserved; unsigned max(A,1-A) reported separately),
for sibling pairs at every level: cluster pairs tiered by curated
relationship AND internal sibling clades (supertype pairs within a
subclass, subclass pairs within a class; node score = column sums /
summed leaf sizes).

Diagnostic only: no thresholds derived, frozen Walk untouched.
Run: python examples/walk_separability.py
"""
import csv
import gzip
import os

import numpy as np
from scipy.io import mmread
from scipy.stats import rankdata

from metaarbor import measure
from metaarbor.kernel import (auroc, lognorm, node_scores,
                              rank_normalize, variable_genes,
                              vote_cache)
from metaarbor.rng import Minstd
from metaarbor.tree import leaves_under
from metaarbor.walk import _boot_delta, select_node

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "wmb_plilaorb")
MA = os.path.join(HERE, "harmonize_demo")
FX = os.path.join(REPO, "pkgs", "fixtures")
OUT = os.path.join(HERE, "walk_separability")
os.makedirs(OUT, exist_ok=True)
SEED = 0
BASE_SEED = 7
N_PAIR_CAP = 200


def rank_rows(x):
    try:
        return rankdata(x, axis=1)
    except TypeError:                      # older scipy
        return np.apply_along_axis(rankdata, 1, x)


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
cls_of_sub = {r[1]: r[0] for r in rows_lv}
sub_of = {r[3]: r[1] for r in rows_lv}
sup_of = {r[3]: r[2] for r in rows_lv}
cls_of = {r[3]: r[0] for r in rows_lv}

# =================== PART 1: Walk ceiling audit ============================
print("Part 1: measuring v2 -> v3 (frozen kernel, harmonize namespaces)")
import json  # noqa: E402
cA, lA, cellsA = load("10Xv2")
cB, lB, cellsB = load("10Xv3")
labA = np.asarray([f"v2|{c['subclass']}" for c in cellsA])
labB = np.asarray([f"v3|{c['cluster']}" for c in cellsB])
m = measure(cA, labA, cB, labB, genes, lib_a=lA, lib_b=lB)
cache = m["cache_a"]
tree_b = json.load(open(os.path.join(MA, "allen_input_trees.json")))["v3"]

# norm matrices for parent-local re-ranking (same hvg as measure)
hvg_idx = [genes.index(g) for g in m["hvg"]]
tn = rank_normalize(lognorm(cA, lA)[:, hvg_idx])       # v2 = test
rn = rank_normalize(lognorm(cB, lB)[:, hvg_idx])       # v3 = train
leaf_cells = {l: np.flatnonzero(labB == l) for l in cache["leaves"]}

_local = {}


def local_cache(split_node):
    """Vote cache re-ranked within ONLY the training cells under
    split_node (the projection prototype's parent-context rule)."""
    if split_node in _local:
        return _local[split_node]
    lvs = [l for l in leaves_under(tree_b, split_node)
           if l in leaf_cells]
    sub = np.concatenate([leaf_cells[l] for l in lvs])
    order = np.argsort(sub)
    sub = sub[order]
    sub_labels = labB[sub]
    n_sub = len(sub)
    Vl = np.zeros((tn.shape[0], len(lvs)))
    ind = np.zeros((n_sub, len(lvs)))
    for j, l in enumerate(lvs):
        ind[:, j] = sub_labels == l
    for s in range(0, tn.shape[0], 2000):
        co = tn[s:s + 2000] @ rn[sub].T
        Vl[s:s + 2000] = (rank_rows(co) / n_sub) @ ind
    out = {"V": Vl, "leaves": lvs}
    _local[split_node] = out
    return out


def loc_scores(lc, node):
    idx = [lc["leaves"].index(l) for l in leaves_under(tree_b, node)
           if l in lc["leaves"]]
    return lc["V"][:, idx].sum(axis=1)


queries = sorted(set(labA))
cls_of_q = {q: cls_of_sub[q.split("|", 1)[1]] for q in queries}
audit = []
for qi, q in enumerate(queries):
    sel = select_node(cache, labA, q, tree_b, seed=BASE_SEED + qi,
                      trace=True)
    positive = labA == q
    sib_mask = np.asarray([cls_of_q[l] == cls_of_q[q] for l in labA])
    for step in sel["path"]:
        if step["override"] or np.isnan(step["sib_lo"]):
            continue
        split = step["id"] if step["stopped"] else \
            tree_b["parent"][step["id"]]
        best, second = step["best"], step["second"]
        frozen_desc = (not step["stopped"]) or \
            (step["stopped"] and step["par_lo"] > 0
             if not np.isnan(step["par_lo"]) else False)
        actual = ("descend" if not step["stopped"] else
                  ("stop_parent_better"
                   if (not np.isnan(step["par_lo"]) and
                       step["par_lo"] > 0) else
                   "stop_not_concentrated"))
        lc = local_cache(split)
        gb = node_scores(cache, leaves_under(tree_b, best))
        gs = node_scores(cache, leaves_under(tree_b, second))
        lb_ = loc_scores(lc, best)
        ls_ = loc_scores(lc, second)
        row = {"query": q, "split": split, "best": best,
               "second": second, "actual": actual,
               "frozen_sib_lo": round(step["sib_lo"], 4)}
        for rank_name, sb, ss in (("global", gb, gs),
                                  ("local", lb_, ls_)):
            for neg_name, mask in (("all", np.ones(len(labA), bool)),
                                   ("sibs", sib_mask | positive)):
                d = _boot_delta(sb[mask], ss[mask], positive[mask],
                                Minstd(BASE_SEED + qi), 200)
                lo = float(np.quantile(d, 0.05))
                row[f"{rank_name}_{neg_name}_lo"] = round(lo, 4)
                row[f"{rank_name}_{neg_name}_desc"] = lo > 0.01
        audit.append(row)
with open(os.path.join(OUT, "q1_walk_audit.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(audit[0]))
    w.writeheader()
    w.writerows(audit)

stops = [r for r in audit if r["actual"] == "stop_not_concentrated"]
descs = [r for r in audit if r["actual"] == "descend"]
agree = np.mean([r["global_all_desc"] == (r["actual"] == "descend")
                 for r in audit if r["actual"] != "stop_parent_better"])
print(f"\nWalk audit: {len(audit)} bootstrap splits "
      f"({len(stops)} stops-not-concentrated, {len(descs)} descends); "
      f"(global,all) vs frozen agreement {agree:.2%} "
      f"(RNG-sensitivity calibration)")
for var in ("global_all", "global_sibs", "local_all", "local_sibs"):
    conv = sum(r[f"{var}_desc"] for r in stops)
    lost = sum(not r[f"{var}_desc"] for r in descs)
    print(f"  {var:12s}: stops->supported descent {conv}/{len(stops)}"
          f" | descends un-supported {lost}/{len(descs)}")

# =================== PART 2: sibling distinctness ==========================
print("\nPart 2: donor-held-out within-v3 sibling audit")
donB = np.asarray([c["donor_label"] for c in cellsB])
clusB = np.asarray([c["cluster"] for c in cellsB])
rs = np.random.RandomState(SEED)
donors = sorted(set(donB))
rs.shuffle(donors)
train_don = set(donors[: len(donors) // 2])
tr = np.asarray([d in train_don for d in donB])
te = ~tr
print(f"  donors: {len(train_don)} train / "
      f"{len(donors) - len(train_don)} test; cells {tr.sum()}/{te.sum()}")
eb = lognorm(cB, lB)
hvg2 = variable_genes(eb[tr], eb[tr], genes, 1000)   # TRAIN-only HVGs
nt2 = rank_normalize(eb[np.ix_(te, hvg2)])
nr2 = rank_normalize(eb[np.ix_(tr, hvg2)])
cache2 = vote_cache(nt2, nr2, clusB[tr])
Vn = cache2["V"] / cache2["leaf_sizes"]              # frozen vote scale
col2 = {l: i for i, l in enumerate(cache2["leaves"])}
labT = clusB[te]


def tier(a, b):
    if sup_of[a] == sup_of[b]:
        return "same_supertype"
    if sub_of[a] == sub_of[b]:
        return "same_subclass"
    if cls_of[a] == cls_of[b]:
        return "same_class"
    return "cross_class"


def h2h(score_a, score_b, mask_a, mask_b):
    both = mask_a | mask_b
    signed = auroc((score_a - score_b)[both], mask_a[both])
    return float(signed), float(max(signed, 1 - signed))


clusters = sorted(set(clusB) & set(cache2["leaves"]))
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
        ma, mb = labT == a, labT == b
        if ma.sum() < 5 or mb.sum() < 5:
            continue
        s, u = h2h(Vn[:, col2[a]], Vn[:, col2[b]], ma, mb)
        q2.append({"level": "cluster", "tier": tname, "a": a, "b": b,
                   "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                   "signed": round(s, 4), "unsigned": round(u, 4)})
# internal sibling clades: supertype pairs within subclass, subclass
# pairs within class (node scores = col sums / summed leaf sizes)
members = {}
for r in rows_lv:
    members.setdefault(("supertype", r[2]), set()).add(r[3])
    members.setdefault(("subclass", r[1]), set()).add(r[3])
parent_of_node = {("supertype", r[2]): r[1] for r in rows_lv}
parent_of_node.update({("subclass", r[1]): r[0] for r in rows_lv})


def node_score_cells(node):
    cols = [col2[c] for c in members[node] if c in col2]
    if not cols:
        return None, None
    sz = cache2["leaf_sizes"][cols].sum()
    return cache2["V"][:, cols].sum(axis=1) / sz, \
        np.isin(labT, sorted(members[node]))


for level in ("supertype", "subclass"):
    nodes = sorted(n for n in members if n[0] == level)
    by_par = {}
    for n in nodes:
        by_par.setdefault(parent_of_node[n], []).append(n)
    for par, sibs in sorted(by_par.items()):
        for i, a in enumerate(sibs):
            for b in sibs[i + 1:]:
                sa, ma = node_score_cells(a)
                sb_, mb = node_score_cells(b)
                if sa is None or sb_ is None or \
                        ma.sum() < 5 or mb.sum() < 5:
                    continue
                s, u = h2h(sa, sb_, ma, mb)
                q2.append({"level": f"{level}_sibling", "tier": par,
                           "a": a[1], "b": b[1],
                           "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                           "signed": round(s, 4),
                           "unsigned": round(u, 4)})
with open(os.path.join(OUT, "q2_sibling_separability.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(q2[0]))
    w.writeheader()
    w.writerows(q2)
for grp in ("same_supertype", "same_subclass", "same_class",
            "cross_class"):
    sel = [r for r in q2 if r["level"] == "cluster" and
           r["tier"] == grp]
    if not sel:
        continue
    sg = [r["signed"] for r in sel]
    un = [r["unsigned"] for r in sel]
    print(f"  cluster {grp:15s} n={len(sel):3d} signed med="
          f"{np.median(sg):.3f} (reversed<0.5: "
          f"{sum(1 for x in sg if x < 0.5)}) unsigned med="
          f"{np.median(un):.3f} (<0.6: "
          f"{sum(1 for x in un if x < 0.6)})")
for level in ("supertype_sibling", "subclass_sibling"):
    sel = [r for r in q2 if r["level"] == level]
    if sel:
        un = [r["unsigned"] for r in sel]
        sg = [r["signed"] for r in sel]
        print(f"  {level:22s} n={len(sel):3d} signed med="
              f"{np.median(sg):.3f} unsigned med={np.median(un):.3f} "
              f"(<0.6: {sum(1 for x in un if x < 0.6)})")
print("wrote", OUT)
