"""Walk-v2 prototype: the three-question architecture integrated —
global plausibility + local forward discrimination + reverse-child
coverage — and the Allen full-map gate against the frozen baseline.

The frozen Walk is UNCHANGED and remains the baseline; this is a
prototype in examples/, not a package method.

DECISION RULE (prespecified; fixed before the gate was run). At every
multi-child split, with best/second chosen by the FROZEN leaf-size-
normalized votes (navigation unchanged) and the frozen vote-override
retained as-is:
  1. guard = reverse-child coverage at this split (committed reverse
     calls, canonical; classification and rule exactly as frozen in
     walkv2_reverse_coverage.py: >= 2 children map to the query ->
     VETO with provenance; winner maps and every sibling affirmatively
     maps elsewhere -> PERMIT; anything else -> UNRESOLVED).
  2. If guard == VETO -> STOP here (applies to frozen-concentrated
     descents too: the decisive test showed such vetoes correct five
     frozen partial over-descents).
  3. elif the FROZEN concentration test passes (global scores, frozen
     stream, frozen margins) -> frozen parent-better check, then
     descend (the frozen path, now guard-protected).
  4. elif the LOCAL contrast passes (parent-local re-ranks, source-
     sibling negatives, frozen bootstrap rule on a separate stream)
     AND guard == PERMIT -> descend (the recovery path: local
     sensitivity is only ever exercised with affirmative reverse
     clearance).
  5. else STOP (unresolved-breadth flagged when local supported but
     guard was UNRESOLVED).
RNG: the frozen stream (Minstd(base+qi)) is consumed by the frozen
tests exactly as in select_node; local tests consume an independent
stream Minstd(base+qi+10007). Frozen defaults throughout; the final
min_auroc match gate is unchanged.

ALLEN FULL-MAP GATE (all 23 v2 subclass queries onto the inferred v3
tree; placements classified vs curated truth as EXACT / coarse /
inside-truth (= partial over-descent) / off):
  G1 no query degrades from a frozen EXACT placement;
  G2 Walk-v2 EXACT count >= frozen EXACT count;
  G3 Walk-v2 inside-truth count <= frozen (fewer partial
     over-descents);
  G4 Walk-v2 off count <= frozen (no new wrong lineage);
  G5 every veto carries explicit reverse-child provenance.
All five must hold or the prototype is rejected.

Run: python examples/walkv2_prototype.py
"""
import csv
import gzip
import json
import os

import numpy as np
from scipy.io import mmread
from scipy.stats import rankdata

from metaarbor import measure
from metaarbor.kernel import (auroc, lognorm, node_scores,
                              rank_normalize)
from metaarbor.rng import Minstd
from metaarbor.tree import leaves_under
from metaarbor.walk import _boot_delta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "wmb_plilaorb")
MA = os.path.join(HERE, "harmonize_demo")
FX = os.path.join(REPO, "pkgs", "fixtures")
OUT = os.path.join(HERE, "walkv2")
BASE_SEED = 7


def rank_rows(x):
    try:
        return rankdata(x, axis=1)
    except TypeError:
        return np.apply_along_axis(rankdata, 1, x)


def load(tag):
    counts = np.asarray(mmread(os.path.join(
        DATA, f"counts_{tag}.mtx")).todense()).T
    lib = np.loadtxt(os.path.join(DATA, f"lib_{tag}.txt"))
    with open(os.path.join(DATA, f"cells_{tag}.csv")) as fh:
        cells = list(csv.DictReader(fh))
    return counts, lib, cells


print("measuring v2 -> v3 (frozen kernel)...")
genes = open(os.path.join(DATA, "genes.txt")).read().split()
cA, lA, cellsA = load("10Xv2")
cB, lB, cellsB = load("10Xv3")
labA = np.asarray([f"v2|{c['subclass']}" for c in cellsA])
labB = np.asarray([f"v3|{c['cluster']}" for c in cellsB])
m = measure(cA, labA, cB, labB, genes, lib_a=lA, lib_b=lB)
cache = m["cache_a"]
hvg_idx = [genes.index(g) for g in m["hvg"]]
tn = rank_normalize(lognorm(cA, lA)[:, hvg_idx])
rn = rank_normalize(lognorm(cB, lB)[:, hvg_idx])
tree = json.load(open(os.path.join(MA, "allen_input_trees.json")))["v3"]
tv2 = json.load(open(os.path.join(MA, "allen_input_trees.json")))["v2"]
dec_rev = json.load(open(os.path.join(MA,
                                      "allen_decisions.json")))["v3>v2"]
canon = json.load(open(os.path.join(MA, "allen_canonical.json")))
lv = list(csv.reader(gzip.open(os.path.join(FX, "tree_levels_b.csv.gz"),
                               "rt")))
truth = {}
for r in lv[1:]:
    truth.setdefault(f"v2|{r[1]}", set()).add(f"v3|{r[3]}")
cls_of_sub = {f"v2|{r[1]}": r[0] for r in lv[1:]}
leaf_cells = {l: np.flatnonzero(labB == l) for l in cache["leaves"]}

_local = {}


def local_cache(split):
    if split in _local:
        return _local[split]
    lvs = [l for l in leaves_under(tree, split) if l in leaf_cells]
    sub = np.sort(np.concatenate([leaf_cells[l] for l in lvs]))
    lab_sub = labB[sub]
    Vl = np.zeros((tn.shape[0], len(lvs)))
    ind = np.column_stack([(lab_sub == l).astype(float) for l in lvs])
    for s in range(0, tn.shape[0], 2000):
        co = tn[s:s + 2000] @ rn[sub].T
        Vl[s:s + 2000] = (rank_rows(co) / len(sub)) @ ind
    _local[split] = {"V": Vl, "leaves": lvs}
    return _local[split]


def loc_scores(lc, node):
    idx = [lc["leaves"].index(l) for l in leaves_under(tree, node)
           if l in lc["leaves"]]
    return lc["V"][:, idx].sum(axis=1)


def v2_ancestors(x):
    out = []
    while tv2["parent"].get(x) not in (None, "root"):
        x = tv2["parent"][x]
        out.append(x)
    return out


def guard(split, q):
    kids = tree["children"].get(split, [])
    maps, elsewhere, uninf = [], [], []
    for c in kids:
        r = dec_rev.get(canon["v3"].get(c, c))
        if not r or not r.get("matched") or r.get("selected") is None:
            uninf.append(c)
        elif r["selected"] == q:
            maps.append(c)
        elif r["selected"] in v2_ancestors(q):
            uninf.append(c)
        else:
            elsewhere.append(c)
    if len(maps) >= 2:
        return "VETO", maps
    if len(maps) == 1 and len(elsewhere) == len(kids) - 1:
        return "PERMIT", maps
    return "UNRESOLVED", maps


def select_node_v2(query, seed):
    positive = labA == query
    rng_f = Minstd(seed)
    rng_l = Minstd(seed + 10007)
    ms = cache["V"][positive] / cache["leaf_sizes"]
    top_leaf = np.asarray(cache["leaves"])[ms.argmax(axis=1)]
    sib_mask = np.asarray([cls_of_sub.get(l) == cls_of_sub.get(query)
                           for l in labA]) | positive

    def votes_for(kids):
        return np.asarray([np.isin(top_leaf,
                                   leaves_under(tree, k)).sum()
                           for k in kids]) / len(top_leaf)

    def n_scores(node):
        return node_scores(cache, leaves_under(tree, node))

    current, ledger = "root", []
    while True:
        kids = tree["children"].get(current, [])
        if not kids:
            break
        if len(kids) == 1:
            current = kids[0]
            continue
        v = votes_for(kids)
        order = np.argsort(-v, kind="stable")
        best, second = kids[order[0]], kids[order[1]]
        override = v[order[0]] >= 0.9
        g, prov = guard(current, query)
        if g == "VETO":
            ledger.append((current, "veto", ";".join(prov)))
            break
        if override:
            current = best
            continue
        d_f = _boot_delta(n_scores(best), n_scores(second), positive,
                          rng_f, 200)
        frozen_ok = np.quantile(d_f, 0.05) > 0.01
        if frozen_ok:
            d_p = _boot_delta(n_scores(current), n_scores(best),
                              positive, rng_f, 200)
            if current != "root" and np.quantile(d_p, 0.05) > 0:
                ledger.append((current, "parent_better", ""))
                break
            current = best
            continue
        lc = local_cache(current)
        sb, ss = loc_scores(lc, best), loc_scores(lc, second)
        d_l = _boot_delta(sb[sib_mask], ss[sib_mask],
                          positive[sib_mask], rng_l, 200)
        local_ok = np.quantile(d_l, 0.05) > 0.01
        if local_ok and g == "PERMIT":
            ledger.append((current, "local_descend", ";".join(prov)))
            current = best
            continue
        ledger.append((current,
                       "unresolved_breadth" if local_ok else
                       "not_concentrated", ""))
        break
    sel_auc = np.nan if current == "root" else \
        auroc(n_scores(current), positive)
    matched = np.isfinite(sel_auc) and sel_auc >= 0.6
    return {"selected": current if matched else None,
            "auroc": float(sel_auc), "ledger": ledger}


def truth_rel(node, T):
    if node is None:
        return "unmatched"
    s = set(leaves_under(tree, node))
    if s == T:
        return "EXACT"
    if s > T:
        return "coarse"
    if s < T:
        return "inside-truth"
    return "off"


from metaarbor.walk import select_node  # noqa: E402

queries = sorted(set(labA))
rows = []
for qi, q in enumerate(queries):
    frozen = select_node(cache, labA, q, tree, seed=BASE_SEED + qi)
    v2sel = select_node_v2(q, BASE_SEED + qi)
    T = truth.get(q, set())
    rows.append({"query": q,
                 "frozen_sel": frozen["selected"],
                 "frozen_rel": truth_rel(frozen["selected"], T),
                 "v2_sel": v2sel["selected"],
                 "v2_rel": truth_rel(v2sel["selected"], T),
                 "v2_ledger": " | ".join(
                     f"{a}:{b}{(':' + c) if c else ''}"
                     for a, b, c in v2sel["ledger"])})
    print(f"{q[:30]:30s} frozen={rows[-1]['frozen_rel']:13s} "
          f"v2={rows[-1]['v2_rel']:13s} "
          f"{rows[-1]['v2_ledger'][:60]}")
with open(os.path.join(OUT, "prototype_allen_map.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def count(col, val):
    return sum(1 for r in rows if r[col] == val)


print("\nplacement counts (frozen -> walk-v2):")
for rel in ("EXACT", "coarse", "inside-truth", "off", "unmatched"):
    print(f"  {rel:13s} {count('frozen_rel', rel):2d} -> "
          f"{count('v2_rel', rel):2d}")
g1 = all(r["v2_rel"] == "EXACT" for r in rows
         if r["frozen_rel"] == "EXACT")
g2 = count("v2_rel", "EXACT") >= count("frozen_rel", "EXACT")
g3 = count("v2_rel", "inside-truth") <= count("frozen_rel",
                                              "inside-truth")
g4 = count("v2_rel", "off") <= count("frozen_rel", "off")
g5 = True
for r in rows:
    for entry in r["v2_ledger"].split(" | "):
        if ":veto" in entry and not entry.split(":veto", 1)[1]:
            g5 = False
print(f"\nG1 no EXACT degraded: {'PASS' if g1 else 'FAIL'}")
print(f"G2 EXACT count non-decreasing: {'PASS' if g2 else 'FAIL'}")
print(f"G3 partial over-descents non-increasing: "
      f"{'PASS' if g3 else 'FAIL'}")
print(f"G4 off-lineage non-increasing: {'PASS' if g4 else 'FAIL'}")
print(f"G5 vetoes carry provenance: {'PASS' if g5 else 'FAIL'}")
verdict = g1 and g2 and g3 and g4 and g5
msg = ("PASS — Walk-v2 prototype accepted on Allen; amygdala "
       "pairwise rerun is the next step" if verdict else
       "FAIL — prototype rejected; frozen Walk stands")
print(f"\nGATE: {msg}")
