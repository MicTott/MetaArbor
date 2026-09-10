"""Walk-v2 prototype, revision 2 — A DETERMINISTIC TWO-STAGE
ESTIMATOR, gated on the Allen full map.

    Stage 1: the frozen Walk, run in BOTH directions (the committed
             reverse decisions are frozen-Walk output).
    Stage 2: each forward walk locally refined using Stage-1 reverse
             coverage — three questions at every split: global
             plausibility (frozen), local forward discrimination
             (parent-local re-ranks, SOURCE-TREE sibling negatives),
             reverse-child coverage (veto / permit / unresolved).
This is NOT a mutually-defined reciprocity system: Stage-2 decisions
never feed back into the reverse direction.

REVISION 2 — specification repairs after review of the first gate
run (whose PASS is VOID; its numbers are recorded at the end of this
docstring). Repairs only; the gate criteria and seeds are unchanged,
plus the reviewer-required contract additions:
 R-a CURATED-TRUTH LEAKAGE REMOVED: local-contrast negatives are now
     the SOURCE INPUT TREE's sibling context — leaves under the
     query's inferred-tree parent (climbing unary chains; root
     context = all leaves) — not the curated class map, which is
     scoring fixture only.
 R-b WINNER-RETURN ENFORCED: PERMIT now requires the single mapping
     child to BE the vote-winning child (maps == [best]); the first
     version could descend into A on reverse evidence about B.
     Adversarial regression test: tests/test_walkv2_guard.py.
 R-c INTERNAL-QUERY GENERALIZATION: a reverse landing INSIDE the
     query's descendant subtree counts as maps_to_Q (equality-only
     was leaf-safe but wrong in general); an ancestor landing stays
     uninformative.
 R-d FULL OUTPUT CONTRACT: the frozen final compactness gate
     (min_compact=0.7) is applied and scored for BOTH arms;
     unmatched/root outcomes are constrained (G6) and
     compactness-discordant outcomes constrained (G7); the
     compatible-placement DEPTH DEFICIT (hops between the selected
     node and the smallest node containing all of truth) is
     reported for both arms.

DECISION RULE at every multi-child split (frozen navigation and
vote-override unchanged; frozen stream for frozen tests, independent
stream for local tests):
  guard == VETO                         -> stop (provenance ledgered)
  frozen concentration + parent checks  -> descend (frozen path)
  local contrast (source-tree sibling negatives) supported
      AND guard == PERMIT (winner-return) -> descend (recovery path)
  otherwise                             -> stop (unresolved-breadth
                                           flagged when local was
                                           supported)

GATE (unchanged criteria G1-G5, plus the contract additions):
  G1 no query degrades from a frozen EXACT placement
  G2 EXACT count non-decreasing
  G3 inside-truth (partial over-descent) count non-increasing
  G4 off-lineage count non-increasing
  G5 every veto carries reverse-child provenance
  G6 unmatched/root outcomes non-increasing
  G7 compactness-discordant outcomes non-increasing
Accept whatever result follows; no further repairs without review.

VOIDED FIRST-RUN RECORD (leaky negatives + unenforced winner-return):
EXACT 12->14, inside-truth 4->1, off 1->1, "PASS" — invalid; the
Endo coarse->EXACT recovery in particular was driven by the leaky
local contrast and must re-earn itself under source-tree negatives.

Run: python examples/walkv2_prototype.py
"""
import csv
import gzip
import json
import os

import numpy as np
from scipy.stats import rankdata


def rank_rows(x):
    try:
        return rankdata(x, axis=1)
    except TypeError:
        return np.apply_along_axis(rankdata, 1, x)


# ---- pure, testable guard machinery (no module-level data) ----------------
def tree_ancestors(tree, x):
    out = []
    while tree["parent"].get(x) not in (None, "root"):
        x = tree["parent"][x]
        out.append(x)
    return out


def classify_reverse(sel, q, src_tree):
    """Classify one reverse landing relative to source query q."""
    if sel == q or q in tree_ancestors(src_tree, sel):
        return "maps_to_Q"                 # exact, or inside Q's subtree
    if sel == "root" or sel in tree_ancestors(src_tree, q):
        return "uninformative"             # coarse call covering Q
    return "maps_elsewhere"                # (root covers everything:
                                           #  never affirmative
                                           #  elsewhere-evidence)


def guard(split, q, best, tgt_tree, rev_dec, canon_map, src_tree,
          leaves_under_fn):
    """Reverse-child coverage decision at a target split.
    VETO: >= 2 children map to Q. PERMIT: the single mapping child IS
    the vote winner AND every sibling affirmatively maps elsewhere.
    Otherwise UNRESOLVED."""
    kids = tgt_tree["children"].get(split, [])
    maps, elsewhere = [], []
    for c in kids:
        r = rev_dec.get(canon_map.get(c, c))
        if not r or not r.get("matched") or r.get("selected") is None:
            continue
        cls = classify_reverse(r["selected"], q, src_tree)
        if cls == "maps_to_Q":
            maps.append(c)
        elif cls == "maps_elsewhere":
            elsewhere.append(c)
    if len(maps) >= 2:
        return "VETO", maps
    if maps == [best] and len(elsewhere) == len(kids) - 1:
        return "PERMIT", maps
    return "UNRESOLVED", maps


def source_sibling_leaves(q, src_tree, leaves_under_fn):
    """The source input tree's sibling context: leaves under Q's
    parent, climbing unary chains; root context -> all leaves."""
    p = src_tree["parent"].get(q)
    while p not in (None, "root"):
        lvs = set(leaves_under_fn(src_tree, p))
        if lvs != {q}:
            return lvs
        p = src_tree["parent"].get(p)
    return set(src_tree["leaves"])


if __name__ == "__main__":
    from scipy.io import mmread

    from metaarbor import measure
    from metaarbor.kernel import (auroc, lognorm, node_scores,
                                  rank_normalize)
    from metaarbor.rng import Minstd
    from metaarbor.tree import leaves_under
    from metaarbor.walk import _boot_delta, compactness, select_node

    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    DATA = os.path.join(REPO, "data", "wmb_plilaorb")
    MA = os.path.join(HERE, "harmonize_demo")
    FX = os.path.join(REPO, "pkgs", "fixtures")
    OUT = os.path.join(HERE, "walkv2")
    BASE_SEED = 7

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
    it = json.load(open(os.path.join(MA, "allen_input_trees.json")))
    tree, tv2 = it["v3"], it["v2"]
    dec_rev = json.load(open(os.path.join(
        MA, "allen_decisions.json")))["v3>v2"]
    canon = json.load(open(os.path.join(MA, "allen_canonical.json")))
    lv = list(csv.reader(gzip.open(
        os.path.join(FX, "tree_levels_b.csv.gz"), "rt")))
    truth = {}
    for r in lv[1:]:
        truth.setdefault(f"v2|{r[1]}", set()).add(f"v3|{r[3]}")
    leaf_cells = {l: np.flatnonzero(labB == l)
                  for l in cache["leaves"]}

    _local = {}

    def local_cache(split):
        if split in _local:
            return _local[split]
        lvs = [l for l in leaves_under(tree, split) if l in leaf_cells]
        sub = np.sort(np.concatenate([leaf_cells[l] for l in lvs]))
        lab_sub = labB[sub]
        Vl = np.zeros((tn.shape[0], len(lvs)))
        ind = np.column_stack([(lab_sub == l).astype(float)
                               for l in lvs])
        for s in range(0, tn.shape[0], 2000):
            co = tn[s:s + 2000] @ rn[sub].T
            Vl[s:s + 2000] = (rank_rows(co) / len(sub)) @ ind
        _local[split] = {"V": Vl, "leaves": lvs}
        return _local[split]

    def loc_scores(lc, node):
        idx = [lc["leaves"].index(l)
               for l in leaves_under(tree, node) if l in lc["leaves"]]
        return lc["V"][:, idx].sum(axis=1)

    def select_node_v2(query, seed):
        positive = labA == query
        rng_f, rng_l = Minstd(seed), Minstd(seed + 10007)
        ms = cache["V"][positive] / cache["leaf_sizes"]
        top_leaf = np.asarray(cache["leaves"])[ms.argmax(axis=1)]
        sibs = source_sibling_leaves(query, tv2, leaves_under)
        sib_mask = np.isin(labA, sorted(sibs)) | positive

        def votes_for(kids):
            return np.asarray([np.isin(
                top_leaf, leaves_under(tree, k)).sum()
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
            g, prov = guard(current, query, best, tree, dec_rev,
                            canon["v3"], tv2, leaves_under)
            if g == "VETO":
                ledger.append((current, "veto", ";".join(prov)))
                break
            if v[order[0]] >= 0.9:
                current = best
                continue
            d_f = _boot_delta(n_scores(best), n_scores(second),
                              positive, rng_f, 200)
            if np.quantile(d_f, 0.05) > 0.01:
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
                ledger.append((current, "local_descend",
                               ";".join(prov)))
                current = best
                continue
            ledger.append((current, "unresolved_breadth" if local_ok
                           else "not_concentrated", ""))
            break
        sel_auc = np.nan if current == "root" else \
            auroc(n_scores(current), positive)
        matched = np.isfinite(sel_auc) and sel_auc >= 0.6
        selected = current if matched else None
        comp = (compactness(cache, positive, tree, selected)
                if matched else np.nan)
        return {"selected": selected, "auroc": float(sel_auc),
                "at_root": current == "root",
                "compactness": float(comp) if comp == comp else None,
                "ledger": ledger}

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

    def depth_of(node):
        d, x = 0, node
        while tree["parent"].get(x) not in (None, "root"):
            d += 1
            x = tree["parent"][x]
        return d

    def smallest_containing(T):
        cur = "root"
        while True:
            nxt = [c for c in tree["children"].get(cur, [])
                   if set(leaves_under(tree, c)) >= T]
            if not nxt:
                return cur
            cur = nxt[0]

    def deficit(node, T):
        if node is None:
            return None
        if not set(leaves_under(tree, node)) >= T:
            return None                    # not a compatible placement
        return depth_of(smallest_containing(T)) - depth_of(node)

    queries = sorted(set(labA))
    rows = []
    for qi, q in enumerate(queries):
        fr = select_node(cache, labA, q, tree, seed=BASE_SEED + qi)
        fcomp = (compactness(cache, labA == q, tree, fr["selected"])
                 if fr["matched"] else np.nan)
        v2r = select_node_v2(q, BASE_SEED + qi)
        T = truth.get(q, set())
        rows.append({
            "query": q,
            "frozen_sel": fr["selected"],
            "frozen_rel": truth_rel(fr["selected"], T),
            "frozen_discordant": bool(fr["matched"] and
                                      fcomp == fcomp and fcomp < 0.7),
            "frozen_deficit": deficit(fr["selected"], T),
            "v2_sel": v2r["selected"],
            "v2_rel": truth_rel(v2r["selected"], T),
            "v2_discordant": bool(v2r["selected"] is not None and
                                  v2r["compactness"] is not None and
                                  v2r["compactness"] < 0.7),
            "v2_deficit": deficit(v2r["selected"], T),
            "v2_ledger": " | ".join(
                f"{a}:{b}{(':' + c) if c else ''}"
                for a, b, c in v2r["ledger"])})
        print(f"{q[:30]:30s} frozen={rows[-1]['frozen_rel']:13s} "
              f"v2={rows[-1]['v2_rel']:13s} "
              f"{rows[-1]['v2_ledger'][:58]}")
    with open(os.path.join(OUT, "prototype_allen_map.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def count(col, val):
        return sum(1 for r in rows if r[col] == val)

    print("\nplacement counts (frozen -> walk-v2):")
    for rel in ("EXACT", "coarse", "inside-truth", "off",
                "unmatched"):
        print(f"  {rel:13s} {count('frozen_rel', rel):2d} -> "
              f"{count('v2_rel', rel):2d}")
    fd = [r["frozen_deficit"] for r in rows
          if r["frozen_deficit"] is not None]
    vd = [r["v2_deficit"] for r in rows if r["v2_deficit"] is not None]
    print(f"compatible-placement depth deficit: frozen sum="
          f"{sum(fd)} (n={len(fd)}) -> v2 sum={sum(vd)} (n={len(vd)})")
    ndis_f = sum(1 for r in rows if r["frozen_discordant"])
    ndis_v = sum(1 for r in rows if r["v2_discordant"])
    print(f"compactness-discordant: frozen {ndis_f} -> v2 {ndis_v}")

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
    g6 = count("v2_rel", "unmatched") <= count("frozen_rel",
                                               "unmatched")
    g7 = ndis_v <= ndis_f
    for name, ok in (("G1 no EXACT degraded", g1),
                     ("G2 EXACT non-decreasing", g2),
                     ("G3 partial over-descents non-increasing", g3),
                     ("G4 off-lineage non-increasing", g4),
                     ("G5 veto provenance", g5),
                     ("G6 unmatched/root non-increasing", g6),
                     ("G7 discordant non-increasing", g7)):
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    verdict = g1 and g2 and g3 and g4 and g5 and g6 and g7
    msg = ("PASS — Walk-v2 rev2 accepted on Allen" if verdict else
           "FAIL — prototype rejected; frozen Walk stands")
    print(f"\nGATE: {msg}")
