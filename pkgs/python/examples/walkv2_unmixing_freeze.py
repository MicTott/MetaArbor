"""Walk-v2 breadth guard v4: simplex-constrained unmixing — the ONE
further attempt after the three recorded per-cell failures
(walkv2_breadth_freeze.py). If this fails its gate, the per-cell
breadth rule is abandoned and the frozen Walk is retained; no further
statistics will be iterated around these simulations.

WHY THIS IS A DIFFERENT ESTIMAND, not a tweak: the failed rules
counted noisy per-cell winners against a scalar null, but confusion
is CHILD-ASYMMETRIC (a pure population spills toward its particular
nearest sibling). Unmixing models the query population's mean
parent-local vote vector as a simplex mixture of PURE-CHILD
centroids:
    w_hat = argmin_{w >= 0, sum w = 1} || v_bar_Q - C w ||^2
Each child's centroid carries its own confusion signature, so a pure
child's spill toward a similar sibling is explained by its OWN
centroid rather than misread as composition.

PRESPECIFIED — fixed before any simulation output existed:

Construction (leakage-controlled): training ranks come from reference
donor 0 only; centroids are DONOR-HELD-OUT (reference donor 1's pure
child populations, scored by the identical parent-local machinery);
HVGs from the same measure() call as before (query vs reference —
no evaluation-cell labels are used). Feature = the K-vector of
per-child leaf-size-normalized mean local votes. Simplex constraint
enforced by augmented least squares (sum-to-one row, weight 1e3,
scipy nnls).

Decision statistic: LCB(w_(2)) — the 5th percentile, over B=200
seeded bootstrap resamples of the query's cells, of the SECOND-
largest mixture weight. RULE(tau): SPANS iff LCB(w_(2)) > tau.
Grid: tau in {0.02, 0.05, 0.10, 0.15, 0.20}.

Abstention safeguards (fixed constants, not tuned): return
UNRESOLVED — never a spans/contained call — when the relative
reconstruction residual ||v_bar - C w||/||v_bar|| > 0.5 or the
centroid matrix condition number > 1e3. Both are recorded per world
regardless, as are: per-donor point weights (stability), and a
donor-swap replicate (train and centroid donors exchanged) whose
second-weight difference is recorded as centroid-donor sensitivity.

Criteria (bsd=0.5 worlds = the measured Allen operating regime, as
calibrated and recorded in walkv2_breadth_freeze.py; bsd=1.0 is the
reported stress tier): over contained cases (minority 0, 0.05)
false-span rate <= 10%; over spanning cases (0.20, 0.30, 0.50)
spans rate >= 95%; the 0.10 transition zone is reported only.
Unresolved worlds count AGAINST whichever side they fall on (a
contained-world unresolved is not a false span, but a spanning-world
unresolved is a missed veto — vetoes must be affirmative).
Selection seeds {0,1,2}; validation seeds {7,8} consumed ONLY if a
rule passes selection.

Run: python examples/walkv2_unmixing_freeze.py
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walkv2_breadth_freeze import (BATCH, CONTAINED,  # noqa: E402
                                   MINORITY, SPANNING, rank_rows,
                                   world)
from metaarbor import measure  # noqa: E402
from metaarbor.kernel import lognorm, rank_normalize  # noqa: E402
from metaarbor.tree import leaves_under  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "walkv2")
GRID_TAU = (0.02, 0.05, 0.10, 0.15, 0.20)
SELECT_SEEDS, VALID_SEEDS = (0, 1, 2), (7, 8)
RESID_MAX, COND_MAX, RHO, B = 0.5, 1e3, 1e3, 200


def simplex_ls(C, v):
    """min ||v - C w||, w >= 0, sum w = 1 (augmented nnls)."""
    A = np.vstack([C, RHO * np.ones((1, C.shape[1]))])
    b = np.concatenate([v, [RHO]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else w


def unmix(seed, r_minor, bsd, imbalance, small):
    ref_X, ref_lab, ref_don, q_X, q_lab, q_don, tree, genes = world(
        seed, r_minor, bsd, imbalance, small)
    m = measure(q_X, q_lab, ref_X, ref_lab, genes)
    hvg = [genes.index(g) for g in m["hvg"]]
    tn = rank_normalize(lognorm(q_X)[:, hvg])
    rn = rank_normalize(lognorm(ref_X)[:, hvg])
    kids = tree["children"]["family:F1"]
    f1 = leaves_under(tree, "family:F1")

    def features(norm_rows, train_don):
        sub = np.flatnonzero((ref_don == train_don) &
                             np.isin(ref_lab, f1))
        lab_sub = ref_lab[sub]
        co = norm_rows @ rn[sub].T
        w = rank_rows(co) / len(sub)
        out = np.zeros((norm_rows.shape[0], len(kids)))
        for j, k in enumerate(kids):
            cols = np.isin(lab_sub, leaves_under(tree, k))
            out[:, j] = w[:, cols].sum(axis=1) / cols.sum()
        return out

    def run_side(train_don, cent_don):
        F_q = features(tn, train_don)
        cent_cells = (ref_don == cent_don) & np.isin(ref_lab, f1)
        F_c = features(rn[cent_cells], train_don)
        lab_c = ref_lab[cent_cells]
        C = np.column_stack([
            F_c[np.isin(lab_c, leaves_under(tree, k))].mean(axis=0)
            for k in kids])
        qmask = q_lab == "Q"
        v = F_q[qmask].mean(axis=0)
        wt = simplex_ls(C, v)
        resid = float(np.linalg.norm(v - C @ wt) / np.linalg.norm(v))
        cond = float(np.linalg.cond(C))
        rs = np.random.RandomState(seed * 7 + 1)
        qi = np.flatnonzero(qmask)
        w2s = []
        for _ in range(B):
            take = rs.choice(qi, len(qi), replace=True)
            wb = simplex_ls(C, F_q[take].mean(axis=0))
            w2s.append(np.sort(wb)[-2])
        lcb = float(np.quantile(w2s, 0.05))
        per_donor = []
        for d in sorted(set(q_don[qmask])):
            sel = qmask & (q_don == d)
            wd = simplex_ls(C, F_q[sel].mean(axis=0))
            per_donor.append(float(np.sort(wd)[-2]))
        return {"w": [round(float(x), 4) for x in wt],
                "w2": round(float(np.sort(wt)[-2]), 4),
                "lcb_w2": round(lcb, 4), "resid": round(resid, 4),
                "cond": round(cond, 1),
                "per_donor_w2": [round(x, 4) for x in per_donor]}

    a = run_side(0, 1)
    b = run_side(1, 0)                 # donor-swap replicate
    a["swap_dw2"] = round(abs(a["w2"] - b["w2"]), 4)
    a["unresolved"] = bool(a["resid"] > RESID_MAX or
                           a["cond"] > COND_MAX)
    return a


def run(seeds):
    rows = []
    for seed, r, bsd in itertools.product(seeds, MINORITY, BATCH):
        imbalance = (seed + int(bsd * 10)) % 2 == 0
        small = seed % 3 == 2
        u = unmix(seed, r, bsd, imbalance, small)
        rows.append({"seed": seed, "minority": r, "batch_sd": bsd,
                     "imbalance": imbalance, "small": small, **{
                         k: (json.dumps(v) if isinstance(v, list)
                             else v) for k, v in u.items()}})
        print(f"  seed={seed} r={r:.2f} bsd={bsd}  w={u['w']} "
              f"lcb_w2={u['lcb_w2']} resid={u['resid']} "
              f"swap_dw2={u['swap_dw2']}"
              f"{'  UNRESOLVED' if u['unresolved'] else ''}")
    return rows


def spans(row, tau):
    if row["unresolved"] in (True, "True"):
        return None
    return float(row["lcb_w2"]) > tau


if __name__ == "__main__":
    print("selection worlds:")
    sel = run(SELECT_SEEDS)
    with open(os.path.join(OUT, "unmix_selection.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel[0]))
        w.writeheader()
        w.writerows(sel)
    grid = []
    for tau in GRID_TAU:
        cont = [r for r in sel if r["minority"] in CONTAINED
                and r["batch_sd"] == 0.5]
        spn = [r for r in sel if r["minority"] in SPANNING
               and r["batch_sd"] == 0.5]
        fv = np.mean([spans(r, tau) is True for r in cont])
        mv = np.mean([spans(r, tau) is not True for r in spn])
        grid.append({"tau": tau, "false_span": round(float(fv), 3),
                     "missed_veto": round(float(mv), 3),
                     "ok": fv <= 0.10 and mv <= 0.05})
    with open(os.path.join(OUT, "unmix_grid.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    passing = [g for g in grid if g["ok"]]
    if not passing:
        print("\nNO tau satisfies both criteria — frontier:")
        for g in grid:
            print(f"  tau={g['tau']} false_span={g['false_span']} "
                  f"missed_veto={g['missed_veto']}")
        print("v4 FAILED: per the prespecified stopping rule, the "
              "per-cell breadth guard is abandoned; frozen Walk is "
              "retained.")
        raise SystemExit(0)
    best = sorted(passing, key=lambda g: g["tau"])[0]
    TAU = best["tau"]
    print(f"\nSELECTED (frozen): tau={TAU} "
          f"(false_span={best['false_span']}, "
          f"missed_veto={best['missed_veto']})")
    print("\nvalidation worlds (untouched until now):")
    val = run(VALID_SEEDS)
    with open(os.path.join(OUT, "unmix_validation.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(val[0]))
        w.writeheader()
        w.writerows(val)
    for bsd, tier in ((0.5, "operating regime"), (1.0, "STRESS")):
        print(f"  [{tier}, bsd={bsd}]")
        for grp, name in ((CONTAINED, "contained"),
                          ((0.10,), "transition"),
                          (SPANNING, "spanning")):
            rs_ = [r for r in val if r["minority"] in grp
                   and r["batch_sd"] == bsd]
            if not rs_:
                continue
            sp = np.mean([spans(r, TAU) is True for r in rs_])
            un = np.mean([spans(r, TAU) is None for r in rs_])
            print(f"    {name:10s} n={len(rs_):2d} spans={sp:.2f} "
                  f"unresolved={un:.2f}")
    with open(os.path.join(OUT, "frozen_unmix.json"), "w") as fh:
        json.dump({"tau": TAU, "resid_max": RESID_MAX,
                   "cond_max": COND_MAX, "B": B,
                   "statistic": "LCB(second simplex weight), "
                                "donor-held-out centroids",
                   "selected_on": {"seeds": list(SELECT_SEEDS),
                                   "regime": "batch_sd=0.5"}},
                  fh, indent=1)
    print("\nwrote frozen_unmix.json — next gate: the five Allen "
          "exact stops (protection) + coarse stops (recovery).")
