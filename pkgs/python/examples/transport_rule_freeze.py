"""Freeze the Transport relationship-classification rule (the step
between transport_relations.py and the four-arm comparison).

PRESPECIFIED — everything in this header was fixed before any selection
output was computed, and nothing here may be revised after validation.

Rule family (two thresholds), applied to a node pair's source capture c
and target coverage v:
    equal             : c >= HI and v >= HI
    source_in_target  : c >= HI and v <  HI
    target_in_source  : v >= HI and c <  HI
    partial           : LO <= c < HI and LO <= v < HI
    unrelated         : otherwise
Fixed guards (constants, NOT tuned): a non-unrelated call additionally
requires lift >= 2 and source mass_survival >= 0.5, else the pair is
'unrelated'. Robustness guard: the class is computed under recursive
AND abundance marginals; disagreement demotes the pair to 'unstable'
(an abstention — reported, and counted against recall). The uniform
convention is a negative control and never enters classification.

Grid: HI in {0.70, 0.75, 0.80, 0.85, 0.90},
      LO in {0.15, 0.20, 0.25, 0.30}.

Selection data (and nothing else): simulation seeds {0,1,2} plus the
Allen source=v2 direction. Criterion: maximize the mean of
(sim macro-F1 over 5 classes, pooled over selection seeds) and
(Allen-v2 macro-F1 over its 4 classes; 'partial' is absent there by
construction). Ties -> larger HI, then larger LO (more conservative).

Validation data (untouched by selection): simulation seeds {5,6} and
the Allen source=v3 direction (the transpose reading of the SAME
coupling — held out as a direction, but sharing the coupling, so the
fully independent validation is the held-out seeds; both are reported).

Simulation: consensus.simulate.simulate_donors(K=2) worlds, relabeled
to engineered partitions of the latent leaves so ALL five classes exist
(20 latent leaves in 5 families):
    F1: atlas A fine (4 leaves), atlas B one coarse label
    F2: mirrored (A coarse, B fine)
    F3: both fine, identical         -> leaf-level equal pairs
    F4: crosscut  A {s1,s2},{s3,s4}  vs  B {s2,s3},{s1,s4}
        -> every cross pair shares exactly one latent leaf: partial
    F5: both fine, identical
Family nodes give node-level equal pairs; cross-family pairs are
disjoint. S comes from the frozen kernel measure(); the coupling from
the frozen FUGW configuration; trees are the two-level label trees.

Run: python examples/transport_rule_freeze.py     (needs pot)
"""
import csv
import gzip
import itertools
import json
import os

import numpy as np

from metaarbor import measure, tree_from_levels, tree_weights, write_csv
from metaarbor.consensus.simulate import simulate_donors
from metaarbor.fugw import solve
from metaarbor.tree import leaf_path_dist, leaves_under

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "..", "fixtures")
DATA = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "transport_relations")
GRID_HI = (0.70, 0.75, 0.80, 0.85, 0.90)
GRID_LO = (0.15, 0.20, 0.25, 0.30)
LIFT_MIN, SURVIVAL_MIN = 2.0, 0.5
SELECT_SEEDS, VALID_SEEDS = (0, 1, 2), (5, 6)
CLASSES = ("equal", "source_in_target", "target_in_source", "partial",
           "disjoint")


# ---- engineered partitions over the latent leaves -------------------------
def partition(atlas):
    """label -> latent leaf set, per the prespecified design."""
    fine = lambda f: {f"{f}.s{j}": {f"{f}.s{j}"} for j in (1, 2, 3, 4)}
    p = {}
    if atlas == "A":
        p.update(fine("F1"))
        p["F2.coarse"] = {f"F2.s{j}" for j in (1, 2, 3, 4)}
        p.update(fine("F3"))
        p["F4.x12"] = {"F4.s1", "F4.s2"}
        p["F4.x34"] = {"F4.s3", "F4.s4"}
        p.update(fine("F5"))
    else:
        p["F1.coarse"] = {f"F1.s{j}" for j in (1, 2, 3, 4)}
        p.update(fine("F2"))
        p.update(fine("F3"))
        p["F4.y23"] = {"F4.s2", "F4.s3"}
        p["F4.y14"] = {"F4.s1", "F4.s4"}
        p.update(fine("F5"))
    return p


def label_tree(part):
    fam = lambda lab: lab.split(".")[0]
    return tree_from_levels(sorted((fam(l), l) for l in part),
                            ["family", "label"])


def relation(sa, sb):
    inter = sa & sb
    if not inter:
        return "disjoint"
    if sa == sb:
        return "equal"
    if sa < sb:
        return "source_in_target"
    if sb < sa:
        return "target_in_source"
    return "partial"


# ---- quadrant machinery (identical definitions to transport_relations) ----
def nodes_of(tree, leaf_order):
    parent = tree["parent"]
    ix = {l: i for i, l in enumerate(leaf_order)}
    out = {}
    for n in parent:
        if parent.get(n) is None:
            continue
        lvs = [l for l in leaves_under(tree, n) if l in ix]
        if lvs:
            out[n] = np.asarray(sorted(ix[l] for l in lvs))
    return out


def quadrant_rows(pi, wA, A_nodes, B_nodes):
    total = pi.sum()
    row_out, col_in = pi.sum(axis=1), pi.sum(axis=0)
    out = {}
    for na, ia in A_nodes.items():
        src = row_out[ia].sum()
        surv = src / wA[ia].sum() if wA[ia].sum() > 0 else 0.0
        for nb, ib in B_nodes.items():
            joint = pi[np.ix_(ia, ib)].sum()
            tgt = col_in[ib].sum()
            c = joint / src if src > 0 else 0.0
            v = joint / tgt if tgt > 0 else 0.0
            share = tgt / total
            out[(na, nb)] = (c, v, c / share if share > 0 else 0.0, surv)
    return out


def classify(q, hi, lo):
    c, v, lift, surv = q
    if c >= hi and v >= hi:
        cls = "equal"
    elif c >= hi:
        cls = "source_in_target"
    elif v >= hi:
        cls = "target_in_source"
    elif c >= lo and v >= lo:
        cls = "partial"
    else:
        return "disjoint"
    if lift < LIFT_MIN or surv < SURVIVAL_MIN:
        return "disjoint"
    return cls


def robust_classify(qr, qa, hi, lo):
    a, b = classify(qr, hi, lo), classify(qa, hi, lo)
    return a if a == b else "unstable"


def macro_f1(pairs, hi, lo, present_classes):
    """pairs: list of (q_recursive, q_abundance, truth)."""
    stats = {c: [0, 0, 0] for c in present_classes}   # tp, fp, fn
    n_unstable = 0
    for qr, qa, truth in pairs:
        pred = robust_classify(qr, qa, hi, lo)
        if pred == "unstable":
            n_unstable += 1
            if truth in stats:
                stats[truth][2] += 1
            continue
        for c in present_classes:
            if pred == c and truth == c:
                stats[c][0] += 1
            elif pred == c:
                stats[c][1] += 1
            elif truth == c:
                stats[c][2] += 1
    f1s = []
    for c, (tp, fp, fn) in stats.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), n_unstable, stats


# ---- simulation worlds ----------------------------------------------------
def sim_pairs(seed):
    sim = simulate_donors(K=2, n_family=5, n_sub=4, seed=seed,
                          cells_per_leaf=60, batch_sd=0.5)
    parts = {"A": partition("A"), "B": partition("B")}
    data = {}
    for k, d in zip(("A", "B"), sim["donors"]):
        lab_of = {leaf: lab for lab, s in parts[k].items() for leaf in s}
        data[k] = {"counts": d["counts"],
                   "labels": np.asarray([lab_of[l] for l in d["latent"]])}
    genes = [str(i) for i in range(sim["donors"][0]["counts"].shape[1])]
    m = measure(data["A"]["counts"], data["A"]["labels"],
                data["B"]["counts"], data["B"]["labels"], genes)
    S = m["costs"]["S"]
    qn, cols = m["costs"]["rows"], sorted(set(data["B"]["labels"]))
    # leaf_costs cols follow cache_a['leaves'] == sorted(set(labels_b))
    assert m["costs"]["cols"] == cols
    ta, tb = label_tree(parts["A"]), label_tree(parts["B"])
    CA_h, la = leaf_path_dist(ta)
    CB_h, lb = leaf_path_dist(tb)
    CA = CA_h[np.ix_([la.index(q) for q in qn],
                     [la.index(q) for q in qn])]
    CB = CB_h[np.ix_([lb.index(c) for c in cols],
                     [lb.index(c) for c in cols])]
    wa_m, wb_m = tree_weights(ta), tree_weights(tb)
    w_rec = (np.asarray([wa_m[q] for q in qn]),
             np.asarray([wb_m[c] for c in cols]))
    ca = np.asarray([(data["A"]["labels"] == q).sum() for q in qn],
                    dtype=float)
    cb = np.asarray([(data["B"]["labels"] == c).sum() for c in cols],
                    dtype=float)
    w_ab = (ca / ca.sum(), cb / cb.sum())
    A_nodes, B_nodes = nodes_of(ta, qn), nodes_of(tb, cols)
    # truth in latent-leaf space (family nodes included)
    latent_of = {}
    for k in ("A", "B"):
        for lab, s in parts[k].items():
            latent_of[lab] = set(s)
        for fam in ("F1", "F2", "F3", "F4", "F5"):
            latent_of[f"family:{fam}"] = {f"{fam}.s{j}"
                                          for j in (1, 2, 3, 4)}
    quads = {}
    for name, (wA, wB) in (("recursive", w_rec), ("abundance", w_ab)):
        pi, _ = solve(1 - S, CA, CB, wA, wB)
        quads[name] = quadrant_rows(pi, np.asarray(wA), A_nodes, B_nodes)
    out = []
    for key in quads["recursive"]:
        na, nb = key
        out.append((quads["recursive"][key], quads["abundance"][key],
                    relation(latent_of[na], latent_of[nb])))
    return out


# ---- Allen slices from the committed relation table -----------------------
def allen_pairs(direction):
    with gzip.open(os.path.join(OUT, "relation_table.csv.gz"), "rt") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["convention"], {})[(r["source"], r["target"])] = r
    # transposed direction needs colmass-side survival/lift, which the
    # table lacks -> recompute both couplings exactly as
    # transport_relations.py does (deterministic; same fixtures)
    import transport_relations as tr   # executes its solves on import
    out = []
    mirror = {"source_in_target": "target_in_source",
              "target_in_source": "source_in_target"}
    for conv in ("recursive", "abundance"):
        pi = tr.couplings[conv]
        wA = dict(tr.CONVENTIONS)[conv][0]
        wB = dict(tr.CONVENTIONS)[conv][1]
        if direction == "v2_source":
            q = quadrant_rows(pi, np.asarray(wA),
                              {k: v[0] for k, v in tr.A_nodes.items()},
                              {k: v[0] for k, v in tr.B_nodes.items()})
        else:
            q = quadrant_rows(pi.T, np.asarray(wB),
                              {k: v[0] for k, v in tr.B_nodes.items()},
                              {k: v[0] for k, v in tr.A_nodes.items()})
        if conv == "recursive":
            qr = q
        else:
            qa = q
    for key in qr:
        na, nb = key
        if direction == "v2_source":
            truth = by["recursive"][(na, nb)]["truth"]
        else:
            t = by["recursive"][(nb, na)]["truth"]
            truth = mirror.get(t, t)
        out.append((qr[key], qa[key], truth))
    return out


if __name__ == "__main__":
    print("building selection data (sim seeds "
          f"{SELECT_SEEDS} + Allen v2-source)...")
    sim_sel = []
    for s in SELECT_SEEDS:
        sim_sel += sim_pairs(s)
        print(f"  sim seed {s} done ({len(sim_sel)} pairs)")
    allen_sel = allen_pairs("v2_source")
    print(f"  Allen v2-source: {len(allen_sel)} pairs")

    results = []
    for hi, lo in itertools.product(GRID_HI, GRID_LO):
        f1_sim, u_sim, _ = macro_f1(sim_sel, hi, lo, CLASSES)
        f1_al, u_al, _ = macro_f1(allen_sel, hi, lo,
                                  tuple(c for c in CLASSES
                                        if c != "partial"))
        results.append({"HI": hi, "LO": lo,
                        "macroF1_sim": round(f1_sim, 4),
                        "macroF1_allen_v2": round(f1_al, 4),
                        "criterion": round((f1_sim + f1_al) / 2, 4),
                        "unstable_sim": u_sim, "unstable_allen": u_al})
    results.sort(key=lambda r: (-r["criterion"], -r["HI"], -r["LO"]))
    write_csv(results, os.path.join(OUT, "rule_grid.csv"))
    best = results[0]
    HI, LO = best["HI"], best["LO"]
    print(f"\nSELECTED (frozen): HI={HI} LO={LO} "
          f"criterion={best['criterion']} "
          f"(sim {best['macroF1_sim']}, allen_v2 "
          f"{best['macroF1_allen_v2']})")

    print("\nvalidation (untouched by selection):")
    sim_val = []
    for s in VALID_SEEDS:
        sim_val += sim_pairs(s)
    f1, u, stats = macro_f1(sim_val, HI, LO, CLASSES)
    print(f"  held-out sim seeds {VALID_SEEDS}: macro-F1={f1:.4f} "
          f"(unstable {u}/{len(sim_val)})")
    for c, (tp, fp, fn) in stats.items():
        print(f"    {c:18s} tp={tp:3d} fp={fp:3d} fn={fn:3d}")
    allen_val = allen_pairs("v3_source")
    f1v, uv, statsv = macro_f1(allen_val, HI, LO,
                               tuple(c for c in CLASSES
                                     if c != "partial"))
    print(f"  Allen v3-source (held-out direction; shares the "
          f"coupling): macro-F1={f1v:.4f} (unstable {uv}/{len(allen_val)})")
    for c, (tp, fp, fn) in statsv.items():
        print(f"    {c:18s} tp={tp:3d} fp={fp:3d} fn={fn:3d}")

    frozen = {"HI": HI, "LO": LO, "lift_min": LIFT_MIN,
              "survival_min": SURVIVAL_MIN,
              "robustness": "recursive+abundance agreement",
              "selected_on": {"sim_seeds": list(SELECT_SEEDS),
                              "allen_direction": "v2_source"},
              "validation": {
                  "sim_seeds": list(VALID_SEEDS),
                  "sim_macroF1": round(f1, 4),
                  "allen_v3_macroF1": round(f1v, 4)}}
    with open(os.path.join(OUT, "frozen_rule.json"), "w") as fh:
        json.dump(frozen, fh, indent=1)
    print("\nwrote rule_grid.csv + frozen_rule.json — the rule is now "
          "FROZEN; amygdala and Stage 3 consume frozen_rule.json only.")
