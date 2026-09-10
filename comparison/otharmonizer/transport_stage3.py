"""Stage 3: the four-arm comparison — does Transport characterization
add truth-consistent structure beyond the Walk evidence already in hand?

Footing: everything runs on the SAME inputs as the Walk arms — the
committed inferred trees (examples/harmonize_demo/allen_input_trees.json)
and the frozen kernel measurement (measure(), n_hvg=1000) on
data/wmb_plilaorb with the same "v2|"/"v3|" label namespaces harmonize
used. The earlier relationship table (curated trees + fixture S) was the
vocabulary validation; THIS script is the estimator comparison, so the
footing must match the Walk arms exactly.

Arms (constructions fixed before any score was computed):
  1 walk_core          — the strict consensus cut (certified core).
  2 walk_frontier      — the committed containment frontier
                         (containment_frontier.csv, structure-kept
                         columns), cited at every threshold.
  3 transport_alone    — built ONLY from Transport relations under the
                         frozen rule (frozen_rule.json):
                         (a) 'equal' node pairs, ordered by min(capture,
                         coverage) desc, greedily accepted subject to
                         one-merge-per-node and order-isomorphism with
                         accepted merges (a1 anc a2 <=> b1 anc b2);
                         (b) backbone = accepted merges under
                         nearest-accepted-ancestor parentage;
                         (c) every unplaced label attaches flat beneath
                         the backbone image of its DEEPEST
                         source_in_target target (ties -> larger
                         min(capture, coverage)); backbone image = the
                         target's merge, else its nearest merged
                         input-tree ancestor.
  4 core_plus_transport — trunk = the certified core (arm 1); non-core
                         labels attach flat via the SAME deepest-
                         supported-containment selection as arm 3, with
                         targets mapped into the core exactly as the
                         Walk frontier maps its evidence targets
                         (direct certified member, else nearest
                         certified input-tree ancestor).

Relations: frozen_rule.json only (HI/LO + lift>=2, survival>=0.5,
recursive<->abundance robustness). Both directions are computed from
the one coupling (v2-source and transposed v3-source quadrants); each
label's containment uses its own atlas's source direction. No
threshold, rule, or construction may be adjusted after scores exist.

Scoring: the existing star-proof metrics, structure kept (labeled
leaves, anonymous internals preserved): TRIP_REC primary, COPH
secondary, vs the same curated reference as rescore_current.py; label
coverage stated per arm (selective evaluation); wrong-lineage rate for
attachment arms = fraction of attached labels whose target node's
member labels have a majority curated CLASS different from the
attached label's curated class (prespecified secondary).

Run: PYTHONPATH=../../pkgs/python/src python transport_stage3.py
(needs pot; data/wmb_plilaorb present)
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pkgs", "python",
                                "src"))
from metrics import (MyNode, cophenetic_spearman,  # noqa: E402
                     triplet_scores)
from metaarbor import measure  # noqa: E402
from metaarbor.consensus.cut import consensus_cut  # noqa: E402
from metaarbor.fugw import solve  # noqa: E402
from metaarbor.tree import (ancestors, leaf_path_dist,  # noqa: E402
                            leaves_under, tree_weights)

D = os.path.join(HERE, "..", "..", "data", "wmb_plilaorb")
MA = os.path.join(HERE, "..", "..", "pkgs", "python", "examples",
                  "harmonize_demo")
TR = os.path.join(HERE, "..", "..", "pkgs", "python", "examples",
                  "transport_relations")
RULE = json.load(open(os.path.join(TR, "frozen_rule.json")))
HI, LO = RULE["HI"], RULE["LO"]
LIFT_MIN, SURV_MIN = RULE["lift_min"], RULE["survival_min"]

# ---- data, truth, reference (identical to rescore_current.py) -------------
with open(os.path.join(D, "cells_10Xv3.csv")) as fh:
    cellsB = list(csv.DictReader(fh))
truth = {c["cluster"]: c["subclass"] for c in cellsB}
with open(os.path.join(D, "cells_10Xv2.csv")) as fh:
    cellsA = list(csv.DictReader(fh))
subclasses_v2 = {c["subclass"] for c in cellsA}
labels = {"v2": {f"v2|{s}" for s in subclasses_v2},
          "v3": {f"v3|{c}" for c in truth}}
itrees = json.load(open(os.path.join(MA, "allen_input_trees.json")))
tree_nodes = json.load(open(os.path.join(MA,
                                         "metaarbor_tree_current.json")))
# curated class of every label, for the wrong-lineage secondary
cls_of_sub = {}
lvname = os.path.join(HERE, "..", "..", "pkgs", "fixtures",
                      "tree_levels_b.csv.gz")
import gzip  # noqa: E402
lv = list(csv.reader(gzip.open(lvname, "rt")))
for r in lv[1:]:
    cls_of_sub[r[1]] = r[0]


def curated_class(lab):
    ds, name = lab.split("|", 1)
    sub = name if ds == "v2" else truth[name]
    return cls_of_sub.get(sub)


def ref_tree():
    root = MyNode("root")
    by = {}
    for cl, s in truth.items():
        by.setdefault(s, []).append(cl)
    for s in sorted(by):
        cls = sorted(by[s])
        if s in subclasses_v2 and len(cls) == 1:
            root.addkid(MyNode(f"v2-{s}&v3-{cls[0]}"))
            continue
        sn = MyNode(f"v2-{s}" if s in subclasses_v2 else f"v3only-{s}")
        root.addkid(sn)
        for cl in cls:
            sn.addkid(MyNode(f"v3-{cl}"))
    return root


REF = ref_tree()
disp = lambda lab: f"{lab.split('|', 1)[0]}-{lab.split('|', 1)[1]}"

# ---- the coupling on the inferred-tree footing ----------------------------
def load(tag):
    counts = np.asarray(mmread(os.path.join(
        D, f"counts_{tag}.mtx")).todense()).T
    lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
    return counts, lib


genes = open(os.path.join(D, "genes.txt")).read().split()
cA, lA = load("10Xv2")
cB, lB = load("10Xv3")
labA = np.asarray([f"v2|{c['subclass']}" for c in cellsA])
labB = np.asarray([f"v3|{c['cluster']}" for c in cellsB])
print("measuring (frozen kernel, n_hvg=1000)...")
m = measure(cA, labA, cB, labB, genes, lib_a=lA, lib_b=lB)
S = m["costs"]["S"]
qn, cols = m["costs"]["rows"], m["costs"]["cols"]
ta, tb = itrees["v2"], itrees["v3"]
CA_h, la = leaf_path_dist(ta)
CB_h, lb = leaf_path_dist(tb)
CA = CA_h[np.ix_([la.index(q) for q in qn], [la.index(q) for q in qn])]
CB = CB_h[np.ix_([lb.index(c) for c in cols], [lb.index(c) for c in cols])]
wa_m, wb_m = tree_weights(ta), tree_weights(tb)
w_rec = (np.asarray([wa_m[q] for q in qn]),
         np.asarray([wb_m[c] for c in cols]))
na_ct = np.asarray([(labA == q).sum() for q in qn], dtype=float)
nb_ct = np.asarray([(labB == c).sum() for c in cols], dtype=float)
w_ab = (na_ct / na_ct.sum(), nb_ct / nb_ct.sum())
pis = {}
for conv, (wA, wB) in (("recursive", w_rec), ("abundance", w_ab)):
    pi, gap = solve(1 - S, CA, CB, wA, wB)
    pis[conv] = pi
    print(f"  solved {conv}: mass={pi.sum():.4f}")

# ---- quadrants + frozen-rule relations, both directions -------------------
def nodes_of(tree, leaf_order):
    ix = {l: i for i, l in enumerate(leaf_order)}
    out = {}
    for n in tree["parent"]:
        if tree["parent"].get(n) is None:
            continue
        lvs = [l for l in leaves_under(tree, n) if l in ix]
        if lvs:
            out[n] = np.asarray(sorted(ix[l] for l in lvs))
    return out


A_nodes, B_nodes = nodes_of(ta, qn), nodes_of(tb, cols)


def quadrants(pi, wA, S_nodes, T_nodes):
    total = pi.sum()
    ro, ci = pi.sum(axis=1), pi.sum(axis=0)
    out = {}
    for na, ia in S_nodes.items():
        src = ro[ia].sum()
        surv = src / wA[ia].sum() if wA[ia].sum() > 0 else 0.0
        for nb, ib in T_nodes.items():
            joint = pi[np.ix_(ia, ib)].sum()
            tgt = ci[ib].sum()
            c = joint / src if src > 0 else 0.0
            v = joint / tgt if tgt > 0 else 0.0
            share = tgt / total
            out[(na, nb)] = (c, v, c / share if share > 0 else 0.0, surv)
    return out


def classify(q):
    c, v, lift, surv = q
    if c >= HI and v >= HI:
        cls = "equal"
    elif c >= HI:
        cls = "source_in_target"
    elif v >= HI:
        cls = "target_in_source"
    elif c >= LO and v >= LO:
        cls = "partial"
    else:
        return "disjoint"
    if lift < LIFT_MIN or surv < SURV_MIN:
        return "disjoint"
    return cls


def robust(qr, qa):
    a, b = classify(qr), classify(qa)
    return a if a == b else "unstable"


Q = {}
for direc, (Sn, Tn, flip) in {
        "v2_source": (A_nodes, B_nodes, False),
        "v3_source": (B_nodes, A_nodes, True)}.items():
    qs = {}
    for conv in ("recursive", "abundance"):
        pi = pis[conv].T if flip else pis[conv]
        wA = (w_rec if conv == "recursive" else w_ab)[1 if flip else 0]
        qs[conv] = quadrants(pi, np.asarray(wA), Sn, Tn)
    Q[direc] = {k: (qs["recursive"][k], qs["abundance"][k]) for k in
                qs["recursive"]}

rel = {d: {k: robust(*v) for k, v in Q[d].items()} for d in Q}
n_eq = sum(1 for v in rel["v2_source"].values() if v == "equal")
n_ct = sum(1 for v in rel["v2_source"].values()
           if v == "source_in_target")
n_ct2 = sum(1 for v in rel["v3_source"].values()
            if v == "source_in_target")
print(f"frozen-rule relations: {n_eq} equal (v2-source view), "
      f"{n_ct} v2-in-v3 containments, {n_ct2} v3-in-v2 containments")

# ---- arm 1: certified core ------------------------------------------------
strict = consensus_cut(tree_nodes, level="strict", leaf_labels=labels)
cert_member = {(ds, mm): i for i, nd in strict["nodes"].items()
               for ds, mm in nd["members"].items()}
core_labels = {mm for (ds, mm) in cert_member if mm in labels[ds]}


def core_tree(extra=None):
    kids = {i: list(nd["children"]) for i, nd in strict["nodes"].items()}
    root = MyNode("root")

    def build(i, pn):
        nd = strict["nodes"][i]
        pp = [disp(mm) for ds, mm in sorted(nd["members"].items())
              if mm in labels[ds]]
        node = MyNode("&".join(pp)) if pp else MyNode(f"__a{i}__")
        pn.addkid(node)
        for ds, lab in (extra or {}).get(i, []):
            node.addkid(MyNode(disp(lab)))
        for c_ in sorted(kids.get(i, [])):
            build(c_, node)
    for r in sorted(strict["roots"]):
        build(r, root)
    return root


# ---- deepest-supported-containment selection ------------------------------
def deepest_containment(lab, ds):
    """The label's deepest source_in_target target in the other tree,
    from its own atlas's source direction; ties -> larger min(c, v)."""
    direc = "v2_source" if ds == "v2" else "v3_source"
    other_tree = tb if ds == "v2" else ta
    best = None
    for (na, nb), cls in rel[direc].items():
        if na != lab or cls != "source_in_target":
            continue
        depth = len(ancestors(other_tree, nb))
        c, v, _, _ = Q[direc][(na, nb)][0]
        key = (depth, min(c, v))
        if best is None or key > best[0]:
            best = (key, nb)
    return best[1] if best else None


# ---- arm 4: core + transport attachments ---------------------------------
def to_core(nb, other_ds):
    tree = tb if other_ds == "v3" else ta
    node = cert_member.get((other_ds, nb))
    x = nb
    while node is None:
        x = tree["parent"].get(x)
        if x in (None, "root"):
            return None
        node = cert_member.get((other_ds, x))
    return node


extra4, att4 = {}, []
for ds in ("v2", "v3"):
    other = "v3" if ds == "v2" else "v2"
    for lab in sorted(labels[ds]):
        if lab in core_labels:
            continue
        tgt = deepest_containment(lab, ds)
        if tgt is None:
            continue
        node = to_core(tgt, other)
        if node is None:
            continue
        extra4.setdefault(node, []).append((ds, lab))
        att4.append((lab, node))
arm4 = core_tree(extra4)
n4 = len(core_labels) + len(att4)

# ---- arm 3: transport-alone ----------------------------------------------
cands = sorted(((min(q[0][0], q[0][1]), k)
                for k, q in Q["v2_source"].items()
                if rel["v2_source"][k] == "equal"), reverse=True)
anc_a = {n: set(ancestors(ta, n)) for n in A_nodes}
anc_b = {n: set(ancestors(tb, n)) for n in B_nodes}
merged, used_a, used_b = [], set(), set()
for _score, (na, nb) in cands:
    if na in used_a or nb in used_b:
        continue
    ok = True
    for ma, mb in merged:
        rel_a = (ma in anc_a[na], na in anc_a.get(ma, set()))
        rel_b = (mb in anc_b[nb], nb in anc_b.get(mb, set()))
        if rel_a != rel_b:
            ok = False
            break
    if ok:
        merged.append((na, nb))
        used_a.add(na)
        used_b.add(nb)
merge_of_a = {a: i for i, (a, b) in enumerate(merged)}
parent3 = {}
for i, (a, b) in enumerate(merged):
    par, x = None, ta["parent"].get(a)
    while x not in (None, "root"):
        if x in merge_of_a:
            par = merge_of_a[x]
            break
        x = ta["parent"].get(x)
    parent3[i] = par
placed3 = set()
merge_names = []
for i, (a, b) in enumerate(merged):
    pp = [disp(x) for x in (a, b) if x in labels["v2"] | labels["v3"]]
    placed3.update(x for x in (a, b)
                   if x in labels["v2"] | labels["v3"])
    merge_names.append("&".join(pp) if pp else f"__m{i}__")
extra3, att3 = {}, []
for ds in ("v2", "v3"):
    for lab in sorted(labels[ds]):
        if lab in placed3:
            continue
        tgt = deepest_containment(lab, ds)
        if tgt is None:
            continue
        tree = tb if ds == "v2" else ta
        mo = merge_of_a if ds == "v3" else {b: i for i, (a, b)
                                            in enumerate(merged)}
        node, x = mo.get(tgt), tgt
        while node is None:
            x = tree["parent"].get(x)
            if x in (None, "root"):
                break
            node = mo.get(x)
        if node is None:
            continue
        extra3.setdefault(node, []).append(lab)
        att3.append((lab, node))
root3 = MyNode("root")
mn = {}
for i in range(len(merged)):
    mn[i] = MyNode(merge_names[i])
for i in range(len(merged)):
    (mn[parent3[i]] if parent3[i] is not None else root3).addkid(mn[i])
for i, labs in extra3.items():
    for lab in sorted(labs):
        mn[i].addkid(MyNode(disp(lab)))
n3 = len(placed3) + len(att3)

# ---- wrong-lineage secondary ----------------------------------------------
def wrong_lineage(att, node_members):
    bad = 0
    for lab, node in att:
        mem = node_members(node)
        classes = [curated_class(x) for x in mem if curated_class(x)]
        if not classes:
            continue
        maj = max(set(classes), key=classes.count)
        if curated_class(lab) != maj:
            bad += 1
    return bad / len(att) if att else 0.0


wl4 = wrong_lineage(att4, lambda n: [mm for (ds, mm), i in
                                     cert_member.items() if i == n and
                                     mm in labels[ds]])
wl3 = wrong_lineage(att3, lambda i: [x for x in merged[i]
                                     if x in labels["v2"] | labels["v3"]])

# ---- score ---------------------------------------------------------------
total_labels = len(labels["v2"]) + len(labels["v3"])
rows = []
for arm, tree_, n in (("walk_core", core_tree(), len(core_labels)),
                      ("transport_alone", root3, n3),
                      ("core_plus_transport", arm4, n4)):
    rec, _ = triplet_scores(tree_, REF)
    coph = cophenetic_spearman(tree_, REF)
    rows.append({"arm": arm, "n_labels": n,
                 "coverage": round(n / total_labels, 3),
                 "TRIP_REC_K": round(rec, 4), "COPH_K": round(coph, 4)})
    print(f"{arm:22s} labels={n:3d} cov={n/total_labels:.3f} "
          f"TRIP_REC={rec:.4f} COPH={coph:.4f}")
print(f"wrong-lineage: transport_alone {wl3:.3f} "
      f"({len(att3)} attachments) | core+transport {wl4:.3f} "
      f"({len(att4)} attachments)")
print("\nwalk_frontier (committed containment_frontier.csv, "
      "structure-kept):")
for r in csv.DictReader(open(os.path.join(HERE,
                                          "containment_frontier.csv"))):
    print(f"  t={r['threshold']:>4s} labels={r['n_labels']:>3s} "
          f"cov={r['coverage']} TRIP_REC={r['TRIP_REC_K']} "
          f"COPH={r['COPH_K']}")
    rows.append({"arm": f"walk_frontier_t{r['threshold']}",
                 "n_labels": int(r["n_labels"]),
                 "coverage": float(r["coverage"]),
                 "TRIP_REC_K": float(r["TRIP_REC_K"]),
                 "COPH_K": float(r["COPH_K"])})
with open(os.path.join(HERE, "transport_stage3.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote transport_stage3.csv")
