"""Transport relationship table + vocabulary validation on Allen
(Stages 1-2 of the bounded Transport-characterization project).

Question: do rolled-up FUGW coupling masses distinguish the relationship
classes MetaArbor needs (equivalence / containment each way / partial
overlap / unrelated) where truth is known?

PRESPECIFIED before any output was computed — none of the following may
be revised after looking at results:

1. Coupling: the FROZEN Transport configuration (fugw.py: rho=0.3,
   alpha=0.9, eps=0, hop structure, tree-intrinsic recursive marginals).
   VERSION-DRIFT FINDING, discovered while anchoring this script: POT
   0.9.6 and 0.9.7 agree with each other bit-for-bit but neither
   reproduces the committed allen_demo/metric_fugw/pi_hop.csv.gz (max
   entry diff 5.5e-3; total mass 0.9957 vs 0.9592) — the committed
   couplings came from an environment no longer present, and pyproject
   pins only pot>=0.9.4. The committed files are left untouched as the
   historical record; THIS experiment re-anchors on current POT, whose
   version is recorded in the output dir, and the drift is reported as
   a defect to close (pin POT + regenerate anchors in a dedicated
   commit). Same-environment re-solve is exactly reproducible.
2. Node pairs: every node of the v2 curated tree (class -> subclass) x
   every node of the v3 curated tree (class -> subclass -> supertype ->
   cluster), roots excluded. Analysis focuses beneath shared classes;
   depth columns let readers exclude trivial coarse pairs.
3. Quantities per (A, B), with D() = leaf descendants, from coupling pi:
     joint          = sum of pi over D(A) x D(B)
     capture        = joint / (total mass leaving A's leaves)
     coverage       = joint / (total mass arriving in B's leaves)
     mass_survival  = (total mass leaving A) / (marginal mass of A)
     target_share   = (total mass arriving in B) / (total mass)
     lift           = capture / target_share   (independence baseline)
4. Truth classes from the curated cluster->subclass->class map, by the
   set relation between clusters-under-A and clusters-under-B:
     equal / source_in_target / target_in_source / partial / disjoint.
5. Marginal conventions: recursive tree-intrinsic (PRIMARY, committed);
   abundance = per-side cell-count proportions (prespecified
   sensitivity); uniform-per-leaf (known-biased NEGATIVE CONTROL — a
   relation changing only under uniform does not undermine the primary).
6. Rule-selection split, declared now for the later freezing stage: any
   future classification rule will be chosen using ONLY the source=v2
   (coarse->fine) direction plus simulation; the source=v3 direction is
   held out for validation. No rule is defined in this script.
7. Batch-condition stability is NOT reported: the donor/platform splits
   live in the legacy R-era analyses and are not wired into the Python
   Transport path; sensitivity here = marginal conventions only (stated
   limitation, not a silent omission).

Run: python examples/transport_relations.py   (needs pot)
"""
import csv
import gzip
import os

import numpy as np

from metaarbor import tree_from_levels, tree_weights, write_csv
from metaarbor.fugw import solve
from metaarbor.tree import leaf_path_dist, leaves_under

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
FX = os.path.join(REPO, "pkgs", "fixtures")
DATA = os.path.join(REPO, "data", "wmb_plilaorb")
OUT = os.path.join(HERE, "allen_demo", "metric_fugw")


def rcsv(name):
    with gzip.open(os.path.join(FX, name), "rt") as fh:
        return list(csv.reader(fh))


# ---- shared inputs (identical to metric_fugw_experiment.py) ---------------
srows = rcsv("S_sub.csv.gz")
cols = srows[0][1:]
qn = [r[0] for r in srows[1:]]
S = np.asarray([[float(x) for x in r[1:]] for r in srows[1:]])
M = 1 - S
lv = rcsv("tree_levels_b.csv.gz")
tree_b = tree_from_levels([tuple(r) for r in lv[1:]], lv[0])
cls_of = {r[1]: r[0] for r in lv[1:]}
tree_a = tree_from_levels(sorted({(cls_of[q], q) for q in qn}),
                          ["class", "subclass"])
CA_h, la = leaf_path_dist(tree_a)
CB_h, lb = leaf_path_dist(tree_b)
CA = CA_h[np.ix_([la.index(q) for q in qn], [la.index(q) for q in qn])]
CB = CB_h[np.ix_([lb.index(c) for c in cols], [lb.index(c) for c in cols])]

# ---- marginal conventions -------------------------------------------------
wA_map, wB_map = tree_weights(tree_a), tree_weights(tree_b)
w_recursive = (np.asarray([wA_map[q] for q in qn]),
               np.asarray([wB_map[c] for c in cols]))
w_uniform = (np.ones(len(qn)) / len(qn), np.ones(len(cols)) / len(cols))


def abundance_weights():
    def count(tag, key):
        out = {}
        with open(os.path.join(DATA, f"cells_{tag}.csv")) as fh:
            for c in csv.DictReader(fh):
                out[c[key]] = out.get(c[key], 0) + 1
        return out
    ca = count("10Xv2", "subclass")
    cb = count("10Xv3", "cluster")
    a = np.asarray([ca[q] for q in qn], dtype=float)
    b = np.asarray([cb[c] for c in cols], dtype=float)
    return a / a.sum(), b / b.sum()


CONVENTIONS = [("recursive", w_recursive)]
if os.path.isdir(DATA):
    CONVENTIONS.append(("abundance", abundance_weights()))
else:
    print("NOTE: data/wmb_plilaorb absent — abundance sensitivity skipped")
CONVENTIONS.append(("uniform", w_uniform))

# ---- solve; determinism anchor for the primary ----------------------------
couplings = {}
for name, (wA, wB) in CONVENTIONS:
    pi, gap = solve(M, CA, CB, wA, wB)
    couplings[name] = pi
    print(f"solved {name}: mass={pi.sum():.4f} pq_gap={gap:.1e}")
import ot as _ot
with gzip.open(os.path.join(OUT, "pi_hop.csv.gz"), "rt") as fh:
    rows_ = list(csv.reader(fh))
committed = np.asarray([[float(x) for x in r[1:]] for r in rows_[1:]])
drift = float(np.abs(couplings["recursive"] - committed).max())
if drift > 1e-8:
    print(f"NOTE: POT {_ot.__version__} does not reproduce the committed "
          f"pi_hop (max entry diff {drift:.1e}) — see docstring; this "
          "run anchors on the current version.")
else:
    print("determinism check: recursive coupling matches committed "
          "pi_hop")
OUT2 = os.path.join(HERE, "transport_relations")
os.makedirs(OUT2, exist_ok=True)
write_csv(couplings["recursive"], os.path.join(OUT2, "pi_recursive.csv.gz"),
          row_names=qn, col_names=cols)
with open(os.path.join(OUT2, "SOLVER_VERSION.txt"), "w") as fh:
    fh.write(f"POT {_ot.__version__}\nnumpy {np.__version__}\n"
             f"drift_vs_committed_pi_hop {drift:.3e}\n")

# ---- node inventories -----------------------------------------------------
def nodes_of(tree, leaf_order):
    """All non-root nodes -> (leaf index array, depth). Depth = hops
    from root."""
    parent = tree["parent"]
    kids = {}
    for n, p in parent.items():
        kids.setdefault(p, []).append(n)
    ix = {l: i for i, l in enumerate(leaf_order)}
    out = {}
    def depth(n):
        d, x = 0, n
        while parent.get(x) is not None:
            d += 1
            x = parent[x]
        return d
    for n in parent:
        if parent.get(n) is None:      # root: excluded (prespec #2)
            continue
        lvs = [l for l in leaves_under(tree, n) if l in ix]
        if lvs:
            out[n] = (np.asarray(sorted(ix[l] for l in lvs)), depth(n))
    return out


A_nodes = nodes_of(tree_a, qn)
B_nodes = nodes_of(tree_b, cols)
print(f"node inventory: {len(A_nodes)} v2-side, {len(B_nodes)} v3-side")

# truth: clusters under each node, via the curated map
clusters_of_sub = {}
for r in lv[1:]:
    clusters_of_sub.setdefault(r[1], set()).add(r[3])


def truth_clusters_A(n):
    subs = [l for l in leaves_under(tree_a, n)]
    return set().union(*(clusters_of_sub[s] for s in subs))


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


# ---- Stage 1: the relationship table --------------------------------------
rows = []
for conv, _w in CONVENTIONS:
    pi = couplings[conv]
    total = pi.sum()
    row_out = pi.sum(axis=1)           # mass leaving each source leaf
    col_in = pi.sum(axis=0)            # mass arriving at each target leaf
    wA = dict(CONVENTIONS)[conv][0]
    for na, (ia, da) in sorted(A_nodes.items()):
        src_out = row_out[ia].sum()
        surv = src_out / wA[ia].sum() if wA[ia].sum() > 0 else 0.0
        sa = truth_clusters_A(na)
        for nb, (ib, db) in sorted(B_nodes.items()):
            joint = pi[np.ix_(ia, ib)].sum()
            tgt_in = col_in[ib].sum()
            capture = joint / src_out if src_out > 0 else 0.0
            coverage = joint / tgt_in if tgt_in > 0 else 0.0
            share = tgt_in / total
            rows.append({
                "convention": conv, "source": na, "target": nb,
                "truth": relation(sa, set(leaves_under(tree_b, nb))),
                "joint": round(joint, 6),
                "capture": round(capture, 4),
                "coverage": round(coverage, 4),
                "mass_survival": round(surv, 4),
                "target_share": round(share, 4),
                "lift": round(capture / share, 3) if share > 0 else np.nan,
                "src_depth": da, "tgt_depth": db,
                "src_leaves": len(ia), "tgt_leaves": len(ib)})
write_csv(rows, os.path.join(OUT2, "relation_table.csv.gz"))
print(f"wrote relation_table.csv ({len(rows)} rows, "
      f"{len(CONVENTIONS)} conventions)")

# ---- Stage 2: does the geometry separate the truth classes? ---------------
prim = [r for r in rows if r["convention"] == "recursive"]
print("\ntruth-class geometry (recursive marginals, medians "
      "[capture, coverage, lift]):")
for cls in ("equal", "source_in_target", "target_in_source", "partial",
            "disjoint"):
    sel = [r for r in prim if r["truth"] == cls]
    if not sel:
        # NOTE: 'partial' is empty BY CONSTRUCTION on Allen — both
        # curated trees are the same taxonomy at different depths, so
        # every node pair is nested or disjoint. Partial overlap can
        # only be validated on simulation (per the prespecified
        # rule-selection scheme), not on this benchmark.
        print(f"  {cls:18s} n=   0  (absent by construction here)")
        continue
    med = lambda k: float(np.median([r[k] for r in sel]))
    print(f"  {cls:18s} n={len(sel):4d}  capture={med('capture'):.3f} "
          f"coverage={med('coverage'):.3f} lift={med('lift'):.2f}")

# most-specific-supported-target check (descriptive; no rule defined):
# for each v2 SUBCLASS (leaf source), walk its true v3 subclass node's
# root path and report where capture stays high while coverage rises.
print("\nancestor profiles for 3 prespecified example subclasses\n"
      "(most clusters / median / singleton):")
sizes = sorted(qn, key=lambda q: len(clusters_of_sub[q]))
examples = [sizes[-1], sizes[len(sizes) // 2], sizes[0]]
parent_b = tree_b["parent"]
for q in examples:
    ia = A_nodes[q][0]
    pi = couplings["recursive"]
    src_out = pi[ia].sum()
    # deepest true node = the v3 subclass node ("subclass:<name>"),
    # then its ancestors up to (not including) root
    chain, x = [], f"subclass:{q}"
    while x is not None and x in B_nodes:
        chain.append(x)
        x = parent_b.get(x)
    print(f"  source v2|{q} ({len(clusters_of_sub[q])} clusters):")
    for nb in chain:
        ib, db = B_nodes[nb]
        joint = pi[np.ix_(ia, ib)].sum()
        cap = joint / src_out if src_out > 0 else 0
        cov = joint / pi[:, ib].sum() if pi[:, ib].sum() > 0 else 0
        print(f"    depth {db} {nb[:40]:40s} capture={cap:.3f} "
              f"coverage={cov:.3f}")
