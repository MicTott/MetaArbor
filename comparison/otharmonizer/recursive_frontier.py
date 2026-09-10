"""Recursive reconciliation of the containment frontier — the
reversible hierarchy proposal, validated on Allen truth.

The flat frontier hoists every one-way-supported label to the nearest
CERTIFIED ancestor of its evidence target, producing broad fans. This
construction instead assembles the deepest tree supported by the
existing constraints, with NO new molecular thresholds and NO Walk
rerun:

  vertices    : canonical nodes of both input trees, with certified
                members contracted into their certified core node
                (core equivalences are the ONLY equivalences — v1
                proposes no new merges);
  ancestry    : every non-core vertex keeps its input-tree canonical
                parent (contracted) — input ancestry is PRESERVED,
                never overridden by evidence;
  attachment  : only a component ROOT (a vertex whose canonical parent
                chain tops out) may attach, beneath the contracted
                image of ITS OWN one-way Walk call (support >= t) —
                the image may be a non-core vertex, which is where
                recursive depth comes from (atlas clade under atlas
                clade under core);
  cycles      : attachments are applied in descending support
                (deterministic tie-break by name); any attachment that
                would create a cycle is SKIPPED and ledgered as
                conflicting — never resolved by force;
  conflicts   : every non-root vertex with its own matched call is
                audited after assembly: if the call's image is NOT an
                ancestor of the vertex, the vertex is flagged
                evidence_conflict (counted, listed, NOT moved);
  unplaced    : components with no surviving attachment stay at the
                global root — absence is unresolved, not hidden.

Scoring: structure-kept triplet recovery vs the curated reference ON
MATCHED LABELS — at each threshold, the reference is restricted to
the flat frontier's own label set and three trees are scored on it:
the flat frontier, the recursive proposal, and the full assembly
(input parentage). Acceptance: the recursive proposal must score >=
the flat frontier at every threshold on matched labels; a drop
rejects the assembly rule on Allen before amygdala sees it.

VALIDATION FINDINGS (Allen; recorded after the run — neither the
construction nor the criterion was changed in response):
- ACCEPTANCE, by its letter, FAILS: at t=0.99 recursive scores
  0.9388 vs the frontier's 0.9393 (-0.0005, single-triplet
  resolution) on matched labels. At the other eight thresholds
  recursive >= frontier, by up to +0.012 (t=0.0: 0.9326 vs 0.9205).
  The criterion is reported as stated; whether a one-threshold
  5e-4 dip rejects the proposal is a judgment left explicitly to
  the maintainer, not resolved here by adding a tolerance.
- COVERAGE-ARTIFACT CORRECTION (the substantive finding): the full
  assembly scores 0.9504 on the frontier's own 85-label set —
  identical to the frontier's 0.9500. The earlier claim that
  evidence attachment beats input parentage was a COVERAGE artifact
  (different label sets), not a placement effect: on matched labels,
  parentage, evidence attachment, and recursion are equivalent on
  Allen (within ~0.01 everywhere; recursion == assembly to ~4
  decimals throughout).
- Allen CANNOT demonstrate recursion's depth gains: preserved
  ancestry plus Allen's rich certified core leaves exactly ONE
  root-stranded component to attach, so the recursive tree
  essentially reproduces the assembly (full-coverage TRIP_REC 0.9117
  vs 0.9116). The substrate recursion exists for — fragmented
  forests — is the amygdala inhibitory case (58 components), which
  is where the depth demonstration must come from. Allen's role
  here is safety: no cycles, ancestry preserved, placements
  triplet-equivalent to the two established constructions.
- The post-assembly audit flags ~68 non-root vertices whose own
  call points outside their ancestry position; on Allen these
  conflicts are triplet-neutral (moving them, as the flat frontier
  does, neither helps nor hurts on matched labels). Triplet-neutral
  on this SHALLOW reference is not the same as biologically harmless;
  the audit ledger travels with the tree for that reason.
- TRIPLET AUDIT of the t=0.99 gate failure: the -0.0005 is a NET of
  9 triplets (404 frontier-only-correct vs 395 recursive-only-
  correct, out of ~800 discordant placements flowing both ways in
  ~129k triplets), dominated by the deep-layer IT continuum labels
  (v2-003 L5/6 IT TPE-ENT and neighbors) whose evidence targets and
  input parentage are two partially-correct positions for genuinely
  ambiguous populations. No logical error; per the reviewer's
  decision tree this is an unavoidable near-tie: the recorded gate
  failure stands and the method proceeds labeled EXPLORATORY.
- FUNCTIONAL-RECOVERY GATE (test_recursive.py) PASSES: on a
  fragmented synthetic K=3 world the assembly reconnects the
  stranded component at the right clade, preserves all input
  ancestry, cycle-skips the mutual call, leaves the incomparable-
  parent clade unresolved (multi_parent ledger), creates no new
  equivalences, flags-but-never-moves audit conflicts, and is
  invariant to input insertion order. The gate also CAUGHT A REAL
  BUG: the core-vertex sentinel was the bare string "C", colliding
  with any dataset named C (now CORE = "__CORE__"). Allen results
  are bit-identical before/after the fix — the bug was latent, not
  expressed on v2/v3.

Run: PYTHONPATH=../../pkgs/python/src python recursive_frontier.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pkgs", "python",
                                "src"))
from metrics import (MyNode, cophenetic_spearman,  # noqa: E402
                     triplet_scores)

# core-vertex namespace sentinel: must be impossible as a dataset name
# (a bare "C" collided with a dataset named C — caught by the
# functional-recovery gate in test_recursive.py)
CORE = "__CORE__"
import containment_frontier as cf  # noqa: E402  (committed sweep runs)

labels = cf.labels
strict = cf.strict
cert_member = cf.cert_member
canon, itrees, decisions = cf.canon, cf.itrees, cf.decisions
REF = cf.REF
ALL_LABELS = sorted({f"{ds.replace('|','')}-{m.split('|',1)[-1]}"
                     for ds in labels for m in labels[ds]})
disp = lambda ds, n: f"{ds}-{n.split('|', 1)[-1]}"


def recursive_assemble(strict_nodes, cert_member_, itrees_, canon_,
                       decisions_, t):
    """The K-generic assembly. Vertices = canonical nodes with
    certified members contracted; ancestry preserved; only component
    roots attach, at their own call targets; a root with two or more
    INCOMPARABLE surviving targets is left unresolved (multi_parent
    ledger) rather than resolved; cycles are skipped and ledgered.
    Deterministic and insertion-order invariant (all iteration is
    sorted)."""
    datasets = sorted(itrees_)

    def cparent(ds, n):
        p = itrees_[ds]["parent"].get(n)
        while p not in (None, "root"):
            pc = canon_[ds].get(p, p)
            if pc != n:
                return pc
            p = itrees_[ds]["parent"].get(p)
        return None

    def image(ds, n):
        i = cert_member_.get((ds, n))
        return (CORE, i) if i is not None else (ds, n)

    parent_of, kind = {}, {}
    for i in sorted(strict_nodes):
        nd = strict_nodes[i]
        parent_of[(CORE, i)] = ((CORE, nd["parent"])
                               if nd["parent"] in strict_nodes
                               else "ROOT")
        kind[(CORE, i)] = "core"
    pending = []
    for ds in datasets:
        for n in sorted(set(canon_[ds].values())):
            v = image(ds, n)
            if v[0] == CORE:
                continue
            p = cparent(ds, n)
            if p is None:
                pending.append((ds, n))
            else:
                parent_of[v] = image(ds, p)
            kind[v] = "input"

    def find_top(v, extra):
        seen = set()
        while True:
            if v in seen:
                return None
            seen.add(v)
            p = extra.get(v, parent_of.get(v, "ROOT"))
            if p == "ROOT":
                return v
            v = p

    def calls_of(ds, n):
        out = []
        for other in datasets:
            if other == ds:
                continue
            rec = decisions_.get(f"{ds}>{other}", {}).get(n)
            if not rec or not rec.get("matched") or \
                    rec.get("selected") is None:
                continue
            supp = float(rec["support"])
            if supp < t:
                continue
            w = image(other, canon_[other].get(rec["selected"],
                                               rec["selected"]))
            out.append((supp, other, w))
        return sorted(out, key=lambda x: (-x[0], x[1], str(x[2])))

    def comparable(w1, w2, extra):
        a, x = set(), w1
        while x != "ROOT":
            a.add(x)
            x = extra.get(x, parent_of.get(x, "ROOT"))
        if w2 in a:
            return True
        x = w2
        while x != "ROOT":
            if x == w1:
                return True
            x = extra.get(x, parent_of.get(x, "ROOT"))
        return False

    # roots processed by best support desc (deterministic)
    order = sorted(pending,
                   key=lambda u: (-(calls_of(*u)[0][0]
                                    if calls_of(*u) else -1.0), u))
    attach_edges, cycle_skipped, multi_parent = {}, [], []
    for u in order:
        cands = calls_of(*u)
        if not cands:
            continue
        imgs = sorted({w for _s, _o, w in cands}, key=str)
        if len(imgs) > 1:
            pairwise_ok = all(comparable(a, b, attach_edges)
                              for i_, a in enumerate(imgs)
                              for b in imgs[i_ + 1:])
            if not pairwise_ok:
                multi_parent.append((u, imgs))
                continue
            # all comparable -> deepest (the one with most ancestors)
            def depth(w):
                d, x = 0, w
                while x != "ROOT":
                    d += 1
                    x = attach_edges.get(x, parent_of.get(x, "ROOT"))
                return d
            w = max(imgs, key=lambda x: (depth(x), str(x)))
        else:
            w = imgs[0]
        trial = dict(attach_edges)
        trial[u] = w
        if find_top(w, trial) is None:
            cycle_skipped.append((u, w, cands[0][0]))
            continue
        attach_edges[u] = w
    for u in pending:
        parent_of.setdefault(u, "ROOT")
    for u, w in attach_edges.items():
        parent_of[u] = w

    def ancestors_of(v):
        out, x = set(), parent_of.get(v, "ROOT")
        while x != "ROOT":
            out.add(x)
            x = parent_of.get(x, "ROOT")
        return out

    audit = {"consistent": 0, "conflict": 0}
    conflict_vertices = []
    for ds in datasets:
        for n in sorted(set(canon_[ds].values())):
            v = image(ds, n)
            if v[0] == CORE or (ds, n) in attach_edges:
                continue
            for _s, _o, w in calls_of(ds, n):
                if w == v or w in ancestors_of(v):
                    audit["consistent"] += 1
                else:
                    audit["conflict"] += 1
                    conflict_vertices.append((v, w))
    return {"parent_of": parent_of, "kind": kind, "pending": pending,
            "attach_edges": attach_edges,
            "cycle_skipped": cycle_skipped,
            "multi_parent": multi_parent, "audit": audit,
            "conflict_vertices": conflict_vertices}


def build(t):
    r = recursive_assemble(strict["nodes"], cert_member, itrees, canon,
                           decisions, t)
    parent_of = r["parent_of"]
    attach_edges = r["attach_edges"]
    audit, conflicted = r["audit"], r["cycle_skipped"]

    # ---- render to label space (structure kept) -------------------------
    kids = {}
    for v, p in parent_of.items():
        kids.setdefault(p, []).append(v)

    def name_of(v):
        if v[0] == CORE:
            nd = strict["nodes"][v[1]]
            pp = [disp(ds, m) for ds, m in sorted(nd["members"].items())
                  if m in labels[ds]]
            return "&".join(pp) if pp else f"__c{v[1]}__"
        ds, n = v
        return (disp(ds, n) if n in labels[ds]
                else f"__{ds}_{n}__")

    root = MyNode("root")
    depth_of_label = {}

    def emit(v, pn, d):
        node = MyNode(name_of(v))
        pn.addkid(node)
        if not name_of(v).startswith("__"):
            for lab in name_of(v).split("&"):
                depth_of_label[lab] = d
        for c_ in sorted(kids.get(v, []), key=str):
            emit(c_, node, d + 1)
    for v in sorted(kids.get("ROOT", []), key=str):
        emit(v, root, 1)

    placed = set()
    for u in attach_edges:
        stack = [u]
        while stack:
            v = stack.pop()
            if v[0] != CORE and v[1] in labels[v[0]]:
                placed.add(disp(*v))
            stack.extend(kids.get(v, []))
    for (ds, m) in cert_member:
        if m in labels[ds]:
            placed.add(disp(ds, m))
    # labels descending from core vertices without evidence attachment
    for i in strict["nodes"]:
        stack = list(kids.get((CORE, i), []))
        while stack:
            v = stack.pop()
            if v == "ROOT" or isinstance(v, str):
                continue
            if v[0] != CORE and v[1] in labels[v[0]]:
                placed.add(disp(*v))
            stack.extend(kids.get(v, []))
    return {"tree": root, "placed": placed,
            "n_attach": len(attach_edges),
            "conflicted_attach": conflicted, "audit": audit,
            "depths": depth_of_label}


def ref_subset(allowed):
    """Curated reference restricted to a display-label set."""
    truth, sub2 = cf.truth, cf.subclasses_v2
    root = MyNode("root")
    by = {}
    for cl, s in truth.items():
        by.setdefault(s, []).append(cl)
    for s in sorted(by):
        cls = [c for c in sorted(by[s]) if f"v3-{c}" in allowed]
        has2 = s in sub2 and f"v2-{s}" in allowed
        if not cls and not has2:
            continue
        if has2 and len(by[s]) == 1 and cls:
            root.addkid(MyNode(f"v2-{s}&v3-{cls[0]}"))
            continue
        sn = MyNode(f"v2-{s}" if has2 else f"__f_{s}__")
        root.addkid(sn)
        for cl in cls:
            sn.addkid(MyNode(f"v3-{cl}"))
    return root


def assembly_tree():
    import json
    tn = json.load(open(os.path.join(
        HERE, "..", "..", "pkgs", "python", "examples",
        "harmonize_demo", "metaarbor_tree_current.json")))
    kids, roots = {}, []
    for i, nd in tn.items():
        p = nd.get("parent")
        (roots if p is None else kids.setdefault(p, [])).append(i)
    root = MyNode("root")

    def rec_(i, pn):
        nd = tn[i]
        pp = [f"{ds}-{m.split('|', 1)[-1]}"
              for ds, m in sorted(nd.get("members", {}).items())
              if m in labels.get(ds, ())]
        node = MyNode("&".join(pp)) if pp else MyNode(f"__a{i}__")
        pn.addkid(node)
        for c_ in sorted(kids.get(i, [])):
            rec_(c_, node)
    for r in sorted(roots):
        rec_(r, root)
    return root


if __name__ == "__main__":
    ASM = assembly_tree()
    rows = []
    for t in (1.01, 1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.0):
        r = build(t)
        fhave = ({f"{ds}-{m.split('|',1)[-1]}" for (ds, m) in cert_member
                  if m in labels[ds]} |
                 {f"{ds}-{lab.split('|',1)[-1]}"
                  for ds, lab, _n, s_ in cf.attach if s_ >= t})
        R = ref_subset(fhave)
        rec_f, _ = triplet_scores(cf.frontier_tree(t, keep_structure=True),
                                  R)
        rec_r, _ = triplet_scores(r["tree"], R)
        rec_a, _ = triplet_scores(ASM, R)
        full_r, _ = triplet_scores(r["tree"], REF)
        pd = [d for l, d in r["depths"].items() if l in r["placed"]]
        rows.append({"threshold": t, "matched_labels": len(fhave),
                     "REC_frontier": round(rec_f, 4),
                     "REC_recursive": round(rec_r, 4),
                     "REC_assembly": round(rec_a, 4),
                     "REC_recursive_full": round(full_r, 4),
                     "attachments": r["n_attach"],
                     "mean_depth": round(sum(pd) / len(pd), 2) if pd else 0,
                     "max_depth": max(pd) if pd else 0,
                     "audit_conflicts": r["audit"]["conflict"],
                     "audit_consistent": r["audit"]["consistent"],
                     "cycle_skipped": len(r["conflicted_attach"])})
        print(f"t={t:<5} matched={len(fhave):3d}  frontier={rec_f:.4f} "
              f"recursive={rec_r:.4f} assembly={rec_a:.4f} | "
              f"recursive full(126)={full_r:.4f}  attach={r['n_attach']} "
              f"audit +{r['audit']['consistent']}/-{r['audit']['conflict']}"
              f" cyc={len(r['conflicted_attach'])}")
    with open(os.path.join(HERE, "recursive_frontier.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    verdict = all(r["REC_recursive"] >= r["REC_frontier"] - 1e-9
                  for r in rows)
    print(f"\nACCEPTANCE (recursive >= flat frontier at every t, matched "
          f"labels): {'PASS' if verdict else 'FAIL'}")
    print("wrote recursive_frontier.csv")

