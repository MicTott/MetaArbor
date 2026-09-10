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
  does, neither helps nor hurts on matched labels).

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
import containment_frontier as cf  # noqa: E402  (committed sweep runs)

labels = cf.labels
strict = cf.strict
cert_member = cf.cert_member
canon, itrees, decisions = cf.canon, cf.itrees, cf.decisions
REF = cf.REF
ALL_LABELS = sorted({f"{ds.replace('|','')}-{m.split('|',1)[-1]}"
                     for ds in labels for m in labels[ds]})
disp = lambda ds, n: f"{ds}-{n.split('|', 1)[-1]}"


def cparent(ds, n):
    """Canonical parent of canonical node n in its input tree."""
    p = itrees[ds]["parent"].get(n)
    while p not in (None, "root"):
        pc = canon[ds].get(p, p)
        if pc != n:
            return pc
        p = itrees[ds]["parent"].get(p)
    return None


def image(ds, n):
    i = cert_member.get((ds, n))
    return ("C", i) if i is not None else (ds, n)


def build(t):
    # ---- vertices and ancestry ------------------------------------------
    parent_of, kind = {}, {}
    for i, nd in strict["nodes"].items():
        parent_of[("C", i)] = (("C", nd["parent"])
                               if nd["parent"] in strict["nodes"]
                               else "ROOT")
        kind[("C", i)] = "core"
    pending = []                       # component roots awaiting evidence
    for ds in ("v2", "v3"):
        for n in sorted(set(canon[ds].values())):
            v = image(ds, n)
            if v[0] == "C":
                continue
            p = cparent(ds, n)
            if p is None:
                pending.append((ds, n))
            else:
                parent_of[v] = image(ds, p)
            kind[v] = "input"

    # ---- evidence attachments for component roots -----------------------
    def find_top(v, extra):
        seen = set()
        while True:
            if v in seen:
                return None            # cycle sentinel
            seen.add(v)
            p = extra.get(v, parent_of.get(v, "ROOT"))
            if p == "ROOT":
                return v
            v = p

    cands = []
    for ds, n in pending:
        other = "v3" if ds == "v2" else "v2"
        rec = decisions.get(f"{ds}>{other}", {}).get(n)
        if not rec or not rec.get("matched") or rec.get("selected") is None:
            continue
        supp = float(rec["support"])
        if supp < t:
            continue
        w = image(other, canon[other].get(rec["selected"],
                                          rec["selected"]))
        cands.append((-supp, ds, n, w))
    cands.sort()
    attach_edges, conflicted = {}, []
    for negs, ds, n, w in cands:
        u = (ds, n)
        trial = dict(attach_edges)
        trial[u] = w
        if find_top(w, trial) is None:
            conflicted.append((u, w, -negs))
            continue
        attach_edges[u] = w
    for ds, n in pending:
        parent_of.setdefault((ds, n), "ROOT")
    for u, w in attach_edges.items():
        parent_of[u] = w

    # ---- post-assembly evidence audit (flag, never move) ----------------
    def ancestors_of(v):
        out, x = set(), parent_of.get(v, "ROOT")
        while x != "ROOT":
            out.add(x)
            x = parent_of.get(x, "ROOT")
        return out

    audit = {"consistent": 0, "conflict": 0}
    conflict_vertices = []
    for ds in ("v2", "v3"):
        other = "v3" if ds == "v2" else "v2"
        for n in sorted(set(canon[ds].values())):
            v = image(ds, n)
            if v[0] == "C" or (ds, n) in attach_edges:
                continue
            rec = decisions.get(f"{ds}>{other}", {}).get(n)
            if not rec or not rec.get("matched") or \
                    rec.get("selected") is None:
                continue
            w = image(other, canon[other].get(rec["selected"],
                                              rec["selected"]))
            if w == v or w in ancestors_of(v):
                audit["consistent"] += 1
            else:
                audit["conflict"] += 1
                conflict_vertices.append(v)

    # ---- render to label space (structure kept) -------------------------
    kids = {}
    for v, p in parent_of.items():
        kids.setdefault(p, []).append(v)

    def name_of(v):
        if v[0] == "C":
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
            if v[0] != "C" and v[1] in labels[v[0]]:
                placed.add(disp(*v))
            stack.extend(kids.get(v, []))
    for (ds, m) in cert_member:
        if m in labels[ds]:
            placed.add(disp(ds, m))
    # labels descending from core vertices without evidence attachment
    for i in strict["nodes"]:
        stack = list(kids.get(("C", i), []))
        while stack:
            v = stack.pop()
            if v == "ROOT" or isinstance(v, str):
                continue
            if v[0] != "C" and v[1] in labels[v[0]]:
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
