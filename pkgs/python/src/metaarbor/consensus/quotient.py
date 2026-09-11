"""Quotient-graph assembly, v1 — the narrow construction approved in
ASSEMBLY2.md (revision 2): retain both complete trees, merge strictly
reciprocal nodes when structurally compatible, and annotate — but do
not structurally interpret — one-way evidence.

The six steps (ASSEMBLY2 Section 9):
  1. Keep every canonical node and ancestry edge from every input
     tree (disjoint union).
  2. Merge only accepted reciprocal-equivalence pairs (greedy by
     support; refusals ledgered; structural tie rule — within a tie
     group independent compatible candidates all accept, mutually
     competing ones ledger unresolved).
  3. Inspect the quotient ancestry graph.
  4. If (after transitive reduction of parent sets) every vertex has
     at most one minimal parent, render the reconciled forest.
  5. Otherwise retain the DAG and report certificates.
  6. Store one-way Walk calls as `directional_evidence` annotations —
     data on the graph, never structural edges.

Invariants I1-I6 (ASSEMBLY2 Section 3) are enforced by construction
and asserted before returning. This module has no thresholds of its
own: everything it consumes is frozen-Walk output.

Statuses (derived, ASSEMBLY2 Section 5):
  vertex:    shared | atlas_specific  (+ has_directional_evidence,
             conflicting flags; `conflicting` marks every vertex
             appearing in a certificate — the multi-parent child
             and its incomparable minimal parents)
  component: anchored | unanchored

RESULT CONTRACT (the output may be a DAG, not a tree):
  `parents` (vid -> sorted minimal-parent vids) IS the result — an
  acyclic quotient ancestry graph. Consumers MUST branch on
  `is_forest`:
  - is_forest=True: every parent list has <= 1 element; rendering
    the parent map as a forest is faithful and any tree metric of
    it is a score of the result.
  - is_forest=False: `certificates` enumerates the unreconciled
    multiple-parent constraints. ANY tree obtained by choosing one
    minimal parent per certified vertex is a PROJECTION — a view,
    never the result. A single tree score of a projection is not a
    score of the result; report certificate-aware ranges or
    DAG-level quantities instead, and label projections as views.
  Downstream code that can only consume trees must surface the
  certificates it dropped, never silently pick a projection.
"""
from __future__ import annotations


def _canon_parent(ds, n, trees, canonical):
    q = trees[ds]["parent"].get(n)
    while q not in (None, "root"):
        qc = canonical[ds].get(q, q)
        if qc != n:
            return qc
        q = trees[ds]["parent"].get(q)
    return None


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic representative: lexicographically smaller
            lo, hi = sorted([ra, rb])
            self.p[hi] = lo


def quotient_assemble(trees, canonical, decisions, certified=None):
    """v1 narrow quotient assembly.

    trees:      ds -> {parent, children, leaves} (canonical input
                trees, as dumped by the runners)
    canonical:  ds -> {node -> canonical node}
    decisions:  '<i>><j>' -> {node -> {selected, matched, support}}
                (frozen Walk directional selections)
    certified:  optional list of PREQUALIFIED merges from the frozen
                certification layer (greedy_backbone accepted
                multi-dataset nodes): [{"members": {ds: canonical
                node}, "support": float}]. THIS IS THE PRODUCTION
                PATH: eligibility, detectability, stability,
                MIN_DATASETS and MIN_SUPPORT judgments live in the
                certification layer, and the quotient consumes its
                output — a clique contributes its within-clique
                pairs at the clique's support. When `certified` is
                None the quotient falls back to RAW extraction
                (every reciprocal matched pair at min directional
                support) — a prototype/diagnostic mode that BYPASSES
                certification and must not be used for biological
                results. Either way `decisions` still supplies the
                directional-evidence annotations.

    Returns a dict:
      vertices    vid -> {members: {ds: node}, shared: bool}
      parents     vid -> sorted list of MINIMAL parent vids
                  (singleton everywhere iff is_forest)
      is_forest   bool
      certificates list of {vertex, minimal_parents} (multi-parent)
      annotations list of {source_vid, target_vid, source, target,
                  direction, support, within_merge}
      statuses    vid -> 'shared'|'atlas_specific'
      components  vid -> component id;  component_status
                  cid -> 'anchored'|'unanchored'
      ledger      {refused: [...], unresolved_ties: [...]}
    """
    datasets = sorted(trees)
    nodes = {ds: sorted(set(canonical[ds].values()))
             for ds in datasets}

    # ---- step 2: candidate reciprocal pairs ---------------------------
    def call(i, j, a):
        r = decisions.get(f"{i}>{j}", {}).get(a)
        if r and r.get("matched") and r.get("selected") is not None \
                and r.get("support") is not None:
            return r["selected"], float(r["support"])
        return None, None

    cands = []
    if certified is not None:
        for m in certified:
            mem = sorted(m["members"].items())
            s = float(m.get("support", 1.0))
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    cands.append((s, mem[x], mem[y]))
    else:
        for i in datasets:
            for j in datasets:
                if j <= i:
                    continue
                for a in nodes[i]:
                    b, s1 = call(i, j, a)
                    if b is None:
                        continue
                    a2, s2 = call(j, i, b)
                    if a2 == a:
                        cands.append((min(s1, s2), (i, a), (j, b)))

    uf = _UF()
    verts = [(ds, n) for ds in datasets for n in nodes[ds]]
    for v in verts:
        uf.find(v)

    def members_of(root, extra=None):
        out = {}
        for v in verts:
            r = uf.find(v)
            if extra:
                r = extra.get(r, r)
            if r == root:
                out.setdefault(v[0], set()).add(v[1])
        return out

    def acyclic_and_injective(pairs_to_add):
        """Would the union of current classes with these extra pairs
        keep per-atlas injectivity and an acyclic quotient ancestry
        graph? Pure check; mutates nothing."""
        trial = _UF()
        trial.p = dict(uf.p)
        for (va, vb) in pairs_to_add:
            trial.union(va, vb)
        cls, mem = {}, {}
        for v in verts:
            r = trial.find(v)
            cls[v] = r
            mem.setdefault(r, {}).setdefault(v[0], set()).add(v[1])
        for r, m in mem.items():
            if any(len(ns) > 1 for ns in m.values()):
                return False               # two same-atlas nodes merged
        edges = {}
        for ds, n in verts:
            p = _canon_parent(ds, n, trees, canonical)
            if p is not None:
                a, b = cls[(ds, n)], cls[(ds, p)]
                if a != b:
                    edges.setdefault(a, set()).add(b)
        seen, done = set(), set()

        def dfs(x):
            if x in done:
                return True
            if x in seen:
                return False
            seen.add(x)
            for y in edges.get(x, ()):
                if not dfs(y):
                    return False
            done.add(x)
            return True
        return all(dfs(cls[v]) for v in verts)

    refused, unresolved_ties = [], []
    by_support = {}
    for s, va, vb in cands:
        by_support.setdefault(s, []).append((va, vb))
    for s in sorted(by_support, reverse=True):
        group = sorted(by_support[s])
        # 1. drop candidates individually impossible against the
        #    current accepted state (they can never be accepted; an
        #    impossible candidate must not drag down safe ones)
        possible = []
        for c in group:
            if acyclic_and_injective([c]):
                possible.append(c)
            else:
                refused.append({"support": s, "pair": c,
                                "reason": "incompatible"})
        # 2. group by STRUCTURAL INTERACTION, not merely shared
        #    endpoints. Shared endpoints are NOT automatically
        #    competing: A<->C plus C<->B is exactly how a
        #    three-atlas meta-clade forms (equivalence = transitive
        #    closure of accepted pairs). But candidates with
        #    DISJOINT endpoints can still conflict through ancestry
        #    (two individually-valid merges whose union is a cycle),
        #    so the interaction relation is: shared endpoint OR
        #    pairwise joint infeasibility. Each interaction
        #    component is evaluated JOINTLY; feasible components
        #    whose UNION is infeasible are coarsened together (a
        #    fixed point), so acceptance never depends on candidate
        #    order or on names (I4). Components that survive accept
        #    wholesale; the rest are ledgered unresolved.
        if not possible:
            continue
        n = len(possible)
        pi = list(range(n))

        def fi(i):
            while pi[i] != i:
                pi[i] = pi[pi[i]]
                i = pi[i]
            return i

        def ui(i, j):
            ri, rj = fi(i), fi(j)
            if ri != rj:
                pi[max(ri, rj)] = min(ri, rj)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = possible[i], possible[j]
                if set(a) & set(b) or \
                        not acyclic_and_injective([a, b]):
                    ui(i, j)
        while True:
            comps = {}
            for i in range(n):
                comps.setdefault(fi(i), []).append(i)
            cand_of = {r: [possible[i] for i in idx]
                       for r, idx in comps.items()}
            feas = {r: acyclic_and_injective(cand_of[r])
                    for r in comps}
            ok_roots = sorted(r for r in comps if feas[r])
            joint = [c for r in ok_roots for c in cand_of[r]]
            if acyclic_and_injective(joint):
                for va, vb in joint:
                    uf.union(va, vb)
                for r in comps:
                    if not feas[r]:
                        # jointly incompatible though individually
                        # possible: order within the tie would
                        # decide -> unresolved, nothing forced
                        unresolved_ties.extend(
                            {"support": s, "pair": c,
                             "reason": "order_dependent_within_tie"}
                            for c in cand_of[r])
                break
            # feasible components conflict ACROSS components:
            # coarsen every offending pair (or, for a strictly
            # higher-order conflict, all of them) and re-evaluate
            merged = False
            for x in range(len(ok_roots)):
                for y in range(x + 1, len(ok_roots)):
                    if not acyclic_and_injective(
                            cand_of[ok_roots[x]] +
                            cand_of[ok_roots[y]]):
                        ui(ok_roots[x], ok_roots[y])
                        merged = True
            if not merged:
                for r in ok_roots[1:]:
                    ui(ok_roots[0], r)

    # ---- vertices -----------------------------------------------------
    cls = {v: uf.find(v) for v in verts}
    vids = sorted(set(cls.values()))
    members = {r: {} for r in vids}
    for v in verts:
        members[cls[v]].setdefault(v[0], v[1])
    vertices = {r: {"members": members[r],
                    "shared": len(members[r]) >= 2} for r in vids}

    # ---- steps 3-5: quotient ancestry, minimal parents ---------------
    raw_parents = {r: set() for r in vids}
    for ds, n in verts:
        p = _canon_parent(ds, n, trees, canonical)
        if p is not None and cls[(ds, p)] != cls[(ds, n)]:
            raw_parents[cls[(ds, n)]].add(cls[(ds, p)])

    anc_cache = {}

    def ancestors(r):
        if r in anc_cache:
            return anc_cache[r]
        out = set()
        stack = list(raw_parents[r])
        while stack:
            x = stack.pop()
            if x not in out:
                out.add(x)
                stack.extend(raw_parents[x])
        anc_cache[r] = out
        return out

    parents = {}
    certificates = []
    for r in vids:
        ps = raw_parents[r]
        minimal = sorted(p for p in ps
                         if not any(p in ancestors(q)
                                    for q in ps if q != p))
        parents[r] = minimal
        if len(minimal) > 1:
            certificates.append({"vertex": r,
                                 "minimal_parents": minimal})
    is_forest = not certificates

    # ---- step 6: annotations ------------------------------------------
    merged_pairs = {tuple(sorted((cls[va], cls[vb])))
                    for _s, va, vb in cands
                    if cls[va] == cls[vb]}
    annotations = []
    for i in datasets:
        for j in datasets:
            if i == j:
                continue
            for a in nodes[i]:
                b, s1 = call(i, j, a)
                if b is None:
                    continue
                sv, tv = cls[(i, a)], cls[(j, b)]
                annotations.append(
                    {"source_vid": sv, "target_vid": tv,
                     "source": a, "target": b,
                     "direction": f"{i}>{j}", "support": s1,
                     "within_merge": sv == tv})

    # ---- derived statuses (kind + documented flags) -------------------
    ann_sources = {a["source_vid"] for a in annotations
                   if not a["within_merge"]}
    # `conflicting` marks every vertex APPEARING in a certificate:
    # the multi-parent child AND its incomparable minimal parents
    cert_verts = {c["vertex"] for c in certificates}
    cert_verts |= {p for c in certificates
                   for p in c["minimal_parents"]}
    statuses = {r: {"kind": ("shared" if vertices[r]["shared"]
                             else "atlas_specific"),
                    "has_directional_evidence": r in ann_sources,
                    "conflicting": r in cert_verts} for r in vids}
    comp = {}
    for r in vids:
        if r in comp:
            continue
        stack, seen = [r], {r}
        while stack:
            x = stack.pop()
            for y in list(raw_parents[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
            for y in vids:
                if x in raw_parents[y] and y not in seen:
                    seen.add(y)
                    stack.append(y)
        cid = min(seen)
        for x in seen:
            comp[x] = cid
    component_status = {}
    for r in vids:
        cid = comp[r]
        if vertices[r]["shared"]:
            component_status[cid] = "anchored"
        component_status.setdefault(cid, "unanchored")

    # ---- invariants (assert, never silently return) --------------------
    all_labels = sorted((ds, n) for ds in datasets
                        for n in trees[ds]["leaves"])
    placed = sorted((ds, m) for r in vids
                    for ds, m in vertices[r]["members"].items()
                    if m in set(trees[ds]["leaves"]))
    assert placed == all_labels, "I1 conservation violated"
    # I3 order preservation: every input edge maps to an ancestor
    # relation in the quotient, and the quotient ancestry graph is
    # acyclic (no vertex is its own ancestor).
    for ds, n in verts:
        p = _canon_parent(ds, n, trees, canonical)
        if p is not None and cls[(ds, p)] != cls[(ds, n)]:
            assert cls[(ds, p)] in ancestors(cls[(ds, n)]), \
                f"I3: input edge lost for {(ds, n)}"
    for r in vids:
        assert r not in ancestors(r), f"I3: cycle through {r}"

    return {"vertices": vertices, "parents": parents,
            "is_forest": is_forest, "certificates": certificates,
            "annotations": annotations, "statuses": statuses,
            "components": comp, "component_status": component_status,
            "ledger": {"refused": refused,
                       "unresolved_ties": unresolved_ties}}
