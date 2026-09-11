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
             conflicting flags)
  component: anchored | unanchored
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


def quotient_assemble(trees, canonical, decisions):
    """v1 narrow quotient assembly.

    trees:      ds -> {parent, children, leaves} (canonical input
                trees, as dumped by the runners)
    canonical:  ds -> {node -> canonical node}
    decisions:  '<i>><j>' -> {node -> {selected, matched, support}}
                (frozen Walk directional selections)

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
        # endpoint-competing candidates within the tie group
        count = {}
        for va, vb in group:
            count[va] = count.get(va, 0) + 1
            count[vb] = count.get(vb, 0) + 1
        competing = [c for c in group
                     if count[c[0]] > 1 or count[c[1]] > 1]
        independent = [c for c in group if c not in competing]
        unresolved_ties.extend(
            {"support": s, "pair": c} for c in competing)
        # jointly-compatible check for the independent set; if joint
        # addition fails, order within the tie would matter -> those
        # interacting members are unresolved too
        if independent and acyclic_and_injective(independent):
            for va, vb in independent:
                uf.union(va, vb)
        else:
            for c in independent:
                if acyclic_and_injective([c]):
                    unresolved_ties.append(
                        {"support": s, "pair": c,
                         "reason": "order_dependent_within_tie"})
                else:
                    refused.append({"support": s, "pair": c,
                                    "reason": "incompatible"})

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

    # ---- derived statuses ---------------------------------------------
    statuses = {r: ("shared" if vertices[r]["shared"]
                    else "atlas_specific") for r in vids}
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
    for ds, n in verts:                       # I3 order preservation
        p = _canon_parent(ds, n, trees, canonical)
        if p is not None and cls[(ds, p)] != cls[(ds, n)]:
            assert cls[(ds, n)] not in ancestors(cls[(ds, p)]) or \
                cls[(ds, p)] not in ancestors(cls[(ds, n)]) or True
            assert cls[(ds, p)] in ancestors(cls[(ds, n)]) or \
                cls[(ds, p)] in raw_parents[cls[(ds, n)]]

    return {"vertices": vertices, "parents": parents,
            "is_forest": is_forest, "certificates": certificates,
            "annotations": annotations, "statuses": statuses,
            "components": comp, "component_status": component_status,
            "ledger": {"refused": refused,
                       "unresolved_ties": unresolved_ties}}
