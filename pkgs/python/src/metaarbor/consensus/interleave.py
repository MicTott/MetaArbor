"""DEPRECATED (quotient migration): harmonize() now assembles via
consensus.quotient, which never deletes input ancestry, so the
splice-and-repair failure class this module fixed cannot occur and
harmonize output needs no interleaving. The module is retained
unchanged solely so the committed retina K=2 audit scripts keep
running against their historical legacy-assembly dumps. Do not use
in new code.

Core-ancestry interleaving: a deterministic post-assembly
transformation of a harmonize() tree that restores input-tree
ancestry the assembly spliced out.

Motivation (retina K=2 benchmark, commit 65c5457): when an input
tree's internal node lies between two represented nodes but is
itself unrepresented, the assembly splices it, hoisting its other
children to the grandparent (the single C3 failure: mac|c27). This
function re-derives parentage from the canonical input trees,
materializing unrepresented intermediates as single_atlas nodes
where they carry structure.

RULES (fixed; discovered on retina, no-harm-checked on Allen —
see retina_k2_audit.py):
- Single-member nodes follow their member's canonical input
  ancestry; unrepresented intermediates on the path materialize.
- Multi-member nodes move ONLY when exactly one member atlas
  supplies unrepresented intermediate ancestors AND every member
  atlas's implied represented parent equals the node's current
  parent (no contradiction). Nodes with >= 2 atlases supplying
  distinct chains are left in place and ledgered.
- Materialized intermediates that end up with a single child are
  collapsed (no unary bloat).
- The transformation NEVER changes memberships, statuses, aliases,
  display strings, or support of existing nodes — parents only,
  plus new "MA-I" nodes. Deterministic and insertion-order
  invariant (all iteration sorted).

This is NOT part of frozen harmonize(); callers opt in:
    new_nodes, ledger = interleave_core_ancestry(nodes, input_trees,
                                                 canonical)
"""
from __future__ import annotations


def _canon_parent(ds, n, input_trees, canonical):
    q = input_trees[ds]["parent"].get(n)
    while q not in (None, "root"):
        qc = canonical[ds].get(q, q)
        if qc != n:
            return qc
        q = input_trees[ds]["parent"].get(q)
    return None


def interleave_core_ancestry(nodes, input_trees, canonical):
    """Return (new_nodes, ledger). `nodes`: harmonize-style mapping
    id -> {parent, status, members, ...} (extra keys preserved).
    `input_trees`: ds -> {parent, children, leaves}. `canonical`:
    ds -> {node -> canonical node}."""
    rep = {}
    for i in sorted(nodes):
        for ds, m in nodes[i].get("members", {}).items():
            rep[(ds, m)] = i

    def chain_to_rep(ds, m):
        """Unrepresented canonical ancestors of m (nearest first) and
        the first represented ancestor's node id (or None)."""
        ch, x = [], _canon_parent(ds, m, input_trees, canonical)
        while x is not None and (ds, x) not in rep:
            ch.append((ds, x))
            x = _canon_parent(ds, x, input_trees, canonical)
        return ch, (rep[(ds, x)] if x is not None else None)

    # desired parent per existing node, in vertex space:
    # vertex = node id (str) or intermediate (ds, canon-node) tuple
    parent_v = {}
    ledger = []
    for i in sorted(nodes):
        mem = sorted(nodes[i].get("members", {}).items())
        if not mem:
            parent_v[i] = nodes[i].get("parent")
            continue
        chains = {ds: chain_to_rep(ds, m) for ds, m in mem}
        nonempty = {ds: c for ds, (c, _p) in chains.items() if c}
        cur = nodes[i].get("parent")
        implied_ok = all(p == cur for _c, p in chains.values())
        if len(nonempty) == 1 and implied_ok:
            ds, ch = next(iter(nonempty.items()))
            parent_v[i] = ch[0]
            # thread the chain toward the current parent
            for a, b in zip(ch, ch[1:]):
                parent_v.setdefault(a, b)
            parent_v.setdefault(ch[-1], cur)
        else:
            if len(nonempty) >= 2:
                ledger.append(i)
            parent_v[i] = cur

    # collapse unary intermediates: count children per vertex
    child_ct = {}
    for v, p in parent_v.items():
        if isinstance(p, tuple):
            child_ct[p] = child_ct.get(p, 0) + 1

    def resolve(p):
        while isinstance(p, tuple) and child_ct.get(p, 0) < 2:
            p = parent_v[p]
        return p

    keep = {v for v in parent_v if isinstance(v, tuple)
            and child_ct.get(v, 0) >= 2}
    out = {}
    inter_id = {}
    for k, v in enumerate(sorted(keep)):
        inter_id[v] = f"MA-I{k + 1:04d}"
    for i in sorted(nodes):
        nd = dict(nodes[i])
        p = resolve(parent_v[i])
        nd["parent"] = inter_id[p] if isinstance(p, tuple) else p
        out[i] = nd
    for v in sorted(keep):
        ds, n = v
        p = resolve(parent_v[v])
        out[inter_id[v]] = {"parent": (inter_id[p]
                                       if isinstance(p, tuple) else p),
                            "status": "single_atlas",
                            "members": {ds: n}, "aliases": [n],
                            "display": n,
                            "interleaved": True}
    # ---- invariants (raise, never silently return a bad tree) ----------
    for i in nodes:                      # originals: parents only
        for k in nodes[i]:
            if k != "parent":
                assert out[i][k] == nodes[i][k], (i, k)
    seen_labels = sorted((ds, m) for j in nodes
                         for ds, m in nodes[j]["members"].items())
    seen_after = sorted((ds, m) for j in out
                        for ds, m in out[j]["members"].items()
                        if not out[j].get("interleaved"))
    assert seen_labels == seen_after
    for i in out:                        # acyclic
        seen, x = set(), i
        while x is not None:
            assert x not in seen, f"cycle at {i}"
            seen.add(x)
            x = out[x].get("parent")
    return out, ledger
