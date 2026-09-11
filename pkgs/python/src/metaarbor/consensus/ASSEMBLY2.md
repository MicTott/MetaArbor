# Assembly redesign memo: quotient-graph synthesis

Status: DESIGN ONLY — no code changed. Written 2026-09-10 at the
reviewer's request, after the retina K=2 campaign exposed the c27
splice-hoist and its interleaving repair. The evidence layer (frozen
Walk, all molecular gates and thresholds) is untouched throughout
this memo; everything here concerns what happens AFTER directional
evidence exists.

## 1. The problem with the current assembly

The current assembly is a cascade of routing rules — candidate
ordering, affiliates, private consolidation, rejection fallback,
completeness repair, and now core-ancestry interleaving. Each rule
is individually defensible and test-covered, but collectively they
are procedural, and the c27 failure showed the deep reason: the
assembler DELETES structure (unmerged internal nodes) and then needs
repair machinery to put it back. A construction in which structure
is never deleted cannot have that failure class.

## 2. Formal objects

INPUT.
- K datasets. Dataset k has label set `L_k`, a rooted input tree
  `T_k` over `L_k` with internal nodes `N_k`, and bootstrap support
  `s_k : N_k -> [0,1]`. Canonical nodes `C_k` = the quotient of
  `N_k ∪ L_k` by unary chains (existing machinery).
- Directional evidence: for each ordered pair (i, j), a partial map
  `e_{i->j} : C_i ⇀ C_j × [0,1]` — the frozen Walk's selection and
  support, surviving all molecular gates. Semantics of
  `e_{i->j}(a) = (b, s)`: the cells of `a` lie within `b`
  (containment-or-equality), asserted from atlas i's evidence, at
  support `s`.

OUTPUT. An annotated quotient graph `G = (V, E_anc, E_dir, ~)` plus,
when it exists, its forest reduction:
- `~` : an equivalence relation on `⋃_k C_k`, generated ONLY by
  accepted reciprocal pairs (Section 4). `V = (⋃_k C_k) / ~`.
- `E_anc` : the image of every input parent edge. NEVER deleted.
- `E_dir` : accepted one-way evidence edges `[a] -> [b]` with
  support, endpoints NOT merged.
- A conflict ledger of refused merges and incompatibility
  certificates (Section 6).
- Statuses are DERIVED from `G` (Section 5), not produced by
  routing.

## 3. Invariants

- I1 CONSERVATION. Every input label occurs in exactly one vertex;
  every input ancestry edge has an image path in `E_anc`.
  Completeness holds by construction; there is no repair step
  because deletion is impossible.
- I2 EVIDENCE GROUNDING. Every merge is licensed by accepted
  reciprocal evidence at frozen thresholds; every `E_dir` edge is a
  matched one-way selection. Every output relationship carries a
  citable evidence record.
- I3 ORDER PRESERVATION. The quotient map is order-preserving on
  every input tree: `x <_k y` implies `[x] ≤ [y]`. A merge whose
  addition would create a cycle or an order reversal is REFUSED and
  ledgered — never linearized, never silently resolved.
- I4 DETERMINISM. Output is invariant to dataset insertion order
  and label ordering; ties break by (support desc, canonical name).
- I5 CONFLICT TRANSPARENCY. If the combined preorder is not a
  forest, the output is the DAG together with minimal certificates
  (the offending cycle, or the incomparable-minimal-ancestor set).
  A tree is returned only when a tree exists.
- I6 DERIVED STATUS. `shared` / `atlas_specific` / `contained` /
  `conflicting` / `unresolved` are functions of `G` (Section 5),
  never side effects of processing order.

## 4. Acceptance (the one place judgment lives)

Candidate merges are reciprocal pairs
`e_{i->j}(a) = (b, s1)` and `e_{j->i}(b) = (a, s2)` passing the
frozen gates. The acceptance problem is: choose a maximal-weight
subset of candidates whose generated `~` satisfies I3 and per-atlas
injectivity (no vertex holds two nodes of one atlas). This is
combinatorial; the shipped algorithm is greedy by (min support desc,
name) — STATED as an approximation to the maximal consistent set,
which is what the current greedy_backbone already is, minus its
routing duties.

The reconciled-tier extension (same-branch near-miss pairs:
`e_{i->j}(a) = b` with `e_{j->i}(b)` an ancestor/descendant of `a`
on one root path — truth-certified by the retina n04<->n10 case)
changes ONLY this acceptance predicate. It adds no assembly
machinery, which is the cleanest argument for this formulation.

## 5. Statuses become theorems, not branches

- `shared`         : vertex with members from ≥ 2 atlases.
- `atlas_specific` : vertex whose entire component (under `E_anc`)
                     touches one atlas — the current "private" and
                     ordinary single-atlas subtrees, unified.
- `contained`      : source of an accepted `E_dir` edge — the
                     current one-way frontier and "affiliates",
                     unified as directional annotation.
- `conflicting`    : vertex appearing in a certificate.
- `unresolved`     : root of a component with no accepted incident
                     cross-atlas relationship — the current
                     "unplaced", now just a disconnected component.

## 6. When a single tree is impossible

The reduction to a forest exists iff the quotient preorder
(`E_anc` edges as ≤, quotient by `~`) is a forest order. The two
failure certificates:
- CYCLE: mutual containment claims across incomparable clades
  (through any mix of `E_anc` and accepted merges).
- MULTIPLE INCOMPARABLE MINIMAL ANCESTORS: a vertex constrained
  below two ≤-incomparable vertices — the signature of CROSSCUTTING
  partitions. The retina transfer matrix shows this is real
  biology, not an edge case: Macosko's OFF clusters crosscut
  Shekhar's OFF types (BC2/BC3A/BC4/BC1A all majority-assign to
  c28), so a fully resolved common tree of both partitions does
  not exist below the OFF level.
Return in that case: the DAG, the certificates, and (for display
only, ledgered) a maximal-forest view obtained by dropping the
minimal-support `E_dir`/merge edges — clearly labeled as a view,
never as the result.

## 7. What the current rules become

| current rule                    | under quotient assembly        |
|---------------------------------|--------------------------------|
| AUROC/margin/compactness/boot   | unchanged (evidence layer)     |
| canonical-node matching         | unchanged (defines `C_k`)      |
| ancestry compatibility          | I3 — THE acceptance check      |
| dataset-set uniqueness          | per-atlas injectivity in I3    |
| candidate ordering (greedy)     | I4 tie-breaking, stated as     |
|                                 | approximation (Section 4)      |
| affiliates                      | `E_dir` edges (demoted)        |
| private cand. + consolidation   | derived `atlas_specific`       |
| rejection fallback routing      | GONE — nothing ever leaves     |
| completeness repair             | GONE — I1 by construction      |
| core-ancestry interleaving      | GONE — `E_anc` never deleted   |
| unplaced handling               | derived `unresolved` component |

Three mechanisms disappear outright; four demote to derived
concepts; the judgment concentrates in one acceptance predicate.

## 8. The worked examples

RETINA c27 (the motivating failure): `n04` is a vertex from the
start; c27's ancestry edge into it is never deleted. The accepted
merge `[n01, n10]` has ancestry constraints `[n01] ≤ n04 ≤ [n06]`
(mac chain) and `[n01] ≤ [n06]` (she chain, since n10's parent n12
merges with n06) — compatible; the reduction nests
`[n01,n10]` under `n04` under `[n06,n12]` automatically. This is
byte-for-byte the interleaved result (truth TRIP 51/51 on the mac
side) with no interleaving rule in existence. c27's support-1.0 call
to BC3B persists as an `E_dir` edge — the honest containment record
the per-cell audit endorsed.

AMYGDALA INHIBITORY: 58 components with sparse reciprocity emerge as
exactly that — a forest of components plus `E_dir` edges. The
"containment frontier" stops being a construction and becomes a
RENDERING of `E_dir` at a support threshold. Nothing is forced;
`unresolved` is a fact of the graph.

CROSSCUTTING (retina OFF level): the mac and she OFF partitions
yield incomparable-minimal-ancestor certificates below the OFF
vertex — the correct mathematical statement of what the transfer
matrix measured. Current machinery has no way to SAY this; the
quotient output states it as a certificate.

## 9. The half-page algorithm (paper form)

1. Form the disjoint union of the K canonical input trees; keep
   every ancestry edge.
2. Accept reciprocal evidence pairs greedily by support, refusing
   any merge that would violate order preservation or per-atlas
   injectivity (refusals ledgered); quotient by the accepted pairs.
3. Add every surviving one-way selection as a directional
   containment edge, endpoints unmerged.
4. If the resulting preorder is a forest, emit the reconciled
   forest; otherwise emit the DAG with minimal conflict
   certificates. Statuses (shared, atlas-specific, contained,
   conflicting, unresolved) are read off the graph.
Every input label is present by construction; every relationship in
the output carries its evidence record; nothing is ever repaired
because nothing is ever discarded.

## 10. Migration and gates (when implementation is approved)

- The evidence layer and all frozen thresholds are untouched.
- Regression gates: retina C1-C4 with the quotient assembly must
  reproduce the interleaved result (TRIP 1291/1291) with the
  interleaving module deleted; Allen must reproduce the committed
  determinism dump's derived statuses (or the supersession is
  documented node-by-node); amygdala structural counts compared and
  explained; the full synthetic suite (structural review, cut,
  interleave-shaped cases) re-expressed against the invariants.
- The reconciled-tier acceptance extension is a separate,
  subsequently-gated change to Section 4 only.
- Freeze point: after migration, the acceptance predicate and the
  reduction rule are frozen together; new biology must never add a
  branch to the assembly again — if it cannot be expressed as
  evidence (`e`), acceptance, or rendering, it does not belong.
