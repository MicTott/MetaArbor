# Assembly redesign memo: quotient-graph synthesis (revision 2)

Status: MIGRATED — harmonize() assembles via consensus.quotient
(certified merges only) as of the migration commit; routing,
rejection fallback, completeness repair are deleted and interleave
is deprecated. Originally: Revision 2 incorporates the
reviewer's corrections to revision 1 (d7421bf), the most important
of which is SEMANTIC: revision 1 treated one-way Walk selections as
containment edges participating in the quotient preorder. That was
wrong, and this project's own c27 audit is the counterexample —
`c27 -> BC3B` at support 1.0 is an ARGMAX LANDING (c27's cells in
fact spread 0.44/0.34 across BC3B and BC4), not evidence that c27's
cells lie within BC3B. Revision 1's phrase calling that edge an
"honest containment record" was an overinterpretation and is
retracted here. One-way calls are `directional_evidence`
ANNOTATIONS; they never order the graph; a separately validated
containment rule may later promote a subset to the stronger word.

The evidence layer (frozen Walk, all molecular gates and
thresholds) is untouched throughout; everything here concerns what
happens AFTER directional evidence exists.

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
  `e_{i->j}(a) = (b, s)`: `b` is the supported Walk landing
  selected for `a`, at support `s`. This is DIRECTIONAL SIMILARITY
  EVIDENCE and does not itself establish equivalence or
  containment.

OUTPUT. An annotated quotient graph `G = (V, E_anc, A_dir, ~)`
plus, when it exists, its forest reduction:
- `~` : an equivalence relation on `⋃_k C_k`, generated ONLY by
  accepted reciprocal pairs (Section 4). `V = (⋃_k C_k) / ~`.
- `E_anc` : the image of every input parent edge. NEVER deleted.
  E_anc and the merges are the ONLY sources of order: the quotient
  preorder is built from ancestry alone.
- `A_dir` : one-way Walk selections stored as `directional_evidence`
  ANNOTATIONS `[a] ~~> [b]` with support. They are data attached to
  the graph — renderable, filterable, auditable — and are NEVER
  ordering edges. (A future, separately validated containment rule
  may promote a qualified subset; until then the honest name is
  `maps_to`.)
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
  reciprocal evidence at frozen thresholds; every `A_dir`
  annotation is a matched one-way selection carrying its support.
  Every output relationship carries a citable evidence record, and
  no annotation is presented with stronger semantics than the
  evidence supports (a selection is `maps_to`, not containment).
- I3 ORDER PRESERVATION. The quotient map is order-preserving on
  every input tree: `x <_k y` implies `[x] ≤ [y]`. A merge whose
  addition would create a cycle or an order reversal is REFUSED and
  ledgered — never linearized, never silently resolved.
- I4 DETERMINISM. Output is invariant to dataset insertion order
  AND to label renaming; ties break by support then by structural /
  evidence-derived criteria only (Section 4); exact ties are
  ledgered unresolved, never broken by name.
- I5 CONFLICT TRANSPARENCY. If the combined preorder is not a
  forest, the output is the DAG together with minimal certificates
  (the offending cycle, or the incomparable-minimal-ancestor set).
  A tree is returned only when a tree exists.
- I6 DERIVED STATUS. Vertex statuses (`shared`, `atlas_specific`,
  `has_directional_evidence`, `conflicting`) and component statuses
  (`anchored`, `unanchored`) are functions of `G` (Section 5),
  never side effects of processing order.

## 4. Acceptance (the one place judgment lives)

Candidate merges are reciprocal pairs
`e_{i->j}(a) = (b, s1)` and `e_{j->i}(b) = (a, s2)` passing the
frozen gates. The acceptance problem is: choose a subset of candidates whose
generated `~` satisfies I3 and per-atlas injectivity (no vertex
holds two nodes of one atlas). The shipped algorithm is greedy in
descending support and yields a DETERMINISTIC INCLUSION-MAXIMAL
compatible set — no maximum-weight or approximation-ratio claim is
made or needed (none is proven). Tie-breaking must be structural or
evidence-derived, never by label/canonical NAME (name ties violate
label-renaming invariance). Within a tie group, INDEPENDENT
compatible candidates (sharing no endpoint and jointly compatible)
are ALL accepted; only MUTUALLY COMPETING tied candidates — those
sharing an endpoint slot, or whose outcome depends on acceptance
order within the tie — are ledgered unresolved. (Supports of
exactly 1.0 are common; a broad tie rule would discard many valid
merges.)

The reconciled-tier idea (same-branch near-miss pairs, e.g. the
truth-certified retina n04<->n10 case) is NOT a mere parameter of
this predicate: it changes what counts as equivalence EVIDENCE.
It remains a separately specified, separately validated extension —
out of scope for the v1 implementation below.

## 5. Statuses become theorems, not branches

Two levels, kept distinct (a single-atlas vertex beneath a shared
ancestor — c27 — is atlas-specific yet anchored):
VERTEX statuses:
- `shared`         : vertex with members from ≥ 2 atlases.
- `atlas_specific` : vertex whose members come from one atlas
                     (regardless of where it sits).
- `has_directional_evidence` : source of an `A_dir` annotation —
                     the current one-way frontier and "affiliates",
                     unified as annotation (NOT a containment
                     claim).
- `conflicting`    : vertex appearing in a certificate.
COMPONENT statuses (under `E_anc` + merges):
- `anchored`       : component containing ≥ 1 shared vertex.
- `unanchored`     : component with none — the current "unplaced",
                     now simply a disconnected component.

## 6. When a single tree is impossible

The reduction to a forest exists iff the quotient preorder
(`E_anc` edges as ≤, quotient by `~`) is a forest order. The two
failure certificates:
- CYCLE: reciprocal merges identifying nodes in ways incompatible
  with the original ancestry orders (a directed cycle through
  `E_anc` under the quotient).
- MULTIPLE INCOMPARABLE MINIMAL ANCESTORS: a vertex constrained
  below two ≤-incomparable vertices. NOTE ON SCOPE: with ancestry
  as the only order source, this certificate arises from merge
  interactions; single-target directional maps cannot by themselves
  express many-to-many structure, so the quotient graph does NOT
  certify crosscutting partitions (the retina transfer matrix
  SUGGESTS mac/she OFF crosscutting, but that is a cell-level
  observation outside this graph's constraint language). Claiming a
  crosscutting certificate would require many-to-many constraints
  the current evidence maps do not carry — deliberately out of
  scope.
Return in that case: the DAG, the certificates, and (for display
only, ledgered) a maximal-forest view obtained by withholding the
minimal-support MERGES — clearly labeled as a view, never as the
result. (`A_dir` annotations never participate: they carry no
order.)

## 7. What the current rules become

| current rule                    | under quotient assembly        |
|---------------------------------|--------------------------------|
| AUROC/margin/compactness/boot   | unchanged (evidence layer)     |
| canonical-node matching         | unchanged (defines `C_k`)      |
| ancestry compatibility          | I3 — THE acceptance check      |
| dataset-set uniqueness          | per-atlas injectivity in I3    |
| candidate ordering (greedy)     | deterministic inclusion-       |
|                                 | maximal selection (Section 4)  |
| affiliates                      | `A_dir` annotations (demoted)  |
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
`[n01,n10]` under `n04` under `[n06,n12]` automatically. This is EXPECTED to reproduce the interleaved
result (truth TRIP 51/51 on the mac side) with no interleaving
rule in existence; exact reproduction is a MIGRATION GATE, not yet
a fact. c27's support-1.0
call to BC3B persists as a `directional_evidence` ANNOTATION —
which is all the per-cell audit licenses (cells spread 0.44/0.34
across BC3B/BC4; the revision-1 phrase 'honest containment record'
is retracted).

AMYGDALA INHIBITORY: 58 components with sparse reciprocity emerge
as exactly that — a forest of components plus `A_dir` annotations.
The "containment frontier" stops being a construction and becomes a
RENDERING of annotations at a support threshold — explicitly a
view, carrying `maps_to` semantics. Nothing is forced; `unanchored`
is a fact of the graph.

CROSSCUTTING (retina OFF level): NOT certified by this graph (see
Section 6) — the transfer matrix's crosscutting observation lives
at the cell level, outside the graph's constraint language, and is
reported as such.

## 9. The half-page algorithm (paper form, v1 = the reviewer's
narrow quotient)

1. Keep every node and ancestry edge from every input tree
   (disjoint union of canonical trees).
2. Merge only accepted reciprocal-equivalence nodes (greedy by
   support; refusals ledgered; structural tie rule, exact ties
   unresolved).
3. Inspect the quotient ancestry graph.
4. If it is a forest, render the reconciled forest.
5. If not, retain the DAG and report the incompatible
   relationships as certificates.
6. Store one-way Walk calls as `directional_evidence` annotations —
   data on the graph, never structural edges.
Every input label is present by construction; every relationship
carries its evidence record; nothing is repaired because nothing is
discarded; nothing is asserted beyond what the evidence semantics
support.

## 10. Migration and gates (when implementation is approved)

- The evidence layer and all frozen thresholds are untouched.
- Regression gates: retina C1-C4 with the quotient assembly must
  reproduce the interleaved result (TRIP 1291/1291) with the
  interleaving module deleted; Allen must reproduce the committed
  determinism dump's derived statuses (or the supersession is
  documented node-by-node); amygdala structural counts compared and
  explained; the full synthetic suite (structural review, cut,
  interleave-shaped cases) re-expressed against the invariants.
- The reconciled-tier extension (new equivalence-evidence class)
  and any promotion of `directional_evidence` to genuine
  containment are SEPARATE, subsequently specified and validated
  changes — neither ships in v1.
- Freeze point: after migration, the acceptance predicate and the
  reduction rule are frozen together; new biology must never add a
  branch to the assembly again — if it cannot be expressed as
  evidence (`e`), acceptance, or rendering, it does not belong.

## 11. Result contract (the output may be a DAG)

The authoritative result is the minimal-parent map over vertices —
an acyclic quotient ancestry graph — and consumers MUST branch on
`is_forest`. When `is_forest` is true, every vertex has at most one
minimal parent, the forest rendering is faithful, and a tree metric
of it scores the result. When false, the certificates enumerate the
unreconciled multiple-parent constraints; any tree obtained by
choosing one minimal parent per certified vertex is a PROJECTION —
a view, never the result — and a single tree score of a projection
is not a score of the result: report certificate-aware ranges over
compatible projections or DAG-level quantities, and label
projections as views. Downstream code that can only consume trees
must surface the certificates it dropped, never silently pick a
projection. The `conflicting` vertex flag marks every vertex
appearing in a certificate: the multi-parent child and its
incomparable minimal parents. Tie acceptance groups candidates by
STRUCTURAL INTERACTION (shared endpoints or joint infeasibility,
coarsened to a fixed point), not merely shared endpoints, so
disjoint-endpoint ancestry conflicts within a tie are ledgered
unresolved rather than resolved by processing order (I4).

## 12. Ontology freeze (design principle, adopted 2026-09-11)

The user-facing ontology is FOUR statuses and no more:
- merged/shared: reciprocal evidence passed the single
  certification rule;
- dataset-specific: not merged;
- unresolved constraint: ancestry from merged trees cannot be
  represented uniquely (a certificate);
- directional annotation: evidence points somewhere but does not
  justify a merge.

New evidence classes are REJECTED as a design principle. Every
future improvement — balanced subsampling, compactness variants,
alternative contrasts, same-branch handling — must arrive as a
globally validated change to the UNCERTAINTY ESTIMATE or to the
single certification rule, never as a new named relationship
category or decision branch. If an improved estimator makes a
pair's reciprocal evidence pass the same rule, it merges; if not,
it abstains; the measurements explaining the outcome are recorded,
unlabeled. This supersedes the Section 4 reserved "reconciled-tier
evidence class": the same-branch near-miss cases follow the same
path as every other pair. Methods evolve internally; the exposed
ontology stays small and stable.

Balanced within-comparison subsampling is adopted as a DIAGNOSTIC
(does a correspondence survive equal-sized comparison; is a failure
caused by imbalance; at what per-type cell count does certification
power collapse), evaluated by a prespecified synthetic experiment
(downsampled true reciprocal pairs at 20/30/50/100 cells, planted
absent-partner and near-neighbor cases; scored on correct
certification, abstention, false certification) — never as a rescue
tier. Expectation stated in advance: BC1A/BC1B are power
near-misses and balancing will characterize, not fix, the limit.
