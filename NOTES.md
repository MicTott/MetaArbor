# Prototype notes — findings that feed back into DESIGN.md

2026-09-02, first prototype session. Kernel (`R/kernel.R`), tree utilities
(`R/tree.R`), baseline estimator (`R/baseline.R`), simulation (`R/simulate.R`);
validated by `tests/test_kernel.R`, `analysis/01_simulation_validation.R`,
`analysis/02_edge_cases.R`. All gates pass.

## Findings

1. **Root neutrality is exact, and it kills the root-wins problem.** With mean
   rank-standardized votes, the union of all training leaves gives every test
   cell the same score (the sum of all ranks is constant), so the root's AUROC
   is exactly 0.5 and over-merged nodes decay toward chance. The DESIGN §3b
   anti-triviality worry ("scores drift upward with node size, the root wins
   by default") has the *opposite* sign under this scoring: the pressure is
   toward under-merging, not over-merging. Verified as a unit test (up to
   floating-point tie-breaking, which must be rounded away before ranking).

2. **Parent-vs-child ΔAUROC alone cannot resolve the level under saturation.**
   At zero batch effect both parent and child sit at AUROC ≈ 1 against the
   easy one-vs-all background and the test cannot separate them; the walk
   overshoots to a leaf. First observed as the single failure in an otherwise
   clean 32/32 sweep.

3. **Per-cell preference voting is underpowered.** The first fix (each
   positive cell votes for its preferred child; descend on a concentrated
   majority) fails in both directions: sibling similarity makes per-cell
   argmax noisy, so genuine leaf-level queries concentrate only ~0.6-0.7 and
   get blocked, while family-level queries with unbalanced latent substructure
   lean 0.55-0.65 and sneak past any majority threshold. No fixed threshold
   separates the regimes reliably.

4. **The sibling contrast is the right merge test.** Descend into the best
   child only when its one-vs-all AUROC significantly exceeds the
   *second-best sibling's* (paired bootstrap over test cells), and the parent
   is not significantly better than the best child. A leaf-level query
   separates the siblings even at saturation (both near 1, but consistently
   ordered); a family-level query relates to them exchangeably (Δ ≈ 0)
   regardless of saturation. This replaces DESIGN §3b's original "deepest node
   not significantly worse than the best ancestor" formulation, which
   finding 2 shows is insufficient on its own. Per-cell preference shares are
   kept as descriptive output only.

5. **Compactness must be measured in the parent's context.** A correct leaf
   selection legitimately shares family-level affinity with its siblings
   (within-family share ≈ 1/k), so raw within-selection affinity share
   misflags correct leaf matches as discordant. Scatter *within* the selected
   node's parent subtree is what family structure predicts; scatter *across*
   the parent's boundary is discordance. For top-level selections (parent =
   root) the selection's own subtree is the context.

6. **Validation results.** 32/32 correct node selections across batch_sd ∈
   {0, 0.5, 1.0} in both directions (coarse→fine-tree and fine→coarse-tree);
   12/12 leaf-level queries descend to their own leaf in the fine-vs-fine
   configuration; a family removed from the fine atlas is called unmatched;
   a two-family mixture label stops at the root and is called discordant.
   Symmetrized AUROC of true pairs is flat (~0.95) across the batch sweep —
   the integration-free batch-robustness premise holds in simulation.

## Known limitations (deliberate, tracked)

- **Forests (DESIGN §3b item 6) not implemented.** A query equal to a strict
  subset of a family (e.g. two of three subtypes) selects the strongest
  single leaf instead of the two-leaf union — the correct target is a forest,
  which the descent cannot express yet.
- **Hierarchical FDR** is approximated by descent gating only (children tested
  only where the walk reaches them); formal calibration pending.
- Bootstrap resamples cells; donor-level resampling awaits real data with
  donor structure (Allen benchmark).
- ~~FUGW estimator (§3c) not started~~ — implemented same day
  (`R/fugw.R` + `python/fugw_run.py`, POT 0.9.7 via a venv;
  `TN_PYTHON=<path>` selects the interpreter). Simulation validation
  (`analysis/04_fugw_sim.R`, batch_sd 0.5): 4/4 correct family assignments
  at α ∈ {0.3, 0.5, 0.8} with ~100% of each family's mass inside its own
  subtree, and P–Q gap ≈ 5e-9 (POT's two-coupling relaxation converged to
  the exact solution). Novel-family case: F4 absent from B leaves F4's
  transported mass depressed (0.69 vs ≥ 0.94 relative to marginal) under
  unbalanced marginals — the intended "weak but real vs atlas-specific"
  mechanism. α mapping to POT (linear-only convention): α_pot = α/(1−α),
  documented in `python/fugw_run.py`.

## Allen PL-ILA-ORB benchmark — first real-data results (2026-09-02)

Setup: A = 10Xv2 (22,067 cells) at subclass; B = 10Xv3 (13,842 cells), curated
tree over 103 kept clusters (floor ≥30/platform, cap 300); 575 joint HVGs;
scored against the curated subclass→cluster truth. `analysis/03_wmb_benchmark.R`.

- **Stage 2 — the premise holds on real data.** 100/103 self-RBH across the
  real 10Xv2↔10Xv3 platform gap; median self-AUROC 0.998; all 103 above 0.9.
- **Stage 4**: cell-vote compact support median 0.99, 22/23 subclasses ≥ 0.7
  (min 0.58, `003 L5/6 IT TPE-ENT Glut` — plausibly a genuine dissection-
  boundary case).
- **Stage 5 baseline**: 21/23 subclasses exact leaf-set matches; 84/103
  clusters find their parent subclass. The two misses are one-level
  near-misses with ~0.98 compactness (Lamp5 one supertype deep; Endo one
  class shallow) — threshold sensitivity, not wrong branches.
- **Stage 6 FUGW head-to-head: baseline 21/23, FUGW best 12/23**
  (α ∈ {0.3..0.9}, ρ=1, ε=0; best at α=0.9 ≈ molecular-only; median
  own-subtree mass 0.75–0.91, rarely clearing the 0.9 assignment bar).
  On this benchmark the transparent baseline clearly wins — the outcome the
  design treats as "FUGW is unnecessary complexity" unless fair tuning
  (ρ sweep, cost calibration transforms, assignment threshold) closes the
  gap. Numerical note: ε>0 needs POT's log-domain sinkhorn at this scale
  (exp overflow otherwise); ε=0 with the mm solver is stable and blur-free.

**Decisions after this run (2026-09-02):** TreeNeighbor's direct-selection
baseline is now the *primary method*; FUGW is demoted to ablation/challenger.
FUGW performing best near α→1 (molecular-only) suggests its tree term was
actively hurting rather than adding information — worth one diagnostic look
during the ablation, not a rescue effort. The estimator is **frozen** (see
R/baseline.R header) through the batch-condition validation: the 21/23 was
obtained while iterating decision rules against this same run, so it is a
development-set figure until the frozen rules reproduce it under splits they
were never tuned on.

**Scope caveats to carry:**

- "Integration-free robustness holds" currently means: *within the jointly
  curated Allen taxonomy, across a real chemistry gap*. Both sides share one
  annotation pipeline and one clustering; arbitrary cross-atlas robustness
  (independent labs, pipelines, taxonomies) is untested until the Siletti-
  style cross-atlas case runs.
- The reverse direction is the weaker one (84/103 vs 21/23 forward); the
  19 failures need classification (wrong branch / adjacent near-miss /
  premature stop / unmatched) before the forward number is quoted alone.
  `analysis/06_batch_conditions.R` classifies them per condition.

## Batch-condition validation — frozen estimator (2026-09-02)

`analysis/06_batch_conditions.R`, estimator frozen before running (no rule or
threshold touched afterward). Random half-split and donor-held-out split are
within 10Xv2; platform is the 10Xv2-vs-10Xv3 development run's saved outputs.

| condition | clusters | self-RBH | med self-AUROC | sib margin | fwd exact | rev parent | wrong-branch / adjacent / premature / unmatched |
|---|---|---|---|---|---|---|---|
| random (near-zero batch) | 103 | 103 (100%) | 0.998 | 0.013 | 21/23 | 87/103 | 0 / 5 / 11 / 0 |
| donor-held-out (mild) | 100 | 100 (100%) | 0.998 | 0.014 | 22/23 | 83/100 | 0 / 6 / 11 / 0 |
| platform 10Xv2↔v3 (strong) | 103 | 100 (97%) | 0.998 | 0.012 | 21/23 | 84/103 | 0 / 8 / 11 / 0 |

Findings:

- **The curve is flat.** Sibling margin and self-AUROC do not move across
  three orders of batch severity; the platform gap costs 3 RBH points and
  nothing else measurable. The frozen rules reproduce 21-22/23 forward on
  splits they were never tuned on — the forward figure is now validated, not
  a development artifact.
- **The reverse asymmetry is not batch-driven.** 16-19 reverse failures at
  every severity, including near-zero batch. Classification: **zero
  wrong-branch and zero unmatched anywhere**. The failures are 11 premature
  stops (constant across conditions — the same deep-layer IT clusters, plus
  Vip/Lamp5/Sst singletons and COP) and 5-8 adjacent same-class near-misses
  (L4/5→L5→L6 IT layer neighbors; COP→OPC; L6 CT→L6b). This is the known
  IT-layer continuum and the oligodendrocyte lineage intermediate — taxonomy
  geometry, not estimator noise and not batch. A flat reverse tree gives the
  walk no node that can express "IT, layer ambiguous"; the forward
  direction's deep tree is exactly what absorbs this, which is the method's
  own argument in miniature.
- The two forward misses (Lamp5, Endo) also persist at near-zero batch —
  threshold/floor geometry, not batch.

Three real-data findings that changed the estimator (all folded into code):

7. **Union-AUROC is size-biased at coarse splits.** Root neutrality at
   scale: a large heterogeneous child's union score dilutes toward constant,
   so one-vs-all AUROC misranks children of unequal size (L5 ET picked a
   wrong small class while 0% of its cells voted there). Navigation is now
   **vote-guided**: each positive cell votes for its argmax leaf; a child
   collects votes of leaves in its subtree; the walk enters the plurality
   child. AUROC remains the stop statistic.
8. **Statistical vs practical significance.** With 22k cells the bootstrap
   makes ΔAUROC = 0.001 significant and the walk overshoots to single
   leaves. The sibling contrast now requires a practical margin
   (`margin = 0.01` AUROC units) — an explicit calibration parameter.
9. **A clear-votes override.** Distinct siblings can both saturate
   one-vs-all (Endo vs Peri under class Vascular); when ≥90% of cells vote
   for one child's subtree, the walk descends without the AUROC test.

## Figure set from the frozen benchmark (2026-09-02)

`analysis/07-12`, outputs in `figures/` with plotting tables in
`figures/tables/`. Estimator untouched; the trace instrumentation added to
`tn_select_node` is recording-only, proven by re-running both full maps and
asserting identity with the saved benchmark (07). Two findings came out of
the figure work itself:

10. **Threshold sensitivity (fig 5):** the frozen point sits on a plateau —
    forward 20-22/23 and reverse 77-86/103 across margin ∈ [0.0025, 0.04]
    and override ∈ [0.80, disabled], with **zero wrong-branch errors in the
    entire grid**. The vote override earns its keep: disabling it costs 7
    reverse recoveries. Slightly looser settings do marginally better here,
    but per the freeze, no retuning — noted only as sensitivity.
11. **The GW tree term is not what hurts FUGW (fig 6).** Controlled
    comparison at matched cost/marginals/regularization: tree-only GW
    recovers 0-1/23 (the taxonomy path metric alone carries almost no
    assignment signal); molecular-only unbalanced OT gets 4-10/23; FUGW gets
    7-14/23 — i.e. the tree term consistently *helps* the transport model,
    refuting the earlier "tree term actively hurting" reading of the α
    sweep. What keeps the whole transport family far below direct selection
    (21/23) is coupling diffuseness: median row entropy ~1.6 nats (mass
    spread over ~5 effective leaves), so the 0.9-mass assignment rarely
    commits. The failure is the relaxation, not the structure prior.

12. **FUGW post-mortem (analysis/13, tables fig7_*): the 0.90-mass decision
    rule, not the transport, was the main handicap.** Per-query decomposition
    of every sweep coupling: (a) *underconfidence is the dominant effect* —
    threshold-free argmax-family accuracy exceeds the 0.90-mass count by a
    median of 5 queries across all FUGW settings, and at the sweep's best
    setting (raw cost, ρ=0.3, α=0.9, selected post hoc) reaches 21/23 —
    tying the frozen TreeNeighbor — with L2/3 IT missing the bar at 0.888.
    (b) The 14 confident queries show textbook coarse-to-fine splitting:
    mass ≈ 1.0 inside the true family, spread over ~60-100% of its true
    leaves. (c) The residual failures are structured, not random: leaked
    mass crosses to *adjacent* families along the same taxonomy boundaries
    as the baseline's reverse errors (IT continuum, L6b↔L6 CT, OPC↔COP),
    and effective-leaves ratios ≫ 1 occur exactly for singleton-target
    families (Car3 4.2×, chandelier 3.6×, Chodl 4.8×) — transport smears
    rare families. Only 2/23 outright argmax failures at the best setting
    (Pvalb chandelier, Sst Chodl) — queries the baseline gets right;
    complementary failure sets. (d) *Entropic regularization rejected as the
    cause*: the sweep ran at ε=0 (mm solver); an explicit ε probe leaves
    accuracy flat while increasing diffuseness. The diffuseness is
    structural (KL marginal relaxation + genuinely correlated neighbors).
    Closing wording for the ablation: transport with an argmax-family
    readout is competitive but never better than direct selection, is far
    more setting-sensitive (argmax range 10-21 across the sweep), and
    uniquely mishandles rare singleton families. Demotion stands.

13. **The singleton smearing was a uniform-marginal capacity artifact — and
    fixing it rescues FUGW entirely (analysis/14, tables fig7_marg*).**
    Capacity arithmetic (prespecified): under uniform marginals a source
    subclass carries 1/23 of mass but a singleton target family may hold
    only 1/103 — a 103/23 = 4.48× mismatch, matching the observed singleton
    effective-leaf ratios (4.2× / 3.6× / 4.8×; observed singleton
    true-masses 0.29-0.41 sit just above the hard 0.223 capacity bound,
    the pay-penalty-and-spill signature). Three marginal schemes, identical
    grid, no tuning:
    - *uniform*: median argmax 14 [10-21]; singletons smeared (ratio 4.1).
    - *abundance* (donor-balanced): does NOT rescue — slightly worse
      (median 12); cell abundance ≠ transport capacity.
    - *hierarchy-balanced* (each target family equal total mass, split
      among its own leaves — uses only B's own taxonomy): **median argmax
      23/23 [21-23]; all 8 singletons confident with effective-leaf ratio
      exactly 1.00; zero cross-family at the prespecified reference
      setting.** At α = 0.1, five (calibration, ρ) combinations reach
      23/23 argmax AND 23/23 *confident* — exceeding the frozen
      TreeNeighbor's 21/23. Under these marginals the GW term is genuinely
      additive (molecular-only UOT: 23 argmax but 19-20 confident; FUGW at
      α=0.1: 23/23 both; tree-only GW still ~2-3, so molecular remains
      essential).
    **Correction (same day, item 14):** the "hierarchy-balanced" scheme in
    analysis/14 is ORACLE LEVEL BALANCING — it equalizes mass across B's
    subclass level, a level the analyst chose knowing the source labels are
    subclasses. It passes the mechanical leak tests (never references the
    source; source-swap leaves weights unchanged) but the level *selection*
    is where the knowledge entered. Item 14 replaces it with the legitimate
    intrinsic version.
    **Status change: FUGW is un-demoted** — from ablation to co-equal
    challenger. The layered diagnosis now reads: threshold rule cost ~5-7
    queries; uniform marginals cost the singletons and the stability
    (10-21 → 21-23 across settings); with both fixed, transport recovery is
    perfect on this benchmark. **Caveat that bounds the claim:** the
    capacity match is *exact* here because both atlases share one taxonomy
    (23 source subclasses ↔ 23 target families); in a genuine cross-atlas
    setting the match is approximate, so this is an upper bound. Before any
    ordering claim, hierarchy-FUGW must pass the same frozen battery
    TreeNeighbor passed (three batch conditions) and the cross-taxonomy
    test. Methodological takeaway for the paper: *marginal design is the
    hidden hyperparameter of unbalanced OT on hierarchies* — arguably a
    contribution in its own right.

14a. **Terminology and restraints (user, 2026-09-02).** The contribution is
    named **refinement-invariant marginals**: *splitting one annotation into
    additional subtypes redistributes its existing branch mass but does not
    increase that branch's total transport capacity.* Two claims stay
    restrained: (i) "graceful degradation" on disagreeing taxonomies is a
    HYPOTHESIS — FUGW is nonlinear and degradation could be abrupt around
    specific structural conflicts; the amygdala case measures it, we do not
    assume it. (ii) Recursive equal splitting is principled but not uniquely
    correct: it treats sibling branches as equally important *conceptual
    units*, not equally abundant *biological populations* — appropriate for
    annotation harmonization, and stated explicitly. The Allen benchmark
    proves correctness under compatible hierarchies, not automatic
    reconciliation of incompatible ones.

    **FUGW FROZEN (2026-09-02)** at the reference configuration chosen
    BEFORE the intrinsic comparison — not a retrospective 23/23 setting:
    cost = 1 − S_sub (raw), ρ = 0.3, α = 0.9, ε = 0 (mm solver),
    refinement-invariant (intrinsic recursive) marginals on both sides,
    readouts = argmax-family accuracy and mass-based confidence categories.
    Next: the same three-condition batch battery the walk estimator passed,
    with assignment and coupling-concentration stability; then packaging for
    JHPCE; then both frozen methods to the amygdala trees.

14. **Intrinsic recursive marginals legitimize the FUGW rescue
    (analysis/15, `tn_tree_weights` in R/tree.R).** The prespecifiable
    version of hierarchy balancing: mass 1 at each tree's root, split
    equally at every internal node, computed for BOTH atlases independently
    — a pure function of each tree, no level selection, no reference to the
    paired atlas. Unit-tested for the four leak criteria: source
    independence (structural — the function takes only the tree),
    refinement invariance (splitting a leaf into 10 children leaves its
    branch total unchanged), name invariance, and per-node mass
    conservation. Result on the identical grid, no tuning: **median argmax
    23/23 [20-23], confident 20 [16-21], all 8 singletons recovered with
    effective-leaf ratio 1.00; reference setting 23 argmax / 21 confident /
    0 cross-family** — statistically indistinguishable from the oracle
    scheme. The mechanism is emergent, not encoded: intrinsic weights are
    NOT uniform (0.016-0.111 across subclasses), yet source-branch and
    target-family capacities agree to machine precision (max diff 7e-18)
    because the two trees share their class→subclass prefix, so the
    recursive splits coincide. On atlases with genuinely different
    branching structures the alignment will be approximate and unbalanced
    OT must absorb the residual — which is the appropriate behavior, and
    exactly what the amygdala/cross-taxonomy case will measure. Principle
    for the paper: *a branch's transport capacity must not increase merely
    because its annotators divided it into more clusters*; recursive
    tree-intrinsic marginals implement that invariance with no oracle
    input.

15. **Frozen FUGW passes the three-condition battery flat (analysis/16,
    results/fugw_battery_*).** Frozen configuration (pre-intrinsic
    reference: raw cost, ρ=0.3, α=0.9, ε=0, refinement-invariant marginals
    both sides): **argmax-family 23/23 at every condition** (random half-
    split, donor-held-out, cross-platform), identical confidence profile
    everywhere (21 confident + 2 underconfident, zero cross-family, all 8
    singletons recovered, median family entropy 0), and **23/23 assignment
    stability** — every query maps to the same family across all three
    conditions. The two persistently underconfident queries are 003 L5/6 IT
    TPE-ENT and 005 L5 IT CTX — the deep-layer IT continuum again, at
    near-zero batch too: the same taxonomy geometry that bounds the walk
    estimator. On the forward task under compatible hierarchies, frozen
    FUGW is now flatter than the frozen walk (23/23/23 vs 21/22/21); the
    walk retains the reverse direction, per-decision statistics, and
    interpretability. Both frozen methods proceed to packaging and then the
    amygdala trees; the cross-taxonomy case decides the ordering — its
    degradation behavior is measured, not assumed (item 14a restraint).

16. **Packaged (pkgs/), python-primary with an R companion (2026-09-02).**
    Decision: python is the primary implementation (`pkgs/treeneighbor-py`,
    AnnData-first API + array core; FUGW native via optional POT) because
    FUGW is python, atlases ship as h5ad, and the MetaNeighbor ecosystem
    seat is open; the R companion (`pkgs/TreeNeighbor-R`) repackages the
    validated code, pure R, no FUGW. Top-level `R/` + `analysis/` stay
    frozen as the historical record. Cross-language chain of custody:
    bootstraps use a shared portable MINSTD stream (exact in R doubles)
    with per-query seeds and a documented draw order. Gates all passed:
    (a) packaged-R walk reproduces the saved frozen benchmark EXACTLY
    (23/23 fwd + 103/103 rev selections and relations identical under the
    new RNG); (b) python == packaged-R on every query, both directions,
    real caches + simulation end-to-end (HVG set equal, S within 1e-9);
    (c) packaged FUGW reproduces the frozen battery platform result to the
    digit (23/23 argmax, 21 confident, same two underconfident IT queries,
    same P-Q gap). R package passes R CMD INSTALL + its test suite; python
    package pip-installs with 7/7 tests. JHPCE install instructions in
    pkgs/README.md. Note: macOS case-insensitive filesystems collapse
    `treeneighbor/` and `TreeNeighbor/` into one directory — hence the
    `-py` / `-R` suffixes.

17. **Renamed to MetaArbor (2026-09-02); numerical behavior unchanged and
    re-proven.** Framework = MetaArbor; modes = MetaArbor-Walk (both
    languages) and MetaArbor-Transport (python-only). Python distribution +
    import namespace `metaarbor` (`pkgs/python/`); R package `MetaArbor`
    with `ma_*` functions (`pkgs/r/` — lowercase dirs because macOS
    case-insensitive filesystems merge `metaarbor/`/`MetaArbor/`). The
    pre-rename state is preserved at git tag `v0.1-treeneighbor-final`
    (repo git-initialized for the purpose; ~5 GB of rebuildable caches
    gitignored with documented rebuild paths). The old name persists only
    in the frozen research record (top-level R/, analysis/, DESIGN.md with
    a naming note, NOTES history). Serialization was already rename-proof:
    RDS holds plain lists, fixtures are CSV, python never pickles. All
    gates re-run post-rename with ZERO differences: clean R install +
    tests; clean-venv python install + 7/7 tests against regenerated
    fixtures; R identity gate IDENTICAL (23/23 fwd, 103/103 rev);
    cross-language parity green; Transport gate identical to the digit
    (23/23, 21 confident, same two underconfident IT queries, P-Q gap
    8.2e-09). New additive feature: a **common node-evidence table**
    (`metaarbor.node_evidence` / `ma_node_evidence`) — one row per (query,
    split, child) with vote fraction, child one-vs-all AUROC, reverse-fold
    directional AUROC, sibling/parent bootstrap lower bounds,
    override/margin verdicts, decision, transport-mass share (python), and
    the selected relation; evidence rows re-derive the walk with the same
    per-query seeds, so recorded decisions are exactly the map's.

18. **Interpretation & visualization layer built and FROZEN (2026-09-02);
    python is the standalone primary (user decision — R deferred).**
    Additive only: no estimator, threshold, RNG or frozen preset changed;
    the walk gained recording-only decision-support stats (fraction of the
    decisive test's existing bootstrap draws agreeing with the decision),
    verified non-interfering by the identity gate. New python surface:
    `walk_summary` / `transport_summary` / `alignment_summary` (one row per
    query; six documented threshold-free agreement categories), accessors
    (`vote_fraction_matrix`, `node_auroc_matrix` — documented as an
    interpretability view with size/saturation biases, `family_mass`,
    `walk_traces`, `write_csv`), five packaged matplotlib plots
    (`plot_alignment_tree`, `plot_evidence_heatmap`,
    `plot_transport_heatmap`, `plot_query_path`, benchmark-only
    `plot_error_tree` + `classify_outcome`), and `result_bundle` (table +
    figures in one call, PNG+PDF, native objects returned). Gates: walk
    summary selections identical to the saved frozen benchmark;
    row-normalized transport mass sums to 1 for every massed query;
    summaries and categories deterministic; both benchmark directions
    render (vignette `examples/allen_interpretation.py` = the render gate;
    outputs in examples/allen_demo). Allen agreement: 14 agree + 9
    same_branch_different_depth + 0 conflicting — the 9 are node-identity
    chain-collapse effects (Walk selects a singleton subclass's leaf,
    Transport names the subclass node), reported as-is rather than
    threshold-smoothed. 11/11 python tests pass. The R numeric
    interpretation code written earlier in the session remains in-tree but
    is unvalidated and deferred with the rest of the R package.

Figure inventory: fig1 (six annotated walks — the Endo page shows a miss
mechanism end-to-end: root override at vote 0.98, then saturated vascular
siblings inside the margin one level above truth), fig2 (error topology on
the curated tree; the reverse premature-stop fan converges on the root from
the deep-layer IT block), fig3 (vote heatmaps; IT continuum bleeds across
subclass boundaries, GABAergic/NN blocks crisp), fig4 (three batch
conditions with raw counts), fig5 (sensitivity), fig6 (FUGW decomposition).

## Allen benchmark groundwork

- Frontal-cortex ROIs in WMB-10X metadata: `PL-ILA-ORB`
  (prelimbic/infralimbic/orbital) and `ACA` (anterior cingulate); `AI`
  (agranular insular) optional. Resolves the §4.1 verification flag.
- Expression lives in per-region h5ad packages (WMB-10Xv2-Isocortex-1..4,
  WMB-10Xv3-Isocortex-1..2, ~8-12 GB each, raw + log2 variants); the cells'
  `matrix_label` column says which package each cell needs, so the benchmark
  downloads only the packages that actually contain frontal cells.
- `data/wmb_frontal_cell_metadata.csv` + `data/wmb_frontal_summary.json`:
  per-cell metadata (labels, donor, platform, package) for the frontal
  subset, streamed from the 20230630 release without storing the 1.4 GB
  source file.
- Measured counts: core PFC (ACA + PL-ILA-ORB) = 209,603 cells (10Xv2
  128,818 / 10Xv3 80,785; 25 donors overall) spanning 16 classes /
  74 subclasses / 220 supertypes / 582 clusters; with AI added, 309,126
  cells and 19 / 107 / 291 / 712. Subclass-vs-cluster ≈ 8:1 — the designed
  coarse:fine regime. Frontal cells sit in six expression packages
  (v2 Isocortex-1..4, v3 Isocortex-1..2), ~50 GB total for the raw h5ads.
