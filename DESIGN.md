# TreeNeighbor: MetaNeighbor-fused tree alignment for cross-atlas annotation harmonization

**Status: design document.** No code exists yet. This document specifies the method,
its statistical rationale, and the benchmark that will decide whether its ambitious
half earns its complexity. Working title `TreeNeighbor`; rename freely.

**One-sentence framing.** An integration-free framework for aligning independently
constructed cell-type taxonomies, using MetaNeighbor-derived molecular evidence with
either direct hierarchical selection (the interpretable baseline) or structure-aware
unbalanced optimal transport (the ambitious estimator, justified only if it improves
held-out hierarchy recovery beyond the baseline).

---

## 1. Motivation and gap

Atlases annotate the same tissue at different resolutions: one lab publishes ~30
coarse types, another ~300 fine clusters. The correct correspondence is usually
that a coarse type maps to a *family* of fine clusters — a subtree, not a leaf.
Any harmonization method that scores label pairs one-to-one dilutes exactly the
signal it needs: if coarse type $A$ is the union of fine clusters $B_1, B_2, B_3$,
each $B_j$ matches $A$ well-but-not-perfectly and no single pairwise score reveals
the family.

### 1.1 The current field

The most recent method, **OTHarmonizer** (Bioinformatics 2026; btag506), aligns
annotations with partial optimal transport and infers equal / parent–child / novel
relations from the transport matrices. Its benchmark (ten scenarios: five Symsim
simulations, three cell-type-specific HLCA subsets, whole-lung HLCA and heart hECA
at atlas scale) is the reference evaluation for this problem, and its own results
define the gap:

| Method | Similarity basis | Batch strategy | Hierarchy strategy | Reported weakness |
|---|---|---|---|---|
| scHPL | classifier confusion (kNN/SVM) | none intrinsic | progressive tree learning | sensitive to batch effects |
| treeArches | scHPL on scArches/scVI latent | scVI integration | progressive tree learning | inherits integration quality; misses annotations |
| CellHint | predictive clustering trees | distance re-weighting | tree assembly | non-tree structures in mixed-granularity settings; struggled on complex real data |
| OTHarmonizer | Euclidean distances in scVI latent space | scVI integration | flat partial OT + post-hoc decision rules | hierarchy recovery depends on integration quality; atlas-scale failure |

Three findings from the OTHarmonizer paper matter most here. First, every
benchmarked method scored well only when batch effects were mild; strong batch
effects degraded all of them. Second, at atlas scale none of the methods supported
an end-to-end automated process — expert curation was still required. Third,
OTHarmonizer's transport costs are Euclidean distances in an scVI-integrated latent
space, so its robustness is *inherited from the integration step*, not intrinsic:
where scVI under- or over-corrects, the costs are wrong and the hierarchy follows.

### 1.2 The missing foundation

MetaNeighbor (Crow et al. 2018, *Nat Commun*; Fischer et al. 2021, *Nat Protoc*)
measures cross-dataset cell-type replicability without any integration step. It
builds a cell–cell Spearman-correlation network on highly variable genes,
re-ranks and standardizes the network edge weights to [0, 1], scores each cell by
the fraction of its (weighted) neighbors carrying a given label, and evaluates
label recovery with AUROC.

The precise claim — and no more — is this: MetaNeighbor is **integration-free and
rank-based, with demonstrated cross-dataset robustness** to lab of origin,
sequencing technology, and clustering pipeline. It does not mathematically erase
per-cell technical distortions, and its behavior under *strong* batch effects is a
demonstrated tendency to be stress-tested here, not a theorem. But it is exactly
the kind of evidence OTHarmonizer lacks: molecular similarity that never passes
through a learned embedding whose failure the downstream hierarchy would inherit.

Nobody has used this measurement as the evidence inside a tree-alignment method.
That is the gap this project fills.

A practical note that strengthens the case: the MetaNeighbor package has been
deprecated and removed from Bioconductor 3.23 (last stable release in Bioc 3.21;
the development version remains on GitHub at gillislab/MetaNeighbor). The voting
kernel is small — a rank-standardized correlation network and a matrix–vector
product — so the prototype should reimplement it natively rather than depend on a
deprecated package. This also gives us the per-cell vote cache the method needs
(§3a), which the packaged functions do not expose.

---

## 2. Problem formalization

Two atlases provide independently constructed trees $T_A$ and $T_B$ over their own
cell populations, potentially at different depths and resolutions. Leaves are the
observed terminal populations ($L_A$, $L_B$); internal nodes are the taxonomies'
groupings. The estimand is a **soft, many-to-many correspondence** between the two
trees.

The framing commitment, which drives every later choice: **one-to-many mappings
are the expected representation of resolution difference, not exceptions.** A
30-type atlas set against a 300-type atlas *should* produce couplings in which one
coarse type distributes its mass over several fine leaves. A formalism that treats
such splitting as error will fight the biology.

### 2.1 Candidate formalisms

| Formalism | What it does | Fit |
|---|---|---|
| Tree edit distance | minimum-cost node insertions/deletions/relabelings | intuitive, but arbitrary edit costs dominate; treats resolution difference as edits to be paid for |
| Cophenetic / tree-distance comparison | compares pairwise distances induced by the trees | requires leaves to already correspond — assumes the answer |
| Triplet / quartet agreement | do matched leaves keep the same local branching? | good **evaluation** metric after alignment; not an alignment method |
| Hierarchical optimal transport | coarse node distributes mass across fine nodes | natural for coarse-to-fine matching |
| Fused Gromov–Wasserstein | aligns nodes using molecular similarity *and* within-tree relational structure | the most flexible existing machinery |
| Latent consensus-tree model | both trees as dataset-specific prunings of a shared latent tree | statistically elegant; substantially harder to fit; deferred |

Transport treats mass splitting as legitimate; tree-edit distance mistakes it for
edits. That is why the ambitious estimator (§3c) is transport-based, and why
triplet/quartet agreement is retained on the evaluation side only (§4).

### 2.2 Output relations

For compatibility with OTHarmonizer's evaluation (AH-F1 over pairwise relations,
TEDS, PCBS), the final output is a harmonized relation graph labeling node pairs
$(u \in T_A, v \in T_B)$ as one of:

- **equal** — bidirectionally concentrated, near-total mutual mass;
- **parent–child (family)** — $u$'s mass concentrates inside the subtree under $v$
  (or vice versa) but spreads across its children;
- **split** — $u$'s mass divides across several children of $v$ in comparable
  shares (the informative special case of parent–child);
- **novel / unmatched** — mass largely unassigned under unbalanced transport, or
  no node passes the baseline's tests;
- **structurally discordant** — molecular support scattered across unrelated
  branches; reported as a finding (mixed source label, unstable tree, convergent
  expression, or incompatible annotation axes), never forced into a clean relation.

---

## 3. Method: one measurement, two estimators

The architecture separates *what is measured* from *how correspondence is
estimated*. The measurement layer is shared; the two estimators consume it
differently, and §4 adjudicates between them.

### 3a. Shared measurement layer: MetaNeighbor leaf costs

Cross-atlas MetaNeighbor voting is run at the **terminal populations (leaves)
only**. Highly variable genes are selected jointly (`variableGenes` semantics:
genes variable in both datasets), the rank-standardized network is built, and
neighbor-voting AUROCs are computed for every cross-atlas leaf pair.

**Directionality.** MetaNeighbor scores are directional: the vignette states
explicitly that AUROC scores across testing and training folds will not be
identical, because each test cell type is scored against the heterogeneity of its
own dataset. So for leaves $i \in L_A$, $j \in L_B$ there are two measurements,
$\mathrm{AUC}_{A \to B}(i,j)$ and $\mathrm{AUC}_{B \to A}(j,i)$. The provisional
symmetrized cost is

$$M_{ij} \;=\; 1 - \tfrac{1}{2}\left(\mathrm{AUC}_{A \to B}(i,j) + \mathrm{AUC}_{B \to A}(j,i)\right).$$

**Calibration is an open design question, not an established answer.** AUROC is
bounded, chance-centered at 0.5, and not a metric; transport theory is developed
for distance-like costs. Candidate transforms to compare empirically on the
benchmark: the raw form above; rescaling so chance (0.5) maps to maximal
dissimilarity, $M_{ij} = \max(0, 1 - 2(S_{ij} - 0.5))$ with
$S_{ij}$ the symmetrized AUROC; and a rank-transform of $S_{ij}$ within each row/
column pair. The choice is settled by ground-truth recovery and stability (§4),
and the doc's position is that whichever transform wins must win on held-out data.

**The vote cache.** Let $W$ be the fixed rank-standardized weight matrix between
test cells and training cells and $D$ its row sums. The voting score of test cell
$c$ for a training set $U$ is $s_U(c) = (W \mathbf{1}_U)(c) / D(c)$ — linear in
$\mathbf{1}_U$. Hence for disjoint training populations $U, V$:

$$s_{U \cup V} = s_U + s_V.$$

Caching per-cell scores $s_{\{j\}}$ for every training leaf $j$ therefore permits
**inexpensive re-scoring of any union of leaves** — any internal node — by summing
cached columns and recomputing the AUROC from the summed scores (a sort, not a new
network). Two boundaries on this claim, stated exactly:

1. **AUROCs are not additive.** $\mathrm{AUROC}(s_U + s_V) \neq
   \mathrm{AUROC}(s_U) + \mathrm{AUROC}(s_V)$. The cache makes union *scores*
   cheap; every union's AUROC is recomputed from those scores.
2. The identity holds for a **fixed network and fixed test background**. Scoring
   variants that change the comparison population (e.g. `one_vs_best`, which
   evaluates against the best-competing type only) or that would re-rank the
   network for the merged labels must be recomputed under their own definitions;
   the cache is exact for the plain one-vs-all vote and a fixed $W$.

The full derivation with these caveats is Appendix A. FUGW itself (§3c) consumes
only the leaf-by-leaf matrix $M$; the union-scoring machinery serves the baseline.

### 3b. Baseline estimator: hierarchical MetaNeighbor (direct selection)

The interpretable procedure, and the **required baseline** — not an afterthought.
Every FUGW result in §4 is read against it.

For each query population $i$ in one atlas, walk the other atlas's tree top-down:

1. **Score** $i$ against each internal node (leaf-union) via the vote cache.
2. **Merge test** at each split *(revised twice per prototype and benchmark
   findings — see NOTES.md)*: navigation is **vote-guided** — each positive
   cell votes for its argmax mean-rank training leaf, a child collects the
   votes falling in its subtree, and the walk enters the plurality child
   (one-vs-all AUROC of a union is size-biased: large heterogeneous children
   dilute toward chance, root-neutrality at scale). The **stop decision** is
   the **sibling contrast**: descend only when the entered child's AUROC
   exceeds its best sibling's by a *practical margin* (not mere significance —
   with tens of thousands of cells, ΔAUROC = 0.001 is "significant" and the
   walk overshoots) under a paired bootstrap, and the parent is not
   significantly better; a ≥90% vote share for one child overrides the AUROC
   test, since distinct siblings can both saturate one-vs-all. A leaf-level
   query separates siblings even at saturation; a family-level query relates
   to them exchangeably ($\Delta \approx 0$). The original formulation
   ("deepest node not significantly worse than the best ancestor") fails under
   saturation, and pure per-cell preference testing is underpowered for the
   stop decision — votes navigate, AUROC contrasts decide.
3. **Anti-triviality.** Prototyping reversed this concern: with mean
   rank-standardized votes the union of *all* training leaves gives every test
   cell an identical score, so the root's AUROC is exactly 0.5 and over-merged
   nodes decay toward chance — the root cannot win by default, and the pressure
   is toward under-merging. The sibling-contrast test in step 2 is what
   arbitrates the level; one-vs-best sibling scoring remains available as a
   reporting variant.
4. **Multiplicity.** The hypotheses are nested on a tree: test children only where
   the parent's test rejects, controlling a hierarchical FDR (top-down
   tree-structured testing in the style of Yekutieli), with permutation nulls
   (label shuffles) calibrating each test.
5. **Reciprocity.** Run both directions ($T_A$ populations onto $T_B$ and vice
   versa); relations are asserted only where the directions agree.
6. **Forests.** When no single subtree covers $i$ but two or three do jointly,
   report the forest — a straddling coarse type is a finding (§2.2, discordant),
   not a failure.

Output: per-population selected node(s) with test-based confidence, assembled into
the §2.2 relation graph.

### 3c. Ambitious estimator: MetaNeighbor-informed FUGW

A global coupling estimated by fused unbalanced Gromov–Wasserstein transport, in
which every proposed match informs every other match.

**Support: leaves only.** The transported objects are the terminal populations
$L_A, L_B$ — never leaves and internal nodes together in one distribution.
Internal nodes overlap their descendants; assigning cell-count-derived marginals
to all nodes would count the same cells repeatedly. Internal nodes enter only
through (i) the within-tree distances and (ii) post-hoc mass aggregation (§3d).

**Objective (idealized single-coupling form).** With molecular costs $M_{ij}$
(§3a), within-tree distances $C^A_{ii'}$ (between leaves of $T_A$; cophenetic or
path distance — compare both) and $C^B_{jj'}$, leaf weights $w_A, w_B$ (§3e), and
coupling $\pi \geq 0$:

$$\min_{\pi \ge 0}\;
\alpha \sum_{ij} M_{ij}\,\pi_{ij}
\;+\; (1-\alpha) \sum_{i i' j j'} \left(C^A_{ii'} - C^B_{jj'}\right)^2 \pi_{ij}\,\pi_{i'j'}
\;+\; \rho_1\,\mathrm{KL}(\pi_{\#1} \,\|\, w_A)
\;+\; \rho_2\,\mathrm{KL}(\pi_{\#2} \,\|\, w_B)
\;+\; \varepsilon\,\mathrm{Reg}(\pi).$$

The Gromov term is **quadratic over pairs of pairs**, and that is not a detail:
before a correspondence exists there is no meaningful direct structural distance
between a node of $T_A$ and a node of $T_B$. GW instead asks whether pairs of
populations close in $T_A$ are transported to pairs similarly close in $T_B$ —
it rewards ancestry preservation by construction, without ever comparing the trees
node-to-node directly.

**Why the structure term can help, concretely.** Suppose coarse type $A$ scores
$0.86, 0.83, 0.80$ against fine leaves $B_1, B_2, B_3$ (one compact family) and
$0.79$ against $B_4$ on a distant branch. Molecular evidence alone barely
separates $B_3$ from $B_4$. The GW term sees that $B_1$–$B_3$ are mutually close
while $B_4$ is not, and that the mappings of $A$'s neighboring coarse types also
support the $B_1/B_2/B_3$ branch — so the coherent family is preferred without
pretending the AUROC gap is decisive. Conversely, when apparent matches scatter
across unrelated branches, the coupling should *not* manufacture a clean family;
that scatter surfaces as discordance (§2.2).

**Unbalanced marginals** (the KL terms) let mass go unmatched at a cost —
absorbing populations absent from one atlas, genuinely novel types, and sampling
differences — which is the mechanism separating "weak but real counterpart" from
"probably atlas-specific."

**Implementation reality.** POT (`ot.gromov.fused_unbalanced_gromov_wasserstein`)
implements this family, with two provisos to carry into the prototype: POT's
formulation optimizes a *pair* of couplings $(P, Q)$ by block coordinate descent —
a relaxation whose two couplings coincide at the exact solution — and its $\alpha$
convention weights the linear term only (no $(1-\alpha)$ on the Gromov term), so
our $\alpha$ must be mapped onto POT's arguments explicitly. Solver options: mm
(default), Sinkhorn, L-BFGS-B; divergence KL or L2. References implemented there:
Thual et al. 2022 (FUGW, NeurIPS), Séjourné et al. 2021 (unbalanced GW), Tran et
al. 2023 (unbalanced co-OT). R access via reticulate, or a native solver later.

**Model class, honestly.** This is not a more detailed version of §3b; it is a
different model class, and the trade-offs should be stated in the doc's terms:

| | Baseline (§3b) | FUGW (§3c) |
|---|---|---|
| unit of inference | each mapping mostly independent | every mapping influences every other |
| optimization | none (testing procedure) | nonconvex, hyperparameter-heavy ($\alpha, \rho, \varepsilon$) |
| interpretability | high — each decision is a test | lower — a global coupling |
| power | limited to local structure | can resolve ambiguity via global coherence |

§4's comparison exists to decide whether that extra power is real.

### 3d. Aggregation and interpretation

FUGW yields a leaf-level coupling $\pi$. Internal-node statements come from
rolling transported mass up both trees:

$$\Gamma(u, v) \;=\; \sum_{i \in \mathrm{Desc}(u)} \; \sum_{j \in \mathrm{Desc}(v)} \pi_{ij},$$

with leaves as their own (sole) descendants. If a coarse population sends ~90% of
its transported mass into the descendants of one node $v$, that supports "family
under $v$"; comparable shares across $v$'s children support "split"; mass left on
the table under the unbalanced penalties supports "novel." **FUGW never directly
announces a parent–child relation** — relations are inferred afterward from the
concentration and reciprocity of rolled-up mass, with thresholds calibrated on the
benchmark's ground truth (§4), mirroring how OTHarmonizer's decision rules sit on
top of its transport matrices but hierarchy-aware from the start.

### 3e. Marginals

Raw cell counts reflect sampling and experimental design as much as biology.
**Uniform-per-population or donor-balanced weights are the primary analysis;
cell-count weights are a sensitivity analysis.** Unbalanced transport relaxes
mismatched marginals but does not make arbitrary initial weights harmless — a
population given 30% of the mass because a sorting step enriched it will still
distort the coupling.

### 3f. Uncertainty and anti-circularity

If the same cells and genes build the trees, define the MetaNeighbor costs, tune
$\alpha$, and evaluate the mapping, the structural and molecular terms are partly
duplicated evidence and the evaluation is circular. The design commits to:

- **Curated trees** for the ground-truth benchmark (§4) — the Allen taxonomy is
  imported, not re-derived.
- Where trees must be inferred (future applications): **training donors (or gene
  splits) build the trees; held-out donors supply the cross-atlas molecular
  evidence** and assess correspondence.
- **Bootstrap tree reconstruction** exposes unstable branches; splits without
  bootstrap support collapse to **polytomies** rather than being treated as
  precise biological branches.
- **Bootstrap cells/donors → ensemble couplings** with per-edge confidence for
  both estimators; permutation nulls for mass-concentration significance.

### 3g. Multi-atlas note

Pairwise FUGW plans need not be transitively consistent: couplings $A \to B$ and
$B \to C$ do not guarantee a compatible $A \to C$. Version 1 therefore estimates
all pairwise couplings and builds a consensus relation graph **only from stable,
transitive relationships**. A true multi-marginal or barycenter formulation is a
later extension, not a v1 promise.

---

## 4. Benchmark: Allen whole-mouse-brain taxonomy, frontal cortex subset

The benchmark must (i) provide exact ground truth for parent–child relations,
(ii) contain real — not simulated — batch structure whose severity can be dialed,
and (iii) stay small enough to iterate on. The Allen Brain Cell atlas (Yao et al.
2023, *Nature*) satisfies all three.

### 4.1 Data

The WMB taxonomy is expert-curated and strictly hierarchical: **34 classes → 338
subclasses → 1,201 supertypes → 5,322 clusters** over ~4M QC-passing cells,
profiled on both **10Xv2 and 10Xv3** chemistries with anatomically defined
CCFv3 microdissections. Access is open (AWS S3, `abc_atlas_access` /
`AbcProjectCache`; no login).

**Subset: frontal cortex.** Restrict to frontal/prefrontal dissections to keep
the problem at the target scale — on the order of tens of subclasses and a few
hundred clusters. *Verified against the 20230630 release metadata:* the
prefrontal dissections are `region_of_interest_acronym` values **`PL-ILA-ORB`**
(prelimbic/infralimbic/orbital areas) and **`ACA`** (anterior cingulate area),
with `AI` (agranular insular) as an optional addition; per-cell `matrix_label`
identifies which expression package (WMB-10Xv2-Isocortex-1..4,
WMB-10Xv3-Isocortex-1..2) each cell needs, so only packages actually containing
frontal cells are downloaded. The frontal subset's per-cell metadata and level
counts live in `data/wmb_frontal_cell_metadata.csv` /
`data/wmb_frontal_summary.json`.

*Measured (20230630 release):* core PFC (`ACA` + `PL-ILA-ORB`) holds 209,603
cells — 128,818 on 10Xv2 and 80,785 on 10Xv3, from 25 donors overall — spanning
16 classes, 74 subclasses, 220 supertypes, and 582 clusters (with `AI` added:
309,126 cells; 19 / 107 / 291 / 712). Subclass vs cluster gives the designed
~8:1 coarse:fine ratio; many of the 582 clusters will be rare
dissection-margin populations, so the benchmark applies a minimum-cells-per-
platform floor before constructing the pseudo-atlases.

### 4.2 Construction

- **Pseudo-atlas A (coarse):** one split of the cells, labeled at **subclass**.
- **Pseudo-atlas B (fine):** the disjoint split, labeled at **cluster** (or
  supertype, as a second configuration).
- **Ground truth:** the curated subclass → cluster nesting — every parent–child,
  equal, and split relation is known exactly.
- **Batch-severity axis (real, not simulated):** (1) random cell splits within
  platform (minimal); (2) donor-disjoint splits within platform (mild);
  (3) 10Xv2 vs 10Xv3 cross-platform splits (strong). Optional stressors on top:
  UMI downsampling and HVG-set perturbation.

### 4.3 Staged progression

Run in order; each stage gates the next and is a result in itself.

1. **Import curated trees** for both pseudo-atlases (pruned to the subset).
2. **Reciprocal-best-hit MetaNeighbor anchors** between the pseudo-atlases at
   matched (cluster-vs-cluster) resolution — the sanity layer establishing that
   the measurement works on this data before any hierarchy is asked of it (the
   MetaNeighbor protocol's guidance: reciprocal top hits and AUROC > 0.9 mark
   strong candidates).
3. **Local structure preservation test:** do RBH-anchored leaves preserve local
   tree distances (triplet agreement over anchor triples)? If not, no alignment
   method will save the configuration; diagnose before proceeding.
4. **Compact-support test:** for each coarse population, does its cross-atlas
   support (top vote recipients) occupy a compact target subtree? This is the
   direct empirical check of the project's founding hypothesis.
5. **Baseline (§3b)** on the mismatched pair (subclass-atlas vs cluster-atlas),
   across the batch-severity axis.
6. **FUGW (§3c)** on the same pairs, and the deciding question: does it improve
   **held-out hierarchy recovery** beyond the baseline — evaluated on donor
   splits never touched during $\alpha/\rho/\varepsilon$ selection? If not, FUGW
   is unnecessary complexity and the baseline *is* the method. If it resolves
   scattered or ambiguous matches into reproducible, biologically coherent
   subtrees, it is the headline methodological contribution.

### 4.4 Comparisons and metrics

Comparators: scHPL, treeArches, CellHint, OTHarmonizer, each run per its own
recommended pipeline (including their integration steps — the point is to compare
*systems*, and the batch-severity curve is where integration-dependence shows).

Metrics, in two tiers:

- **Relation-level (OTHarmonizer-compatible):** AH-F1 (macro-F1 over pairwise
  equal / parent–child / non-relation), TEDS (tree edit distance similarity),
  PCBS (parent–children branch similarity) — enabling direct comparison with
  published numbers and any reproduction of their scenarios.
- **Coupling-level (ours):** transported mass on true ancestor–descendant pairs;
  triplet/quartet agreement of the induced leaf matching; bootstrap stability of
  selected nodes and coupling edges.

The headline figure is **metric vs batch severity**, one curve per method, with
the baseline-vs-FUGW gap called out explicitly.

The eventual paper-grade evaluation adds OTHarmonizer's own ten scenarios (Symsim
simulations; HLCA subsets; whole-lung HLCA and heart hECA) — out of scope for the
first prototype but the reason for keeping metric compatibility from day one.

---

## 5. Novelty and prior art

- **OTHarmonizer** (Bioinformatics 2026): flat partial OT with scVI-latent
  Euclidean costs, hierarchy from post-hoc rules. We replace the batch-fragile
  cost with an integration-free one and make the transport itself
  structure-aware (GW term); many-to-many is first-class rather than
  rule-derived.
- **scHPL / treeArches** (Michielsen et al.): progressive tree learning from
  classifier confusion; treeArches adds scVI. Closest published machinery for
  the *task*; neither uses co-expression-network evidence nor transport.
- **CellHint**: predictive clustering trees; documented non-tree pathologies in
  mixed-granularity settings.
- **OT machinery** (all pre-existing; we contribute the composition, not the
  solvers): FGW (Vayer et al. 2019/2020), unbalanced GW (Séjourné et al. 2021),
  **FUGW** (Thual et al. 2022, NeurIPS — brain alignment; implemented in POT as
  `fused_unbalanced_gromov_wasserstein` with BCD over coupled plans), unbalanced
  co-OT (Tran et al. 2023), unbalanced hierarchical OT for graph matching
  (arXiv:2310.12081), tree-Wasserstein ground metrics (Le et al. 2019).
  Cell-level GW alignments (SCOT; moscot) transport *cells* across modalities —
  related machinery, different estimand.
- **MetaNeighbor** (Crow et al. 2018; Fischer et al. 2021): supplies the
  measurement layer, including `one_vs_best` scoring and the protocol's
  cluster split/merge guidance, which §3b generalizes from flat label sets to
  trees.

**The claim to novelty is the composition** — integration-free co-expression
costs inside structure-aware unbalanced transport, with a leaves-only support
rule, principled aggregation, and anti-circular evaluation — plus the honest
experimental design in which the transparent baseline can win.

---

## 6. Risks and open questions

- **AUROC → cost calibration** (§3a): AUROC is not a metric; the right monotone
  transform is an empirical question and the answer may be benchmark-dependent.
- **Score saturation on large unions** (baseline): merged nodes are easier
  targets; the sibling one-vs-best guard and beat-both-children rule are designed
  for this but need empirical tuning.
- **$\alpha$ ill-posedness**: where a tree contradicts expression (curation
  errors, discordant axes), molecular and structural terms pull apart and the
  coupling becomes $\alpha$-sensitive; report $\alpha$-sweeps, never a single
  magic value.
- **Entropic blur**: regularization spreads mass; rare populations are the first
  casualties. Compare $\varepsilon \to 0$ solvers (POT allows zero) against
  Sinkhorn-regularized runs.
- **FUGW nonconvexity**: BCD reaches local optima; multi-start and
  bootstrap-ensemble agreement are the practical mitigations.
- **Conceptually discordant schemes**: when two atlases partition along different
  axes (spatial vs functional state), purely data-driven harmonization has an
  intrinsic limit — OTHarmonizer's authors say as much, and no cost function
  fixes it. The discordance output class (§2.2) exists to say so rather than
  guess.
- **MetaNeighbor under strong batch effects** is a demonstrated tendency, not a
  guarantee — the batch-severity axis (§4.2) tests the project's own premise, and
  a negative result there would be a real (publishable) boundary on the idea.

---

## 7. Roadmap

1. **This design doc** (current stage — no code, no data).
2. **R prototype, baseline first**: native vote-cache implementation (§3a
   kernel), then §3b; FUGW via POT through reticulate (native solver only if the
   method earns it).
3. **Allen frontal-cortex benchmark** (§4), stages 1–6.
4. **Decision point**: baseline-only method vs baseline+FUGW method, decided by
   held-out hierarchy recovery.
5. **OTHarmonizer scenario reproduction** (Symsim, HLCA, hECA) for the paper.
6. **Package** (Bioconductor conventions) and manuscript.

---

## Appendix A. Additivity of voting scores (and its exact scope)

Let $W \in \mathbb{R}^{n_{\text{test}} \times n_{\text{train}}}$ be the fixed
rank-standardized network weights between test and training cells, and
$D = W\mathbf{1}$ the test-cell degrees. The neighbor-voting score of test cell
$c$ for training population $U$ is

$$s_U(c) = \frac{(W \mathbf{1}_U)(c)}{D(c)}.$$

Because $\mathbf{1}_{U \cup V} = \mathbf{1}_U + \mathbf{1}_V$ for disjoint $U, V$
and the map $\mathbf{1}_U \mapsto W\mathbf{1}_U$ is linear:

$$s_{U \cup V} = D^{-1} W (\mathbf{1}_U + \mathbf{1}_V) = s_U + s_V.$$

Caching $s_{\{j\}}$ for every training leaf $j$ (an $n_{\text{test}} \times |L|$
matrix) therefore yields the score vector of **any internal node** by summing
columns. The node's AUROC is then computed from that summed score vector against
the relevant positive/negative test labels — an $O(n \log n)$ sort, no new
network.

Scope, precisely:

1. **Scores are additive; AUROCs are not.**
   $\mathrm{AUROC}(s_U + s_V) \neq \mathrm{AUROC}(s_U) + \mathrm{AUROC}(s_V)$,
   and a node's AUROC is not any simple function of its leaves' AUROCs. Every
   union's AUROC must be recomputed from the summed scores (cheap, per above).
2. **Fixed network, fixed background.** The identity assumes $W$ is built once
   (rank standardization not redone per merged label set) and the AUROC's test
   background is held fixed. Scoring variants that change the comparison set —
   `one_vs_best` restricts negatives to the best competitor — must be recomputed
   under their own definitions from the same cache, which remains cheap but is
   not a column sum alone.
3. When the test-side populations are also merged, the same cache applies: the
   positive/negative label split changes, the scores do not.

---

## References (to be formatted properly in the manuscript)

- OTHarmonizer: *Bioinformatics* 42(7): btag506 (2026).
- Crow M, Paul A, Ballouz S, Huang ZJ, Gillis J. *Nat Commun* 9:884 (2018).
- Fischer S, Crow M, Harris BD, Gillis J. *Nat Protoc* 16:4031–4067 (2021).
- Yao Z, et al. *Nature* 624:317–332 (2023) — WMB taxonomy; ABC atlas access via
  `abc_atlas_access` (AWS S3 `allen-brain-cell-atlas`).
- Michielsen L, et al. scHPL, *Nat Commun* (2021); treeArches, *Bioinformatics* (2023).
- Domínguez Conde C, et al. CellHint, *Cell* (2023) — verify exact citation.
- Vayer T, et al. FGW (2019/2020); Séjourné T, et al. unbalanced GW, NeurIPS
  (2021); Thual A, et al. FUGW, NeurIPS (2022); Tran et al. unbalanced co-OT,
  AAAI (2023); Le T, et al. tree-Wasserstein, NeurIPS (2019).
- POT: `ot.gromov.fused_unbalanced_gromov_wasserstein` (pythonot.github.io).
- Yekutieli D. Hierarchical FDR testing, *JASA* (2008) — verify exact citation.
