# Retina bipolar K=2 benchmark — prespecification

Committed BEFORE any `infer_tree` or `harmonize` call on these data.
Nothing below may be revised after MetaArbor output exists; the one
generated annex (the Macosko group table) is produced by the frozen
rule in section 4 and committed before harmonization.

## 1. Question

Can frozen MetaArbor (Walk + harmonize, all frozen settings)
reconcile two genuinely independent taxonomies of the same biological
system at unequal resolution — Macosko 2015 whole-retina bipolar
clusters (8 coarse, numbered) vs Shekhar 2016 bipolar types (14
labels / 15 types) — recovering the known coarse-above-fine
hierarchy? This is rung 2 of the validation ladder (split-Allen was
too compatible; amygdala too hostile).

## 2. Data and provenance (all in `data/retina_bipolar/`)

- Macosko counts + cluster ids: Bioconductor ExperimentHub EH2690 /
  EH2691 (scRNAseq package resources; GEO GSE63472); 49,300 cells,
  clusters 1–39. Independent per-cell cluster file
  (mccarrolllab.org retina_clusteridentities.txt, 44,808 cells) kept
  as cross-check only.
- Shekhar counts + type labels: ExperimentHub EH2696 / EH2697 (GEO
  GSE81904; labels are the authors' Single Cell Portal SCP3
  assignments as curated by Bioconductor); 44,994 cells, 19 label
  levels.

## 3. Inclusion rules

- Shekhar: cells whose CLUSTER is one of the 14 bipolar labels
  {RBC, BC1A, BC1B, BC2, BC3A, BC3B, BC4, BC5A, BC5B, BC5C, BC5D,
  BC6, BC7, BC8/9}; MG, AC, Rod/Cone Photoreceptors,
  Doublets/Contaminants, and unlabeled cells are excluded.
- Macosko: cells with cluster in 26..33 (the authors' bipolar
  clusters; Hemberg-curated identity map and Shekhar 2016's
  "~5,500 BCs" reanalysis both attest 26–33 = bipolar).
- Genes: intersection of the two gene name lists, upper-cased for
  matching; ties broken by first occurrence.
- Namespaces: `she|<type>`, `mac|c<26..33>`.

## 4. Truth

### 4a. Shekhar-side hierarchy (published; Shekhar 2016)

    root
    ├─ RBC                      (rod bipolar; Prkca)
    └─ cone bipolar             (Scgn)
       ├─ OFF: BC1A BC1B BC2 BC3A BC3B BC4        (Grik1)
       └─ ON:  BC5A BC5B BC5C BC5D BC6 BC7 BC8/9  (Grm6, Isl1)

### 4b. Macosko-side group truth (frozen marker protocol)

LIMITATION, declared: the per-cluster fine mapping (Shekhar Fig. S2
reanalysis of Macosko cells) could not be independently obtained
from any accessible source, so Macosko clusters carry GROUP-level
truth only, derived by this rule (pseudobulk log1p-CPM per cluster,
z-scores across the 8 clusters, paper-cited markers only):

1. RBC: clusters with z(Prkca) − z(Scgn) > 1.
2. Remaining clusters: ON if (z(Grm6) + z(Isl1))/2 > z(Grik1),
   else OFF.

The resulting table is generated once by `retina_k2_prep.py`,
committed as `retina_k2_macosko_groups.csv`, and never revised.
Expression is used for this truth derivation, but never MetaArbor
output and never cross-dataset information.

### 4c. Truth tree for triplet scoring

Label-space tree: root → RBC-group, CBC → (OFF group with its
Shekhar types; ON group with its Shekhar types); each Macosko
cluster attaches under its 4b group node as a leaf. Reference-
unresolved triplets (within groups on the Macosko side) are excluded
from recovery by construction of the metrics (structure-kept
conventions from comparison/otharmonizer/metrics.py).

## 5. Pipeline (frozen settings throughout; no tuning anywhere)

- `infer_tree` per atlas: n_hvg=2000, n_boot=50, seed=0 (the Allen
  configuration).
- `harmonize` K=2: n_hvg=1000, n_boot=200, stability propagated —
  frozen thresholds untouched.
- Decision dumps saved exactly as allen_frontier_dump.py does
  (selections, canonical maps, input trees) for containment and
  audit work.
- Primary estimator: FROZEN Walk. (Walk-v2 arms may be run later as
  a separate exploratory comparison; they are not part of this
  prespecification.)

## 6. Report and prespecified checks

Report: reciprocal meta-clades and their members; per-label status
(backbone / private / single_atlas / unplaced); containment (one-way)
counts and the frontier sweep; completeness accounting; structure-
kept triplet recovery and cophenetic vs the 4c truth tree for (a)
each inferred input tree, (b) the harmonized assembly, (c) strict and
default cuts; wrong-group placements (a Shekhar type of one group
sharing a reciprocal meta-clade with a Macosko cluster of another
group); stability = one seed replicate (seed 977) plus atlas-order
swap, reporting decision agreement.

Prespecified checks (pass/fail stated; failures reported, not
repaired):
- C1 completeness invariant holds (every input label accounted).
- C2 zero wrong-GROUP reciprocal merges.
- C3 assembly triplet recovery of the Shekhar-side truth >= the
  Shekhar input tree's own recovery − 0.02 (harmonization must not
  damage input structure; the Allen structural-retention criterion).
- C4 at least one Macosko cluster acquires a cross-atlas
  relationship (reciprocal or one-way) to a Shekhar type of the
  matching group — the minimum sign of coarse-fine reconciliation.

OTHarmonizer comparison and K=3 (Mouse Retina Cell Atlas) are
subsequent steps, not part of this K=2 prespecification.
