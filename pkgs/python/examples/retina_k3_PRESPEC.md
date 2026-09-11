# Retina bipolar K=3 benchmark — prespecification

Committed BEFORE any `infer_tree` or `harmonize` call on the MRCA
data. Nothing below may be revised after MetaArbor output exists.
Everything frozen in the K=2 prespecification stays frozen here; the
K=2 artifacts (input trees, Macosko group table, truth) are REUSED
verbatim, never rederived.

## 1. Question

Two questions, one run:

1. Can frozen MetaArbor reconcile THREE independently constructed
   taxonomies of the same system — Macosko 2015 coarse clusters (8),
   Shekhar 2016 types (14 labels / 15 types), MRCA 2024 types (15) —
   recovering both the coarse-above-fine hierarchy (K=2's result)
   and the published fine-fine correspondence between Shekhar and
   MRCA?
2. The reviewer's pre-migration gate for quotient assembly: one real
   K=3 dataset through `quotient_assemble`, scored under the DAG
   result contract (ASSEMBLY2 Section 11). The quotient is the
   PRIMARY assembly of this benchmark; the legacy `harmonize()`
   assembly is reported alongside for comparison.

## 2. Data and provenance

Macosko and Shekhar: exactly the K=2 subsets, reused from
`examples/retina_k2/` (mac_bipolar.mtx, she_bipolar.mtx, labels,
shared_genes.txt) and `data/retina_bipolar/`. The K=2 input trees
(`retina_k2_input_trees.json`) are reused verbatim.

MRCA: "MRCA- scRNA-seq of the mouse retina - bipolar cell
subclass.h5ad" from Zenodo record 10815031 (CC-BY-4.0), the data of
Li et al. 2024 iScience "Comprehensive single-cell atlas of the
mouse retina" (Rui Chen lab; PMID 38328114), saved as
`data/retina_bipolar/mrca_bipolar.h5ad`. File facts recorded at
prespecification: 147,523 cells x 32,152 genes; `raw/X` holds
integer counts (used); `X` is normalized (not used); genes matched
by `var/feature_name`; fine labels in `obs/author_cell_type`
(15 categories: BC1A, BC1B, BC2, BC3A, BC3B, BC4, BC5A, BC5B, BC5C,
BC5D, BC6, BC7, BC8, BC9, RBC — Shekhar's nomenclature, adopted by
the MRCA authors, with BC8/BC9 resolved where Shekhar reported a
BC8/9 mixture).

## 3. Inclusion rules (independence)

MRCA integrated public datasets, INCLUDING Shekhar GSE81904 itself
(19,862 of the 147,523 BC cells) and Ma et al. 2023 (658 cells).
Including those would make the third atlas partially the second.

- KEEP exactly the cells with `obs/accession == "GSE243413"` — the
  newly generated Chen-lab data (127,003 cells; references
  Chen_CD73 122,563 + WT_CD73 4,333 + CD90.1 107). This excludes
  every cell from Shekhar 2016 and Ma 2023.
- Labels: `obs/author_cell_type`, namespace `mrca|<type>`.
- Per-type cell counts after exclusion, recorded now: BC1A 30,
  BC1B 35, BC2 216, BC3A 7,602, BC3B 10,593, BC4 154, BC5A 16,885,
  BC5B 9,145, BC5C 6,149, BC5D 4,259, BC6 18,788, BC7 14,372,
  BC8 2,753, BC9 2,614, RBC 33,408.
- SUBSAMPLING (computational only, frozen before any run): types
  with more than 2,000 kept cells are downsampled to 2,000 with
  `numpy.random.default_rng(0)`; types at or below 2,000 keep all
  cells. Expected total ~22.4k cells.
- Genes: intersection of the K=2 shared gene list with MRCA
  `feature_name`, upper-cased, first occurrence wins (the K=2
  rule). The three-atlas analysis runs on this common panel; the
  mac/she matrices are re-subset by column to it (their K=2 input
  TREES are still reused unchanged — tree inference happened on the
  K=2 panel and is not repeated).

DECLARED LIMITATION: the new MRCA data are CD73-enriched, leaving
four OFF types rare (BC1A, BC1B, BC2, BC4; 30-216 cells). Frozen
MetaArbor abstaining on those types is an acceptable outcome;
misassigning them is not. All counts reported with raw denominators.

## 4. Truth

### 4a. Shekhar-side hierarchy — unchanged from K=2

    root
    ├─ RBC                      (Prkca)
    └─ cone bipolar             (Scgn)
       ├─ OFF: BC1A BC1B BC2 BC3A BC3B BC4        (Grik1)
       └─ ON:  BC5A BC5B BC5C BC5D BC6 BC7 BC8/9  (Grm6, Isl1)

### 4b. MRCA-side hierarchy — same tree by published name identity

The MRCA authors state their 15 BC clusters correspond to the
previously annotated (Shekhar) types with no novel type. Groups:
RBC; OFF = {BC1A, BC1B, BC2, BC3A, BC3B, BC4}; ON = {BC5A, BC5B,
BC5C, BC5D, BC6, BC7, BC8, BC9}.

### 4c. Macosko group truth — the COMMITTED K=2 table

`retina_k2/retina_k2_macosko_groups.csv` is reused verbatim
(c26 RBC_group; c27-c29 OFF; c30-c33 ON). It is not rederived.

### 4d. Fine-fine correspondence (the new truth K=2 did not have)

- `she|X <-> mrca|X` for the 13 name-identical types
  {RBC, BC1A, BC1B, BC2, BC3A, BC3B, BC4, BC5A, BC5B, BC5C, BC5D,
  BC6, BC7}.
- `she|BC8_9` corresponds to the FAMILY {mrca|BC8, mrca|BC9}
  (Shekhar could not separate them; MRCA can). Scored separately
  from the 13, as family-consistent placement.

### 4e. Truth tree for triplet scoring

root → RBC_group, CBC; CBC → OFF, ON. Within RBC_group a cherry
{she|RBC, mrca|RBC}; within OFF/ON one cherry per name-identical
pair {she|X, mrca|X}, plus the triple {she|BC8_9, mrca|BC8,
mrca|BC9} under ON. Macosko clusters attach directly under their 4c
group node (unresolved within group). This is strictly finer truth
than K=2's.

## 5. Pipeline (frozen settings; no tuning anywhere)

- MRCA input tree: `infer_tree` with n_hvg=2000, n_boot=50, seed=0
  (identical to K=2) on the Section 3 subset over the common panel.
  Macosko and Shekhar input trees: the committed K=2 trees, reused.
- `harmonize` K=3 (datasets ordered mac, she, mrca): n_hvg=1000,
  n_boot=200, all frozen thresholds untouched. Decision dumps
  (selections, canonical maps, input trees) saved in the K=2 format.
- PRIMARY assembly: `quotient_assemble` on those dumps. Legacy
  `harmonize()` assembly reported alongside.
- Scoring follows the result contract: if the quotient is a forest,
  score the forest; if a DAG, report certificates and score over
  ALL compatible forest projections (exhaustive when the
  certificate count allows, as in the Allen gate).

## 6. Prespecified checks (pass/fail stated; failures reported,
never repaired post hoc)

- C1 completeness: every input label of all three atlases accounted
  (quotient I1 asserts this by construction; the legacy assembly is
  checked as in K=2).
- C2 zero wrong-GROUP reciprocal merges, for all three pairs.
- C3 structural retention: triplet recovery of the 4e truth
  restricted to each fine atlas's own labels must be >= that input
  tree's own recovery − 0.02 (she side and mrca side separately).
  For a DAG result the bound must hold at the MINIMUM over
  compatible projections.
- C4 every Macosko cluster acquires at least one cross-atlas
  relationship (reciprocal merge or directional annotation) to a
  she or mrca node of its matching group.
- C5 fine-fine recovery: (a) zero cross-group she<->mrca reciprocal
  merges; (b) at least 8 of the 13 name-identical pairs form
  reciprocal merges; (c) name-mismatched WITHIN-group merges are
  reported with supports, not auto-failed (BC5 subtypes are known
  hard); (d) BC8_9: family-consistent placement of she|BC8_9
  relative to {mrca|BC8, mrca|BC9} reported separately.
- C6 quotient K=3 gate (the migration requirement): internal
  invariant assertions pass; every vertex holding members of all
  three atlases is group-consistent; forest-vs-DAG stated per the
  contract with certificates and ledger printed in full; the
  three-atlas meta-clade count reported.

## 7. Stability

- Atlas-order permutation: rerun with order (mrca, she, mac);
  report pairwise decision agreement as in K=2.
- Seed replicate: seed 977 arm, decision agreement reported.

OTHarmonizer K=3 comparison is a subsequent step, not part of this
prespecification.
