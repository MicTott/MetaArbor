# MetaArbor vs OTHarmonizer: frozen matched comparison

Allen WMB PL-ILA-ORB, 10Xv2 at subclass (23 labels, 22k cells) vs 10Xv3
at cluster (103 labels, 13.8k cells), identical inputs to both methods:
raw counts on the shared 4,724-gene panel plus each atlas's own label
column. Curated class labels and the curated cluster->subclass nesting
were held out of both methods' construction and used only for scoring.
MetaArbor was frozen (tag `v0.8-harmonize-frozen`) BEFORE OTHarmonizer
was run, so nothing here was tuned against the competitor.

## Arms

- **MetaArbor**: `harmonize()` at frozen settings, both input trees
  inferred from expression (`pkgs/python/examples/
  allen_compare_metaarbor.py`). Two runs: base_seed 211 (primary) and
  977 (stability). Dataset-order invariance is structural (sorted keys).
- **OTHarmonizer** (github.com/Duck-Boss/OTHarmonizer, Bioinformatics
  2026 btag506): tutorial pipeline verbatim — normalize 1e4, log1p,
  HVG (batch-aware, subset), scVI 80 epochs (seed 0, trained once and
  reused across all runs), `do_harmonization(latent, ...)`
  (`oth_allen.py`). Runs: its own automatic order (chose v2-first),
  explicit v2-first, explicit v3-first, plus 5 seeded replicates per
  order (`oth_replicates.py`) because `get_metacells` samples 100
  cells/annotation with unseeded `np.random.choice` — the method is
  stochastic even at fixed order and fixed latent.

## Scoring (`score_compare.py`, one function for every tree)

Per v3 cluster: find its node, climb to the first ancestor carrying v2
labels, compare that ancestor's v2 subclass set to curated truth:
exact / consistent_coarse (contains truth; breadth reported) /
wrong_lineage / root (no v2 ancestor) / missing. Topology agreement
uses OTHarmonizer's own TEDS/PCBS/AH-F1 against one shared curated
reference tree (single-cluster subclasses encoded as EQUAL nodes);
MetaArbor's tree is projected to the same label space (anonymous
inferred internals spliced; internal-clade members do not masquerade
as equality groups; affiliates excluded). Stability = per-cluster
predicted-set agreement between runs.

## Results (n = 103 clusters; full tables in comparison_*.csv)

| | exact | compat.-coarse | wrong | root | missing |
|---|---|---|---|---|---|
| MetaArbor (2 seeds) | 73–75 | 23–25 (median breadth 5) | 4 | 1 | 0 |
| OTHarmonizer (12 runs) | 71–84 (mean 78.7) | 0 | 2–8 | 13–27 (mean 19.7) | 0 |

- Placed consistent with curated truth (exact + compatible):
  MetaArbor 98/103 vs OTHarmonizer 71–84/103.
- Exact only: OTHarmonizer mean 78.7 vs MetaArbor 73–75 — a ~4-cluster
  edge that sits inside OTHarmonizer's own run-to-run band (71–84).
- Wrong lineage: comparable (4 vs mean 4.8).
- Unresolved at root: MetaArbor 1 vs OTHarmonizer mean 19.7 (19%).
- Topology: PCBS favors MetaArbor (0.914/0.920, above all 12
  OTHarmonizer runs, max 0.908); AH-F1 comparable (0.900/0.905 vs
  0.845–0.936); TEDS slightly favors OTHarmonizer (its best 0.840 vs
  0.748/0.765; OTHarmonizer mean ~0.78).
- Stability: MetaArbor 97.1% cross-seed agreement (deterministic at
  fixed seed). OTHarmonizer 79.2% within-order (range 71.8–85.4%) and
  78.7% between-order — ~21% of clusters change their predicted parent
  between runs on identical inputs; insertion order adds little beyond
  the sampling noise.

## Read

The methods sit at different points on the commit-vs-abstain axis:
OTHarmonizer commits every placed cluster to a single subclass and
sends the rest to root; MetaArbor abstains into compatible coarse
positions (concentrated under the v2 inferred tree's own unresolved
5-subclass IT clade) and leaves almost nothing unresolved. Where they
disagree in kind: OTHarmonizer leaves ~20 clusters with no parent at
all; MetaArbor keeps them in truth-compatible positions with the
under-resolution visible and quantified.

## Caveats

- One benchmark pair, mild batch axis (10Xv2 vs 10Xv3, same lab and
  tissue) — favorable terrain for OTHarmonizer's scVI-dependent costs;
  the batch-severity axis where MetaArbor's integration-free evidence
  should separate is untested here.
- scVI trained once and shared across all OTHarmonizer runs, removing
  its training variance from the stability estimate (favorable to
  OTHarmonizer).
- The scVI latent was fit on tutorial-preprocessed (log-normalized)
  input with a zinb likelihood, exactly as the OTHarmonizer tutorial
  does; the warning it triggers is theirs, reproduced faithfully.
- MetaArbor's environment: Python 3.12, numpy 1.26.4, scipy 1.13.1,
  POT 0.9.3, scvi-tools 1.2.1, anndata 0.10.9, scArches 0.6.1.
