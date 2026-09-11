# retina_k2_audit_figure.png — provenance

Seven-panel audit figure (reviewer-specified): Macosko input tree,
Shekhar input tree, group-level truth, MetaArbor interleaved
(committed interleave_core_ancestry applied to the frozen
retina_k2_tree.json), OTHarmonizer default / mac-first / she-first.

Encodings: truth-group background bands (RBC/OFF/ON); label text and
marker colored by dataset (Macosko dark red, Shekhar dark blue);
reciprocal or explicit merges as black squares with black bold text;
certified backbone edges black; containment-supported single-atlas
placements dashed gray (label's own one-way call matched); inherited
input ancestry light gray; OTH root-dumped labels in a shaded region
below each OTH panel. All panels show all 22 labels in a shared
canonical vertical order (truth-group order; children sorted by
minimum canonical rank).

Raw triplet denominators (correct / reference-resolved), computed
from the committed artifacts with the structure-kept metrics:

  arm              truth        she-retention  mac-retention
  mac_input           51/51          0/0            51/51
  she_input          309/309      309/309            0/0
  MA_assembly       1203/1291    309/309           43/51
  MA_interleaved    1291/1291    309/309           51/51
  OTH_default        177/1291     23/309            0/51
  OTH_mac_first      203/1291     31/309            0/51
  OTH_she_first      103/1291      0/309            0/51

Note the reviewer's small-denominator concern: mac-retention rests
on 51 resolved triplets, so OTH's 0/51 (all three runs) is a genuine
zero, not a small-sample artifact; MA's assembly deficit was 43/51
and the interleaved repair reaches 51/51.
