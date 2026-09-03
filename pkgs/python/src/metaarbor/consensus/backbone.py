"""Greedy consensus backbone (DESIGN.md step 6). CONTRACT STUBS.

Hierarchical greedy (corrected): process ancestors before descendants,
ranking candidates WITHIN each parent context (global support sorting is
pathological and cannot guarantee parents precede children). Candidate
classes separated first — BACKBONE (>= 2 datasets, sufficient eligible
support), PRIVATE (stable in one dataset + no correspondence elsewhere +
high predicted detection power in powered others, all three required),
UNKNOWN-elsewhere (never counted against, never called private).
Eligibility from `eligibility.call_v2` (Beta-binomial LOO posterior).
Accept iff `poset.compatible`; emit conflicts; polytomies where support
fails. Output: backbone + private branches + conflict graph + provenance
table keyed by stable canonical IDs (MA-C####).
"""
from __future__ import annotations


def greedy_backbone(candidates, trees, eligibility_calls):
    raise NotImplementedError("v1 contract; lands after sim gates")


def provenance_table(backbone, synonyms):
    raise NotImplementedError("v1 contract; lands after sim gates")
