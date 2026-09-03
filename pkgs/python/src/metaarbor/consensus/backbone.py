"""Greedy consensus backbone (DESIGN.md step 6). CONTRACT STUBS.

Rank candidates by (supporting / eligible) donors — eligibility from
`eligibility.call` — accept in order iff `poset.compatible` with all
accepted; emit conflicts to the conflict graph; stop resolving where
support fails (polytomies, never forced binary splits). Output: backbone
+ private branches + conflict graph + provenance table keyed by stable
canonical IDs (MA-C####; display names and synonyms never change the ID).
"""
from __future__ import annotations


def greedy_backbone(candidates, trees, eligibility_calls):
    raise NotImplementedError("v1 contract; lands after sim gates")


def provenance_table(backbone, synonyms):
    raise NotImplementedError("v1 contract; lands after sim gates")
