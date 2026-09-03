"""MetaArbor interpretation layer: one-row-per-query alignment summaries and
the joint Walk/Transport agreement categories.

This layer exposes and combines quantities the frozen estimators already
compute — it introduces no new thresholds and never alters a decision.
Walk rows re-derive the walk with the same per-query seeds as
`baseline_map`, so summarized decisions are exactly the map's decisions
(re-asserted by the interpretation gate).

Agreement categories (documented, threshold-free — estimator vs estimator,
no ground truth involved):
  agree                        same selected node
  topologically_equivalent     different node ids with IDENTICAL descendant
                               leaf sets (unary-chain collapse) — a notation
                               artifact, not a disagreement
  same_branch_different_depth  one selection is a proper ancestor of the
                               other (genuinely different depth)
  conflicting_branch           selections on disjoint branches
  walk_only                    Walk matched; Transport has no mass
  transport_only               Transport has mass; Walk unmatched
  both_unmatched               neither commits
Transport mass bins reuse the frozen readout (>= 0.9 confident,
>= 0.5 moderate, else diffuse); they describe concentration, not
correctness.
"""
from __future__ import annotations

import numpy as np

from .tree import ancestors, leaves_under
from .walk import baseline_map, select_node


def _node_depth(tree, node):
    return len(ancestors(tree, node))


def walk_summary(cache, test_labels, tree, S_dir, base_seed=7, **walk_kwargs):
    """One row per query for MetaArbor-Walk: selected node, its depth and
    parent, the stopping split's best/second sibling AUROCs and mean
    sibling ΔAUROC, the bootstrap frequency of the final decision (fraction
    of the decisive test's existing draws agreeing with the decision taken;
    NaN when the vote override decided), and the relation."""
    test_labels = np.asarray(test_labels)
    queries = sorted(set(test_labels))
    mapping = {m["query"]: m for m in
               baseline_map(cache, test_labels, tree, S_dir,
                            base_seed=base_seed, **walk_kwargs)}
    rows = []
    for qi, q in enumerate(queries):
        sel = select_node(cache, test_labels, q, tree, seed=base_seed + qi,
                          trace=True, **walk_kwargs)
        m = mapping[q]
        last = sel["path"][-1] if sel["path"] else None
        best_auc = second_auc = np.nan
        if last is not None and sel["trace"]:
            stop_rows = [t for t in sel["trace"]
                         if t["split_at"] == sel["trace"][-1]["split_at"]]
            by_child = {t["child"]: t["child_auroc"] for t in stop_rows}
            best_auc = by_child.get(last["best"], np.nan)
            second_auc = by_child.get(last["second"], np.nan)
        if last is None:
            support = np.nan
        elif last["override"]:
            support = np.nan            # override decided; vote is the support
        elif last["stopped"] and not np.isnan(last["par_gt0"]) and last["par_lo"] > 0:
            support = last["par_gt0"]   # stopped because parent clearly better
        elif last["stopped"]:
            support = 1.0 - last["sib_gt_margin"]  # stopped: draws <= margin
        else:
            support = last["sib_gt_margin"]        # descended into a leaf
        rows.append({
            "query": q,
            "walk_selected": m["selected"],
            "walk_depth": (np.nan if m["selected"] is None
                           else _node_depth(tree, m["selected"])),
            "walk_parent": (None if m["selected"] is None
                            else tree["parent"][m["selected"]]),
            "walk_auroc": m["auroc"],
            "walk_best_sib_auroc": best_auc,
            "walk_second_sib_auroc": second_auc,
            "walk_sib_delta": np.nan if last is None else last["sib_delta"],
            "walk_vote": np.nan if last is None else last["vote"],
            "walk_override": None if last is None else bool(last["override"]),
            "walk_decision_support": support,
            "walk_relation": m["relation"],
        })
    return rows


def transport_summary(pi, row_names, col_names, family_of_leaf, tree=None):
    """One row per query for MetaArbor-Transport: argmax family (and its
    node id in `tree` when supplied), row-normalized mass there, effective
    target leaves from row entropy, and the frozen mass-concentration bin.
    `family_of_leaf`: dict target leaf -> its grouping label (an explicit
    level of the target taxonomy, chosen by the caller)."""
    fams = sorted(set(family_of_leaf.values()))
    fam_idx = {f: [j for j, c in enumerate(col_names)
                   if family_of_leaf[c] == f] for f in fams}
    rows = []
    for i, q in enumerate(row_names):
        r = np.asarray(pi[i], dtype=float)
        tot = r.sum()
        if tot <= 0:
            rows.append({"query": q, "transport_family": None,
                         "transport_node": None, "transport_mass": 0.0,
                         "transport_eff_leaves": np.nan,
                         "transport_bin": "unmatched"})
            continue
        p = r / tot
        fam_mass = {f: float(p[fam_idx[f]].sum()) for f in fams}
        best = max(fam_mass, key=fam_mass.get)
        mass = fam_mass[best]
        pn = p[p > 0]
        eff = float(np.exp(-(pn * np.log(pn)).sum()))
        node = None
        if tree is not None:
            cand = [n for n in tree["parent"]
                    if n != "root" and
                    sorted(l for l in leaves_under(tree, n) if l in col_names)
                    == sorted(l for l, f in family_of_leaf.items() if f == best)]
            node = min(cand, key=lambda n: _node_depth(tree, n)) if cand else None
        rows.append({"query": q, "transport_family": best,
                     "transport_node": node, "transport_mass": mass,
                     "transport_eff_leaves": eff,
                     "transport_bin": ("confident" if mass >= 0.9 else
                                       "moderate" if mass >= 0.5 else
                                       "diffuse")})
    return rows


def agreement(walk_selected, transport_node, tree):
    """The documented agreement category for one query. Raw node identities
    are preserved in the summary columns; this category only interprets
    them."""
    w, t = walk_selected, transport_node
    if w is None and t is None:
        return "both_unmatched"
    if w is None:
        return "transport_only"
    if t is None:
        return "walk_only"
    if w == t:
        return "agree"
    if leaves_under(tree, w) == leaves_under(tree, t):
        return "topologically_equivalent"
    if w in ancestors(tree, t) or t in ancestors(tree, w):
        return "same_branch_different_depth"
    return "conflicting_branch"


def alignment_summary(walk_rows, transport_rows, tree):
    """Join the two per-query summaries and add the agreement category."""
    trans = {r["query"]: r for r in transport_rows}
    out = []
    for w in walk_rows:
        t = trans.get(w["query"], {"transport_family": None,
                                   "transport_node": None,
                                   "transport_mass": np.nan,
                                   "transport_eff_leaves": np.nan,
                                   "transport_bin": "unmatched"})
        row = dict(w)
        row.update({k: v for k, v in t.items() if k != "query"})
        row["agreement"] = agreement(w["walk_selected"],
                                     t.get("transport_node"), tree)
        out.append(row)
    return out
