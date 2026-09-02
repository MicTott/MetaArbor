"""The frozen TreeNeighbor walk estimator (NOTES.md items 4, 7-9; frozen
2026-09-02). Votes navigate, AUROC contrasts decide:

- navigation: each positive cell votes for its argmax mean-rank training
  leaf; a child collects the votes falling in its subtree; the walk enters
  the plurality child (union-AUROC is size-biased at coarse splits)
- override: >= 0.90 vote share for one child descends without the AUROC test
  (distinct siblings can both saturate one-vs-all)
- stop: descend only when the entered child's AUROC beats the second-best
  sibling's by a practical margin (0.01) under a paired bootstrap, and the
  parent is not significantly better
- compactness: fraction of positive cells whose argmax leaf lies in the
  selected node's parent-context subtree

Bootstrap draws use the portable MINSTD stream with a per-query seed
(base_seed + query rank in the sorted query list), consuming indices in a
fixed documented order, so R and Python implementations make bit-comparable
decisions.

FROZEN defaults: alpha=0.05, n_boot=200, min_auroc=0.6, margin=0.01,
vote_override=0.9, min_compact=0.7. Changing them on validation data
un-freezes the estimator; do not.
"""
from __future__ import annotations

import numpy as np

from .kernel import auroc, node_scores, node_mean_scores
from .rng import Minstd
from .tree import ancestors, leaves_under


def _boot_delta(scores1, scores2, positive, rng, n_boot=200):
    """Paired bootstrap of auroc(scores1) - auroc(scores2), resampling test
    cells stratified by positive/negative. Draw order (must match R):
    per iteration, n_pos positive indices then n_neg negative indices."""
    ip = np.flatnonzero(positive)
    ineg = np.flatnonzero(~positive)
    n_pos, n_neg = len(ip), len(ineg)
    pos_mask = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    delta = np.empty(n_boot)
    for b in range(n_boot):
        sp = ip[rng.indices(n_pos, n_pos)]
        sn = ineg[rng.indices(n_neg, n_neg)]
        idx = np.concatenate([sp, sn])
        delta[b] = auroc(scores1[idx], pos_mask) - auroc(scores2[idx], pos_mask)
    return delta


def select_node(cache, test_labels, query, tree, seed,
                alpha=0.05, n_boot=200, min_auroc=0.6,
                margin=0.01, vote_override=0.9):
    positive = np.asarray(test_labels) == query
    rng = Minstd(seed)
    ms = cache["V"][positive] / cache["leaf_sizes"]
    top_leaf = np.asarray(cache["leaves"])[ms.argmax(axis=1)]

    def votes_for(kids):
        return np.asarray([
            np.isin(top_leaf, leaves_under(tree, k)).sum() for k in kids
        ]) / len(top_leaf)

    def n_scores(node):
        return node_scores(cache, leaves_under(tree, node))

    def n_auc(node):
        return auroc(n_scores(node), positive)

    current = "root"
    path = []
    while True:
        kids = tree["children"][current]
        if not kids:
            break
        if len(kids) == 1:
            current = kids[0]
            continue
        v = votes_for(kids)
        order = np.argsort(-v, kind="stable")
        best, second = kids[order[0]], kids[order[1]]
        sib_lo = par_lo = np.nan
        override = v[order[0]] >= vote_override
        if override:
            stop = False
        else:
            d_sib = _boot_delta(n_scores(best), n_scores(second),
                                positive, rng, n_boot)
            sib_lo = np.quantile(d_sib, alpha)
            concentrated = sib_lo > margin
            if current != "root" and concentrated:
                d_par = _boot_delta(n_scores(current), n_scores(best),
                                    positive, rng, n_boot)
                par_lo = np.quantile(d_par, alpha)
            stop = (not concentrated) or (not np.isnan(par_lo) and par_lo > 0)
        path.append({"id": current if stop else best,
                     "vote": float(v[order[0]]), "sib_lo": float(sib_lo),
                     "par_lo": float(par_lo), "override": bool(override),
                     "stopped": bool(stop)})
        if stop:
            break
        current = best
    sel_auc = np.nan if current == "root" else n_auc(current)
    matched = np.isfinite(sel_auc) and sel_auc >= min_auroc
    return {"query": query, "selected": current if matched else None,
            "auroc": float(sel_auc), "matched": bool(matched),
            "at_root": current == "root", "final": current, "path": path}


def compactness(cache, positive, tree, selected):
    parent = tree["parent"][selected]
    ctx = selected if parent in (None, "root") else parent
    ms = cache["V"][positive] / cache["leaf_sizes"]
    top_leaf = np.asarray(cache["leaves"])[ms.argmax(axis=1)]
    return float(np.isin(top_leaf, leaves_under(tree, ctx)).mean())


def baseline_map(cache, test_labels, tree, S_dir, base_seed=7,
                 alpha=0.05, n_boot=200, min_auroc=0.6,
                 min_compact=0.7, margin=0.01, vote_override=0.9):
    """Map every test population onto `tree`. `S_dir`: dict query ->
    {leaf: symmetrized AUROC}, used only for the no-signal call at root
    stops. Per-query seed = base_seed + rank in the sorted query list."""
    test_labels = np.asarray(test_labels)
    queries = sorted(set(test_labels))
    out = []
    for qi, q in enumerate(queries):
        sel = select_node(cache, test_labels, q, tree, seed=base_seed + qi,
                          alpha=alpha, n_boot=n_boot, min_auroc=min_auroc,
                          margin=margin, vote_override=vote_override)
        comp = (compactness(cache, test_labels == q, tree, sel["selected"])
                if sel["matched"] else np.nan)
        has_signal = max(S_dir[q].values()) >= min_auroc
        is_leaf = sel["matched"] and not tree["children"][sel["selected"]]
        if sel["at_root"]:
            relation = "discordant" if has_signal else "unmatched"
        elif not sel["matched"]:
            relation = "unmatched"
        elif not np.isnan(comp) and comp < min_compact:
            relation = "discordant"
        elif is_leaf:
            relation = "leaf"
        else:
            relation = "family"
        out.append({"query": q, "selected": sel["selected"],
                    "auroc": sel["auroc"], "compactness": comp,
                    "relation": relation})
    return out
