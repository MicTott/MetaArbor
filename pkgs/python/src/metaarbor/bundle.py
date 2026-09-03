"""The default MetaArbor result bundle: summary table + annotated alignment
tree + evidence heatmap (+ transport heatmap and disagreement query-path
plots when a coupling is supplied), written as PNG and PDF, with native
matplotlib objects returned for restyling."""
from __future__ import annotations

import os

import numpy as np

from .accessors import walk_traces, write_csv
from .phylogram import plot_alignment_phylogram
from .plots import (plot_alignment_tree, plot_evidence_heatmap,
                    plot_query_path, plot_transport_heatmap)
from .summary import alignment_summary, transport_summary, walk_summary


def result_bundle(cache, test_labels, tree, S_dir, out_dir, prefix="metaarbor",
                  pi=None, pi_rows=None, pi_cols=None, family_of_leaf=None,
                  base_seed=7, formats=("png", "pdf"), dpi=200,
                  query_paths="disagreements", **walk_kwargs):
    """Run the frozen interpretation layer end to end. Returns
    {"summary": rows, "figures": {name: (fig, ax)}, "files": [paths]}."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    wk = walk_summary(cache, test_labels, tree, S_dir, base_seed=base_seed,
                      **walk_kwargs)
    if pi is not None:
        ts = transport_summary(pi, pi_rows, pi_cols, family_of_leaf, tree=tree)
        rows = alignment_summary(wk, ts, tree)
    else:
        rows = [dict(r, transport_family=None, transport_node=None,
                     transport_mass=np.nan, transport_eff_leaves=np.nan,
                     transport_bin="unmatched",
                     agreement="walk_only" if r["walk_selected"] else
                               "both_unmatched") for r in wk]
    files.append(write_csv(rows, os.path.join(out_dir,
                                              f"{prefix}_summary.csv")))
    figures = {}

    def save(name, fig):
        for ext in formats:
            p = os.path.join(out_dir, f"{prefix}_{name}.{ext}")
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            files.append(p)

    fig, ax = plot_alignment_phylogram(rows, tree)
    figures["alignment_phylogram"] = (fig, ax)
    save("alignment_phylogram", fig)

    fig, ax = plot_evidence_heatmap(
        cache, test_labels, tree, family_of_leaf=family_of_leaf,
        walk_selected={r["query"]: r["walk_selected"] for r in rows},
        transport_node={r["query"]: r.get("transport_node") for r in rows}
        if pi is not None else None)
    figures["evidence_heatmap"] = (fig, ax)
    save("evidence_heatmap", fig)

    if pi is not None:
        fig, ax = plot_transport_heatmap(pi, pi_rows, pi_cols, tree,
                                         family_of_leaf=family_of_leaf)
        figures["transport_heatmap"] = (fig, ax)
        save("transport_heatmap", fig)

    if query_paths:
        targets = ([r["query"] for r in rows
                    if r["agreement"] not in ("agree",)]
                   if query_paths == "disagreements" else list(query_paths))
        if targets:
            tr = walk_traces(cache, test_labels, tree, base_seed=base_seed,
                             **walk_kwargs)
            for q in targets:
                prow = None
                if pi is not None and q in list(pi_rows):
                    prow = np.asarray(pi)[list(pi_rows).index(q)]
                fig, axes = plot_query_path(tr, q, transport_row=prow,
                                            col_names=pi_cols, tree=tree)
                safe = "".join(ch if ch.isalnum() else "_" for ch in q)[:40]
                figures[f"query_path_{safe}"] = (fig, axes)
                save(f"query_path_{safe}", fig)
    return {"summary": rows, "figures": figures, "files": files}
