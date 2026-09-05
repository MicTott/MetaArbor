"""Offline audit of unanchored labels in a harmonize_k3 run — the model
is NOT changed; this only classifies the saved one-way Walk evidence.

An input label is UNANCHORED when its assembled component contains no
multi-atlas meta-clade (it lives in a purely single-atlas tree of the
forest). Each unanchored label is classified (MetaNeighbor-style
terminology; the `evidence` column carries the annotation):

  one_way_match        a supported one-way Walk selection exists;
                       evidence = into_shared_family (agreeing
                       landings inside an existing shared family — the
                       decisive class) or outside_families (real
                       signal, but the target is itself unanchored)
  conflicting_matches  supported placements imply genuinely
                       incompatible families across atlas pairs
  no_supported_match   no supported placement anywhere; evidence =
                       distributed_evidence when the frozen Walk's
                       compactness gate fired (the vote mass was too
                       dispersed to support a concentrated match — NOT
                       proof of conflicting biology), else none
  insufficient_power   no supported placement and fewer than
                       --min-cells cells (cannot distinguish absence
                       from undersampling)
  atlas_specific       the label's lineage is absent from every other
                       atlas per user-supplied metadata (--absent CSV
                       with columns: atlas,label; without the file this
                       category cannot be assessed and is not assigned)

Usage:
    python audit_unanchored.py <run_dir> [--absent absent.csv]
                               [--min-cells 30]

Outputs in <run_dir>/:
  audit.csv               per-label classification with the evidence
  audit_panel.png         category x atlas panel + the decisive count
  backbone_collapsed.png  shared backbone: meta-clades + attached
                          private branches (single-atlas fine detail
                          collapsed)
  full_forest.png         the complete assembly under a synthetic
                          universe root (drawn dashed: it asserts
                          nothing biological)
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from metaarbor.style import OKABE_ITO, SIZES, TEXT_MID, TREE_GRAY, save_pub
from metaarbor.consensus.plot_reconciled import DS_COLORS, STATUS_MARK
from metaarbor.consensus.diagnostics import _style as pub_style


def load(run_dir):
    def j(name):
        with open(os.path.join(run_dir, name)) as fh:
            return json.load(fh)
    return (j("tree.json"), j("decisions.json"), j("canonical.json"),
            j("input_trees.json"), j("cell_counts.json"))


def components(nodes):
    """{node_id: component_id}; components of the assembled forest."""
    comp, cid = {}, 0
    kids = {}
    for i, nd in nodes.items():
        kids.setdefault(nd["parent"], []).append(i)
    for r in kids.get(None, []):
        cid += 1
        stack = [r]
        while stack:
            v = stack.pop()
            comp[v] = cid
            stack.extend(kids.get(v, []))
    return comp


def anchored_components(nodes, comp):
    return {comp[i] for i, nd in nodes.items()
            if len(nd["members"]) >= 2}


def member_index(nodes):
    idx = {}
    for i, nd in nodes.items():
        for ds, m in nd["members"].items():
            idx.setdefault((ds, m), i)
    return idx


def nearest_family(nodes, start):
    """Climb assembled parents from `start` to the nearest node with
    members from >= 2 atlases."""
    p = start
    while p is not None:
        if len(nodes[p]["members"]) >= 2:
            return p
        p = nodes[p]["parent"]
    return None


def ancestor_line(nodes, a, b):
    """True if a == b or one is an assembled-tree ancestor of the
    other."""
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        p = nodes[y]["parent"]
        while p is not None:
            if p == x:
                return True
            p = nodes[p]["parent"]
    return False


def target_family(nodes, midx, itrees, canon, dt, selected):
    """Map a one-way selection target (canonical node of atlas dt) to
    the nearest shared family in the assembly."""
    node = midx.get((dt, selected))
    if node is None:
        # climb dt's own input tree until a represented ancestor
        p = itrees[dt]["parent"].get(selected)
        while p not in (None, "root"):
            pc = canon[dt].get(p, p)
            if (dt, pc) in midx:
                node = midx[(dt, pc)]
                break
            p = itrees[dt]["parent"].get(p)
    return nearest_family(nodes, node) if node is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--absent", default=None,
                    help="CSV atlas,label rows: lineages known absent "
                         "from that atlas (metadata ground truth)")
    ap.add_argument("--min-cells", type=int, default=30)
    args = ap.parse_args()
    nodes, decisions, canon, itrees, cells = load(args.run_dir)

    absent = {}
    if args.absent:
        with open(args.absent) as fh:
            for row in csv.DictReader(fh):
                absent.setdefault(row["label"], set()).add(row["atlas"])

    comp = components(nodes)
    anch = anchored_components(nodes, comp)
    midx = member_index(nodes)
    atlases = sorted(itrees)

    rows = []
    for ds in atlases:
        for leaf in itrees[ds]["leaves"]:
            c = canon[ds].get(leaf, leaf)
            node = midx.get((ds, c)) or midx.get((ds, leaf))
            if node is None or comp.get(node) in anch:
                continue                          # anchored (or absent)
            n_cells = cells[ds].get(leaf, 0)
            fams, sels = {}, {}
            discordant = []
            for dt in atlases:
                if dt == ds:
                    continue
                rec = decisions.get(f"{ds}>{dt}", {}).get(c, {})
                if rec.get("relation") == "discordant":
                    # the frozen Walk found signal but gated the
                    # placement as incompatible with the target topology
                    discordant.append(dt)
                if rec.get("matched") and rec.get("selected"):
                    sels[dt] = (rec["selected"], rec.get("support"))
                    fam = target_family(nodes, midx, itrees, canon, dt,
                                        rec["selected"])
                    if fam is not None:
                        fams[dt] = fam
            others = [dt for dt in atlases if dt != ds]
            evidence = ""
            if absent.get(leaf.split("|", 1)[-1]) and \
                    all(dt in absent[leaf.split("|", 1)[-1]]
                        for dt in others):
                cat = "atlas_specific"
            elif fams:
                vals = list(fams.values())
                ok = all(ancestor_line(nodes, vals[0], v)
                         for v in vals[1:])
                if ok:
                    cat, evidence = "one_way_match", "into_shared_family"
                else:
                    cat = "conflicting_matches"
            elif sels:
                cat, evidence = "one_way_match", "outside_families"
            elif discordant:
                # compactness-gated: the evidence was too DISTRIBUTED to
                # support a concentrated match — this is absence of a
                # supported match, not proof of conflicting biology
                cat, evidence = "no_supported_match", "distributed_evidence"
            elif n_cells >= args.min_cells:
                cat = "no_supported_match"
            else:
                cat = "insufficient_power"
            rows.append({
                "atlas": ds, "label": leaf.split("|", 1)[-1],
                "category": cat, "evidence": evidence,
                "n_cells": n_cells,
                "canonical_used": c if c != leaf else "",
                "discordant_pairs": "; ".join(discordant),
                "selections": "; ".join(
                    f"{dt}:{s[0]}({s[1]:.2f})" if s[1] is not None
                    else f"{dt}:{s[0]}" for dt, s in sorted(sels.items())),
                "implied_families": "; ".join(
                    f"{dt}:{nodes[f]['display']}[{f}]"
                    for dt, f in sorted(fams.items())),
            })

    with open(os.path.join(args.run_dir, "audit.csv"), "w",
              newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    tab = Counter((r["atlas"], r["category"]) for r in rows)
    cats = ["one_way_match", "conflicting_matches",
            "no_supported_match", "insufficient_power",
            "atlas_specific"]
    print(f"unanchored labels: {len(rows)}")
    for ds in atlases:
        line = "  ".join(f"{c}={tab.get((ds, c), 0)}" for c in cats)
        print(f"  {ds}: {line}")
    coh = sum(1 for r in rows if r["category"] == "one_way_match" and
              r["evidence"] == "into_shared_family")
    print(f"DECISIVE: {coh}/{len(rows)} unanchored labels are "
          "one-way matches into an existing shared family "
          "(representable only by asymmetric fine-to-coarse "
          "attachment, not by strict reciprocity)")
    if not args.absent:
        print("note: atlas_specific not assessed (no --absent "
              "metadata supplied)")

    # ---- renders ---------------------------------------------------------
    colors = {k: DS_COLORS[i % len(DS_COLORS)]
              for i, k in enumerate(atlases)}

    def draw(ax, sub, universe=False, label_aliases=True):
        kids = {}
        for i, nd in sub.items():
            kids.setdefault(nd["parent"] if nd["parent"] in sub else None,
                            []).append(i)
        order, depth = [], {}

        def dfs(v, d):
            depth[v] = d
            ks = sorted(kids.get(v, []),
                        key=lambda x: (sub[x]["status"],
                                       sub[x]["display"]))
            if not ks:
                order.append(v)
            for cch in ks:
                dfs(cch, d + 1)
        for r in sorted(kids.get(None, []),
                        key=lambda x: (sub[x]["status"],
                                       sub[x]["display"])):
            dfs(r, 1)
        ys = {v: k for k, v in enumerate(order)}

        def yof(v):
            if v in ys:
                return float(ys[v])
            return float(np.mean([yof(cch) for cch in kids[v]]))
        for i in sub:
            x1, y1 = depth[i], yof(i)
            p = sub[i]["parent"] if sub[i]["parent"] in sub else None
            if p is not None:
                ax.plot([depth[p], depth[p], x1], [yof(p), y1, y1],
                        color=TREE_GRAY, lw=0.7)
            elif universe:
                ax.plot([0.2, 0.2, x1], [yof(i), y1, y1],
                        color=TREE_GRAY, lw=0.7, ls=":")
            nd = sub[i]
            mark, _f = STATUS_MARK.get(nd["status"], ("s", "full"))
            if len(nd["members"]) >= 2:
                ax.scatter([x1], [y1], s=24, marker="o",
                           color="#4a4a4a", zorder=3)
            else:
                ds1 = next(iter(nd["members"]), None)
                ax.scatter([x1], [y1], s=20, marker=mark,
                           facecolor=("white" if nd["status"] !=
                                      "private" else
                                      OKABE_ITO["vermillion"]),
                           edgecolor=(colors.get(ds1, "#4a4a4a")),
                           linewidth=1.0, zorder=3)
            texts = []
            for a in (nd["aliases"] if label_aliases
                      else [nd["display"]]):
                base = str(a)
                mark_aff = base.startswith("≈ ")
                b = base[2:] if mark_aff else base
                ds1 = b.split("|", 1)[0] if "|" in b else None
                texts.append((("≈ " if mark_aff else "") +
                              b.split("|", 1)[-1].split(":")[-1],
                              colors.get(ds1, "#4a4a4a")))
            for j, (txt, col) in enumerate(texts):
                ax.text(x1 + 0.1, y1 + (j - (len(texts) - 1) / 2) * .42,
                        txt, fontsize=4.8, va="center", color=col)
        if universe and kids.get(None):
            ax.text(0.15, np.mean([yof(r) for r in kids[None]]),
                    "universe\n(synthetic)", fontsize=5.5, ha="right",
                    va="center", color=TEXT_MID, style="italic")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    with pub_style() as plt:
        # 1) collapsed shared backbone: meta-clades + private + ancestors
        keep = {i for i, nd in nodes.items()
                if len(nd["members"]) >= 2 or nd["status"] == "private"}
        for i in list(keep):
            p = nodes[i]["parent"]
            while p is not None:
                keep.add(p)
                p = nodes[p]["parent"]
        sub = {i: nodes[i] for i in keep}
        fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(sub))))
        draw(ax, sub, label_aliases=True)
        n_meta = sum(len(nd['members']) >= 2 for nd in sub.values())
        ax.set_title(f"shared backbone: {n_meta} meta-clades + private "
                     "branches (single-atlas detail collapsed)",
                     loc="left", fontsize=SIZES["axis"],
                     fontweight="bold")
        save_pub(fig, os.path.join(args.run_dir, "backbone_collapsed"),
                 formats=("png",), dpi=200)

        # 2) complete forest under a synthetic universe root
        fig, ax = plt.subplots(
            figsize=(8, max(5, 0.16 * len(nodes))))
        draw(ax, dict(nodes), universe=True, label_aliases=True)
        ax.set_title("complete assembly — synthetic universe root "
                     "(dashed) joins the forest; it asserts nothing "
                     "biological", loc="left", fontsize=SIZES["axis"],
                     fontweight="bold")
        save_pub(fig, os.path.join(args.run_dir, "full_forest"),
                 formats=("png",), dpi=150)

        # 3) audit panel
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(cats))
        bottom = np.zeros(len(cats))
        for ds in atlases:
            vals = np.array([tab.get((ds, c), 0) for c in cats],
                            dtype=float)
            ax.bar(x, vals, bottom=bottom, color=colors[ds], label=ds,
                   width=0.6)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cats],
                           fontsize=6)
        ax.set_ylabel("unanchored labels", fontsize=SIZES["axis"])
        ax.legend(fontsize=SIZES["legend"], frameon=False)
        ax.set_title(f"unanchored-label audit (n={len(rows)}): "
                     f"{coh} point coherently into a shared family",
                     loc="left", fontsize=SIZES["axis"],
                     fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        save_pub(fig, os.path.join(args.run_dir, "audit_panel"),
                 formats=("png",), dpi=200)
    print("wrote audit.csv, audit_panel.png, backbone_collapsed.png, "
          "full_forest.png ->", args.run_dir)


if __name__ == "__main__":
    main()
