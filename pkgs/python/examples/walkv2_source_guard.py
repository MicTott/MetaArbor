"""Walk-v2 source-children breadth guard — exploratory accuracy check
on the committed Allen dumps (no rule frozen; raw signal only).

The guard: where a source node has labeled children with their own
one-way Walk calls, the parent spans the target split if different
children's calls land on different target branches. For every v3
internal canonical node with >= 2 children carrying matched calls
into v2: guard says CONTAINED when all children's targets lie on one
root-path in the v2 tree, SPANS otherwise. Truth: the node's curated
cluster set maps to > 1 v2 subclass.

Result on the committed dumps (recorded 2026-09-10): 76 evaluable
nodes; guard-SPANS precision 1.00 (14/14), recall 0.56 (11 true
spanning nodes missed when children's calls collapse onto one coarse
target). Perfect precision is the property a veto needs: when this
guard fires it is always right; its misses fall through to the
per-cell breadth statistic (still unfrozen — see
walkv2_breadth_freeze.py for the recorded v1-v3 failures).

Run: python examples/walkv2_source_guard.py
"""
import csv
import gzip
import json
import os

from metaarbor.tree import leaves_under

HERE = os.path.dirname(os.path.abspath(__file__))
MA = os.path.join(HERE, "harmonize_demo")
FX = os.path.join(HERE, "..", "..", "fixtures")

dec = json.load(open(os.path.join(MA, "allen_decisions.json")))["v3>v2"]
canon = json.load(open(os.path.join(MA, "allen_canonical.json")))
trees = json.load(open(os.path.join(MA, "allen_input_trees.json")))
tb, tv2 = trees["v3"], trees["v2"]
lv = list(csv.reader(gzip.open(os.path.join(FX, "tree_levels_b.csv.gz"),
                               "rt")))
sub_of = {f"v3|{r[3]}": f"v2|{r[1]}" for r in lv[1:]}


def truth_subs(node):
    return {sub_of[l] for l in leaves_under(tb, node) if l in sub_of}


def path(x):
    out = [x]
    while tv2["parent"].get(x) not in (None, "root"):
        x = tv2["parent"][x]
        out.append(x)
    return out


rows = []
for n in sorted(set(canon["v3"].values())):
    kids = sorted({canon["v3"].get(k, k)
                   for k in tb["children"].get(n, [])} - {n})
    if len(kids) < 2:
        continue
    tgts = []
    for k in kids:
        r = dec.get(k)
        if r and r.get("matched") and r.get("selected"):
            tgts.append(r["selected"])
    if len(tgts) < 2:
        continue
    same_branch = all(t == tgts[0] or t in path(tgts[0])
                      or tgts[0] in path(t) for t in tgts)
    guard = "contained" if same_branch else "spans"
    truth = "spans" if len(truth_subs(n)) > 1 else "contained"
    rows.append({"node": n, "n_child_calls": len(tgts),
                 "guard": guard, "truth": truth})
with open(os.path.join(HERE, "walkv2", "source_guard_allen.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
tp = sum(1 for r in rows if r["guard"] == "spans"
         and r["truth"] == "spans")
fp = sum(1 for r in rows if r["guard"] == "spans"
         and r["truth"] == "contained")
fn = sum(1 for r in rows if r["guard"] == "contained"
         and r["truth"] == "spans")
print(f"nodes {len(rows)} | spans precision "
      f"{tp / (tp + fp) if tp + fp else float('nan'):.2f} "
      f"({tp}/{tp + fp}) | recall {tp / (tp + fn):.2f}")
