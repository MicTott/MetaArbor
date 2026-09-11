"""Quotient from CERTIFIED merges — retina K=2 and K=3 (review items
1-2): the certification layer (frozen greedy_backbone: eligibility,
detectability, stability, MIN_DATASETS, MIN_SUPPORT) outputs the
prequalified reciprocal merges; quotient assembly consumes those and
preserves the complete trees. Raw-mode quotient (every matched
reciprocal pair) is rerun alongside purely as the diagnostic
contrast — it is no longer a result.

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/retina_quotient_certified.py
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "comparison",
                                "otharmonizer"))
from metrics import MyNode, triplet_scores  # noqa: E402
from metaarbor.consensus.candidates import canonical_nodes  # noqa
from metaarbor.consensus.harmonize import harmonize  # noqa: E402
from metaarbor.consensus.quotient import quotient_assemble  # noqa


def certified_of(harm):
    """Prequalified reciprocal merges from the frozen certification
    layer: accepted multi-dataset backbone nodes."""
    out = []
    for nd in harm["backbone"]["nodes"]:
        if nd["status"] == "backbone" and len(nd["members"]) >= 2:
            out.append({"members": dict(nd["members"]),
                        "support": float(nd["mean_boot_support"]),
                        "id": nd["id"]})
    return out


def dec_out(harm):
    sel_out = {}
    for (ki, kj), recs in harm["decisions"]["selections"].items():
        sel_out[f"{ki}>{kj}"] = {
            n: {"selected": r["selected"],
                "matched": bool(r["matched"]),
                "support": (None if r["support"] is None or
                            r["support"] != r["support"]
                            else float(r["support"]))}
            for n, r in recs.items()}
    return sel_out


def summarize(tag, g, group_of):
    sh = sum(1 for v in g["vertices"].values() if v["shared"])
    tri = [r for r, v in g["vertices"].items()
           if len(v["members"]) == 3]
    wrong = [r for r, v in g["vertices"].items() if v["shared"] and
             len({group_of[m] for m in v["members"].values()
                  if m in group_of}) > 1]
    print(f"{tag}: vertices={len(g['vertices'])} shared={sh} "
          f"three_atlas={len(tri)} forest={g['is_forest']} "
          f"certs={len(g['certificates'])} wrong_group={len(wrong)}")
    for c in g["certificates"]:
        print("   cert:",
              sorted(g["vertices"][c["vertex"]]["members"].items()))
    return {"vertices": len(g["vertices"]), "shared": sh,
            "three_atlas": len(tri), "forest": g["is_forest"],
            "certs": len(g["certificates"]),
            "wrong_group": len(wrong)}


# ===================== K=2 =====================
print("===== K=2 (retina_k2 panel, 20,808 genes) =====")
O2 = os.path.join(HERE, "retina_k2")
it2 = json.load(open(os.path.join(O2, "retina_k2_input_trees.json")))
truth2 = json.load(open(os.path.join(O2, "retina_k2_truth.json")))
group2 = {l: g for g, ls in truth2.items() for l in ls}
genes2 = [g for g in open(os.path.join(
    O2, "shared_genes.txt")).read().split("\n") if g]
X2 = {k: np.asarray(mmread(os.path.join(
    O2, f"{k}_bipolar.mtx")).todense()) for k in ("mac", "she")}
lab2 = {k: np.loadtxt(os.path.join(O2, f"{k}_labels.txt"), dtype=str)
        for k in ("mac", "she")}
trees2 = {k: {"parent": it2[k]["parent"],
              "children": it2[k]["children"],
              "leaves": it2[k]["leaves"]} for k in ("mac", "she")}
stab2 = {(k, n): float(v) for k in ("mac", "she")
         for n, v in it2[k]["support"].items()}
harm2 = harmonize({k: {"counts": X2[k], "labels": lab2[k],
                       "gene_names": genes2,
                       "lib": X2[k].sum(axis=1)}
                   for k in ("mac", "she")}, trees2, n_hvg=1000,
                  n_boot=200, stability=stab2)
cert2 = certified_of(harm2)
print("certified merges:")
for m in cert2:
    print("  ", sorted(m["members"].items()),
          f"support={m['support']:.3f}")
json.dump(cert2, open(os.path.join(
    O2, "retina_k2_certified_merges.json"), "w"), indent=1)
canon2 = {k: canonical_nodes(trees2[k])[1] for k in trees2}
d2 = dec_out(harm2)
g2c = quotient_assemble(trees2, canon2, d2, certified=cert2)
g2r = quotient_assemble(trees2, canon2, d2)
s2c = summarize("K2 certified", g2c, group2)
s2r = summarize("K2 raw (diagnostic)", g2r, group2)


def render(g, group_of, choice=None):
    choice = choice or {}
    kids, roots = {}, []
    for r, ps in g["parents"].items():
        if ps:
            kids.setdefault(choice.get(r, ps[0]), []).append(r)
        else:
            roots.append(r)
    root = MyNode("root")

    def emit(r, pn):
        labs = sorted(m.replace("|", "-") for m in
                      g["vertices"][r]["members"].values()
                      if m in group_of)
        node = MyNode("&".join(labs) if labs else f"__{r}__")
        pn.addkid(node)
        for c in sorted(kids.get(r, []), key=str):
            emit(c, node)
    for r in sorted(roots, key=str):
        emit(r, root)
    return root


def all_projections(g):
    certs = g["certificates"]
    if not certs or len(certs) > 8:
        return [dict()]
    vxs = [c["vertex"] for c in certs]
    return [dict(zip(vxs, combo)) for combo in
            itertools.product(*[c["minimal_parents"]
                                for c in certs])]


def truth_tree_k2():
    root = MyNode("root")
    rbc = MyNode("__rbc__")
    root.addkid(rbc)
    for l in truth2["RBC_group"]:
        rbc.addkid(MyNode(l.replace("|", "-")))
    cbc = MyNode("__cbc__")
    root.addkid(cbc)
    for grp in ("OFF", "ON"):
        gn = MyNode(f"__{grp}__")
        cbc.addkid(gn)
        for l in truth2[grp]:
            gn.addkid(MyNode(l.replace("|", "-")))
    return root


REF2 = truth_tree_k2()
sc = [triplet_scores(render(g2c, group2, ch), REF2)[0]
      for ch in all_projections(g2c)]
print(f"K2 certified TRIP vs truth: min={min(sc):.4f} "
      f"max={max(sc):.4f} over {len(sc)} projection(s)")

# ===================== K=3 =====================
print("\n===== K=3 (common panel, 18,128 genes) =====")
O3 = os.path.join(HERE, "retina_k3")
it3 = json.load(open(os.path.join(O3, "retina_k3_input_trees.json")))
truth3 = json.load(open(os.path.join(O3, "retina_k3_truth.json")))
group3 = {}
for grp, d in truth3.items():
    for ch in d["cherries"]:
        for l in ch:
            group3[l] = grp
    for l in d["mac"]:
        group3[l] = grp
DS = ("mac", "she", "mrca")
genes3 = [g for g in open(os.path.join(
    O3, "common_genes.txt")).read().split("\n") if g]
X3 = {k: np.asarray(mmread(os.path.join(
    O3, f"{k}_bipolar.mtx")).todense()) for k in DS}
lab3 = {k: np.loadtxt(os.path.join(O3, f"{k}_labels.txt"), dtype=str)
        for k in DS}
trees3 = {k: {"parent": it3[k]["parent"],
              "children": it3[k]["children"],
              "leaves": it3[k]["leaves"]} for k in DS}
stab3 = {(k, n): float(v) for k in DS
         for n, v in it3[k]["support"].items()}
harm3 = harmonize({k: {"counts": X3[k], "labels": lab3[k],
                       "gene_names": genes3,
                       "lib": X3[k].sum(axis=1)} for k in DS},
                  trees3, n_hvg=1000, n_boot=200, stability=stab3)
cert3 = certified_of(harm3)
print("certified merges:")
for m in cert3:
    print("  ", sorted(m["members"].items()),
          f"support={m['support']:.3f}")
json.dump(cert3, open(os.path.join(
    O3, "retina_k3_certified_merges.json"), "w"), indent=1)
canon3 = {k: canonical_nodes(trees3[k])[1] for k in trees3}
d3 = dec_out(harm3)
g3c = quotient_assemble(trees3, canon3, d3, certified=cert3)
g3r = quotient_assemble(trees3, canon3, d3)
s3c = summarize("K3 certified", g3c, group3)
s3r = summarize("K3 raw (diagnostic)", g3r, group3)

NAME_PAIRS = ["RBC", "BC1A", "BC1B", "BC2", "BC3A", "BC3B", "BC4",
              "BC5A", "BC5B", "BC5C", "BC5D", "BC6", "BC7"]
vid_of = {}
for r, v in g3c["vertices"].items():
    for d_, m in v["members"].items():
        vid_of[(d_, m)] = r
hits = [t for t in NAME_PAIRS
        if vid_of.get(("she", f"she|{t}")) ==
        vid_of.get(("mrca", f"mrca|{t}"))
        and vid_of.get(("she", f"she|{t}")) is not None]
print(f"K3 certified name-identical she<->mrca merges: "
      f"{len(hits)}/13 {sorted(hits)}")


def truth_tree_k3(keep_atlas=None):
    keep = (lambda l: keep_atlas is None or
            l.startswith(f"{keep_atlas}|"))
    root = MyNode("root")

    def grp_node(gname, parent):
        gn = MyNode(f"__{gname}__")
        parent.addkid(gn)
        for ch in truth3[gname]["cherries"]:
            ls = [l for l in ch if keep(l)]
            if len(ls) >= 2:
                cn = MyNode("__ch__")
                gn.addkid(cn)
                for l in ls:
                    cn.addkid(MyNode(l.replace("|", "-")))
            else:
                for l in ls:
                    gn.addkid(MyNode(l.replace("|", "-")))
        for l in truth3[gname]["mac"]:
            if keep(l):
                gn.addkid(MyNode(l.replace("|", "-")))
    grp_node("RBC_group", root)
    cbc = MyNode("__cbc__")
    root.addkid(cbc)
    grp_node("OFF", cbc)
    grp_node("ON", cbc)
    return root


def render3(g, choice, keep_atlas=None):
    keep = (lambda l: keep_atlas is None or
            l.startswith(f"{keep_atlas}|"))
    kids, roots = {}, []
    for r, ps in g["parents"].items():
        if ps:
            kids.setdefault(choice.get(r, ps[0]), []).append(r)
        else:
            roots.append(r)
    root = MyNode("root")

    def emit(r, pn):
        labs = sorted(m.replace("|", "-") for m in
                      g["vertices"][r]["members"].values()
                      if m in group3 and keep(m))
        node = MyNode("&".join(labs) if labs else f"__{r}__")
        pn.addkid(node)
        for c in sorted(kids.get(r, []), key=str):
            emit(c, node)
    for r in sorted(roots, key=str):
        emit(r, root)
    return root


for k in ("she", "mrca"):
    REF_k = truth_tree_k3(k)
    scores = [triplet_scores(render3(g3c, ch, k), REF_k)[0]
              for ch in all_projections(g3c)]
    print(f"K3 certified {k}-side TRIP: min={min(scores):.4f} "
          f"max={max(scores):.4f} over {len(scores)} projections")
REF3 = truth_tree_k3()
fs = [triplet_scores(render3(g3c, ch), REF3)[0]
      for ch in all_projections(g3c)]
print(f"K3 certified FULL-truth TRIP: min={min(fs):.4f} "
      f"max={max(fs):.4f}")
json.dump({"k2": {"certified": s2c, "raw": s2r,
                  "trip_min": min(sc), "trip_max": max(sc)},
           "k3": {"certified": s3c, "raw": s3r,
                  "name_pairs": len(hits),
                  "full_trip_min": min(fs),
                  "full_trip_max": max(fs)}},
          open(os.path.join(O3, "retina_quotient_certified.json"),
               "w"), indent=1)
print("done")
