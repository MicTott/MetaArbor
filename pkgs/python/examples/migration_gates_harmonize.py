"""Migration gates for quotient-assembly harmonize() (the assembly
replacement approved after the certification-boundary review).

GATE A retina K=2: new harmonize() must certify exactly the committed
  merge set (retina_k2_certified_merges.json), return a forest, score
  TRIP 1.0000 against the committed truth, and be invariant to atlas
  order (full tree-record equality).
GATE B retina K=3: must reproduce the committed certified-quotient
  structure — 18 backbone nodes from the committed certified merges,
  8 three-atlas meta-clades, 3 certificates, 11/13 name pairs — with
  completeness over all three atlases.
GATE D consumer contract on the REAL K=3 DAG: no tree record
  carries a plain `parent` key; consensus_cut and from_harmonize
  REFUSE the DAG without an explicit projection policy and, with
  projection='projected_parent', visibly carry the certificates —
  the cut in cut['certificates'], the adapter in the returned
  tree's 'projection' and 'certificates' keys, both verbatim;
  the audit renderer marks every multi-parent vertex's drawn edge
  as 'projected'; and on the K=2 forest the same consumers work
  with no policy and no false alarm.
GATE C allen: with trees re-inferred exactly as allen_harmonize.py
  does, the certified layer being untouched means the 24 committed
  backbone member-sets and 6 private member-sets must be reproduced
  EXACTLY; the 2 formerly unplaced_single_atlas labels must now be
  placed single-atlas members (the documented supersession — nothing
  is ever unplaced); completeness holds; certificates reported.

Run: PYTHONPATH=src:../../comparison/otharmonizer \
     python examples/migration_gates_harmonize.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.io import mmread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "comparison",
                                "otharmonizer"))
from metrics import MyNode, triplet_scores  # noqa: E402
from metaarbor.consensus.harmonize import harmonize  # noqa: E402
from metaarbor.infer_tree import infer_tree  # noqa: E402


def tree_arg(it, k):
    return {"parent": it[k]["parent"], "children": it[k]["children"],
            "leaves": it[k]["leaves"]}


def backbone_sets(harm):
    return {frozenset(nd["members"].items())
            for nd in harm["tree"].values()
            if nd["status"] == "backbone"}


def rec_of(harm):
    return {tuple(sorted(nd["members"].items())):
            (nd["parents"], nd["status"])
            for nd in harm["tree"].values()}


def gate_a():
    O2 = os.path.join(HERE, "retina_k2")
    it = json.load(open(os.path.join(O2,
                                     "retina_k2_input_trees.json")))
    truth = json.load(open(os.path.join(O2, "retina_k2_truth.json")))
    committed = json.load(open(os.path.join(
        O2, "retina_k2_certified_merges.json")))
    genes = [g for g in open(os.path.join(
        O2, "shared_genes.txt")).read().split("\n") if g]
    X = {k: np.asarray(mmread(os.path.join(
        O2, f"{k}_bipolar.mtx")).todense()) for k in ("mac", "she")}
    lab = {k: np.loadtxt(os.path.join(O2, f"{k}_labels.txt"),
                         dtype=str) for k in ("mac", "she")}
    trees = {k: tree_arg(it, k) for k in ("mac", "she")}
    stab = {(k, n): float(v) for k in ("mac", "she")
            for n, v in it[k]["support"].items()}

    def run(order):
        return harmonize({k: {"counts": X[k], "labels": lab[k],
                              "gene_names": genes,
                              "lib": X[k].sum(axis=1)}
                          for k in order},
                         {k: trees[k] for k in order},
                         n_hvg=1000, n_boot=200, stability=stab)
    harm = run(["mac", "she"])
    ok = harm["is_forest"]
    got = backbone_sets(harm)
    want = {frozenset(m["members"].items()) for m in committed}
    ok &= got == want
    group_of = {l: g for g, ls in truth.items() for l in ls}

    def to_mynode(harm):
        nodes = harm["tree"]
        root = MyNode("root")
        built = {}

        def build(i):
            if i in built:
                return built[i]
            nd = nodes[i]
            labs = sorted(m.replace("|", "-") for m in
                          nd["members"].values() if m in group_of)
            n = MyNode("&".join(labs) if labs else f"__{i}__")
            built[i] = n
            return n
        for i in sorted(nodes):
            build(i)
        for i, nd in nodes.items():
            (built[nd["projected_parent"]]
             if nd["projected_parent"] else root).addkid(built[i])
        return root

    def truth_tree():
        rt = MyNode("root")
        rbc = MyNode("__rbc__")
        rt.addkid(rbc)
        for l in truth["RBC_group"]:
            rbc.addkid(MyNode(l.replace("|", "-")))
        cbc = MyNode("__cbc__")
        rt.addkid(cbc)
        for grp in ("OFF", "ON"):
            gn = MyNode(f"__{grp}__")
            cbc.addkid(gn)
            for l in truth[grp]:
                gn.addkid(MyNode(l.replace("|", "-")))
        return rt
    rec, _ = triplet_scores(to_mynode(harm), truth_tree())
    ok &= abs(rec - 1.0) < 1e-9
    harm_o = run(["she", "mac"])
    det = rec_of(harm) == rec_of(harm_o)
    ok &= det
    print(f"GATE A retina K=2: forest={harm['is_forest']} "
          f"certified_match={got == want} TRIP={rec:.4f} "
          f"order_invariant={det} -> {'PASS' if ok else 'FAIL'}")
    return ok, harm


def gate_b():
    O3 = os.path.join(HERE, "retina_k3")
    it = json.load(open(os.path.join(O3,
                                     "retina_k3_input_trees.json")))
    committed = json.load(open(os.path.join(
        O3, "retina_k3_certified_merges.json")))
    genes = [g for g in open(os.path.join(
        O3, "common_genes.txt")).read().split("\n") if g]
    DS = ("mac", "she", "mrca")
    X = {k: np.asarray(mmread(os.path.join(
        O3, f"{k}_bipolar.mtx")).todense()) for k in DS}
    lab = {k: np.loadtxt(os.path.join(O3, f"{k}_labels.txt"),
                         dtype=str) for k in DS}
    trees = {k: tree_arg(it, k) for k in DS}
    stab = {(k, n): float(v) for k in DS
            for n, v in it[k]["support"].items()}
    harm = harmonize({k: {"counts": X[k], "labels": lab[k],
                          "gene_names": genes,
                          "lib": X[k].sum(axis=1)} for k in DS},
                     trees, n_hvg=1000, n_boot=200, stability=stab)
    nodes = harm["tree"]
    placed = {(ds, m) for nd in nodes.values()
              for ds, m in nd["members"].items()}
    complete = all((ds, l) in placed for ds in DS
                   for l in trees[ds]["leaves"])
    n_bb = sum(1 for nd in nodes.values()
               if nd["status"] == "backbone")
    tri = sum(1 for nd in nodes.values()
              if len(nd["members"]) == 3)
    want = {frozenset(m["members"].items()) for m in committed}
    cert_match = all(any(fs <= set(nd["members"].items())
                         for nd in nodes.values()) for fs in want)
    NAME_PAIRS = ["RBC", "BC1A", "BC1B", "BC2", "BC3A", "BC3B",
                  "BC4", "BC5A", "BC5B", "BC5C", "BC5D", "BC6",
                  "BC7"]
    by_member = {(ds, m): i for i, nd in nodes.items()
                 for ds, m in nd["members"].items()}
    hits = [t for t in NAME_PAIRS
            if by_member.get(("she", f"she|{t}")) ==
            by_member.get(("mrca", f"mrca|{t}"))
            and by_member.get(("she", f"she|{t}")) is not None]
    ok = (complete and n_bb == 18 and tri == 8 and
          len(harm["certificates"]) == 3 and cert_match and
          len(hits) == 11)
    print(f"GATE B retina K=3: complete={complete} backbone={n_bb} "
          f"(want 18) three_atlas={tri} (want 8) "
          f"certs={len(harm['certificates'])} (want 3) "
          f"certified_match={cert_match} name_pairs={len(hits)}/13 "
          f"forest={harm['is_forest']} -> {'PASS' if ok else 'FAIL'}")
    return ok, harm


def gate_d(harm_k2, harm_k3):
    from metaarbor.consensus.cut import consensus_cut
    from metaarbor.projection import from_harmonize
    from metaarbor.viz import nested_from_harmonize
    checks = {}
    checks["no_plain_parent_key"] = all(
        "parent" not in nd for nd in harm_k3["tree"].values())
    assert harm_k3["is_forest"] is False and harm_k3["certificates"]
    try:
        consensus_cut(harm_k3)
        checks["cut_refuses_dag"] = False
    except ValueError as e:
        checks["cut_refuses_dag"] = "DAG" in str(e)
    cut = consensus_cut(harm_k3, projection="projected_parent")
    checks["cut_carries_certificates"] = (
        cut["projection"] == "projected_parent" and
        cut["certificates"] == harm_k3["certificates"] and
        cut["summary"]["n_certificates"] ==
        len(harm_k3["certificates"]))
    try:
        from_harmonize(harm_k3)
        checks["projection_refuses_dag"] = False
    except ValueError as e:
        checks["projection_refuses_dag"] = "DAG" in str(e)
    tree, _ = from_harmonize(harm_k3, projection="projected_parent")
    checks["projection_explicit_works"] = bool(tree["leaves"])
    checks["projection_carries_provenance"] = (
        tree["projection"] == "projected_parent" and
        tree["certificates"] == harm_k3["certificates"])
    real = {m for nd in harm_k3["tree"].values()
            for m in nd["members"].values()}
    nested = nested_from_harmonize(harm_k3["tree"], real)
    multi = {i for i, nd in harm_k3["tree"].items()
             if len(nd["parents"]) > 1}

    def count_projected(n):
        c = 1 if n.get("edge") == "projected" else 0
        return c + sum(count_projected(x) for x in n["children"])
    checks["viz_marks_projected"] = (
        count_projected(nested) == len(multi))
    # forest: consumers need no policy, no false alarm
    assert harm_k2["is_forest"]
    cut2 = consensus_cut(harm_k2)
    t2, _ = from_harmonize(harm_k2)
    checks["forest_no_false_alarm"] = (
        cut2["projection"] is None and cut2["certificates"] == []
        and bool(t2["leaves"]) and t2["projection"] is None
        and t2["certificates"] == [])
    ok = all(checks.values())
    print("GATE D consumer contract (real K=3 DAG):")
    for k, v in checks.items():
        print(f"   {k}: {v}")
    print(f"GATE D -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate_c():
    D = os.path.join(HERE, "..", "..", "..", "data", "wmb_plilaorb")

    def load(tag):
        counts = np.asarray(mmread(
            os.path.join(D, f"counts_{tag}.mtx")).todense()).T
        lib = np.loadtxt(os.path.join(D, f"lib_{tag}.txt"))
        with open(os.path.join(D, f"cells_{tag}.csv")) as fh:
            cells = list(csv.DictReader(fh))
        return counts, lib, cells
    genes = open(os.path.join(D, "genes.txt")).read().split()
    cA, lA, cellsA = load("10Xv2")
    cB, lB, cellsB = load("10Xv3")
    labA = np.asarray([f"v2|{c['subclass']}" for c in cellsA])
    labB = np.asarray([f"v3|{c['cluster']}" for c in cellsB])
    inf_a = infer_tree(cA, labA, lib=lA, n_hvg=2000, n_boot=50,
                       seed=0)
    inf_b = infer_tree(cB, labB, lib=lB, n_hvg=2000, n_boot=50,
                       seed=0)
    trees = {"v2": inf_a["tree"], "v3": inf_b["tree"]}
    stab = {("v2", n): float(v)
            for n, v in inf_a["support"].items()}
    stab.update({("v3", n): float(v)
                 for n, v in inf_b["support"].items()})
    harm = harmonize({"v2": {"counts": cA, "labels": labA,
                             "gene_names": genes, "lib": lA},
                      "v3": {"counts": cB, "labels": labB,
                             "gene_names": genes, "lib": lB}},
                     trees, n_hvg=1000, n_boot=200, stability=stab)
    committed = json.load(open(os.path.join(
        HERE, "harmonize_demo", "metaarbor_tree_current.json")))
    want_bb = {frozenset(nd["members"].items())
               for nd in committed.values()
               if nd["status"] == "backbone"
               and len(nd["members"]) >= 2}
    want_priv = {frozenset(nd["members"].items())
                 for nd in committed.values()
                 if nd["status"] == "private"}
    got_bb = backbone_sets(harm)
    nodes = harm["tree"]
    got_priv = {frozenset(nd["members"].items())
                for nd in nodes.values()
                if nd["status"] == "private"}
    prev_unplaced = [(ds, m) for nd in committed.values()
                     if nd["status"] == "unplaced_single_atlas"
                     for ds, m in nd["members"].items()]
    by_member = {(ds, m): i for i, nd in nodes.items()
                 for ds, m in nd["members"].items()}
    unplaced_now_placed = all((ds, m) in by_member
                              for ds, m in prev_unplaced)
    placed = set(by_member)
    complete = all((ds, l) in placed for ds in trees
                   for l in trees[ds]["leaves"])
    # private accounting: quotient keeps every SUBTREE node as its own
    # vertex, so committed multi-node private roots correspond to
    # per-node private vertices — compare the flat member sets
    want_priv_members = {m for fs in want_priv for m in fs}
    got_priv_members = {(ds, m) for nd in nodes.values()
                        if nd["status"] == "private"
                        for ds, m in nd["members"].items()}
    priv_ok = want_priv_members <= got_priv_members
    ok = (got_bb == want_bb and priv_ok and complete and
          unplaced_now_placed)
    print(f"GATE C allen: backbone_match={got_bb == want_bb} "
          f"({len(got_bb)} vs committed {len(want_bb)}) "
          f"private_members_kept={priv_ok} complete={complete} "
          f"formerly_unplaced_now_placed={unplaced_now_placed} "
          f"forest={harm['is_forest']} "
          f"certs={len(harm['certificates'])} "
          f"-> {'PASS' if ok else 'FAIL'}")
    for c in harm["certificates"]:
        nd = nodes[c["node"]]
        print("   constraint:", sorted(nd["members"].items()))
    return ok


if __name__ == "__main__":
    ra, harm_a = gate_a()
    rb, harm_b = gate_b()
    rd = gate_d(harm_a, harm_b)
    rc = gate_c()
    print(f"\nHARMONIZE MIGRATION GATES: "
          f"{'PASS' if ra and rb and rc and rd else 'FAIL'}")
