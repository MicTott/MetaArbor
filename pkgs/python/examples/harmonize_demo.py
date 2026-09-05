"""The K=2 visual proof gate for harmonize(): four cases whose merged
trees must VISIBLY show the intended behavior (reviewer-specified):

  a) equivalent labels collapse to one node with aliases
  b) a coarse type becomes an internal node above finer types
  c) a genuine private type remains a private branch
  d) incompatible evidence becomes conflicts, not forced splits

Each case renders input trees beside the merged output with every
original label. Assertions accompany every case — the figure must show
it AND the structure must verify it.

Usage: python examples/harmonize_demo.py
"""
import os

import numpy as np

from metaarbor import tree_from_levels
from metaarbor.consensus.harmonize import harmonize
from metaarbor.consensus.plot_reconciled import plot_reconciled_tree
from metaarbor.style import save_pub

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "harmonize_demo")
os.makedirs(OUT, exist_ok=True)
GENES = [f"g{i}" for i in range(900)]


def simulate(key, leaves, family_of, n_per=70, batch_seed=0, sub_lfc=1.3,
             latent_map=None, label_level=None):
    """Family+subtype expression for `leaves` (latent programs shared via
    latent_map so different atlases' cells come from the same biology)."""
    rs = np.random.RandomState(batch_seed)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    pool = np.random.RandomState(98).permutation(len(GENES))
    fams = sorted(set(family_of.values()))
    fam_idx = {f: pool[i * 40:(i + 1) * 40] for i, f in enumerate(fams)}
    latents = sorted(set((latent_map or {}).get(l, l) for l in leaves))
    sub_idx = {s: pool[len(fams) * 40 + i * 15:
                       len(fams) * 40 + (i + 1) * 15]
               for i, s in enumerate(latents)}
    batch = rs.lognormal(0, 0.4, len(GENES))
    blocks, labels = [], []
    for l in leaves:
        lat = (latent_map or {}).get(l, l)
        mu = base.copy()
        mu[fam_idx[family_of[l]]] *= np.exp(1.2)
        mu[sub_idx[lat]] *= np.exp(sub_lfc)
        lam = np.outer(rs.gamma(10, 0.1, n_per), mu * batch)
        blocks.append(rs.poisson(lam))
        lab = family_of[l] if label_level == "family" else l
        labels += [lab] * n_per
    return {"counts": np.vstack(blocks).astype(float),
            "labels": np.asarray(labels), "gene_names": GENES}


def run_case(name, datasets, trees, checks, names):
    print(f"\n=== case {name} ===")
    harm = harmonize(datasets, trees, n_hvg=700, n_boot=100,
                     trust_trees=True)
    nodes = harm["tree"]
    by_status = {}
    for nd in nodes.values():
        by_status[nd["status"]] = by_status.get(nd["status"], 0) + 1
    print("  nodes:", by_status, "| conflicts:",
          len([c for c in harm["conflicts"]
               if c.get("class") == "genuine_conflict"]),
          "| affiliates:", len(harm["affiliates"]))
    ok = checks(harm)
    print("  CASE:", "PASS" if ok else "FAIL")
    fig, _ = plot_reconciled_tree(harm, trees, dataset_names=names)
    save_pub(fig, os.path.join(OUT, name), formats=("png",), dpi=200)
    return ok


# ---- case a: equivalent labels collapse with aliases -----------------------
def case_a():
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2) for j in (1, 2)}
    leaves = sorted(fam)
    tr = {k: tree_from_levels([(f"{k}|{fam[l]}", f"{k}|{l}")
                               for l in leaves], ["family", "leaf"])
          for k in ("A", "B")}
    ds = {}
    for i, k in enumerate(("A", "B")):
        d = simulate(k, leaves, fam, batch_seed=i)
        d["labels"] = np.asarray([f"{k}|{l}" for l in d["labels"]])
        ds[k] = d

    def checks(h):
        # every leaf meta-clade holds BOTH atlases' labels as aliases
        leafnodes = [nd for nd in h["tree"].values()
                     if nd["status"] == "backbone" and not nd["children"]]
        two_alias = sum(len(nd["aliases"]) >= 2 for nd in leafnodes)
        return two_alias >= 4 and all(
            len(nd["members"]) == 2 for nd in leafnodes)
    return run_case("a_equivalent_collapse", ds, tr, checks,
                    {"A": "atlas A", "B": "atlas B"})


# ---- case b: coarse label becomes an internal node above fine types --------
def case_b():
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2, 3) for j in (1, 2, 3)}
    leaves = sorted(fam)
    tr_b = tree_from_levels([(f"B|{fam[l]}", f"B|{l}") for l in leaves],
                            ["family", "leaf"])
    tr_a = tree_from_levels([(f"A|F{i}",) for i in (1, 2, 3)], ["leaf"])
    dA = simulate("A", leaves, fam, batch_seed=3, label_level="family")
    dA["labels"] = np.asarray([f"A|{l}" for l in dA["labels"]])
    dB = simulate("B", leaves, fam, batch_seed=4)
    dB["labels"] = np.asarray([f"B|{l}" for l in dB["labels"]])
    ds = {"A": dA, "B": dB}

    def checks(h):
        # coarse A labels form internal meta-clades; fine B structure sits
        # BENEATH them (as backbone if reciprocal, else single_atlas)
        fams_ok, fine_below = 0, 0
        for nd in h["tree"].values():
            if nd["status"] == "backbone" and "A" in nd["members"]:
                if nd["children"]:
                    fams_ok += 1
                    fine_below += sum(
                        "B" in h["tree"][c]["members"]
                        for c in nd["children"])
        return fams_ok == 3 and fine_below >= 6
    return run_case("b_coarse_above_fine", ds,
                    {"A": tr_a, "B": tr_b}, checks,
                    {"A": "coarse atlas (3 types)",
                     "B": "fine atlas (9 types)"})


# ---- case c: genuine private branch persists -------------------------------
def case_c():
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2, 3) for j in (1, 2)}
    leaves = sorted(fam)
    leaves_a = [l for l in leaves if not l.startswith("F3")]
    tr = {}
    tr["A"] = tree_from_levels([(f"A|{fam[l]}", f"A|{l}")
                                for l in leaves_a], ["family", "leaf"])
    tr["B"] = tree_from_levels([(f"B|{fam[l]}", f"B|{l}")
                                for l in leaves], ["family", "leaf"])
    dA = simulate("A", leaves_a, fam, batch_seed=5)
    dA["labels"] = np.asarray([f"A|{l}" for l in dA["labels"]])
    dB = simulate("B", leaves, fam, batch_seed=6)
    dB["labels"] = np.asarray([f"B|{l}" for l in dB["labels"]])
    ds = {"A": dA, "B": dB}

    def checks(h):
        priv = [nd for nd in h["tree"].values()
                if nd["status"] == "private"]
        return any("F3" in a for nd in priv for a in nd["aliases"])
    return run_case("c_private_branch", ds, tr, checks,
                    {"A": "atlas A (no F3)", "B": "atlas B (has F3)"})


# ---- case d: incompatible topology -> conflicts, not forced splits ---------
def case_d():
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2) for j in (1, 2)}
    leaves = sorted(fam)
    # B's tree deliberately mis-groups across families
    wrong = {"F1.s1": "G1", "F2.s1": "G1", "F1.s2": "G2", "F2.s2": "G2"}
    tr = {"A": tree_from_levels([(f"A|{fam[l]}", f"A|{l}")
                                 for l in leaves], ["family", "leaf"]),
          "B": tree_from_levels([(f"B|{wrong[l]}", f"B|{l}")
                                 for l in leaves], ["family", "leaf"])}
    dA = simulate("A", leaves, fam, batch_seed=7)
    dA["labels"] = np.asarray([f"A|{l}" for l in dA["labels"]])
    dB = simulate("B", leaves, fam, batch_seed=8)
    dB["labels"] = np.asarray([f"B|{l}" for l in dB["labels"]])
    ds = {"A": dA, "B": dB}

    def checks(h):
        # the true signature of incompatible grouping layers: leaf
        # correspondences survive; NEITHER atlas's family layer is
        # adopted (no family meta-clade forms); the leaves sit in a ROOT
        # POLYTOMY; both atlases' family nodes are reported as unplaced
        # internal structure. Abstention -> polytomy, never forced splits.
        leaf_bb = [nd for nd in h["tree"].values()
                   if nd["status"] == "backbone" and
                   len(nd["members"]) == 2 and not nd["children"]]
        root_poly = sum(nd["parent"] is None for nd in leaf_bb)
        fam_metaclades = [nd for nd in h["tree"].values()
                         if nd["status"] == "backbone" and nd["children"]]
        unplaced = sum(len(v)
                       for v in h["unplaced_internals"].values())
        return (len(leaf_bb) >= 3 and root_poly >= 3 and
                not fam_metaclades and unplaced >= 3)
    return run_case("d_conflict_polytomy", ds, tr, checks,
                    {"A": "atlas A (true families)",
                     "B": "atlas B (crossed families)"})


if __name__ == "__main__":
    results = {"a": case_a(), "b": case_b(), "c": case_c(), "d": case_d()}
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"case {k}: {'PASS' if v else 'FAIL'}")
    print("figures in", OUT)
