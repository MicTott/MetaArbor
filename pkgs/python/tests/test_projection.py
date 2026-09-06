"""Projection prototype gates:

1. PARITY — the projector's vote computation equals kernel.vote_cache.
2. ACCURACY — cells from a held-out batch-shifted dataset route to the
   correct leaf; wrong-family confident assignments ~ 0.
3. ABSTENTION — cells carrying only the FAMILY program (no subtype
   signal) resolve at the family, not confidently at a leaf.
4. NOVELTY — cells from a family absent from the reference resolve
   shallow and score low on out-of-reference evidence.
5. DETERMINISM — identical inputs give identical outputs.
"""
import numpy as np
import pytest

from metaarbor import tree_from_levels
from metaarbor.kernel import lognorm, rank_normalize, variable_genes, \
    vote_cache
from metaarbor.projection import (_votes_and_maxcorr, build_projector,
                                  project)

GENES = [f"g{i}" for i in range(900)]
FAM = {f"F{i}.s{j}": f"F{i}" for i in (1, 2, 3) for j in (1, 2)}


def sim(leaves, batch_seed, n_per=60, fam_only=None, extra_latents=None):
    """Family+subtype programs shared across simulated atlases; cells in
    `fam_only` families carry NO subtype program (parent-level cells);
    `extra_latents` adds novel-family programs unknown to others."""
    rs = np.random.RandomState(batch_seed)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    pool = np.random.RandomState(98).permutation(len(GENES))
    fams = sorted(set(FAM.values()) | set(extra_latents or []))
    fam_idx = {f: pool[i * 40:(i + 1) * 40] for i, f in enumerate(fams)}
    subs = sorted(FAM)
    sub_idx = {s: pool[len(fams) * 40 + i * 15:
                       len(fams) * 40 + (i + 1) * 15]
               for i, s in enumerate(subs)}
    batch = rs.lognormal(0, 0.4, len(GENES))
    blocks, labels = [], []
    for l in leaves:
        f = FAM.get(l, l)
        mu = base.copy()
        mu[fam_idx[f]] *= np.exp(1.2)
        if l in FAM and f not in (fam_only or []):
            mu[sub_idx[l]] *= np.exp(1.3)
        lam = np.outer(rs.gamma(10, 0.1, n_per), mu * batch)
        blocks.append(rs.poisson(lam))
        labels += [l] * n_per
    return (np.vstack(blocks).astype(float), np.asarray(labels))


def ref_tree():
    return tree_from_levels([(FAM[l], l) for l in sorted(FAM)],
                            ["family", "leaf"])


@pytest.fixture(scope="module")
def world():
    tree = ref_tree()
    xa, la = sim(sorted(FAM), batch_seed=11)
    xb, lb = sim(sorted(FAM), batch_seed=12)
    proj = build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES, "name": "A"},
         {"counts": xb, "labels": lb, "gene_names": GENES, "name": "B"}],
        tree, cap_per_leaf=40, seed=0)
    return tree, proj


def test_vote_parity_with_kernel():
    xa, la = sim(sorted(FAM), batch_seed=1, n_per=20)
    xq, _ = sim(sorted(FAM), batch_seed=2, n_per=10)
    ea, eq = lognorm(xa), lognorm(xq)
    hvg = variable_genes(eq, ea, GENES, 400)
    qn, rn = rank_normalize(eq[:, hvg]), rank_normalize(ea[:, hvg])
    ours = _votes_and_maxcorr(qn, rn, la)
    ref = vote_cache(qn, rn, la)
    assert ours["leaves"] == ref["leaves"]
    assert np.allclose(ours["V"], ref["V"])
    assert np.allclose(ours["leaf_sizes"], ref["leaf_sizes"])


def test_heldout_accuracy_and_no_wrong_family(world):
    tree, proj = world
    xq, lq = sim(sorted(FAM), batch_seed=33)      # unseen batch
    out = project(proj, xq, GENES)
    leaf_acc = np.mean(out["best_leaf"] == lq)
    fam_true = np.asarray([FAM[l] for l in lq])
    fam_pred = np.asarray([FAM.get(b, b) for b in out["best_leaf"]])
    fam_acc = np.mean(fam_pred == fam_true)
    assert leaf_acc >= 0.9, leaf_acc
    assert fam_acc >= 0.98, fam_acc
    # a cell may abstain to its family, but a cell RESOLVED to leaf
    # depth must essentially never sit in the wrong family
    deep = out["resolved_depth"] >= 2
    wrong_fam_deep = np.mean(fam_pred[deep] != fam_true[deep]) \
        if deep.any() else 0.0
    assert wrong_fam_deep <= 0.01, wrong_fam_deep
    # most held-out cells should resolve to full depth in this easy sim
    assert np.mean(deep) >= 0.7, np.mean(deep)


def test_family_only_cells_abstain_at_family(world):
    tree, proj = world
    # F1 cells carry ONLY the family program: subtype split unsupported
    xq, lq = sim([l for l in sorted(FAM) if FAM[l] == "F1"],
                 batch_seed=44, fam_only=["F1"])
    out = project(proj, xq, GENES)
    fam_pred = np.asarray([FAM.get(b, b) for b in out["best_leaf"]])
    assert np.mean(fam_pred == "F1") >= 0.95        # family still right
    # the confident walk should STOP at the family for most cells
    at_family = out["resolved_node"] == "family:F1"
    assert np.mean(at_family) >= 0.6, np.mean(at_family)
    # and essentially never claim a leaf confidently
    deep = out["resolved_depth"] >= 2
    assert np.mean(deep) <= 0.25, np.mean(deep)


def test_novel_family_shallow_and_low_oor(world):
    tree, proj = world
    xq, _ = sim(["FX"], batch_seed=55, extra_latents=["FX"])
    out_nov = project(proj, xq, GENES)
    xk, _ = sim(sorted(FAM), batch_seed=56)
    out_in = project(proj, xk, GENES)
    # novel-family cells resolve shallower than in-reference cells
    assert (np.median(out_nov["resolved_depth"]) <
            np.median(out_in["resolved_depth"]))
    # out-of-reference evidence separates novel from in-reference cells
    from metaarbor.kernel import auroc
    score = np.concatenate([out_in["top_leaf_mean"],
                            out_nov["top_leaf_mean"]])
    is_in = np.concatenate([np.ones(len(out_in["top_leaf_mean"])),
                            np.zeros(len(out_nov["top_leaf_mean"]))])
    assert auroc(score, is_in == 1) >= 0.8
    # novel cells never resolve confidently to a leaf
    assert np.mean(out_nov["resolved_depth"] >= 2) <= 0.1


def test_determinism(world):
    tree, proj = world
    xq, _ = sim(sorted(FAM), batch_seed=77, n_per=15)
    a = project(proj, xq, GENES)
    b = project(proj, xq, GENES)
    assert list(a["best_leaf"]) == list(b["best_leaf"])
    assert np.allclose(a["path_score"], b["path_score"])
    assert list(a["resolved_node"]) == list(b["resolved_node"])
